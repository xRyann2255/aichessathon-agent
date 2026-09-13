"""The search as numba kernels over the board in `kernel.py`.

One root call per window of the deepening loop in `agent.py`: `search_root` orders the
root moves, searches each with `negamax` inside the window it is given, and returns the
best. Leaves are settled by `quiesce`. Both are fail-soft alpha-beta over pseudo-legal
moves that are made and then tested for leaving the king in check.

Principal variation search: the first move takes the full window. Later moves probe a
zero window, then take the full window if they raise alpha without reaching beta.
Null moves are confined to zero-window nodes.

Moves in the main search are tried in stages: the transposition table's move, then the
captures and promotions ordered by the victim's worth plus a capture history, then the
quiet moves with the killer of the ply first and the rest by a butterfly history plus a
continuation history, keyed by the move that led to the node and the quiet move's piece
and target (`cont_index`), so a quiet's score is conditioned on the move it answers. The
quiet moves are only generated when nothing before them cut the node. The histories use
the gravity update, which keeps every entry within HISTORY_MAX, and the move that cut a
node gains a bonus while the moves tried before it lose as much, in the butterfly and
the continuation tables alike. The move that led to a node travels in the picker's
slice of the ply (`PK_PREV`), written by the parent before each child; it is 0 after a
null move and at the root, and a 0 skips the read and the update.

The transposition table is a flat array of one int64 word per entry, indexed by the low
bits of the hash; the word packs an 18-bit tag from the hash's high bits with the move,
depth, bound, generation and score. An entry is replaced when it is empty, holds the
same position, is from an earlier move of the game, or holds a shallower search. Mate
scores are stored relative to the node. A tag match on another position happens once in
262,144 probes of a full entry; the stored move is re-read from the board and checked
against the position before it is tried, so a false hit costs at most a wrong score at a
depth that happens to satisfy the probe.

Quiescence probes the same table before standing pat and stores its result at depth 0.

Late move reductions: from LMR_MIN_DEPTH, a quiet move tried after the first move of a
node, the killer excepted and never in check, is searched first to a depth reduced by
the log-by-log table `lmr_table` builds, on a zero window, and only to full depth when
that search beats alpha. The reduced depth never falls below one, so no child drops
into quiescence.

Reverse futility: below RFP_MAX_DEPTH, a node out of check whose static evaluation
clears beta by a margin growing with depth returns at once, before the null move and
before any move is generated; the null move then reuses that evaluation.

A repetition of a position on the search path or earlier in the game, and a halfmove
clock at 100 or more without checkmate, score as a draw before the node is evaluated:
zero when the game is level, and otherwise the contempt margin against the side the
root's last score says stands better (`draw_score`), so the search plays on when ahead.
The game's earlier positions arrive as hashes from `agent.py`, oldest first.

Time: the clock is read through ctypes once every `stride` nodes (a power of two), and a
deadline sets the stop flag, which every node checks on entry and after every child
returns; the unwinding scores are garbage and the root ignores them. The read is gated
on the node count so LLVM cannot hoist it out of the search, and the flag lives in the
search state array that every node also writes its node count to.

Every array here reaches the kernels as an argument, for the reason `kernel.py` gives:
numba would freeze a global array into the compiled code. The arrays travel behind one
`Search` object, so a recursive call passes one word (numba 0.67's StructRefModel is one
meminfo pointer) instead of seven per array (25
array arguments measured 26 ns a call against 1.2 behind a jitclass,
`notes/measurements/2026-09-06-kernel-profile.md`). The hot kernels take the object
too, through the `_s` entries below `new_search`, and read their arrays from it inside
their own bodies, so a node pushes the handle rather than the descriptors of every
array a call reads. The object holds references to the
arrays `agent.py` allocates, so a write from Python reaches the kernels as before. The
search state `ss` carries the counters, the stop flag, the stride mask, the deadline and
the results.
"""

from __future__ import annotations

import ctypes
import math
import sys
from typing import Any

import llvmlite.binding as llvm  # type: ignore[import-untyped]
import numpy as np
from llvmlite import ir as llvm_ir
from numba import int8, int16, int32, int64, njit, types, uint8, uint64
from numba.core import cgutils, imputils
from numba.core.runtime.nrtdynmod import meminfo_data_ty
from numba.experimental import structref
from numba.extending import intrinsic, lower_getattr_generic

import bitboards as tables
from kernel import (
    A1,
    A8,
    BISHOP,
    BLACK,
    BP,
    CACHE,
    CASTLING,
    DOUBLE_PUSH,
    EMPTY,
    EN_PASSANT,
    H1,
    H8,
    HALF,
    HASH,
    KING,
    KNIGHT,
    MAX_PLY,
    MOVES_PER_PLY,
    NORMAL,
    OCC_ALL,
    OCC_BLACK,
    OCC_WHITE,
    ONE,
    PAWN,
    PLY,
    QUEEN,
    ROOK,
    ST_EG,
    ST_MG,
    ST_PHASE,
    STM,
    TESTING,
    UNDO_EG,
    UNDO_FIELDS,
    UNDO_MG,
    UNDO_PHASE,
    WHITE,
    WP,
    Bitboards,
    Int32s,
    Ints,
    Uint8s,
    attacked,
    attacked_body,
    attackers_to,
    bishop_attacks,
    check_masks_body,
    encode,
    evasion_targets_body,
    gen_noisy_body,
    gen_quiet_body,
    gives_check_body,
    in_check,
    jit,
    last_was_null,
    lsb,
    make_body,
    make_null,
    may_resolve,
    move_captured,
    move_flag,
    move_piece,
    move_promotion,
    move_source,
    move_target,
    needs_legality_test,
    pinned_pieces_body,
    popcount,
    push_promotions,
    push_targets,
    rook_attacks,
    see_ge_body,
    unmake_body,
    unmake_null,
)
from nnue import Int16s, evaluate_net, nn_apply_body, nn_make_body, nn_null

MATE = 100_000
# The net is skipped when the tables' score sits this far outside the window.
LAZY_MARGIN = 400
INFINITY = 1_000_000
MAX_DEPTH = 64
MATE_BOUND = MATE - 4 * MAX_DEPTH

# The search state array.
SS_NODES = 0  # nodes searched since the root call began
SS_STOP = 1  # set once the deadline has passed; every node returns at once after that
SS_STRIDE_MASK = 2  # the clock is read when nodes & mask == 0
SS_DEADLINE = 3  # in clock ticks
SS_ITERATION_BEST = 4  # the best root move of the iteration in progress, or 0
SS_GAME_LENGTH = 5  # how many earlier game positions `game` holds
SS_GENERATION = 6  # the transposition table's generation, one per get_move
SS_ROOT_COUNT = 7  # root moves in `root`
SS_ROOT_DEPTH = 8  # the depth of the iteration in progress, for the check extension
SS_CONTEMPT = 9  # the draw margin for the root side in centipawns: positive when it stands better
SS_LMP_DEPTH = 10  # move-count pruning applies at this depth and under; 0 turns it off
SS_SEE_PRUNE = 12  # 1: quiescence skips a capture the exchanges lose by SEE_QUIESCE_MARGIN; 0 off
SS_SEE_QUIET_DEPTH = 13  # quiet moves the exchanges lose are skipped at this depth and under; 0 off
SS_BAD_CAPTURES = 14  # 1: captures the exchanges lose wait until after the quiet moves; 0 off
SS_NODE_LIMIT = 15  # stop at this many nodes a move when not 0: the fixed-node match mode
SS_PROBCUT_ELIGIBLE = 16  # nodes admitted to the one-capture probe
SS_PROBCUT_ATTEMPTS = 17  # legal captures sent to quiescence
SS_PROBCUT_VERIFICATIONS = 18  # captures that reached the reduced main search
SS_PROBCUT_CUTS = 19  # reduced searches that cleared the raised window
SS_CONT_HISTORY = 20  # 1: the continuation history is read and updated; 0 leaves it out
SS_PV_LEN = 21  # plies of the previous iteration's line kept in pv (#503)
SS_SIZE = 22

# One capture, first at depth eight, checked four plies shallower. The margin is on
# the shipped net's centipawn scale; its pruning risk is measured against a deeper search.
PROBCUT_MIN_DEPTH = 8
PROBCUT_REDUCTION = 4
PROBCUT_MARGIN = 180

# Ordering. The victim's worth on Ethereal's scale for captures in the main search, the
# attacker as a small tie-break and the capture history on top; killers and history for
# quiet moves; and the plain most-valuable-victim order in quiescence.
MVV_AUGMENT = (0, 2400, 2400, 4800, 9600, 0)  # by captured piece type, pawn to king
ORDER_VALUE = (1, 3, 3, 5, 9, 20)
HISTORY_MAX = 16384
HISTORY_BONUS_CAP = 1200
# #501: the malus on a quiet that failed shrinks with how late it sat in the tried order,
# reaching zero at HISTORY_MALUS_SPAN tried quiets, so a late move is blamed less than an
# early one (Stockfish, "Decrease all stats malus according to move count", 2025-01-26,
# +3.8 Elo at its short control). 0 turns the arm off: every failed quiet loses the bonus.
HISTORY_MALUS_SPAN = 24
KILLER_SCORE = 1 << 20

# Late move reductions: Ethereal's table, 0.7844 + ln(depth) * ln(moves) / 2.4696,
# truncated on store, which reduces one to three plies at depth 5 to 9. A quiet move is
# reduced from LMR_MIN_DEPTH once LMR_FULL_MOVES moves of the node have been searched.
LMR_BASE = 0.7844
LMR_DIVISOR = 2.4696
LMR_TABLE_SIDE = 64
LMR_MIN_DEPTH = 3
LMR_FULL_MOVES = 1
# #500: a node counts its children that failed high, and a later move is reduced one ply
# more once that count passes CUTOFF_COUNT_THRESHOLD (Stockfish, "Refine reduction logic
# based on next-ply cutoff count", 2025-12-28, +2.9 Elo at its short control). 0 turns
# the arm off.
CUTOFF_COUNT_THRESHOLD = 3
# Late noisy moves (#520): a capture the exchange loses, tried after the quiet moves in
# the PICK_BAD stage, is reduced by the quiet table's count less one ply, never under
# zero; a winning or level capture is never reduced (Stockfish's capture reduction,
# CPW's page). LMR_NOISY at 0 keeps the reduction to the quiet stage.
LMR_NOISY = 1
# Move-count pruning (late move pruning): at depth LMP_MAX_DEPTH and under, once a node out
# of check with a score better than mate against it has searched lmp_limit(depth) moves
# (Alexandria's non-improving table, 2, 3, 6, 9, 14, 19, 26, 33 at depths 1 to 8), the
# quiet moves left are skipped, at the zero-window nodes only (Alexandria), so the main
# line is searched whole. Above LMP_CHECK_DEPTH the first pruned quiet drains the rest of
# the list unscanned (#405). At depth LMP_CHECK_DEPTH and under a quiet move that gives
# check is kept, told by the kernel's `gives_check` without making it: the mating checks
# the suite missed were pruned before they were seen. Priced at about 80 Elo in Ethereal
# and measured at -126 in a Python engine with pseudo-legal generation; the match decides.
# The depth it applies at sits in the search state, so a test can turn it off.
LMP_MAX_DEPTH = 8
LMP_CHECK_DEPTH = 3
# Quiescence prunes a capture whose exchanges lose more than SEE_QUIESCE_MARGIN by the
# kernel's static exchange evaluation (Stockfish `see_ge(move, -74)` on a 208 pawn, so about
# a third of a pawn), unless the capture gives check: the mating sacrifices the suite missed
# are losing captures with check. In check nothing is pruned. The switch sits in the search
# state, so a test can turn it off.
SEE_QUIESCE_MARGIN = -30
# Row #451. Out of check, quiescence generates noisy moves only, so its `searched == 0`
# verdict lives in the check branch and a stalemated node returned its stand pat where the
# main search returns a draw. A legal loop at every quiet node costs 0.7 to 1.2 ply, so the
# loop runs only behind `stalemate_gate`. Set to 0 before the import to compile the arm out.
QUIESCE_STALEMATE_GATE = 1
# A threshold reaches see_ge as np.int64, never as a bare literal: numba compiles a jitted
# callee once per distinct literal argument, and each compile of see_ge costs half a second.
# In the main search, at a zero-window node out of check at SEE_QUIET_MAX_DEPTH and under,
# a quiet move whose destination the exchanges lose by more than SEE_QUIET_STEP times the
# depth squared is skipped (Stockfish `see_ge(move, -25 * lmrDepth * lmrDepth)` on a 208
# pawn, scaled to this 100 pawn); a move that gives check is exempt, so the sacrificial
# checks the suite wants stay in the tree. The depth sits in the search state for the tests.
SEE_QUIET_MAX_DEPTH = 6
SEE_QUIET_STEP = 15
# Beside it, the losing captures (the picker's last stage, each already under zero by
# SEE): at a zero-window node out of check at SEE_CAPTURE_MAX_DEPTH and under, one whose
# exchanges lose more than SEE_CAPTURE_STEP times the depth is skipped, the margin linear
# in depth as stash-bot ships it (-60 x depth, e8ecbb1, its own SPRT +3.1 STC and +4.7 LTC
# for the linear form; #569, notes/deep-research/2026-09-12-github-sweep-2.md search
# item 4). Both are frozen into the kernel at compile time. Built 12 September for the
# London final.
SEE_CAPTURE_MAX_DEPTH = 6
SEE_CAPTURE_STEP = 60
# A capture the exchanges lose sorts below every other noisy move by this offset and is
# searched after the quiet moves (Stockfish's bad-captures stage): it is rarely best and
# costs the cutoff it delays. Every marked score sits under BAD_CAPTURE_LIMIT.
BAD_CAPTURE = -(1 << 28)
BAD_CAPTURE_LIMIT = -(1 << 27)


def lmr_table() -> Ints:
    """The reduction for every (depth, moves searched) pair, flat, LMR_TABLE_SIDE a side."""
    table = np.zeros(LMR_TABLE_SIDE * LMR_TABLE_SIDE, dtype=np.int64)
    for depth in range(1, LMR_TABLE_SIDE):
        for moves in range(1, LMR_TABLE_SIDE):
            table[depth * LMR_TABLE_SIDE + moves] = int(
                LMR_BASE + math.log(depth) * math.log(moves) / LMR_DIVISOR
            )
    return table


PROMOTION_SCORE = 1 << 21
CAPTURE_STAGE = 1 << 22
PREVIOUS_BEST_SCORE = 1 << 30
# Quiescence orders its list by `quiescence_score`, whose largest value is 16 * 20 - 1.
# The table's move for the node sits above all of them and well below PROMOTION_SCORE,
# whose negative is the sentinel that ends the quiescence list.
QS_TABLE_SCORE = 1 << 16
QUIET_HISTORY_SIZE = 2 * 64 * 64  # [side to move][from][to]
CAPTURE_HISTORY_SIZE = 6 * 64 * 6  # [moving piece type][to][captured piece type]
# The continuation history: [previous piece][previous to][piece][to], the pieces with
# their colour, one level (the brief's 1-ply table, 8.88 +/- 5.20 Elo in tcheran 10.0;
# the 2-ply table measured nothing there and -20.5 +/- 28.2 here). int16 holds it, since
# the gravity update keeps every entry within HISTORY_MAX.
CONT_KEYS = 12 * 64
CONT_HISTORY_SIZE = CONT_KEYS * CONT_KEYS
# The pawn-structure history (#499): a quiet move's worth under the pawn structure it
# was played in, keyed by the top PAWN_HISTORY_BITS bits of the correction table's pawn
# mix, the moving piece and its target; read beside the butterfly and continuation
# tables in the quiet stage and moved with them on a cutoff. PAWN_HISTORY at 0 leaves
# the table unread and unwritten, so a probe can switch the arm off.
PAWN_HISTORY = 1
PAWN_HISTORY_BITS = 9
PAWN_HISTORY_SIZE = (1 << PAWN_HISTORY_BITS) * CONT_KEYS  # [pawn slot][piece][to]

# Correction history: the signed gap between the static evaluation and the score the
# search settled on, kept by pawn structure and side to move, shifts the static
# evaluation of every later node with that structure (Stockfish b4d995d0, the pawn key
# first; the weighted update is Alexandria's, which keeps the table in centipawns and so
# needs no rescaling from an engine whose pawn is 208). A stored value is the correction
# times CORR_GRAIN, capped at CORR_MAX; a node at depth d moves it with weight
# min((d + 1)^2, CORR_WEIGHT_MAX) out of CORR_WEIGHT_SCALE. The slot is a mix of the two
# pawn bitboards, CORR_BITS wide, so no key is kept in make and unmake.
CORR_BITS = 14
CORR_SIZE = 1 << CORR_BITS  # slots a side
CORR_GRAIN = 256
CORR_MAX = 64 * CORR_GRAIN  # the correction is at most 64 centipawns either way
CORR_WEIGHT_SCALE = 256
CORR_WEIGHT_MAX = 128
CORR_MIX_A = 0x9E3779B97F4A7C15
CORR_MIX_B = 0xC2B2AE3D27D4EB4F
CORR_MIX_C = 0xBF58476D1CE4E5B9
# A second table keyed by the pieces under attack: the bitboard of each side's pieces the
# other side attacks, mixed with constants of their own so the two tables' slots do not
# follow each other (tcheran's threat correction history measures +9.30 +/- 5.31 Elo
# beside its pawn key, the brief's correction-history entry). Its shift adds to the pawn
# table's at equal weight, the sum the wiki's examples use, since tcheran publishes no
# blend; the update moves both tables by the same weight under the same guard. The two
# attack maps cost the key a few table lookups a piece at every evaluation.
THREAT_MIX_A = 0xD6E8FEB86659FD93
THREAT_MIX_B = 0xA0761D6478BD642F
THREAT_MIX_C = 0xE7037ED1A0B428DB
NOT_FILE_A = np.uint64(0xFEFEFEFEFEFEFEFE)
NOT_FILE_H = np.uint64(0x7F7F7F7F7F7F7F7F)

# The transposition table: TT_ENTRIES entries of two words, 134 MB. The first word of an
# entry is the packed word below; the second is the score word, the raw net score of the
# position that stored it (row #63).
# 2^23 two-word entries measured +24.6 +/- 19.4 over 184 at the platform clock (exp/88);
# 2^24 one-word entries in the same 134 MB is its own row. The size is frozen into the
# compiled kernel: a smaller table for a memory cap is edited here and compiled cold
# (notes/london-day.md), never set after the import.
TT_ENTRIES = 1 << 23
TT_MASK = TT_ENTRIES - 1
TT_STRIDE = 2  # words an entry: the packed word, then the score word
TT_WORDS = TT_ENTRIES * TT_STRIDE
TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2
# The packed word: the move's source, target, flag and promotion in bits 0 to 15 (its
# piece and captured piece are read back from the board, since a true hit is the same
# position, and the promotion piece is two bits that count only for a pawn reaching the
# last rank), depth plus DEPTH_OFFSET in 16 to 21, bound in 22 to 23, generation in 24
# to 27, score plus SCORE_OFFSET in 28 to 45, and the hash's top 18 bits in 46 to 63 as
# the tag. The index takes the hash's low bits, so the tag is independent of it. A
# stored word is never 0, since the depth field holds at least DEPTH_OFFSET - DEPTH_MIN,
# and 0 marks an empty entry. The score field takes 18 bits: MATE is 100,000 and a
# stored score never exceeds it. The depth field takes 6 bits, DEPTH_MIN to DEPTH_MAX,
# and `tt_store` clamps to it: the deepening loop's cap is MAX_DEPTH, past the field's
# ceiling, so a search deeper than DEPTH_MAX is stored as DEPTH_MAX and cuts one probe
# fewer, never one more; below zero the depth is that of a node in check past the
# extension horizon, which searches its evasions and then quiescence whatever the number,
# so the floor changes no result (the old 7-bit field wrapped the whole word at -17). The
# generation is one per get_move and wraps at 16, so an entry is read as current only
# within the move that stored it; a false tag match happens once in 262,144 probes of a
# full entry.
MOVE_MASK = (1 << 16) - 1
DEPTH_SHIFT = 16
DEPTH_OFFSET = 8
DEPTH_MASK = (1 << 6) - 1
DEPTH_MIN = -DEPTH_OFFSET
DEPTH_MAX = DEPTH_MASK - DEPTH_OFFSET
BOUND_SHIFT = 22
GENERATION_SHIFT = 24
GENERATION_MASK = 15
SCORE_SHIFT = 28
SCORE_OFFSET = 1 << 17
SCORE_MASK = (1 << 18) - 1
TAG_SHIFT = 46
TAG_MASK = (1 << 18) - 1
# The score word: the raw net score in bits 0 to 16 as a 17-bit two's-complement value,
# the threats' correction slot in bits 17 to 31 (CORR_BITS wide plus the side's bit, so
# under 32768), and in 32 to 63 a tag that is the key's top 32 bits with the low bit set,
# so a stored word is never 0 and the empty word 0 matches no key. The index takes the
# key's low bits, so the tag is independent of it. The score is what `evaluate_net`
# returns, before the correction shift, which is why an entry's score outlives the packed
# word beside it: the two are written on their own keys and read on their own tags. A
# score the 17 bits cannot hold is not stored at all, so the next read misses and the head
# runs again; the value the search sees is the same either way.
EVAL_TAG_SHIFT = 32
EVAL_WORD_MASK = (1 << 32) - 1
EVAL_SLOT_SHIFT = 17
EVAL_SLOT_MASK = (1 << 15) - 1
EVAL_SCORE_MASK = (1 << 17) - 1
EVAL_SCORE_SIGN = 1 << 16
EVAL_SCORE_MIN = -(1 << 16)
EVAL_SCORE_MAX = (1 << 16) - 1
EVAL_NONE = 1 << 40  # past every score, so it reads as "no score stored here"

GAME_MAX = 1024  # earlier positions of the game the draw check can see

# Three shortcuts shape the tree. Null move: a node whose static evaluation already reaches
# beta, with the side to move not in check and holding a piece besides pawns, lets the
# opponent move twice and searches the reply to a reduced depth; a reply that still cannot
# get under beta proves the node a cutoff without searching its moves. The reduction is
# NULL_MOVE_BASE plus depth over NULL_MOVE_DIVISOR plus one per NULL_MOVE_EVAL_STEP
# centipawns of evaluation over beta, at most NULL_MOVE_EVAL_MAX of those. Internal
# iterative reduction: a node the table knows nothing about, at IIR_MIN_DEPTH or more, is
# searched a ply shallower, since the entry it writes will order the next visit. Check
# extension, decided in the parent: a checking move whose static exchange is at least
# CHECK_EXTENSION_SEE_MARGIN gets its child a ply more, while the child's ply is under
# CHECK_EXTENSION_PLY_FACTOR times the root depth; a check that hangs material is
# searched at the plain depth (the Crafty rule the brief's check-extension section
# quotes). The child in check never extends itself, so a check is counted once.
NULL_MOVE_MIN_DEPTH = 2
NULL_MOVE_BASE = 3
NULL_MOVE_DIVISOR = 3
NULL_MOVE_EVAL_STEP = 200
NULL_MOVE_EVAL_MAX = 3
IIR_MIN_DEPTH = 4
# Follow the previous iteration's line (#503): after each completed iteration the
# table's moves from the root are walked and the positions' keys kept in `pv`,
# PV_MAX_PLIES at most; a node whose key is the line's at its ply is on it, and there
# the internal iterative reduction and the shallow quiet pruning (the move count and
# the SEE quiet gate) are switched off, so the line is re-searched whole rather than
# cut where the table lost it (Stockfish, 2026-03-18). FOLLOW_PV at 0 leaves the line
# unwalked and unread.
FOLLOW_PV = 1
PV_MAX_PLIES = 32
PV_SIZE = 2 * PV_MAX_PLIES  # the keys, then the moves made to reach them
CHECK_EXTENSION_PLY_FACTOR = 2
CHECK_EXTENSION_SEE_MARGIN = 0

# Reverse futility: below RFP_MAX_DEPTH a node whose static evaluation clears beta by
# depth * (RFP_MARGIN_BASE + RFP_MARGIN_STEP * (depth - 1)) centipawns returns without
# a move generated, at beta plus a third of the excess. The schedule is Stockfish's of
# 2026-05-04 (40 to 76 per ply over depth 1 to 10 on a 208 pawn) rescaled to our 100
# pawn: 19 at depth 1, 100 at depth 4, 264 at depth 8. Never in check, at the root, or
# against a mate bound.
RFP_MAX_DEPTH = 9
RFP_MARGIN_BASE = 19
RFP_MARGIN_STEP = 2
RFP_BLEND_DIVISOR = 3

# Razoring: at the frontier, depth one, a node whose static evaluation sits RAZOR_MARGIN
# centipawns under alpha drops straight to the quiescence search instead of generating
# moves. The score returned is the quiescence search's, not alpha, so the node stays
# fail-soft. Stockfish prices razoring at about 1 Elo and tcheran at 7.73 +/- 4.76, and
# the brief reads the disagreement as a depth effect: razoring fires only at frontier
# nodes, which are a far larger share of a shallow tree than of Stockfish's
# (`sota-low-compute-chess-part-2.md`, the evidence table by the ranked shortlist;
# `search-pruning-low-nps.md`, build order). Never in check, at the root, on a principal
# variation node, or against a mate bound. The margin is the node probe's: at depth 11
# over the 60 openings and platform misses, 200 read 0.974 of main's nodes by the
# per-position median and 0.967 in sum, against 0.990 and 1.068 at 300.
RAZOR_MARGIN = 200

# The halfmove clock a table cutoff is refused from. The key carries no clock, the table
# is kept for the whole game, and a score computed at a low clock is handed back unchanged
# at a high one, where the fifty-move draw is inside the horizon and every line the entry
# scored ends in it. Above this the node is searched instead, which is Stockfish's rule at
# the same 90.
TT_CUTOFF_HALF = 90


@jit(inline=True, hot=True, internal=not TESTING)
def lmp_limit(depth: int) -> int:
    """Moves a node searches at `depth` before move-count pruning skips the quiet ones left."""
    return (3 + depth * depth) // 2


@jit(inline=True, hot=True, internal=not TESTING)
def seventh_rank_push(move: int, us: int) -> bool:
    """A quiet pawn move to the mover's seventh rank, one step from promotion, kept at full depth.

    The wider rule, the sixth rank too, read -41 Elo over 102 games: it opened the tree on every
    advanced push. A pawn one step from queening changes the evaluation more than a reduced
    search sees, and there are few of them.
    """
    if move_piece(move) % 6 != PAWN:
        return False
    rank = move_target(move) >> 3
    return rank == 6 if us == WHITE else rank == 1


@jit(inline=True, internal=not TESTING)
def rfp_margin(depth: int) -> int:
    """How far the static evaluation has to clear beta for the node to return at once."""
    return depth * (RFP_MARGIN_BASE + RFP_MARGIN_STEP * (depth - 1))


@jit()
def rfp_allowed(checked: bool, ply: int, depth: int, alpha: int, beta: int) -> bool:
    """Whether the node may return on its static evaluation alone."""
    if checked or ply == 0 or depth >= RFP_MAX_DEPTH:
        return False
    return beta < MATE_BOUND and alpha > -MATE_BOUND


@jit(internal=True)
def razor_allowed(checked: bool, ply: int, depth: int, alpha: int, beta: int) -> bool:
    """Whether the node may drop to the quiescence search on its static evaluation alone."""
    if checked or ply == 0 or depth != 1:
        return False
    if beta - alpha != 1:  # a zero window: never on a principal variation node
        return False
    return beta < MATE_BOUND and alpha > -MATE_BOUND


# The evaluation tables, from evaluation.py: 12 x 64 middlegame values, then 12 x 64
# endgame values, with black's negated; and the phase weight by piece type.
EV_EG = 768
EV_SIZE = 1536
PHASE_INC = (0, 1, 1, 2, 4, 0)
PHASE_MAX = 24


def tapered_scan(bb: Bitboards, ev: Ints) -> tuple[int, int, int]:
    """The middlegame sum, the endgame sum and the phase, counted over every piece.

    Plain Python, beside `kernel.compute_hash` and for the same reason: it runs once a
    `get_move`, and a jitted copy would cost compile at every import for work the search
    never repeats. It is also the check on the numbers `tapered_make` carries, which the
    tests walk it against after every make and unmake.
    """
    mg = 0
    eg = 0
    phase = 0
    for piece in range(12):
        pieces = int(bb[piece])
        weight = PHASE_INC[piece % 6]
        while pieces:
            square = (pieces & -pieces).bit_length() - 1
            pieces &= pieces - 1
            mg += int(ev[piece * 64 + square])
            eg += int(ev[EV_EG + piece * 64 + square])
            phase += weight
    return mg, eg, phase


def set_tapered(bb: Bitboards, st: Ints, ev: Ints) -> None:
    """Set the state array's tapered numbers from the position on the board.

    `agent.Searcher` calls it after `kernel.load`; from there `make_s` and `unmake_s`
    carry them, so every position the search reaches has them without a loop.
    """
    st[ST_MG], st[ST_EG], st[ST_PHASE] = tapered_scan(bb, ev)


def _clock() -> tuple[Any, bool, int]:
    """The OS monotonic clock as a symbol the kernels call by name; whether it takes a clock
    id; ticks a second.

    A ctypes function pointer would put a constant address into the compiled kernel, which
    numba records as a dynamic global and refuses to cache, and the refusal spreads to
    every kernel that reaches the clock. A symbol declared by name is resolved when the
    compiled code is loaded, so the search chain caches
    (`notes/measurements/2026-09-06-import-cache.md`). The read costs the same 20 ns.
    """
    if sys.platform == "win32":
        kernel32 = ctypes.windll.kernel32
        frequency_of = kernel32.QueryPerformanceFrequency
        frequency_of.restype = ctypes.c_int
        frequency_of.argtypes = (ctypes.c_void_p,)
        frequency = np.zeros(1, np.int64)
        frequency_of(frequency.ctypes.data)
        # kernel32 is in every process; registering the address by name makes the JIT's
        # lookup explicit instead of a search of the loaded modules.
        address = ctypes.cast(kernel32.QueryPerformanceCounter, ctypes.c_void_p).value
        llvm.add_symbol("QueryPerformanceCounter", int(address or 0))
        counter = types.ExternalFunction("QueryPerformanceCounter", types.int32(types.uintp))
        return counter, False, int(frequency[0])
    # glibc's clock_gettime, resolved from the process by the JIT.
    clock_gettime = types.ExternalFunction("clock_gettime", types.int32(types.int32, types.uintp))
    return clock_gettime, True, 1_000_000_000


CLOCK, CLOCK_TAKES_ID, TICKS_PER_SECOND = _clock()
CLOCK_MONOTONIC = 1

if CLOCK_TAKES_ID:

    @njit(cache=CACHE)
    def now(clk: Ints) -> int:
        """Clock ticks; `clk` is the two-word timespec the call writes into."""
        CLOCK(CLOCK_MONOTONIC, clk.ctypes.data)
        return int(clk[0]) * 1_000_000_000 + int(clk[1])

else:

    @njit(cache=CACHE)
    def now(clk: Ints) -> int:
        """Clock ticks; `clk` is the one-word buffer the call writes into."""
        CLOCK(clk.ctypes.data)
        return int(clk[0])


# The arrays of one search, behind one object. Every kernel below takes the object and
# reads the arrays it needs from it; the arrays themselves stay the module-level ones
# `agent.py` allocates, and the object only holds references to them. It is a structref
# and not a jitclass because numba names a jitclass type with the class's address in the
# process, so a kernel typed on one never finds its own compiled code in the disk cache;
# a structref type is named by its fields and does
# (`notes/measurements/2026-09-06-import-cache.md`).
# The staged move picker. Ordering used to sit inside `negamax` as four loops over three
# lists; it is now a state machine over a slice of `pk` per ply, and `negamax` asks it for
# one move at a time and does the recursion itself. The stages and their order are
# unchanged: the table's move alone, then the captures whose exchanges win, then the quiet
# moves, then the captures whose exchanges lose. Nothing here calls back into `negamax`,
# which is what the earlier stage split got wrong: a recursion cycle across functions does
# not survive numba's disk cache.
PICK_TABLE = 0
PICK_NOISY = 1
PICK_QUIET = 2
PICK_BAD = 3
PICK_DONE = 4

PK_STAGE = 0  # the stage the picker will draw from next
PK_LAST = 1  # the stage the move it just returned came from
PK_NEXT = 2  # where the picker will look next
PK_AT = 3  # where the move it just returned sits, for the malus and for skipping it
PK_COUNT = 4  # noisy moves generated
PK_QUIET_BASE = 5
PK_QUIET_COUNT = 6
PK_BAD_FROM = 7  # the first noisy move whose exchange loses
PK_TABLE_MOVE = 8
PK_LIST_BASE = 9  # the start of the list the returned move came from
PK_NOISY_READY = 10
PK_QUIET_READY = 11
# The check masks of the node, built at the ply's first `gives_check_s` and read by every
# call after it. `PK_CHK_READY` is 0 until they are built, which every entry to a node
# restores: `pick_setup` clears the picker's words below, and `quiesce` and `search_root`,
# which have no picker, clear this one by hand.
PK_CHK_READY = 12
PK_CHK_PAWN = 13
PK_CHK_KNIGHT = 14
PK_CHK_BISHOP = 15
PK_CHK_ROOK = 16
PK_CHK_DISCOVERY = 17
PK_CLEAR = 18  # the words `pick_setup` clears
# The move the ply owes its accumulator, 0 when the lanes are up to date. It sits above
# `PK_CLEAR` because entering a node must not forget it: `unmake_s` clears a mark nothing
# read, `nn_flush_s` clears one it applies, and nothing else writes the word.
PK_PEND = 18
# The move that led to the ply's node, for the continuation history: the parent writes
# it before each child, 0 after a null move and at the root. Above `PK_CLEAR` like
# `PK_PEND`, since the child's `pick_setup` must not forget it.
PK_PREV = 19
PK_CUTOFFS = 20  # how many children of this ply's node failed high (#500)
PK_WORDS = 21


@jit(internal=not TESTING)
def pick_setup(s: Search, ply: int, table_move: int) -> int:
    """Prepare the picker for a node and return the table move it will actually try.

    A table move that is a capture or a promotion is taken out of the noisy list, which
    means generating that list first; a quiet one is checked on its own, so a cutoff on it
    never needs the list at all. A move no longer on the board is dropped here.
    """
    bb = s.bb
    sq = s.sq
    st = s.st
    tab = s.tab
    ml = s.ml
    ms = s.ms
    pk = s.pk
    off = ply * PK_WORDS
    base = ply * MOVES_PER_PLY
    for word in range(PK_CLEAR):
        pk[np.uint64(off + word)] = 0
    pk[np.uint64(off + PK_NEXT)] = base
    pk[np.uint64(off + PK_AT)] = base
    pk[np.uint64(off + PK_LIST_BASE)] = base
    if table_move != 0 and (move_captured(table_move) != EMPTY or move_promotion(table_move) != 0):
        count = gen_noisy_s(s, base)
        for i in range(base, base + count):
            ms[np.uint64(i)] = noisy_score(ml[np.uint64(i)], s.hc)
        pk[np.uint64(off + PK_NOISY_READY)] = 1
        found = False
        for i in range(base, base + count):
            if ml[np.uint64(i)] == table_move:
                ml[np.uint64(i)] = ml[np.uint64(base + count - 1)]
                ms[np.uint64(i)] = ms[np.uint64(base + count - 1)]
                count -= 1
                found = True
                break
        pk[np.uint64(off + PK_COUNT)] = count
        if not found:
            table_move = 0  # a stale entry: the capture is no longer on the board
    elif table_move != 0 and not pseudo_legal_quiet(table_move, bb, sq, st, tab):
        table_move = 0
    pk[np.uint64(off + PK_TABLE_MOVE)] = table_move
    pk[np.uint64(off + PK_STAGE)] = PICK_TABLE if table_move != 0 else PICK_NOISY
    return table_move


# Inlined into `negamax`: the profiler reads the call itself at 8.5 ns, 0.712 calls a node.
@jit(inline=True, hot=True, internal=not TESTING)
def pick_next(s: Search, ply: int, us: int, previous: int) -> int:
    """The next move to try, or 0 when the node has none left.

    The stage it came from is left in `pk[PK_LAST]` and its place in the move list in
    `pk[PK_AT]`, so the caller can zero a move it skips and charge the history malus over
    the right part of the list, which `pk[PK_LIST_BASE]` gives.
    """
    ml = s.ml
    ms = s.ms
    pk = s.pk
    ss = s.ss
    off = ply * PK_WORDS
    base = ply * MOVES_PER_PLY
    while True:
        stage = pk[np.uint64(off + PK_STAGE)]
        if stage == PICK_TABLE:
            pk[np.uint64(off + PK_STAGE)] = PICK_NOISY
            pk[np.uint64(off + PK_LAST)] = PICK_TABLE
            pk[np.uint64(off + PK_AT)] = base
            pk[np.uint64(off + PK_LIST_BASE)] = base
            return int(pk[np.uint64(off + PK_TABLE_MOVE)])
        if stage == PICK_NOISY:
            if pk[np.uint64(off + PK_NOISY_READY)] == 0:
                count = gen_noisy_s(s, base)
                for i in range(base, base + count):
                    ms[np.uint64(i)] = noisy_score(ml[np.uint64(i)], s.hc)
                pk[np.uint64(off + PK_COUNT)] = count
                pk[np.uint64(off + PK_NOISY_READY)] = 1
            count = pk[np.uint64(off + PK_COUNT)]
            if pk[np.uint64(off + PK_NEXT)] == base and pk[np.uint64(off + PK_NOISY_READY)] == 1:
                # The captures whose exchanges lose sort last and wait for the quiet
                # moves; the exchange is read once per capture, at interior nodes only.
                pk[np.uint64(off + PK_BAD_FROM)] = base + count
                if ss[SS_BAD_CAPTURES] != 0:
                    for i in range(base, base + count):
                        if ml[np.uint64(i)] != 0 and not see_ge_s(
                            ml[np.uint64(i)],
                            np.int64(0),  # type: ignore[arg-type]
                            s,
                        ):
                            ms[np.uint64(i)] = BAD_CAPTURE + ms[np.uint64(i)]
                pk[np.uint64(off + PK_NOISY_READY)] = 2
            i = pk[np.uint64(off + PK_NEXT)]
            if i >= base + count:
                pk[np.uint64(off + PK_STAGE)] = PICK_QUIET
                continue
            move = pick_s(s, i, base + count)
            if ms[np.uint64(i)] < BAD_CAPTURE_LIMIT:
                pk[np.uint64(off + PK_BAD_FROM)] = i  # the rest lose their exchanges
                pk[np.uint64(off + PK_STAGE)] = PICK_QUIET
                continue
            pk[np.uint64(off + PK_NEXT)] = i + 1
            pk[np.uint64(off + PK_AT)] = i
            pk[np.uint64(off + PK_LIST_BASE)] = base
            pk[np.uint64(off + PK_LAST)] = PICK_NOISY
            return move
        if stage == PICK_QUIET:
            if pk[np.uint64(off + PK_QUIET_READY)] == 0:
                quiet_base = base + pk[np.uint64(off + PK_COUNT)]
                quiet_count = gen_quiet_s(s, quiet_base)
                prow = pawn_history_row(s.bb)
                for i in range(quiet_base, quiet_base + quiet_count):
                    ms[np.uint64(i)] = quiet_score(
                        ml[np.uint64(i)],
                        us,
                        ply,
                        s.killers,
                        s.hq,
                        s.chist,
                        previous,
                        s.phist,
                        prow,
                    )
                pk[np.uint64(off + PK_QUIET_BASE)] = quiet_base
                pk[np.uint64(off + PK_QUIET_COUNT)] = quiet_count
                pk[np.uint64(off + PK_NEXT)] = quiet_base
                pk[np.uint64(off + PK_QUIET_READY)] = 1
            quiet_base = pk[np.uint64(off + PK_QUIET_BASE)]
            quiet_count = pk[np.uint64(off + PK_QUIET_COUNT)]
            i = pk[np.uint64(off + PK_NEXT)]
            if i >= quiet_base + quiet_count:
                pk[np.uint64(off + PK_NEXT)] = pk[np.uint64(off + PK_BAD_FROM)]
                pk[np.uint64(off + PK_STAGE)] = PICK_BAD
                continue
            move = pick_s(s, i, quiet_base + quiet_count)
            pk[np.uint64(off + PK_NEXT)] = i + 1
            if move == pk[np.uint64(off + PK_TABLE_MOVE)]:
                ml[np.uint64(i)] = 0
                continue
            pk[np.uint64(off + PK_AT)] = i
            pk[np.uint64(off + PK_LIST_BASE)] = quiet_base
            pk[np.uint64(off + PK_LAST)] = PICK_QUIET
            return move
        if stage == PICK_BAD:
            count = pk[np.uint64(off + PK_COUNT)]
            i = pk[np.uint64(off + PK_NEXT)]
            if i >= base + count:
                pk[np.uint64(off + PK_STAGE)] = PICK_DONE
                continue
            move = pick_s(s, i, base + count)
            pk[np.uint64(off + PK_NEXT)] = i + 1
            pk[np.uint64(off + PK_AT)] = i
            pk[np.uint64(off + PK_LIST_BASE)] = base
            pk[np.uint64(off + PK_LAST)] = PICK_BAD
            return move
        return 0


SEARCH_SPEC = [
    ("bb", uint64[::1]),  # the bitboards of the position
    ("sq", int64[::1]),  # the piece on each square
    ("st", int64[::1]),  # the position's scalars: side to move, ply, hash, clocks
    ("tab", uint64[::1]),  # the attack tables
    ("keys", int64[::1]),  # the Zobrist keys
    ("ev", int64[::1]),  # the piece-square tables
    ("ftw", int16[:, ::1]),  # the net's first-layer weights
    ("hd", int32[::1]),  # the net's head
    ("acc", int16[::1]),  # the net's accumulators, one pair per ply, int16 by the loader's proof
    ("xs", int32[::1]),  # the head's scratch
    ("hb", int8[::1]),  # the head's byte half: clipped products, packed first-layer weights
    ("kacc", int16[::1]),  # the king-move refresh cache: an accumulator a colour and key
    ("kbb", uint64[::1]),  # the board each cached accumulator was built from
    ("ml", int64[::1]),  # the move lists, one block per ply
    ("ms", int64[::1]),  # the move scores, alongside
    ("undo", int64[::1]),  # the make and unmake stack
    ("path", int64[::1]),  # the hashes of the search path
    ("game", int64[::1]),  # the hashes of the game before the root
    ("killers", int64[::1]),  # the killer move of each ply
    ("hq", int64[::1]),  # the butterfly history of quiet moves
    ("hc", int64[::1]),  # the capture history
    ("lmr", int64[::1]),  # the late-move reduction table
    ("tt", int64[::1]),  # the transposition table
    ("root", int64[::1]),  # the root moves
    ("root_nodes", int64[::1]),  # the nodes each root move took
    ("ss", int64[::1]),  # the search state
    ("clk", int64[::1]),  # the clock's scratch words
    ("kpk", uint8[::1]),  # the king and pawn against king bitbase, or empty
    ("pk", int64[::1]),  # the move picker's state, one slice a ply
    ("corr", int32[::1]),  # the correction history, by pawn structure and side to move
    ("tcorr", int32[::1]),  # the correction history, by the pieces under attack
    ("chist", int16[::1]),  # the continuation history of quiet moves, by the move before
    ("phist", int16[::1]),  # the pawn-structure history of quiet moves
    ("pv", int64[::1]),  # the previous iteration's line as position keys (#503)
]


@structref.register
class SearchType(types.StructRef):
    """The numba type of the search object: its fields, in the order of `SEARCH_SPEC`."""

    def preprocess_fields(self, fields: Any) -> Any:
        return tuple((name, types.unliteral(typ)) for name, typ in fields)  # type: ignore[no-untyped-call]


class Search(structref.StructRefProxy):
    """The position, the tables, the scratch arrays and the state of one search."""

    bb: Any
    sq: Any
    st: Any
    tab: Any
    keys: Any
    ev: Any
    ftw: Any
    hd: Any
    acc: Any
    xs: Any
    hb: Any
    kacc: Any
    kbb: Any
    ml: Any
    ms: Any
    undo: Any
    path: Any
    game: Any
    killers: Any
    hq: Any
    hc: Any
    lmr: Any
    tt: Any
    root: Any
    root_nodes: Any
    ss: Any
    clk: Any
    kpk: Any
    pk: Any
    corr: Any
    tcorr: Any
    chist: Any
    phist: Any
    pv: Any

    def __new__(cls, *arrays: Any) -> Any:
        # numba's own constructor for a structref names a local `st`, which the field of
        # that name shadows, so the object is built by `new_search` below instead.
        return new_search(*arrays)


structref.define_boxing(SearchType, Search)  # type: ignore[no-untyped-call]


@lower_getattr_generic(SearchType)  # type: ignore[no-untyped-call, untyped-decorator]
def search_field(context: Any, builder: Any, typ: Any, value: Any, attr: str) -> Any:
    """A field of the search object, read without numba's runtime.

    numba's own lowering for a structref's field reaches the payload through the runtime,
    so a kernel compiled with `_nrt=False` cannot read one. This lowering, registered for
    our type alone, declares the runtime's data call by hand. The kernels never own the
    arrays, `agent.py` does, and under `_nrt=False` no reference is taken; a caller that
    keeps the runtime (a bench in `tools/profile_kernel.py`) takes one through numba's
    borrowed-return helper, so its own release at exit has a reference to release.
    """
    struct = cgutils.create_struct_proxy(typ)(context, builder, value=value)  # type: ignore[no-untyped-call]
    data_of = cgutils.get_or_insert_function(  # type: ignore[no-untyped-call]
        builder.module, meminfo_data_ty, "NRT_MemInfo_data_fast"
    )
    data_ptr = builder.call(data_of, [struct.meminfo])
    payload_type = typ.get_data_type()
    model = context.data_model_manager[payload_type]
    data_ptr = builder.bitcast(data_ptr, model.get_value_type().as_pointer())
    payload = cgutils.create_struct_proxy(payload_type, kind="data")(context, builder, ref=data_ptr)  # type: ignore[no-untyped-call]
    field = getattr(payload, attr)
    return imputils.impl_ret_borrowed(context, builder, typ.field_dict[attr], field)  # type: ignore[no-untyped-call]


SEARCH_TYPE = SearchType(SEARCH_SPEC)  # type: ignore[no-untyped-call]


@njit(cache=CACHE)  # the one kernel that allocates: the object itself
def new_search(
    bb: Any,
    sq: Any,
    st: Any,
    tab: Any,
    keys: Any,
    ev: Any,
    ftw: Any,
    hd: Any,
    acc: Any,
    xs: Any,
    hb: Any,
    kacc: Any,
    kbb: Any,
    ml: Any,
    ms: Any,
    undo: Any,
    path: Any,
    game: Any,
    killers: Any,
    hq: Any,
    hc: Any,
    lmr: Any,
    tt: Any,
    root: Any,
    root_nodes: Any,
    ss: Any,
    clk: Any,
    kpk: Any,
    pk: Any,
    corr: Any,
    tcorr: Any,
    chist: Any,
    phist: Any,
    pv: Any,
) -> Any:
    """The search object over the arrays given, in the order of `SEARCH_SPEC`.

    The kernels read its fields without taking a reference (`search_field`), so the caller
    keeps the arrays alive for as long as the object is used, which `agent.py` does by
    holding them as module globals.
    """
    s: Any = structref.new(SEARCH_TYPE)  # type: ignore[call-arg]
    s.bb = bb
    s.sq = sq
    s.st = st
    s.tab = tab
    s.keys = keys
    s.ev = ev
    s.ftw = ftw
    s.hd = hd
    s.acc = acc
    s.xs = xs
    s.hb = hb
    s.kacc = kacc
    s.kbb = kbb
    s.ml = ml
    s.ms = ms
    s.undo = undo
    s.path = path
    s.game = game
    s.killers = killers
    s.hq = hq
    s.hc = hc
    s.lmr = lmr
    s.tt = tt
    s.root = root
    s.root_nodes = root_nodes
    s.ss = ss
    s.clk = clk
    s.kpk = kpk
    s.pk = pk
    s.corr = corr
    s.tcorr = tcorr
    s.chist = chist
    s.phist = phist
    s.pv = pv
    return s


# The entries the search calls, one per hot kernel: each reads the arrays it needs from
# the search object inside its own compiled body and runs the kernel's body pasted in
# (the note above `attacked_body` in `kernel.py`), so a call crosses with the handle's
# one word (the structref's meminfo pointer; numba 0.67, `make_s` reads only its first
# field) and the scalars where it crossed with seven words per array. The array
# entries of the same names in `kernel.py` and `nnue.py` are the tests' and the benches'.
@jit(internal=True)
def attacked_s(square: int, by: int, s: Search) -> bool:
    """`attacked` on the search's position."""
    return attacked_body(square, by, s.bb, s.tab)


@jit()
def in_check_s(s: Search) -> bool:
    """`in_check` on the search's position: whether the side to move is in check.

    Every node calls this once, and it takes `attacked_body` itself rather than reaching
    it through `attacked_s`, so the test crosses one compiled call and not two. The
    second boundary was 1.12 ns of the call's 4.65 (`tmp/logs/kernel-check-repeated-split.log`).
    `legal_after_s` keeps its call: it is pasted into its callers already, so the same
    change would paste the body into the move loop as well.
    """
    bb = s.bb
    st = s.st
    us = st[STM]
    return attacked_body(lsb(bb[np.uint64(us * 6 + KING)]), 1 - us, bb, s.tab)


@jit(inline=True, hot=True, internal=not TESTING)
def legal_after_s(s: Search) -> bool:
    """`legal_after` on the search's position: after make, whether the side that just
    moved left its king out of check. Pasted into its callers as `legal_after` is."""
    bb = s.bb
    st = s.st
    mover = 1 - st[STM]
    return not attacked_s(lsb(bb[np.uint64(mover * 6 + KING)]), st[STM], s)


@jit(internal=True)
def pinned_pieces_s(us: int, s: Search) -> np.uint64:
    """`pinned_pieces` on the search's position."""
    return pinned_pieces_body(us, s.bb, s.tab)


@jit()
def gives_check_s(move: int, ply: int, s: Search) -> bool:
    """Whether `move` checks the enemy king, read off the node's check masks.

    The masks do not depend on the move, so they are built at the ply's first call and
    kept in the picker's state for the rest of the node; the profile has 1.27 calls a node
    at 6.8 nanoseconds each against one build. A move checks when its target sits in the
    mask of the piece that lands. Three shapes the masks cannot answer keep the full test,
    which is the answer this always gave: a move from a square that stands between one of
    our sliders and their king, an en passant capture, whose taken pawn leaves a square
    the mover never stood on, and a castling, which moves the rook as well. A promotion
    keeps it too, since the piece that lands is not the piece that blocked the king's view
    of the square it lands on.
    """
    st = s.st
    pk = s.pk
    off = ply * PK_WORDS
    if pk[np.uint64(off + PK_CHK_READY)] == 0:
        pawn, knight, bishop, rook, discovery = check_masks_body(s.bb, st, s.tab)
        # The masks are kept as the words' own int64: the bit test below reads one bit,
        # which an arithmetic shift of a negative word gives as faithfully as a logical one.
        pk[np.uint64(off + PK_CHK_PAWN)] = np.int64(pawn)
        pk[np.uint64(off + PK_CHK_KNIGHT)] = np.int64(knight)
        pk[np.uint64(off + PK_CHK_BISHOP)] = np.int64(bishop)
        pk[np.uint64(off + PK_CHK_ROOK)] = np.int64(rook)
        pk[np.uint64(off + PK_CHK_DISCOVERY)] = np.int64(discovery)
        pk[np.uint64(off + PK_CHK_READY)] = 1
    # EN_PASSANT and CASTLING are the two flags above DOUBLE_PUSH, so one comparison
    # sends both to the full test.
    if move_flag(move) >= EN_PASSANT or move_promotion(move) != 0:
        return gives_check_body(move, s.bb, st, s.tab)
    kind = move_piece(move) % 6
    if kind == PAWN:
        mask = pk[np.uint64(off + PK_CHK_PAWN)]
    elif kind == KNIGHT:
        mask = pk[np.uint64(off + PK_CHK_KNIGHT)]
    elif kind == BISHOP:
        mask = pk[np.uint64(off + PK_CHK_BISHOP)]
    elif kind == ROOK:
        mask = pk[np.uint64(off + PK_CHK_ROOK)]
    elif kind == QUEEN:
        mask = pk[np.uint64(off + PK_CHK_BISHOP)] | pk[np.uint64(off + PK_CHK_ROOK)]
    else:
        mask = np.int64(0)  # a king never checks from the square it lands on
    if (mask >> np.int64(move_target(move))) & np.int64(1):
        return True
    if (pk[np.uint64(off + PK_CHK_DISCOVERY)] >> np.int64(move_source(move))) & np.int64(1):
        return gives_check_body(move, s.bb, st, s.tab)
    return False


@jit(internal=True)
def evasion_targets_s(s: Search) -> np.uint64:
    """`evasion_targets` on the search's position."""
    return evasion_targets_body(s.bb, s.st, s.tab)


@jit()
def see_ge_s(move: int, threshold: int, s: Search) -> bool:
    """`see_ge` on the search's position."""
    return see_ge_body(move, threshold, s.bb, s.st, s.tab)


@jit(internal=True)
def gen_noisy_s(s: Search, base: int) -> int:
    """`gen_noisy` on the search's position, into its move list from `base`."""
    return gen_noisy_body(s.bb, s.sq, s.st, s.tab, s.ml, base)


@jit(internal=True)
def gen_quiet_s(s: Search, base: int) -> int:
    """`gen_quiet` on the search's position, into its move list from `base`."""
    return gen_quiet_body(s.bb, s.sq, s.st, s.tab, s.ml, base)


@jit(internal=True)
def gen_all_s(s: Search, base: int) -> int:
    """`gen_all` on the search's position: the noisy moves, then the quiet ones."""
    count = gen_noisy_s(s, base)
    return count + gen_quiet_s(s, base + count)


@jit(inline=True, internal=not TESTING)
def tapered_make(move: int, st: Ints, undo: Ints, ev: Ints) -> None:
    """Carry the tapered sums and the phase across a move, the old three kept on the undo
    stack beside the castling rights and the hash.

    It runs before `make_body`, so the state is still the position the move is played in:
    `st[STM]` owns the piece and `st[PLY]` is the move's own undo slot. Each square the
    move touches is one table pair added and one subtracted, against the loop over every
    piece a leaf ran before (measured 2026-09-08: 13 ns a call at 32 pieces, of which 8 is
    the walk over the twelve bitboards and 5 the pieces themselves).
    """
    slot = st[PLY] * UNDO_FIELDS
    mg = st[ST_MG]
    eg = st[ST_EG]
    phase = st[ST_PHASE]
    undo[np.uint64(slot + UNDO_MG)] = mg
    undo[np.uint64(slot + UNDO_EG)] = eg
    undo[np.uint64(slot + UNDO_PHASE)] = phase
    us = st[STM]
    source = move_source(move)
    target = move_target(move)
    piece = move_piece(move)
    captured = move_captured(move)
    promotion = move_promotion(move)
    flag = move_flag(move)
    left = piece * 64 + source
    mg -= ev[np.uint64(left)]
    eg -= ev[np.uint64(EV_EG + left)]
    if promotion:
        arrived = (us * 6 + promotion) * 64 + target
        phase += PHASE_INC[promotion]  # the pawn it replaces weighs nothing
    else:
        arrived = piece * 64 + target
    mg += ev[np.uint64(arrived)]
    eg += ev[np.uint64(EV_EG + arrived)]
    if captured != EMPTY:
        captured_square = target
        if flag == EN_PASSANT:
            captured_square = target - 8 if us == WHITE else target + 8
        taken = captured * 64 + captured_square
        mg -= ev[np.uint64(taken)]
        eg -= ev[np.uint64(EV_EG + taken)]
        phase -= PHASE_INC[captured % 6]
    elif flag == CASTLING:
        rook = us * 6 + ROOK
        if target > source:
            rook_source, rook_target = target + 1, target - 1
        else:
            rook_source, rook_target = target - 2, target + 1
        mg += ev[np.uint64(rook * 64 + rook_target)] - ev[np.uint64(rook * 64 + rook_source)]
        eg += (
            ev[np.uint64(EV_EG + rook * 64 + rook_target)]
            - ev[np.uint64(EV_EG + rook * 64 + rook_source)]
        )
    st[ST_MG] = mg
    st[ST_EG] = eg
    st[ST_PHASE] = phase


@jit(inline=True, internal=not TESTING)
def tapered_unmake(st: Ints, undo: Ints) -> None:
    """Put back the three numbers `tapered_make` kept, after `unmake_body` has restored the
    ply. Nothing is recomputed, so the numbers are the ones the position had."""
    slot = st[PLY] * UNDO_FIELDS
    st[ST_MG] = undo[np.uint64(slot + UNDO_MG)]
    st[ST_EG] = undo[np.uint64(slot + UNDO_EG)]
    st[ST_PHASE] = undo[np.uint64(slot + UNDO_PHASE)]


@jit()
def nn_flush_s(s: Search, ply: int) -> None:
    """Apply the updates owed from the deepest ply that is up to date down to `ply`.

    A ply is marked and not updated, so a line of them can be owed at once. They are
    applied oldest first, because each reads the lanes of the ply above it, and the walk
    up stops at the first ply that owes nothing: the root owes nothing, and every ply
    between it and here was either applied here or applied by an earlier reader.
    """
    pk = s.pk
    first = ply
    while pk[np.uint64(first * PK_WORDS + PK_PEND)] != 0:
        first -= 1
    bb = s.bb
    ftw = s.ftw
    hd = s.hd
    acc = s.acc
    for at in range(first + 1, ply + 1):
        word = np.uint64(at * PK_WORDS + PK_PEND)
        move = pk[word]
        pk[word] = 0
        nn_apply_body(move, at, bb, ftw, hd, acc)


@jit(inline=True, hot=True, internal=not TESTING)
def nn_bring_up_s(s: Search, ply: int) -> None:
    """The lanes of `ply` brought up to date if anything is owed at it.

    The three places that read lanes call this first: the head, the null move that copies
    the ply's lanes, and a king move, whose update reads the board and so is applied where
    it is made. Everywhere else the lanes are not read, and the update is never paid.
    """
    if s.pk[np.uint64(ply * PK_WORDS + PK_PEND)] != 0:
        nn_flush_s(s, ply)


@jit()
def make_s(move: int, s: Search) -> None:
    """`make` on the search's position, with the tapered numbers carried across it.

    A king move pays off whatever the plies below it owe, here, on the board they were
    made on. A deferred update reads the two king squares off the board it is applied on,
    which is only the board it was made on for as long as no king has moved since.
    """
    st = s.st
    if move_piece(move) % 6 == KING:
        nn_bring_up_s(s, st[PLY])
    tapered_make(move, st, s.undo, s.ev)
    make_body(move, s.bb, s.sq, st, s.undo, s.keys, s.tab)


@jit()
def unmake_s(move: int, s: Search) -> None:
    """`unmake` on the search's position, with the tapered numbers put back."""
    st = s.st
    s.pk[np.uint64(st[PLY] * PK_WORDS + PK_PEND)] = 0  # a mark no reader took is never paid
    unmake_body(move, s.bb, s.sq, st, s.undo)
    tapered_unmake(st, s.undo)


@jit()
def nn_make_s(move: int, s: Search) -> None:
    """The accumulator update the ply just entered owes, recorded rather than applied.

    The profile has 0.73 makes and 0.38 head calls a node: the evaluation cache and the
    lazy gate answer most nodes without the head, and the update those made fed nothing.
    A mark costs one store, and `nn_bring_up_s` pays the line off where the lanes are
    actually read.

    A king move is applied here instead. Its update can rebuild a perspective from the
    board, and a castling moves the rook, neither of which a deferred update still has the
    board for; applying it at once is also what lets every deferred update read the kings
    off the board it is applied on, which is why `make_s` pays the line off first.
    """
    hd = s.hd
    if hd.shape[0] < 2:
        return
    st = s.st
    ply = st[PLY]
    word = np.uint64(ply * PK_WORDS + PK_PEND)
    if move_piece(move) % 6 == KING:
        # `make_s` brought the plies below up before it moved the king, so the parent's
        # lanes are here and the board this reads is the one the move was made on.
        s.pk[word] = 0
        nn_make_body(move, s.bb, st, s.ftw, hd, s.acc, s.kacc, s.kbb)
        return
    s.pk[word] = move


@jit(inline=True, hot=True, internal=not TESTING)
def count_node(s: Search) -> bool:
    """Count a node, read the clock on the stride, and say whether the search has stopped."""
    ss = s.ss
    nodes = ss[SS_NODES] + 1
    ss[SS_NODES] = nodes
    if (nodes & ss[SS_STRIDE_MASK]) == 0 and (
        now(s.clk) >= ss[SS_DEADLINE] or (ss[SS_NODE_LIMIT] != 0 and nodes >= ss[SS_NODE_LIMIT])
    ):
        ss[SS_STOP] = 1
    return bool(ss[SS_STOP] != 0)


@jit(inline=True, internal=not TESTING)
def pawn_slot(bb: Bitboards, stm: int) -> int:
    """The correction table's slot for the pawn structure, on the side to move's half."""
    h = bb[WP] * np.uint64(CORR_MIX_A) ^ (bb[BP] * np.uint64(CORR_MIX_B))
    h ^= h >> np.uint64(29)
    h *= np.uint64(CORR_MIX_C)
    return int(h >> np.uint64(64 - CORR_BITS)) + stm * CORR_SIZE


@jit(inline=True, hot=True, internal=not TESTING)
def pawn_history_row(bb: Bitboards) -> int:
    """The pawn-structure history's row for the pawn structure: the same mix as the
    correction slot, PAWN_HISTORY_BITS wide, times the keys a row holds. The piece key
    carries the colour, so the side needs no bit of its own."""
    if PAWN_HISTORY == 0:
        return 0
    h = bb[WP] * np.uint64(CORR_MIX_A) ^ (bb[BP] * np.uint64(CORR_MIX_B))
    h ^= h >> np.uint64(29)
    h *= np.uint64(CORR_MIX_C)
    return int(h >> np.uint64(64 - PAWN_HISTORY_BITS)) * CONT_KEYS


@jit(inline=True, internal=not TESTING)
def attack_map(side: int, bb: Bitboards, tab: Bitboards) -> np.uint64:
    """Every square `side` attacks, with the occupancy as it stands in `bb`."""
    base = side * 6
    pawns = bb[np.uint64(base + PAWN)]
    if side == WHITE:
        attacks = ((pawns & NOT_FILE_A) << np.uint64(7)) | ((pawns & NOT_FILE_H) << np.uint64(9))
    else:
        attacks = ((pawns & NOT_FILE_A) >> np.uint64(9)) | ((pawns & NOT_FILE_H) >> np.uint64(7))
    attacks |= tab[np.uint64(tables.KING + lsb(bb[np.uint64(base + KING)]))]
    knights = bb[np.uint64(base + KNIGHT)]
    while knights != 0:
        attacks |= tab[np.uint64(tables.KNIGHT + lsb(knights))]
        knights &= knights - ONE
    occupancy = bb[OCC_ALL]
    diagonal = bb[np.uint64(base + BISHOP)] | bb[np.uint64(base + QUEEN)]
    while diagonal != 0:
        attacks |= bishop_attacks(lsb(diagonal), occupancy, tab)
        diagonal &= diagonal - ONE
    straight = bb[np.uint64(base + ROOK)] | bb[np.uint64(base + QUEEN)]
    while straight != 0:
        attacks |= rook_attacks(lsb(straight), occupancy, tab)
        straight &= straight - ONE
    return np.uint64(attacks)


@jit()
def threat_map(side: int, bb: Bitboards, tab: Bitboards, targets: np.uint64) -> np.uint64:
    """The squares of `targets` that `side` attacks: `attack_map(side) & targets` built
    without the sliders that cannot reach one.

    A slider's attacks under any occupancy are a subset of its attacks on an empty board,
    so a piece whose empty-board ray meets no target adds nothing to the intersection and
    its magic lookup is skipped. Pawns, knights and the king keep their single table read.
    1.53 slider lookups a node fail that test (Codex's hot-path audit of v56, 2026-09-10).
    Out of line on purpose: inlined, the gate's branches grow every caller of `evaluate`.
    Compiled once, for a plain int64 side: `threat_slot` casts the two colour constants
    before the calls, since a module constant passed as an argument is typed as a literal
    and numba then compiles one copy per colour, a second compile at import for a folded
    pawn shift the branch predictor gives away nothing on. `WARMED` holds it to one.
    """
    base = side * 6
    pawns = bb[np.uint64(base + PAWN)]
    if side == WHITE:
        attacks = ((pawns & NOT_FILE_A) << np.uint64(7)) | ((pawns & NOT_FILE_H) << np.uint64(9))
    else:
        attacks = ((pawns & NOT_FILE_A) >> np.uint64(9)) | ((pawns & NOT_FILE_H) >> np.uint64(7))
    attacks |= tab[np.uint64(tables.KING + lsb(bb[np.uint64(base + KING)]))]
    knights = bb[np.uint64(base + KNIGHT)]
    while knights != 0:
        attacks |= tab[np.uint64(tables.KNIGHT + lsb(knights))]
        knights &= knights - ONE
    occupancy = bb[OCC_ALL]
    diagonal = bb[np.uint64(base + BISHOP)] | bb[np.uint64(base + QUEEN)]
    while diagonal != 0:
        square = lsb(diagonal)
        if (tab[np.uint64(tables.BISHOP_RAY + square)] & targets) != 0:
            attacks |= bishop_attacks(square, occupancy, tab)
        diagonal &= diagonal - ONE
    straight = bb[np.uint64(base + ROOK)] | bb[np.uint64(base + QUEEN)]
    while straight != 0:
        square = lsb(straight)
        if (tab[np.uint64(tables.ROOK_RAY + square)] & targets) != 0:
            attacks |= rook_attacks(square, occupancy, tab)
        straight &= straight - ONE
    return np.uint64(attacks & targets)


@jit(inline=True, internal=not TESTING)
def threat_slot(bb: Bitboards, tab: Bitboards, stm: int) -> int:
    """The threat table's slot: each side's pieces the other attacks, on the side to
    move's half."""
    # Plain int64 sides: a constant passed straight in is a literal type to numba and
    # compiles `threat_map` once per colour (see its docstring).
    by_black = int(np.int64(BLACK))
    by_white = int(np.int64(WHITE))
    white = threat_map(by_black, bb, tab, bb[OCC_WHITE])
    black = threat_map(by_white, bb, tab, bb[OCC_BLACK])
    h = white * np.uint64(THREAT_MIX_A) ^ (black * np.uint64(THREAT_MIX_B))
    h ^= h >> np.uint64(29)
    h *= np.uint64(THREAT_MIX_C)
    return int(h >> np.uint64(64 - CORR_BITS)) + stm * CORR_SIZE


@jit(inline=True, internal=not TESTING)
def correction_at(bb: Bitboards, st: Ints, corr: Int32s, tcorr: Int32s, threat: int) -> int:
    """`correction` with the threats' slot already in hand: only the pawn structure's
    slot is built here."""
    return (
        int(corr[np.uint64(pawn_slot(bb, st[STM]))]) + int(tcorr[np.uint64(threat)])
    ) // CORR_GRAIN


@jit(inline=True, internal=not TESTING)
def correction(bb: Bitboards, st: Ints, tab: Bitboards, corr: Int32s, tcorr: Int32s) -> int:
    """The centipawns the two tables add to this position's static evaluation: the pawn
    structure's slot and the threats' slot, summed."""
    return correction_at(bb, st, corr, tcorr, threat_slot(bb, tab, st[STM]))


@jit(inline=True, internal=not TESTING)
def move_slot(table: Int32s, slot: int, gap: int, weight: int) -> None:
    """Move one slot towards `gap` by `weight` out of CORR_WEIGHT_SCALE, capped."""
    value = (
        int(table[np.uint64(slot)]) * (CORR_WEIGHT_SCALE - weight) + gap * CORR_GRAIN * weight
    ) // CORR_WEIGHT_SCALE
    if value > CORR_MAX:
        value = CORR_MAX
    elif value < -CORR_MAX:
        value = -CORR_MAX
    table[np.uint64(slot)] = value


@jit(inline=True, internal=not TESTING)
def update_correction(
    depth: int, gap: int, bb: Bitboards, st: Ints, tab: Bitboards, corr: Int32s, tcorr: Int32s
) -> None:
    """Move both tables' slots towards `gap`, the search score less the corrected static
    evaluation, by a weight that grows with the node's depth."""
    weight = (depth + 1) * (depth + 1)
    if weight > CORR_WEIGHT_MAX:
        weight = CORR_WEIGHT_MAX
    stm = st[STM]
    move_slot(corr, pawn_slot(bb, stm), gap, weight)
    move_slot(tcorr, threat_slot(bb, tab, stm), gap, weight)


@jit()
def evaluate_tables(st: Ints) -> int:
    """Tapered piece-square score in centipawns from the side to move's point of view.

    The three numbers are `set_tapered`'s at the root and `tapered_make`'s below it, so a
    leaf reads them instead of walking the pieces. The phase is clamped here rather than
    where it is carried, since a promotion can push it past a full board.
    """
    mg = int(st[ST_MG])
    eg = int(st[ST_EG])
    phase = int(st[ST_PHASE])
    if phase > PHASE_MAX:
        phase = PHASE_MAX
    score = (mg * phase + eg * (PHASE_MAX - phase)) // PHASE_MAX
    return score if st[STM] == WHITE else -score


# Mop-up against a bare king: Chess 4.x's term (Slate and Atkin 1977), 4.7 times the
# losing king's distance from the centre plus 1.6 times fourteen less the distance
# between the kings, in tenths of a pawn here so the gradient outweighs the tables'
# flat score; for bishop and knight the corner that matters is the bishop's colour, so
# the distance to the nearer corner of that colour stands in for the centre distance.
MOP_UP_EDGE = 47
MOP_UP_KINGS = 16
CORNER_REACH = 7


@jit(internal=True)
def manhattan(a: int, b: int) -> int:
    return abs((a & 7) - (b & 7)) + abs((a >> 3) - (b >> 3))


@jit()
def mop_up(bb: Bitboards, st: Ints) -> int:
    """The mop-up term from the side to move's view, or zero when neither king is bare.

    The winner needs mating material: a rook or a queen, two bishops, or a bishop and a
    knight. Pawns and a lone minor piece leave the tables to speak.
    """
    if popcount(bb[OCC_WHITE]) == 1:
        loser = WHITE
    elif popcount(bb[OCC_BLACK]) == 1:
        loser = BLACK
    else:
        return 0
    winner = 1 - loser
    rooks = popcount(bb[np.uint64(winner * 6 + ROOK)]) + popcount(bb[np.uint64(winner * 6 + QUEEN)])
    bishops = popcount(bb[np.uint64(winner * 6 + BISHOP)])
    knights = popcount(bb[np.uint64(winner * 6 + KNIGHT)])
    if rooks == 0 and bishops < 2 and not (bishops == 1 and knights >= 1):
        return 0
    loser_king = lsb(bb[np.uint64(loser * 6 + KING)])
    winner_king = lsb(bb[np.uint64(winner * 6 + KING)])
    if rooks == 0 and bishops == 1:
        bishop = lsb(bb[np.uint64(winner * 6 + BISHOP)])
        light = ((bishop & 7) + (bishop >> 3)) & 1 == 1
        first, second = (A8, H1) if light else (A1, H8)
        corner = min(manhattan(loser_king, first), manhattan(loser_king, second))
        edge = max(0, CORNER_REACH - corner)
    else:
        file = loser_king & 7
        rank = loser_king >> 3
        edge = max(3 - file, file - 4) + max(3 - rank, rank - 4)
    term = MOP_UP_EDGE * edge + MOP_UP_KINGS * (14 - manhattan(loser_king, winner_king))
    return term if st[STM] == winner else -term


# The king and pawn against king bitbase, `weights/kpk.bin` from `tools/kpk_bitbase.py`:
# one bit per (side to move, white king, black king, pawn on files a to d), set where the
# pawn's side wins with best play, checked against the shipped Syzygy table. A position
# with a black pawn is flipped and recoloured, one with the pawn on files e to h mirrored.
KPK_PAWN_SQUARES = 24
KPK_WIN = 2000  # centipawns: past any material the tables can give, short of a mate


@jit()
def kpk_wins(bb: Bitboards, st: Ints, kpk: Uint8s) -> int:
    """1 when the pawn's side wins a king and pawn against king, 0 when it is drawn, -1
    when the position is any other material (or no bitbase is loaded)."""
    if kpk.shape[0] == 0 or popcount(bb[OCC_ALL]) != 3:
        return -1
    if bb[WP] != 0:
        pawn_side = WHITE
    elif bb[BP] != 0:
        pawn_side = BLACK
    else:
        return -1
    pawn = lsb(bb[np.uint64(pawn_side * 6 + PAWN)])
    strong = lsb(bb[np.uint64(pawn_side * 6 + KING)])
    weak = lsb(bb[np.uint64((1 - pawn_side) * 6 + KING)])
    stm = st[STM]
    if pawn_side == BLACK:
        # Flip the board and the colours: the pawn's side is white in the table.
        pawn ^= 56
        strong ^= 56
        weak ^= 56
        stm = 1 - stm
    if (pawn & 7) >= 4:
        pawn ^= 7
        strong ^= 7
        weak ^= 7
    index = ((stm * 64 + strong) * 64 + weak) * KPK_PAWN_SQUARES + (pawn // 8 - 1) * 4 + (pawn & 7)
    return int((kpk[np.uint64(index >> 3)] >> (index & 7)) & 1)


@jit()
def dead_draw(bb: Bitboards) -> bool:
    """A bare king against material that cannot mate: a lone minor piece, or two knights.

    The tables score such material as a win and the search would trade into it; the
    root's tablebase probe only sees positions the search has already reached.
    """
    for loser in (WHITE, BLACK):
        if popcount(bb[np.uint64(OCC_WHITE + loser)]) != 1:
            continue
        winner = 1 - loser
        if (
            bb[np.uint64(winner * 6 + PAWN)] != 0
            or bb[np.uint64(winner * 6 + ROOK)] != 0
            or bb[np.uint64(winner * 6 + QUEEN)] != 0
        ):
            return False
        bishops = popcount(bb[np.uint64(winner * 6 + BISHOP)])
        knights = popcount(bb[np.uint64(winner * 6 + KNIGHT)])
        return bool(bishops + knights <= 1 or (bishops == 0 and knights == 2))
    return False


@jit()
def eval_is_verdict(bb: Bitboards, st: Ints, kpk: Uint8s) -> bool:
    """True where `evaluate` returns a verdict rather than an estimate.

    Two of its returns are decided rather than estimated: material that cannot mate, a
    draw by the rules; and a king and pawn against king the bitbase settled either way.
    A verdict is not a gap a static evaluation could have closed, so the correction
    history must not learn from it, and the score's size does not say which return it was:
    both draws come back as zero and a win can come back as 1967 (the rule of 57e2625,
    Codex's review of 2026-09-08, finding 1). No Syzygy probe runs inside the kernel; the
    root's probe is in `agent.py` and returns before the search.

    Every node out of check calls this, so it opens with one popcount: `kpk_wins` needs
    three men, and `dead_draw` needs a bare king against a king and at most two knights or
    one minor, which is four. Nothing above four men can be a verdict here.
    """
    if popcount(bb[OCC_ALL]) > 4:
        return False
    if dead_draw(bb):
        return True
    return kpk_wins(bb, st, kpk) >= 0


@jit()
def evaluate(alpha: int, beta: int, s: Search) -> int:
    """The net's score when a net is loaded and the tables' score sits within LAZY_MARGIN
    of the window; the tables' score otherwise; either shifted by the correction history
    for the pawn structure and for the pieces under attack; against a bare king the
    tables plus the mop-up term, since the net has no gradient there. Measured here: the
    gate fires often enough in plain alpha-beta that the tables' cost buys a quarter of
    the speed back.

    Far outside the window the exact value does not change the search's decision, and
    the tables cost a tenth of a microsecond against the head's microseconds; the zero
    windows of the principal variation search make that the common case.

    The raw net score is cached in the word beside the position's table entry and served
    on a tag match, so the head runs once for a position while that word holds it. The
    threats' correction slot rides in the same word, so a hit skips the two attack maps
    that build it as well as the head.
    """
    if dead_draw(s.bb):
        return 0
    st = s.st
    known = kpk_wins(s.bb, st, s.kpk)
    if known == 0:
        return 0  # a king and pawn against king the bitbase calls a draw
    tables = evaluate_tables(st)
    if known == 1:
        # The pawn's side wins: a decisive score with the tables as the gradient, from the
        # side to move's view.
        pawn_side = WHITE if s.bb[WP] != 0 else BLACK
        return tables + KPK_WIN if st[STM] == pawn_side else tables - KPK_WIN
    bonus = mop_up(s.bb, st)
    if bonus != 0:
        return tables + bonus  # a bare king: the tables and the mop-up term, not the net
    hd = s.hd
    if hd.shape[0] < 2 or tables + LAZY_MARGIN <= alpha or tables - LAZY_MARGIN >= beta:
        # The correction history's shift for this pawn structure and these threats, from
        # the side to move's view.
        return tables + correction(s.bb, st, s.tab, s.corr, s.tcorr)
    key = st[HASH]
    score, threat = eval_cached(s.tt, key)
    if score == EVAL_NONE:
        nn_bring_up_s(s, st[PLY])  # the head is about to read this ply's lanes
        score = evaluate_net(st, int(popcount(s.bb[OCC_ALL])), hd, s.acc, s.xs, s.hb)
        threat = threat_slot(s.bb, s.tab, st[STM])
        eval_store(s.tt, key, score, threat)
    # The same shift, with the threats' slot read out of the score word rather than built
    # from the two attack maps.
    return score + correction_at(s.bb, st, s.corr, s.tcorr, threat)


@jit(inline=True, hot=True, internal=not TESTING)
def null_move_reduction(depth: int, margin: int) -> int:
    """Plies taken off the null-move search; margin is the evaluation over beta."""
    by_eval = margin // NULL_MOVE_EVAL_STEP
    if by_eval > NULL_MOVE_EVAL_MAX:
        by_eval = NULL_MOVE_EVAL_MAX
    return NULL_MOVE_BASE + depth // NULL_MOVE_DIVISOR + by_eval


@jit()
def null_move_allowed(checked: bool, ply: int, depth: int, beta: int, s: Search) -> bool:
    """Whether the node may try a null move: never in check, at the root, right after
    another null move, against a mate bound, or without a piece besides pawns and kings."""
    if checked or ply == 0 or depth < NULL_MOVE_MIN_DEPTH or beta >= MATE_BOUND:
        return False
    st = s.st
    if last_was_null(st, s.undo):
        return False
    us = st[STM]
    bb = s.bb
    pieces = bb[np.uint64(OCC_WHITE + us)] & ~(
        bb[np.uint64(us * 6 + PAWN)] | bb[np.uint64(us * 6 + KING)]
    )
    return bool(pieces != 0)


@jit(inline=True, hot=True, internal=not TESTING)
def draw_score(ply: int, s: Search) -> int:
    """The score of a draw at `ply`, from the side to move: the contempt against the root side.

    SS_CONTEMPT holds the margin the root side gives up for a draw, positive when its last
    root score says it stands better, so a repetition, a fifty-move draw or a stalemate
    scores below zero at the root side's plies (the even ones) and above zero at the
    other side's, and the search plays on when ahead and takes the draw when behind.
    """
    contempt = int(s.ss[SS_CONTEMPT])
    if (ply & 1) == 0:
        return -contempt
    return contempt


@jit(inline=True, hot=True, internal=not TESTING)
def repeated(ply: int, s: Search) -> bool:
    """Whether the position at `ply` occurred before, on the path or earlier in the game.

    Only positions since the last capture or pawn move can repeat, so the scan goes back
    at most halfmove-clock plies, two at a time, first along the search path and then
    into the game before the root. One earlier occurrence is enough: the side that wants
    the draw can repeat again.
    """
    st = s.st
    path = s.path
    key = st[HASH]
    path[np.uint64(ply)] = key
    clock = st[HALF]
    game_length = s.ss[SS_GAME_LENGTH]
    game = s.game
    distance = 2
    while distance <= clock:
        if distance <= ply:
            other = path[np.uint64(ply - distance)]
        else:
            index = game_length - (distance - ply)
            if index < 0:
                break
            other = game[np.uint64(index)]
        if other == key:
            return True
        distance += 2
    return False


@jit(internal=True)
def has_legal_move(ply: int, s: Search) -> bool:
    """Whether the side to move has a legal move, generated into the list at `ply`.

    What python-chess asks of the position after an announced fifty-move move: no reply
    means the game ended there, so no claim was available.
    """
    ml = s.ml
    base = ply * MOVES_PER_PLY
    count = gen_all_s(s, base)
    for i in range(count):
        move = ml[np.uint64(base + i)]
        make_s(move, s)
        legal = legal_after_s(s)
        unmake_s(move, s)
        if legal:
            return True
    return False


@jit()
def fifty_move_draw(ply: int, s: Search) -> bool:
    """Whether the referee's fifty-move draw already stands, with the clock at 99 or more.

    The referee asks `board.outcome(claim_draw=True)` before each side is asked to move
    (`harness/referee.py`), and python-chess grants the claim one ply early: at a halfmove
    clock of 99 the side to move is drawn by announcing any legal move that does not zero
    the clock, whether it wants the draw or not. A clock of 100 is drawn on the board and
    needs no announcement. A capture or a pawn move zeroes the clock, so neither can carry
    the claim, and a position whose every legal move zeroes it plays on: testing the clock
    alone would score a real win as a draw. Nor does a move that ends the game carry it:
    python-chess reads the claim through `is_fifty_moves`, which asks the position after
    the announced move for a legal reply, so a nonzeroing move that mates or stalemates
    grants nothing and the mate stands (Codex 102's witness, #467). One claim is enough,
    so a position holding both a mate and a quiet move with a reply is drawn before either
    is played. The generators are reached only at 99, so the call sites test the clock
    first and the cost never lands on an ordinary node.
    """
    st = s.st
    half = st[HALF]
    if half >= 100:
        return True
    if half != 99:
        return False
    ml = s.ml
    base = ply * MOVES_PER_PLY
    # The announced move is tried from this ply's list and its replies from the next
    # ply's, which is the child's own scratch and is written again before it is read.
    if base + 2 * MOVES_PER_PLY > ml.shape[0]:
        return False  # the deepest plies own no list to generate into
    count = gen_quiet_s(s, base)
    for i in range(count):
        move = ml[np.uint64(base + i)]
        if move_piece(move) % 6 == PAWN:
            continue
        make_s(move, s)
        granted = legal_after_s(s) and has_legal_move(ply + 1, s)
        unmake_s(move, s)
        if granted:
            return True
    return False


# The small helpers below are pasted into their callers: a separately compiled function
# costs 40 ms of import whatever its size (`notes/measurements/2026-09-06-import-cache.md`).
@jit(inline=True, hot=True, internal=not TESTING)
def to_tt(score: int, ply: int) -> int:
    """A mate score from the root's point of view becomes one from the node's."""
    if score >= MATE_BOUND:
        return score + ply
    if score <= -MATE_BOUND:
        return score - ply
    return score


@jit(inline=True, hot=True, internal=not TESTING)
def from_tt(score: int, ply: int) -> int:
    """Undo to_tt at the ply the entry is read at."""
    if score >= MATE_BOUND:
        return score - ply
    if score <= -MATE_BOUND:
        return score + ply
    return score


@intrinsic
def prefetch_line(typingctx: Any, array: Any, index: Any) -> Any:
    """Ask the cache for the line of `array[index]` ahead of its use: `llvm.prefetch`.

    The table is 134 MB and its probes are random, so the child's probe is the search's one
    cache miss a node. Issued once a legal move is made, before the accumulator update, the
    line arrives while the update runs.
    """
    sig = types.none(array, index)

    def codegen(context: Any, builder: Any, signature: Any, args: Any) -> Any:
        array_type = signature.args[0]
        values = context.make_array(array_type)(context, builder, value=args[0])
        pointer = cgutils.get_item_pointer(  # type: ignore[no-untyped-call]
            context, builder, array_type, values, [args[1]], wraparound=False
        )
        pointer = builder.bitcast(pointer, cgutils.voidptr_t)
        i32 = llvm_ir.IntType(32)
        fnty = llvm_ir.FunctionType(llvm_ir.VoidType(), [pointer.type, i32, i32, i32])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "llvm.prefetch.p0")  # type: ignore[no-untyped-call]
        builder.call(fn, [pointer, i32(0), i32(3), i32(1)])  # a read, kept in every cache level
        return context.get_dummy_value()

    return sig, codegen


@jit(inline=True, hot=True, internal=not TESTING)
def prefetch(tt: Ints, key: int) -> None:
    """Ask the cache for the table's entry of `key` (`prefetch_line`)."""
    prefetch_line(tt, (key & TT_MASK) * TT_STRIDE)  # type: ignore[call-arg]


@jit(inline=True, internal=not TESTING)
def tt_tag(key: int) -> int:
    return (key >> TAG_SHIFT) & TAG_MASK


@jit(inline=True, internal=not TESTING)
def eval_tag(key: int) -> int:
    """The score word's tag for `key`: the key's top 32 bits, never 0."""
    return ((key >> EVAL_TAG_SHIFT) & EVAL_WORD_MASK) | 1


@jit(inline=True, internal=not TESTING)
def eval_cached(tt: Ints, key: int) -> tuple[int, int]:
    """The raw net score and the threats' slot in the word beside `key`'s entry, or
    EVAL_NONE and 0 when the word holds neither."""
    word = tt[np.uint64((key & TT_MASK) * TT_STRIDE + 1)]
    if ((word >> EVAL_TAG_SHIFT) & EVAL_WORD_MASK) != eval_tag(key):
        return EVAL_NONE, 0
    score = word & EVAL_SCORE_MASK
    if score >= EVAL_SCORE_SIGN:
        score -= 1 << EVAL_SLOT_SHIFT  # the low 17 bits are two's complement
    return int(score), int((word >> EVAL_SLOT_SHIFT) & EVAL_SLOT_MASK)


@jit(inline=True, internal=not TESTING)
def eval_store(tt: Ints, key: int, score: int, slot: int) -> None:
    """Store `score` and the threats' slot `slot` in the word beside `key`'s entry, under
    a tag only `key` matches. A score outside the 17 bits is not stored."""
    if score < EVAL_SCORE_MIN or score > EVAL_SCORE_MAX:
        return
    tagged = (
        (score & EVAL_SCORE_MASK) | (slot << EVAL_SLOT_SHIFT) | (eval_tag(key) << EVAL_TAG_SHIFT)
    )
    tt[np.uint64((key & TT_MASK) * TT_STRIDE + 1)] = tagged


@jit()
def tt_probe(tt: Ints, key: int) -> int:
    """The packed word stored for `key`, or 0."""
    stored = tt[np.uint64((key & TT_MASK) * TT_STRIDE)]
    if stored != 0 and ((stored >> TAG_SHIFT) & TAG_MASK) == tt_tag(key):
        return int(stored)
    return 0


@jit(inline=True, internal=not TESTING)
def compact_move(move: int) -> int:
    """The move's source, target, flag and promotion in 16 bits, for the packed word.

    The promotion piece is KNIGHT to QUEEN (1 to 4) or 0 in the kernel's encoding; here it
    is stored as 0 to 3 and read back as a promotion only when the moving piece is a pawn
    on its way to the last rank, so no bit is spent on whether the move promotes.
    """
    promotion = move_promotion(move)
    if promotion != 0:
        promotion -= 1
    return (move & 4095) | (move_flag(move) << 12) | (promotion << 14)


@jit(inline=True, hot=True, internal=not TESTING)
def tt_move(data: int, sq: Ints, stm: int) -> int:
    """The stored move in the kernel's encoding, its pieces read from the board.

    A true hit is the same position, so the piece on the source square and the piece on
    the target square are the move's own; an en passant capture takes the opponent's pawn
    from a square the board shows empty. On a false tag hit the pieces are whatever stands
    there, and `pick_setup` checks the move against the position before it is tried.
    """
    compact = data & MOVE_MASK
    if compact == 0:
        return 0  # stored without a move: a stand pat, or a node that failed low
    source = compact & 63
    target = (compact >> 6) & 63
    flag = (compact >> 12) & 3
    piece = sq[np.uint64(source)]
    captured = sq[np.uint64(target)]
    if flag == EN_PASSANT:
        captured = BP if stm == WHITE else WP
    promotion = 0
    if (piece == WP and target >= 56) or (piece == BP and target < 8):
        promotion = ((compact >> 14) & 3) + 1
    return encode(source, target, piece, captured, promotion, flag)


@jit(inline=True, internal=not TESTING)
def tt_depth(data: int) -> int:
    return ((data >> DEPTH_SHIFT) & DEPTH_MASK) - DEPTH_OFFSET


@jit(inline=True, hot=True, internal=not TESTING)
def tt_bound(data: int) -> int:
    return (data >> BOUND_SHIFT) & 3


@jit(inline=True, hot=True, internal=not TESTING)
def tt_score(data: int) -> int:
    return ((data >> SCORE_SHIFT) & SCORE_MASK) - SCORE_OFFSET


@jit()
def tt_store(
    tt: Ints, key: int, depth: int, score: int, bound: int, move: int, generation: int
) -> None:
    """Depth-preferred replacement; an entry from an earlier move of the game always goes.

    The depth is clamped to the field before it is compared or packed, so a depth outside
    it never wraps into the neighbouring fields. A write with no move of its own over an
    entry for the same position keeps the move that entry already holds, so a stand pat or
    a fail low does not erase what a deeper visit found.
    """
    if depth > DEPTH_MAX:
        depth = DEPTH_MAX
    elif depth < DEPTH_MIN:
        depth = DEPTH_MIN
    slot = (key & TT_MASK) * TT_STRIDE
    tag = tt_tag(key)
    stored = tt[np.uint64(slot)]
    same_position = stored != 0 and ((stored >> TAG_SHIFT) & TAG_MASK) == tag
    if stored != 0 and tt_depth(stored) > depth:
        same_generation = ((stored >> GENERATION_SHIFT) & GENERATION_MASK) == generation
        if same_position or same_generation:
            return
    compact = compact_move(move)
    if move == 0 and same_position:
        compact = stored & MOVE_MASK  # the move already known for this position is kept
    packed = compact | ((depth + DEPTH_OFFSET) << DEPTH_SHIFT) | (bound << BOUND_SHIFT)
    packed |= (generation & GENERATION_MASK) << GENERATION_SHIFT
    packed |= (score + SCORE_OFFSET) << SCORE_SHIFT
    packed |= tag << TAG_SHIFT
    tt[np.uint64(slot)] = packed


@jit(inline=True, internal=not TESTING)
def history_bonus(depth: int) -> int:
    bonus = 32 * depth * depth
    if bonus > HISTORY_BONUS_CAP:
        bonus = HISTORY_BONUS_CAP
    return bonus if depth > 0 else 0


@jit(inline=True, internal=not TESTING)
def bump(table: Ints | Int16s, index: int, delta: int) -> None:
    """The gravity update: entries move toward the bonus and never leave the bound."""
    entry = table[np.uint64(index)]
    table[np.uint64(index)] = entry + delta - entry * abs(delta) // HISTORY_MAX


@jit(inline=True, internal=not TESTING)
def capture_index(move: int) -> int:
    piece = move_piece(move) % 6
    captured = move_captured(move) % 6
    return (piece * 64 + move_target(move)) * 6 + captured


@jit(inline=True, internal=not TESTING)
def quiet_index(move: int, us: int) -> int:
    return (us * 64 + move_source(move)) * 64 + move_target(move)


@jit(inline=True, internal=not TESTING)
def cont_index(previous: int, move: int) -> int:
    """The continuation history slot of a quiet move played after `previous`."""
    before = move_piece(previous) * 64 + move_target(previous)
    return before * CONT_KEYS + move_piece(move) * 64 + move_target(move)


@jit()
def noisy_score(move: int, hc: Ints) -> int:
    """Order in the noisy stage: the victim's worth, the attacker, the capture history."""
    captured = move_captured(move)
    if captured == EMPTY:
        # A promotion by a push: the queen alongside the big captures, the rest last.
        return MVV_AUGMENT[QUEEN] if move_promotion(move) == QUEEN else -PROMOTION_SCORE
    return int(
        MVV_AUGMENT[captured % 6] - move_piece(move) % 6 + hc[np.uint64(capture_index(move))]
    )


@jit()
def quiet_score(
    move: int,
    us: int,
    ply: int,
    killers: Ints,
    hq: Ints,
    chist: Int16s,
    previous: int,
    phist: Int16s,
    prow: int,
) -> int:
    """Order in the quiet stage: the killer of the ply, then the butterfly history plus
    the continuation history after `previous`, the move that led to the node (0: none),
    plus the pawn-structure history in row `prow` (#499).
    """
    if move == killers[np.uint64(ply)]:
        return KILLER_SCORE
    score = int(hq[np.uint64(quiet_index(move, us))])
    if previous != 0:
        score += int(chist[np.uint64(cont_index(previous, move))])
    if PAWN_HISTORY > 0:
        score += int(phist[np.uint64(prow + move_piece(move) * 64 + move_target(move))])
    return score


@jit(internal=True)
def quiescence_score(move: int) -> int:
    """Most valuable victim first, least valuable attacker; a push promotion counts a pawn."""
    captured = move_captured(move)
    victim = ORDER_VALUE[captured % 6] if captured != EMPTY else ORDER_VALUE[PAWN]
    return 16 * victim - ORDER_VALUE[move_piece(move) % 6]


@jit(inline=True, internal=not TESTING)
def pick_body(ml: Ints, ms: Ints, start: int, end: int) -> int:
    """Swap the best-scored move among ml[start:end] into ml[start] and return it.

    The running maximum's value is kept beside its index rather than read back as
    `ms[best]`: compiled, the read-back is a second load with its own index fixup every
    entry, and dropping it takes the scan from 0.58 to 0.45 ns an entry on this machine
    (`notes/measurements/2026-09-08-pick-scan.md`). The comparison is still strict, so
    the first of equal maxima still wins and the tree is unchanged.
    """
    best = start
    best_score = ms[np.uint64(start)]
    for i in range(start + 1, end):
        score = ms[np.uint64(i)]
        if score > best_score:
            best_score = score
            best = i
    if best != start:
        ml[np.uint64(start)], ml[np.uint64(best)] = ml[np.uint64(best)], ml[np.uint64(start)]
        ms[np.uint64(start)], ms[np.uint64(best)] = ms[np.uint64(best)], ms[np.uint64(start)]
    return int(ml[np.uint64(start)])


@jit()
def pick(ml: Ints, ms: Ints, start: int, end: int) -> int:
    """`pick_body` on arrays passed one by one: the root list, the tests and the benches."""
    return pick_body(ml, ms, start, end)


@jit(internal=True)
def pick_s(s: Search, start: int, end: int) -> int:
    """`pick` on the search's move list and scores."""
    return pick_body(s.ml, s.ms, start, end)


@jit()
def pseudo_legal_quiet(move: int, bb: Bitboards, sq: Ints, st: Ints, tab: Bitboards) -> bool:
    """Whether a quiet move from the table can be played here, castling excluded.

    A table move that was a capture is found in the noisy list; a quiet one is checked on
    its own so the quiet moves need not be generated for it. The king's safety is still
    the legality test after make. A castling move from the table is not tried early; it
    comes up in the quiet stage with its own conditions.
    """
    source = move_source(move)
    target = move_target(move)
    piece = move_piece(move)
    if sq[np.uint64(source)] != piece or sq[np.uint64(target)] != EMPTY or piece // 6 != st[STM]:
        return False
    if move_captured(move) != EMPTY or move_promotion(move) != 0:
        return False
    flag = move_flag(move)
    if flag == CASTLING:
        return False
    kind = piece % 6
    if kind == PAWN:
        forward = 8 if st[STM] == WHITE else -8
        if flag == NORMAL:
            return target == source + forward and target // 8 != 0 and target // 8 != 7
        home = 1 if st[STM] == WHITE else 6
        return (
            flag == DOUBLE_PUSH
            and source // 8 == home
            and target == source + 2 * forward
            and sq[np.uint64(source + forward)] == EMPTY
        )
    if flag != NORMAL:
        return False
    target_bit = np.uint64(1) << np.uint64(target)
    if kind == KING:
        return bool(tab[np.uint64(tables.KING + source)] & target_bit)
    if kind == KNIGHT:
        return bool(tab[np.uint64(tables.KNIGHT + source)] & target_bit)
    occupancy = bb[OCC_ALL]
    reach = np.uint64(0)
    if kind in (BISHOP, QUEEN):
        reach |= bishop_attacks(source, occupancy, tab)
    if kind in (ROOK, QUEEN):
        reach |= rook_attacks(source, occupancy, tab)
    return bool(reach & target_bit)


@jit()
def reward_noisy(move: int, ml: Ints, start: int, stop: int, depth: int, hc: Ints) -> None:
    """The capture that cut gains history; the captures in ml[start:stop] tried before it
    lose as much."""
    if move_captured(move) == EMPTY:
        return
    bonus = history_bonus(depth)
    bump(hc, capture_index(move), bonus)
    for i in range(start, stop):
        other = ml[np.uint64(i)]
        if other != 0 and other != move and move_captured(other) != EMPTY:
            bump(hc, capture_index(other), -bonus)


@jit()
def reward_quiet(
    move: int,
    ml: Ints,
    start: int,
    stop: int,
    depth: int,
    ply: int,
    us: int,
    killers: Ints,
    hq: Ints,
    chist: Int16s,
    previous: int,
    phist: Int16s,
    prow: int,
) -> None:
    """The quiet move that cut becomes the killer and gains history; the quiet moves in
    ml[start:stop] tried before it lose as much. The continuation history after
    `previous` moves the same way, unless `previous` is 0 (a null move, the root, or the
    table switched off). The pawn-structure history's row `prow` moves the same way
    when PAWN_HISTORY is on."""
    killers[np.uint64(ply)] = move
    bonus = history_bonus(depth)
    bump(hq, quiet_index(move, us), bonus)
    if previous != 0:
        bump(chist, cont_index(previous, move), bonus)
    if PAWN_HISTORY > 0:
        bump(phist, prow + move_piece(move) * 64 + move_target(move), bonus)
    tried = 0
    for i in range(start, stop):
        other = ml[np.uint64(i)]
        if other != 0 and other != move:
            malus = bonus
            if HISTORY_MALUS_SPAN > 0:
                malus = bonus * max(0, HISTORY_MALUS_SPAN - tried) // HISTORY_MALUS_SPAN
            bump(hq, quiet_index(other, us), -malus)
            if previous != 0:
                bump(chist, cont_index(previous, other), -malus)
            if PAWN_HISTORY > 0:
                bump(phist, prow + move_piece(other) * 64 + move_target(other), -malus)
            tried += 1


@jit(internal=True)
def pieces_immobile(base: int, ours: np.uint64, occupancy: np.uint64, s: Search) -> bool:
    """Whether every knight, bishop, rook and queen of the side whose pieces start at
    `base` attacks nothing but its own men, so none of them has even a pseudo-legal move.

    The king is left out on purpose: a stalemated king is exactly one whose moves are
    pseudo-legal and all illegal, so testing it would make the gate false in the positions
    the gate is for. Out of line, as `threat_map` is: inlined, three slider loops would
    grow `quiesce` itself, and the loops are reached only after the pawn tests pass.
    """
    bb = s.bb
    tab = s.tab
    free = ~ours
    knights = bb[np.uint64(base + KNIGHT)]
    while knights != 0:
        if (tab[np.uint64(tables.KNIGHT + lsb(knights))] & free) != 0:
            return False
        knights &= knights - ONE
    diagonal = bb[np.uint64(base + BISHOP)] | bb[np.uint64(base + QUEEN)]
    while diagonal != 0:
        if (bishop_attacks(lsb(diagonal), occupancy, tab) & free) != 0:
            return False
        diagonal &= diagonal - ONE
    straight = bb[np.uint64(base + ROOK)] | bb[np.uint64(base + QUEEN)]
    while straight != 0:
        if (rook_attacks(lsb(straight), occupancy, tab) & free) != 0:
            return False
        straight &= straight - ONE
    return True


@jit(inline=True, hot=True, internal=not TESTING)
def stalemate_gate(s: Search) -> bool:
    """Whether the side to move may have no move at all: true only when no pawn pushes to
    an empty square, no pawn attacks a square the other side occupies, and no piece but the
    king has a destination that is not one of its own men.

    Two bitboards answer nearly every node: a side with a free pawn push leaves here, and
    that is almost every node with pawns. A double push needs the single push's square
    empty, so the push test covers it, and a promotion push is a push like any other, so
    the last rank is not masked off. En passant is not tested and a pinned piece counts as
    mobile, which leaves the gate true where the position may still be stalemate and the
    legal loop then answers it; the gate never reports immobility where a move exists, so
    it can only leave the old score standing, never invent a draw.
    """
    bb = s.bb
    us = s.st[STM]
    base = us * 6
    pawns = bb[np.uint64(base + PAWN)]
    occupancy = bb[OCC_ALL]
    if us == WHITE:
        pushes = (pawns << np.uint64(8)) & ~occupancy
        attacks = ((pawns & NOT_FILE_A) << np.uint64(7)) | ((pawns & NOT_FILE_H) << np.uint64(9))
        theirs = bb[OCC_BLACK]
        ours = bb[OCC_WHITE]
    else:
        pushes = (pawns >> np.uint64(8)) & ~occupancy
        attacks = ((pawns & NOT_FILE_A) >> np.uint64(9)) | ((pawns & NOT_FILE_H) >> np.uint64(7))
        theirs = bb[OCC_WHITE]
        ours = bb[OCC_BLACK]
    if (pushes | (attacks & theirs)) != np.uint64(0):
        return False
    return pieces_immobile(base, ours, occupancy, s)


@jit()
def quiesce(alpha: int, beta: int, ply: int, s: Search) -> int:
    """Search captures and queen promotions until the position is quiet; fail-soft.

    In check there is no standing pat: every evasion is searched, quiet ones included, and
    no evasion at all is mate. The transposition table is probed before the stand pat and
    written after the node, at depth 0: most nodes are here, and one reached again by
    another capture order is answered without a second head or capture search. The write
    names the move that raised the score unless the node failed low, and a probe that does
    not cut orders that move first, so the second visit finds the same refutation at once.
    """
    if count_node(s):
        return 0
    if repeated(ply, s):
        return draw_score(ply, s)
    st = s.st
    ml = s.ml
    ms = s.ms
    tt = s.tt
    ss = s.ss
    checked = in_check_s(s)
    if st[HALF] >= 99 and not checked and fifty_move_draw(ply, s):
        return draw_score(ply, s)
    if ply >= MAX_PLY - 2:
        return evaluate(alpha, beta, s)
    key = st[HASH]
    table_move = 0
    entry = tt_probe(tt, key)
    if entry != 0:
        score = from_tt(tt_score(entry), ply)
        bound = tt_bound(entry)
        if st[HALF] < TT_CUTOFF_HALF and (
            bound == TT_EXACT
            or (bound == TT_LOWER and score >= beta)
            or (bound == TT_UPPER and score <= alpha)
        ):
            return score
        # The entry did not cut, so its move goes to the front of the noisy list. It is
        # only taken when the generator produced it here, which is the test below.
        table_move = tt_move(entry, s.sq, st[STM])
    targets = evasion_targets_s(s) if checked else np.uint64(0)
    s.pk[np.uint64(ply * PK_WORDS + PK_CHK_READY)] = 0  # this node's masks are not built yet
    alpha_start = alpha
    # Quiescence stores at depth 0 with no move; computed from ply so numba types them as
    # the int64 the main search passes, and tt_store keeps one signature.
    qdepth = ply - ply
    no_move = qdepth
    best_move = no_move
    base = ply * MOVES_PER_PLY
    if checked:
        best = -INFINITY
        count = gen_all_s(s, base)
        for i in range(base, base + count):
            ms[np.uint64(i)] = (
                quiescence_score(ml[np.uint64(i)])
                if move_captured(ml[np.uint64(i)]) != EMPTY
                else 0
            )
    else:
        # Row #451: a stalemated node out of check used to return its stand pat, since the
        # `searched == 0` verdict below sits inside the check branch and zero searched
        # noisy moves is routine here. The gate is false at nearly every node, so the legal
        # loop is paid only where the side to move may have nothing to play. The loop
        # generates into this ply's own list, which is free at this point: the node's own
        # noisy list is written further down, after the loop, and no child is live.
        if QUIESCE_STALEMATE_GATE != 0 and stalemate_gate(s) and not has_legal_move(ply, s):
            return draw_score(ply, s)
        best = evaluate(alpha, beta, s)
        if best >= beta:
            bound = TT_LOWER if best >= beta else TT_EXACT
            tt_store(tt, key, qdepth, to_tt(best, ply), bound, no_move, ss[SS_GENERATION])
            return best
        if best > alpha:
            alpha = best
        count = gen_noisy_s(s, base)
        for i in range(base, base + count):
            move = ml[np.uint64(i)]
            if move_captured(move) == EMPTY and move_promotion(move) != QUEEN:
                ms[np.uint64(i)] = -PROMOTION_SCORE  # an underpromotion by a push is left out
            elif move == table_move:
                ms[np.uint64(i)] = QS_TABLE_SCORE  # the table's move first; make still tests it
            else:
                ms[np.uint64(i)] = quiescence_score(move)
    searched = 0
    pinned = np.uint64(0)
    pins_ready = False
    for i in range(base, base + count):
        move = pick_s(s, i, base + count)
        if ms[np.uint64(i)] == -PROMOTION_SCORE:
            break
        if (
            not checked
            and ss[SS_SEE_PRUNE] != 0
            and not see_ge_s(move, np.int64(SEE_QUIESCE_MARGIN), s)  # type: ignore[arg-type]
            and not gives_check_s(move, ply, s)
        ):
            continue
        if checked and not may_resolve(move, targets, st[STM]):
            continue  # it leaves the king in check, as the legality test would have said
        if not pins_ready:
            # The pieces pinned to our king, built at the node's first made move: a move
            # by any other piece, the king aside, is legal without the attack test after
            # make (`needs_legality_test`, as negamax's loop has had since #129).
            pinned = np.uint64(0) if checked else pinned_pieces_s(st[STM], s)
            pins_ready = True
        make_s(move, s)
        if needs_legality_test(move, checked, pinned) and not legal_after_s(s):
            unmake_s(move, s)
            continue
        # The accumulator is brought up only for a legal move: a third of the moves made
        # are illegal, and the update before the test fed nothing.
        prefetch(tt, st[HASH])
        nn_make_s(move, s)
        searched += 1
        score = -quiesce(-beta, -alpha, ply + 1, s)
        unmake_s(move, s)
        if ss[SS_STOP]:
            return 0
        if score > best:
            best = score
            best_move = move
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    break
    if checked:
        if searched == 0:
            return -MATE + ply
        if st[HALF] >= 99 and fifty_move_draw(ply, s):
            return draw_score(ply, s)
    if best >= beta:
        bound = TT_LOWER
    elif best <= alpha_start:
        bound = TT_UPPER
    else:
        bound = TT_EXACT
    # A fail low has no move to name; a fail high and an exact score store the move that
    # raised `best`, so the next visit to this position tries it before the rest.
    stored_move = no_move if bound == TT_UPPER else best_move
    tt_store(tt, key, qdepth, to_tt(best, ply), bound, stored_move, ss[SS_GENERATION])
    return best


# `node_probe`'s verdicts: search the node, return the score it gives, or drop to the
# quiescence search (the razor test).
PROBE_SEARCH = 0
PROBE_RETURN = 1
PROBE_RAZOR = 2


# The prelude and the close of `negamax` stand in their own functions, each a leaf
# that takes the search and never calls `negamax` or `quiesce` back: numba types the
# shorter `negamax` and each leaf once, and a cold compile is shorter for it. A leaf may
# not recurse into `negamax`: a recursion cycle across functions does not survive a
# reload from numba's disk cache (CLAUDE.md's numba rules).
@jit()
def node_probe(
    depth: int, alpha: int, beta: int, ply: int, checked: bool, key: int, s: Search
) -> tuple[int, int, int, int, int]:
    """The node's prelude: the table probe, the static evaluation, reverse futility and
    the razor test.

    Returns the verdict, the score to return with it, the table's move, the static
    evaluation and whether a verdict rather than an estimate made that static, as 1 or
    0. A table cutoff and a reverse futility return carry PROBE_RETURN and their
    score; a razor drop carries PROBE_RAZOR and `negamax` runs the quiescence search
    itself; PROBE_SEARCH goes on to the null move and the move loop. The table's move is
    0 on a table cutoff: `negamax` returns the score there and reads no move, so the
    packed word is left undecoded.

    The flag rides in the tuple rather than in an array: the tuple stays homogeneous,
    so numba settles its type with the rest of the return, and no array crosses a call
    boundary that did not cross it before.
    """
    st = s.st
    entry = tt_probe(s.tt, key)
    table_move = 0
    if entry != 0:
        if tt_depth(entry) >= depth and ply > 0 and st[HALF] < TT_CUTOFF_HALF:
            score = from_tt(tt_score(entry), ply)
            bound = tt_bound(entry)
            if (
                bound == TT_EXACT
                or (bound == TT_LOWER and score >= beta)
                or (bound == TT_UPPER and score <= alpha)
            ):
                # 53.1 percent of the nonzero decodes ended here (Codex's hot-path audit
                # of v56, 2026-09-10), and the caller reads only the score.
                return PROBE_RETURN, score, 0, -INFINITY, 0
        table_move = tt_move(entry, s.sq, st[STM])
    # The static evaluation, corrected by the pawn structure's and the threats' history,
    # at every node out of check: reverse futility and the null move read it, and the
    # correction history learns from it at the end of the node.
    static = -INFINITY
    static_verdict = 0
    if not checked:
        static = evaluate(alpha, beta, s)
        static_verdict = 1 if eval_is_verdict(s.bb, st, s.kpk) else 0
    if rfp_allowed(checked, ply, depth, alpha, beta) and static - rfp_margin(depth) >= beta:
        return (
            PROBE_RETURN,
            beta + (static - beta) // RFP_BLEND_DIVISOR,
            table_move,
            static,
            static_verdict,
        )
    if razor_allowed(checked, ply, depth, alpha, beta) and static + RAZOR_MARGIN <= alpha:
        return PROBE_RAZOR, 0, table_move, static, static_verdict
    return PROBE_SEARCH, 0, table_move, static, static_verdict


@jit(internal=True)
def node_close(
    depth: int,
    ply: int,
    checked: bool,
    static: int,
    static_verdict: int,
    alpha_start: int,
    best: int,
    best_move: int,
    cut: bool,
    searched: int,
    key: int,
    s: Search,
) -> int:
    """The node's close after the move loop: the score of a node with no legal move, the
    fifty-move draw, the bound, the correction update and the table store."""
    st = s.st
    if searched == 0:
        return -MATE + ply if checked else draw_score(ply, s)
    if ply > 0 and st[HALF] >= 99 and fifty_move_draw(ply, s):
        # In check with an evasion at the fifty-move mark: the referee's draw.
        return draw_score(ply, s)
    if cut:
        bound = TT_LOWER
    elif best <= alpha_start:
        bound = TT_UPPER
    else:
        bound = TT_EXACT
    # The correction history learns where the score is consistent with the bound: an
    # exact score; a lower bound above the static evaluation; an upper bound below it.
    # A mate score is not a gap the static evaluation could have closed, and it would pin
    # the slot at its cap for every position with that pawn structure or those threats, so
    # it teaches nothing (Stockfish guards the same way). A static evaluation a verdict decided
    # is not an estimate either, and its gap is just as unlearnable: `static_verdict`
    # says so from where the verdict was returned, since the score's size does not.
    # The size test stays as a second guard, for the mop-up path, which is an estimate
    # and can pass KPK_WIN against a bare king with an army: it kept those nodes out
    # before and keeps them out now, so the tree moves only where a verdict decided.
    # Not in check, where there is no static evaluation, and not when a capture or a
    # promotion made the score, since the tables then measured a different structure.
    if (
        not checked
        and static_verdict == 0
        and best > -MATE_BOUND
        and best < MATE_BOUND
        and static > -KPK_WIN
        and static < KPK_WIN
        and not (bound == TT_LOWER and best <= static)
        and not (bound == TT_UPPER and best >= static)
        and (
            bound == TT_UPPER
            or (move_captured(best_move) == EMPTY and move_promotion(best_move) == 0)
        )
    ):
        update_correction(depth, best - static, s.bb, st, s.tab, s.corr, s.tcorr)
    tt_store(s.tt, key, depth, to_tt(best, ply), bound, best_move, s.ss[SS_GENERATION])
    return best


@jit()
def probcut_allowed(
    depth: int, alpha: int, beta: int, ply: int, checked: bool, static: int, s: Search
) -> bool:
    """Admit only deep zero-window nodes with estimates and no contrary table bound."""
    if (
        depth < PROBCUT_MIN_DEPTH
        or ply <= 0
        or beta - alpha != 1
        or checked
        or abs(beta) + PROBCUT_MARGIN >= KPK_WIN
        or abs(static) >= KPK_WIN
        or popcount(s.bb[OCC_WHITE] | s.bb[OCC_BLACK]) <= 5
    ):
        return False
    entry = tt_probe(s.tt, s.st[HASH])
    return not (
        entry != 0
        and tt_depth(entry) >= depth - PROBCUT_REDUCTION + 1
        and tt_bound(entry) != TT_LOWER
        and from_tt(tt_score(entry), ply) < beta + PROBCUT_MARGIN
    )


@jit()
def probcut_capture(threshold: int, s: Search) -> int:
    """The first legal capture meeting SEE, in capture-history order; zero if none."""
    base = s.st[PLY] * MOVES_PER_PLY
    count = gen_noisy_s(s, base)
    pinned = np.uint64(0)
    pins_ready = False
    for i in range(base, base + count):
        s.ms[np.uint64(i)] = noisy_score(s.ml[np.uint64(i)], s.hc)
    for i in range(base, base + count):
        move = pick_s(s, i, base + count)
        if move_captured(move) == EMPTY or move_promotion(move) != 0:
            continue
        if not see_ge_s(move, max(0, threshold), s):
            continue
        if not pins_ready:
            # Probcut never runs in check, so only a king move, an en passant capture or
            # a pinned piece's move needs the attack test after make (#565).
            pinned = pinned_pieces_s(s.st[STM], s)
            pins_ready = True
        make_s(move, s)
        legal = not needs_legality_test(move, False, pinned) or legal_after_s(s)
        unmake_s(move, s)
        if legal:
            return move
    return count - count


@jit(internal=True)
def mate_distance_window(alpha: int, beta: int, ply: int) -> tuple[int, int]:
    """Clamp `(alpha, beta)` to what mate distance allows at `ply`: the side to move
    cannot do worse than being mated right here and cannot mate sooner than the next
    ply. Once a node, not once a move, so the call is cheap next to the block it
    replaces; never calls negamax or quiesce.
    """
    if alpha < -MATE + ply:
        alpha = -MATE + ply
    if beta > MATE - ply - 1:
        beta = MATE - ply - 1
    return alpha, beta


@jit()
def negamax(depth: int, alpha: int, beta: int, ply: int, s: Search) -> int:
    if count_node(s):
        return 0
    if ply > 0 and repeated(ply, s):
        return draw_score(ply, s)
    st = s.st
    keys = s.keys
    hd = s.hd
    acc = s.acc
    ml = s.ml
    undo = s.undo
    killers = s.killers
    hq = s.hq
    hc = s.hc
    tt = s.tt
    ss = s.ss
    checked = in_check_s(s)
    if ply > 0 and st[HALF] >= 99 and not checked and fifty_move_draw(ply, s):
        return draw_score(ply, s)
    if depth <= 0 and not checked:
        return quiesce(alpha, beta, ply, s)
    if ply >= MAX_PLY - 2:
        return evaluate(alpha, beta, s)
    # Mate-distance pruning: the side to move cannot do worse than being mated right here
    # and cannot mate sooner than the next ply, so a window outside that pair holds no
    # score this node can return, and the node is skipped before the table is touched.
    if ply > 0:
        alpha, beta = mate_distance_window(alpha, beta, ply)
        if alpha >= beta:
            return alpha
    key = st[HASH]
    # On the previous iteration's line (#503): the key at this ply is this position's.
    follow = FOLLOW_PV > 0 and ply < ss[SS_PV_LEN] and s.pv[np.uint64(ply)] == key
    verdict, score, table_move, static, static_verdict = node_probe(
        depth, alpha, beta, ply, checked, key, s
    )
    if verdict == PROBE_RETURN:
        return score
    if verdict == PROBE_RAZOR:
        return quiesce(alpha, beta, ply, s)
    if beta - alpha == 1 and null_move_allowed(checked, ply, depth, beta, s) and static >= beta:
        reduction = null_move_reduction(depth, static - beta)
        nn_bring_up_s(s, st[PLY])  # `nn_null` copies this ply's lanes
        make_null(st, undo, keys)
        prefetch(tt, st[HASH])
        nn_null(st, hd, acc)
        s.pk[np.uint64(st[PLY] * PK_WORDS + PK_PEND)] = 0  # the copy is already up to date
        s.pk[np.uint64(st[PLY] * PK_WORDS + PK_PREV)] = 0  # no move for the child's quiets
        score = -negamax(depth - 1 - reduction, -beta, -beta + 1, ply + 1, s)
        unmake_null(st, undo)
        if ss[SS_STOP]:
            return 0
        if score >= beta:
            # A mate found while a move down is not proof of one here.
            return beta if score >= MATE_BOUND else score
    if probcut_allowed(depth, alpha, beta, ply, checked, static, s):
        ss[SS_PROBCUT_ELIGIBLE] += 1
        raised = beta + PROBCUT_MARGIN
        capture = probcut_capture(raised - static, s)
        if capture != 0:
            ss[SS_PROBCUT_ATTEMPTS] += 1
            make_s(capture, s)
            prefetch(tt, st[HASH])
            nn_make_s(capture, s)
            s.pk[np.uint64((ply + 1) * PK_WORDS + PK_PREV)] = capture
            probe = -quiesce(-raised, -raised + 1, ply + 1, s)
            if not ss[SS_STOP] and probe >= raised:
                ss[SS_PROBCUT_VERIFICATIONS] += 1
                probe = -negamax(depth - PROBCUT_REDUCTION, -raised, -raised + 1, ply + 1, s)
            unmake_s(capture, s)
            if ss[SS_STOP]:
                return 0
            if probe >= raised:
                ss[SS_PROBCUT_CUTS] += 1
                # This is a selective bound, never a proved mate or tablebase score.
                # Its table depth describes the verification, not the skipped search.
                lower = TT_LOWER + ply - ply  # int64, as at the ordinary store call
                tt_store(
                    tt,
                    key,
                    depth - PROBCUT_REDUCTION + 1,
                    to_tt(beta, ply),
                    lower,
                    capture,
                    ss[SS_GENERATION],
                )
                return beta
    if table_move == 0 and depth >= IIR_MIN_DEPTH and ply > 0 and not follow:
        depth -= 1
    # A killer left by another branch at the next ply would order a stranger's move first.
    killers[np.uint64(ply + 1)] = 0
    us = st[STM]
    # The pieces pinned to our king, once a node: a move by any other piece, the king
    # aside, is legal without the attack test after make (`needs_legality_test`).
    pinned = np.uint64(0) if checked else pinned_pieces_s(us, s)
    targets = evasion_targets_s(s) if checked else np.uint64(0)
    # The check extension's horizon: a child's ply under CHECK_EXTENSION_PLY_FACTOR
    # times the root depth, so a chain of checks cannot run the tree away from the
    # iteration.
    extend = ply + 1 < CHECK_EXTENSION_PLY_FACTOR * ss[SS_ROOT_DEPTH]
    alpha_start = alpha
    best = -INFINITY
    best_move = 0
    searched = 0
    cut = False

    # The move loop. The picker holds the stages and hands back one move at a time; this
    # loop does the legality test, the pruning that depends on how many moves have been
    # searched, the recursion and the history.
    table_move = pick_setup(s, ply, table_move)
    pk = s.pk
    off = ply * PK_WORDS
    pk[np.uint64(off + PK_WORDS + PK_CUTOFFS)] = 0  # the children's cutoffs, counted below
    lmr = s.lmr
    # The move that led here conditions the quiets' order and their history update; 0
    # leaves the continuation table unread, so the tree is the butterfly's alone.
    previous = pk[np.uint64(off + PK_PREV)] if ss[SS_CONT_HISTORY] != 0 else 0
    while True:
        move = pick_next(s, ply, us, previous)
        if move == 0:
            break
        stage = pk[np.uint64(off + PK_LAST)]
        at = pk[np.uint64(off + PK_AT)]
        if checked and not may_resolve(move, targets, us):
            if stage != PICK_TABLE:
                ml[np.uint64(at)] = 0  # it leaves the king in check: not tried, so no history malus
            continue
        if stage == PICK_QUIET:
            counted_out = (
                depth <= ss[SS_LMP_DEPTH]
                and beta - alpha == 1
                and not checked
                and not follow
                and best > -MATE_BOUND
                and searched >= lmp_limit(depth)
            )
            if counted_out:
                if depth > LMP_CHECK_DEPTH:
                    # Every quiet left is pruned by the count as this one is (the count
                    # and the bound only grow within the stage), so the picker skips to
                    # the losing captures instead of drawing each by a scan of the tail;
                    # the tail is never read again, since the malus stops at the move
                    # that cut and a losing capture's malus covers the noisy list (#405).
                    pk[np.uint64(off + PK_NEXT)] = pk[np.uint64(off + PK_QUIET_BASE)] + pk[
                        np.uint64(off + PK_QUIET_COUNT)
                    ]
                    continue
                if not gives_check_s(move, ply, s):
                    ml[np.uint64(at)] = 0
                    continue
            if (
                depth <= ss[SS_SEE_QUIET_DEPTH]
                and beta - alpha == 1
                and not checked
                and not follow
                and best > -MATE_BOUND
                and not see_ge_s(move, -SEE_QUIET_STEP * depth * depth, s)
                and not gives_check_s(move, ply, s)
            ):
                ml[np.uint64(at)] = 0
                continue
        elif stage == PICK_BAD:
            # A losing capture whose exchanges lose more than the depth allows is skipped
            # at a zero-window node out of check (#569); the good captures passed SEE at
            # zero already, so only the picker's last stage reaches this test.
            if (
                depth <= SEE_CAPTURE_MAX_DEPTH
                and beta - alpha == 1
                and not checked
                and not follow
                and best > -MATE_BOUND
                and not see_ge_s(move, -SEE_CAPTURE_STEP * depth, s)
            ):
                ml[np.uint64(at)] = 0
                continue
        # The check extension, filtered by SEE: a checking move that does not lose
        # material gets its child a ply more, a check that hangs material none.
        extension = 0
        if (
            extend
            and gives_check_s(move, ply, s)
            and see_ge_s(move, np.int64(CHECK_EXTENSION_SEE_MARGIN), s)  # type: ignore[arg-type]
        ):
            extension = 1
        make_s(move, s)
        if needs_legality_test(move, checked, pinned) and not legal_after_s(s):
            unmake_s(move, s)
            if stage != PICK_TABLE:
                ml[np.uint64(at)] = 0  # not tried, so no history malus
            continue
        prefetch(tt, st[HASH])
        nn_make_s(move, s)
        searched += 1
        pk[np.uint64(off + PK_WORDS + PK_PREV)] = move  # the child's slice
        reduction = 0
        late_noisy = LMR_NOISY > 0 and stage == PICK_BAD  # a losing capture, after the quiets
        if (
            (stage == PICK_QUIET or late_noisy)
            and depth >= LMR_MIN_DEPTH
            and searched > LMR_FULL_MOVES
            and not checked
            and move != killers[np.uint64(ply)]
            and not seventh_rank_push(move, us)
        ):
            reduction = lmr[
                np.uint64(
                    min(depth, LMR_TABLE_SIDE - 1) * LMR_TABLE_SIDE
                    + min(searched, LMR_TABLE_SIDE - 1)
                )
            ]
            if (
                CUTOFF_COUNT_THRESHOLD > 0
                and pk[np.uint64(off + PK_WORDS + PK_CUTOFFS)] > CUTOFF_COUNT_THRESHOLD
            ):
                reduction += 1
            if late_noisy:
                reduction = max(0, reduction - 1)  # a ply less than a quiet move (#520)
        child = depth - 1 + extension
        score = alpha + 1  # no reduction: the full-depth search below runs
        if reduction > 0:
            score = -negamax(max(1, child - reduction), -alpha - 1, -alpha, ply + 1, s)
            if ss[SS_STOP]:
                unmake_s(move, s)
                return 0
        if score > alpha:
            # PVS: later moves prove they beat alpha before taking the full window.
            if searched == 1:
                score = -negamax(child, -beta, -alpha, ply + 1, s)
            else:
                score = -negamax(child, -alpha - 1, -alpha, ply + 1, s)
                if not ss[SS_STOP] and alpha < score < beta:
                    score = -negamax(child, -beta, -alpha, ply + 1, s)
        unmake_s(move, s)
        if ss[SS_STOP]:
            return 0
        if score > best:
            best, best_move = score, move
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    cut = True
                    pk[np.uint64(off + PK_CUTOFFS)] += 1  # failed high: the parent reads it
                    if move_captured(move) != EMPTY:
                        reward_noisy(move, ml, pk[np.uint64(off + PK_LIST_BASE)], at, depth, hc)
                    else:
                        reward_quiet(
                            move,
                            ml,
                            pk[np.uint64(off + PK_LIST_BASE)],
                            at,
                            depth,
                            ply,
                            us,
                            killers,
                            hq,
                            s.chist,
                            previous,
                            s.phist,
                            pawn_history_row(s.bb),
                        )
                    break

    return node_close(
        depth,
        ply,
        checked,
        static,
        static_verdict,
        alpha_start,
        best,
        best_move,
        cut,
        searched,
        key,
        s,
    )


@jit()
def root_moves(s: Search) -> int:
    """The legal moves of the root position, written to `root`; returns how many."""
    st = s.st
    ml = s.ml
    root = s.root
    count = gen_all_s(s, st[PLY] * MOVES_PER_PLY)
    legal = 0
    base = st[PLY] * MOVES_PER_PLY
    for i in range(base, base + count):
        move = ml[np.uint64(i)]
        make_s(move, s)
        if legal_after_s(s):
            root[np.uint64(legal)] = move
            legal += 1
        unmake_s(move, s)
    return legal


@jit()
def pv_walk(s: Search, first: int) -> int:
    """The previous iteration's line as position keys (#503): `first`, the iteration's
    best root move (the root itself is never stored in the table), then the table's
    moves while the table holds a legal one, PV_MAX_PLIES at most; returns the
    plies kept, which `ss[SS_PV_LEN]` records. The moves made sit in the second half
    of `pv` for the unwind. A repeated key ends the walk, since the table can hold a
    cycle, and so does a move the generator does not produce (a stale entry)."""
    st = s.st
    ml = s.ml
    pv = s.pv
    length = 0
    made = 0
    if FOLLOW_PV > 0:
        while length < PV_MAX_PLIES:
            key = st[HASH]
            repeated = False
            for i in range(length):
                if pv[np.uint64(i)] == key:
                    repeated = True
            if repeated:
                break
            pv[np.uint64(length)] = key
            length += 1
            move = first
            if length > 1:
                entry = tt_probe(s.tt, key)
                if entry == 0:
                    break
                move = tt_move(entry, s.sq, st[STM])
            if move == 0:
                break
            base = st[PLY] * MOVES_PER_PLY
            count = gen_all_s(s, base)
            found = False
            for i in range(base, base + count):
                if ml[np.uint64(i)] == move:
                    found = True
            if not found:
                break
            make_s(move, s)
            if not legal_after_s(s):
                unmake_s(move, s)
                break
            pv[np.uint64(PV_MAX_PLIES + made)] = move
            made += 1
        for i in range(made - 1, -1, -1):
            unmake_s(pv[np.uint64(PV_MAX_PLIES + i)], s)
    s.ss[SS_PV_LEN] = length
    return length


@jit()
def search_root(
    depth: int, previous_best: int, alpha: int, beta: int, s: Search
) -> tuple[int, int]:
    """Search the root moves to `depth` inside (alpha, beta); returns the best and its score.

    Fail-soft: a score at or below `alpha` means every root move failed low and the move
    returned is only the least bad bound, a score at or above `beta` means the move
    returned failed high and the rest were not searched. The deepening loop in `agent.py`
    widens the window and calls again in both cases; on an open window the score is exact.
    The previous iteration's best move goes first, then the captures by their ordering
    score and the quiet moves by theirs. A root move that gives check without losing
    material searches its reply a ply deeper, the extension `negamax` applies in its own
    loop. The nodes each root move took go to `root_nodes` in the order of `root`, which
    the agent's time manager reads.
    """
    st = s.st
    ms = s.ms
    killers = s.killers
    hq = s.hq
    hc = s.hc
    root = s.root
    root_nodes = s.root_nodes
    ss = s.ss
    count = ss[SS_ROOT_COUNT]
    ss[SS_ROOT_DEPTH] = depth
    us = st[STM]
    root_ply = st[PLY]  # 0, read from the array so the kernels below see an int64, not a literal
    s.path[np.uint64(root_ply)] = st[HASH]
    pk = s.pk
    pk[np.uint64(root_ply * PK_WORDS + PK_CHK_READY)] = 0  # the root's masks, once a call
    pk[np.uint64(root_ply * PK_WORDS + PK_PREV)] = 0  # the root's quiets answer no move
    previous = pk[np.uint64(root_ply * PK_WORDS + PK_PREV)]
    prow = pawn_history_row(s.bb)
    # The root's own lanes are refreshed before the search and no ply owes anything under
    # it. `unmake_s` clears every mark on the way out, and this is the floor under that.
    for at in range(pk.shape[0] // PK_WORDS):
        pk[np.uint64(at * PK_WORDS + PK_PEND)] = 0
    for i in range(count):
        move = root[np.uint64(i)]
        if move == previous_best:
            ms[np.uint64(i)] = PREVIOUS_BEST_SCORE
        elif move_captured(move) != EMPTY or move_promotion(move) != 0:
            ms[np.uint64(i)] = CAPTURE_STAGE + noisy_score(move, hc)
        else:
            ms[np.uint64(i)] = quiet_score(
                move, us, root_ply, killers, hq, s.chist, previous, s.phist, prow
            )
    # The root list is sorted in place, best first, since every move is searched.
    for i in range(count):
        pick(root, ms, i, count)
    ss[SS_ITERATION_BEST] = 0
    killers[np.uint64(root_ply + 1)] = 0
    best_move = root[0]
    best = -INFINITY
    for i in range(count):
        move = root[np.uint64(i)]
        before = ss[SS_NODES]
        # The check extension, as in `negamax`; ply 1 is always under the horizon.
        child = depth - 1
        if (
            root_ply + 1 < CHECK_EXTENSION_PLY_FACTOR * depth
            and gives_check_s(move, root_ply, s)
            and see_ge_s(move, np.int64(CHECK_EXTENSION_SEE_MARGIN), s)  # type: ignore[arg-type]
        ):
            child += 1
        make_s(move, s)
        nn_make_s(move, s)
        pk[np.uint64((root_ply + 1) * PK_WORDS + PK_PREV)] = move
        # PVS at the root uses the same probe and re-search as interior nodes.
        if i == 0:
            score = -negamax(child, -beta, -alpha, root_ply + 1, s)
        else:
            score = -negamax(child, -alpha - 1, -alpha, root_ply + 1, s)
            if not ss[SS_STOP] and alpha < score < beta:
                score = -negamax(child, -beta, -alpha, root_ply + 1, s)
        unmake_s(move, s)
        root_nodes[np.uint64(i)] = ss[SS_NODES] - before
        if ss[SS_STOP]:
            break
        if score > best:
            best = score
            if score > alpha:
                alpha = score
                best_move = move
                ss[SS_ITERATION_BEST] = move
                if alpha >= beta:
                    break
    return best_move, best


# The signatures numba infers for the two recursive searches today, copied from
# `fn.signatures` after a warm import. `compile_recursive` compiles on them, which makes
# numba lock the return type before typing the bodies, so `kernel`'s shortcut can answer
# each recursive call site from it instead of cloning the inference state. They are
# compiled from the warm-up thread and not declared on the decorators, since a decorator
# signature compiles at import, on the main thread, before the init guard exists.
QUIESCE_SIGNATURE = int64(int64, int64, int64, SEARCH_TYPE)
NEGAMAX_SIGNATURE = int64(int64, int64, int64, int64, SEARCH_TYPE)


def compile_recursive() -> None:
    """Compile `quiesce` then `negamax` on their declared signatures.

    Called from the warm-up, on its thread. The order is the call order: `negamax` calls
    `quiesce`, which is then already an exact overload rather than a fresh compile.
    """
    quiesce.compile(QUIESCE_SIGNATURE)  # type: ignore[attr-defined]
    negamax.compile(NEGAMAX_SIGNATURE)  # type: ignore[attr-defined]


WARMED: dict[str, Any] = {
    # The kernel's functions the search alone calls, compiled with it; a second signature
    # here means a call site passes a literal or another type and compiles on the clock.
    # The entries the search calls with the search object, compiled when `negamax`,
    # `quiesce` and `search_root` are typed; `legal_after_s` is pasted into its callers
    # and has no signature of its own. The array entries of the same kernels (`see_ge`,
    # `gives_check`, `pinned_pieces`, `evasion_targets`, `nnue.nn_make`) are the tests'
    # and the benches' and compile on their first call, never on the platform.
    "make_s": make_s,
    "unmake_s": unmake_s,
    "nn_make_s": nn_make_s,
    "nn_flush_s": nn_flush_s,
    "attacked_s": attacked_s,
    "in_check_s": in_check_s,
    "see_ge_s": see_ge_s,
    "gives_check_s": gives_check_s,
    "pinned_pieces_s": pinned_pieces_s,
    "evasion_targets_s": evasion_targets_s,
    "gen_noisy_s": gen_noisy_s,
    "gen_quiet_s": gen_quiet_s,
    "gen_all_s": gen_all_s,
    "pick_s": pick_s,
    "kpk_wins": kpk_wins,
    "eval_is_verdict": eval_is_verdict,
    "pieces_immobile": pieces_immobile,
    # Reached only through a warmed caller, so they compile at import today; listed here
    # so a later change that gives one of them a second signature raises instead of
    # compiling on the game clock (Codex's review of the KPK merge, 2026-09-07). The
    # array entries of the generators, of `make` and of `unmake` are not here: nothing
    # the platform runs calls them, so they compile only when a test or a tool does.
    "attacked": attacked,
    "bishop_attacks": bishop_attacks,
    "rook_attacks": rook_attacks,
    "lsb": lsb,
    "popcount": popcount,
    "in_check": in_check,
    "last_was_null": last_was_null,
    "push_targets": push_targets,
    "push_promotions": push_promotions,
    "make_null": make_null,
    "unmake_null": unmake_null,
    "evaluate_tables": evaluate_tables,
    "attackers_to": attackers_to,
    "new_search": new_search,
    "now": now,
    "evaluate": evaluate,
    "threat_map": threat_map,
    "mop_up": mop_up,
    "manhattan": manhattan,
    "dead_draw": dead_draw,
    # The fifty-move claim and the reply test it makes, both reached from `negamax` and
    # `quiesce` with `(int64, Search)` and so compiled at import; listed so a call site
    # that later passes another type raises here instead of compiling on the clock (#468).
    "fifty_move_draw": fifty_move_draw,
    "has_legal_move": has_legal_move,
    "null_move_allowed": null_move_allowed,
    "rfp_allowed": rfp_allowed,
    "tt_probe": tt_probe,
    "tt_store": tt_store,
    "noisy_score": noisy_score,
    "quiet_score": quiet_score,
    "quiescence_score": quiescence_score,
    "pick": pick,
    "pseudo_legal_quiet": pseudo_legal_quiet,
    "reward_noisy": reward_noisy,
    "reward_quiet": reward_quiet,
    "quiesce": quiesce,
    "pv_walk": pv_walk,
    # negamax's leaves, typed when negamax is; a second signature on one means a call
    # site passes another type and compiles on the clock.
    "node_probe": node_probe,
    "node_close": node_close,
    "mate_distance_window": mate_distance_window,
    "probcut_allowed": probcut_allowed,
    "probcut_capture": probcut_capture,
    "negamax": negamax,
    "root_moves": root_moves,
    "search_root": search_root,
}
