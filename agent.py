"""The submission entrypoint. The platform imports this file and calls get_move.

The search runs in numba: `kernel.py` holds the board, its move generation and make and
unmake over preallocated arrays, `search.py` the alpha-beta search, quiescence, move
ordering, transposition table and draw detection, all compiled at import so nothing is
compiled on the clock. Leaves are scored by the network in `nnue.py`, an integer forward
pass over the weights beside this file whose shape `docs/weights/README.md` sets out. The
tapered piece-square tables in `evaluation.py`, which the kernel reads as one array,
stand in for the net where it is missing or cannot help: when no net is loaded, when the
tables' score already lies far outside the search window, and against a bare king, where
the net has no gradient. python-chess is used at the boundary alone. It parses the FEN
and lists the legal moves the answer is checked against, and get_move returns the move as
a UCI string, which the platform's own runner writes out. `diag.py` prints the lines the
per-game log keeps. The search runs on one thread, and the process opens no socket,
starts no subprocess and writes no file.

Shipped data. Beside this file the zip carries `weights/book.pack`, an opening book of
moves only (no evaluations): polyglot's key and move word per entry, read by `book.py` the
way `chess.polyglot` reads the 16-byte file it was packed from; and `weights/syzygy/`,
the public 3/4-man Syzygy tables and selected 5-man tables read through `chess.syzygy`. The
rules permit both: "A table you ship and read during a game may answer the opening or the
endgame." The book answers no position above move 20 and the tables no position above five
pieces, each read from the FEN we are given. The network under `weights/` was trained by
this team; every move outside the book and the tablebases comes from the search in this
file, `search.py` and `kernel.py`.

This file is the glue: it keeps the arrays the search works in for the whole game, turns
the platform's FEN into the kernel's position, runs the deepening loop one root call per
iteration, manages the clock, remembers the positions the game has been through so the
search can see a repetition coming, and checks every move before returning it.

From depth ASPIRATION_DEPTH each iteration is searched on a window around the last
score and again on a wider one while the score lands outside it. The deepening loop stops
when the soft budget is spent, and the search aborts mid-iteration at the hard budget.
The move returned is the best of the last finished iteration, or the best of the
iteration in progress when the abort came after a root move had beaten it, so a legal
move is in hand from the first iteration.

The referee claims threefold repetition and fifty-move draws itself, counting from the
first position of the game, so the process keeps the hash of every position the game has
been through since its last capture or pawn move, reconstructs the opponent's reply from
the next FEN, and hands the list to the search, which scores any repetition of one of
those positions, or of a position on the search path, as a draw before it evaluates the
node. A side that is ahead then avoids the repetition and a side that is behind seeks it.

Before any of that, the position is looked up in the opening book (`book.py`): a
polyglot file of moves for the platform's curated starting positions and the replies the
field plays from them, built offline from public engine analysis. A hit is played at
once and the clock it saves goes to later moves; a miss costs one hash lookup.
"""

import os
import threading

# One thread. numpy's OpenBLAS starts a worker per visible CPU when it loads (four on
# the platform, whose core is a cgroup quota over four visible CPUs) and the workers
# spin after every call, so diag's matmul bench spent the quota on threads the search
# never uses: in Docker at the platform's shape, four threads and 21 ms of CPU for 7 ms
# of wall, against one thread and 3 ms pinned. Set before numpy loads: nothing numeric
# is imported above this line, and the runner imports only the standard library.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time

# The import's own clock, read before the modules below load and compile: the tablebase's
# guarded open at the end of the import reads it against tablebase.EAGER_UNDER_S. The
# stamp sits between imports, which ruff's E402 would flag for the rest of the file.
# ruff: noqa: E402
IMPORT_STARTED = time.monotonic()

from pathlib import Path

import chess
import numpy as np

import diag
import kernel
import nnue
import search
from book import BOOK
from evaluation import kernel_tables
from tablebase import TABLEBASE, eager_at

MATE = search.MATE
MATE_BOUND = search.MATE_BOUND
INFINITY = search.INFINITY
MAX_DEPTH = search.MAX_DEPTH

# Time management. The clock keeps RESERVE_MS at all times and, while the game is open,
# FLOOR_MS above it: every budget comes out of what is left above the floor, and at the
# floor a move gets the increment alone, so the clock is never planned below the floor.
# Each move gets a soft budget, after which no new iteration starts, and a hard budget,
# at which the search aborts mid-iteration with the best move so far; the hard budget is
# four times the soft one and never more than a fixed share of the clock above the floor.
# A negative or zero clock is read as GARBAGE_CLOCK_MS.
# The clock these constants assume is the contract's 120 s plus 0.5 s a move; the edit
# for a different clock, a ply cap or a smaller zip on the day of the final, and the
# check for each, are in notes/london-day.md.
# One knob for a shorter clock on the day of the final (notes/london-day.md): the reserve,
# the floor and the floor's two budgets are the contract clock's, scaled together by
# CLOCK_SCALE (the revealed base over 120,000: 0.25 for 30 s, 0.0833 for 10 s), so a floor
# lowered for a short game never carries budgets that outrun the clock above it (a floor
# of 400 ms under budgets of 400 and 450 ms left 150 ms at a 600 ms clock, under the
# reserve: codex-lane's finding 4, 2026-09-11). The reserve never falls under
# RESERVE_MIN_MS, since the runner's charge and get_move's own work do not shrink with the
# clock, and `allocate` clamps both budgets to the usable clock whatever the constants say.
CLOCK_SCALE = 1.0
RESERVE_MIN_MS = 50
RESERVE_MS = max(RESERVE_MIN_MS, int(200 * CLOCK_SCALE))
INCREMENT_MS = 500  # the contract's increment; get_move is not told it
# The soft budget never falls under this share of the increment, at any clock: the
# increment refills it, and a manager that lets the budget taper toward nothing plays
# depth-one moves without flagging (Stockfish 18 at 9 s + 0.1 s, issue #6639, fix
# 3c04b5c). Half, so that under the floor the other half still rebuilds the bank as the
# taper does, and above the floor at 120 s + 0.5 s the floor's own budgets already
# exceed it and nothing changes.
INCREMENT_SLICE = 0.5
# The increment read from the clock instead: the clock one call is handed, less what the
# previous call left it (its clock less the time it spent), is the increment the runner
# credited, its own charge of a few milliseconds inside. Two consecutive readings within
# OBSERVED_INCREMENT_AGREE_MS of each other that differ from INCREMENT_MS by more than
# OBSERVED_INCREMENT_SWITCH_MS replace it for the game, so a clock with another increment
# is followed without an edit (notes/london-day.md). At the contract's clock the reading
# is 496 to 498 ms and at the local platform-equivalent control's 336 ms it is inside
# the threshold, so neither changes a budget.
OBSERVED_INCREMENT_AGREE_MS = 50
OBSERVED_INCREMENT_SWITCH_MS = 250
OBSERVED_INCREMENT_MAX_MS = 10_000  # a reading past this is a new game, not an increment
# The floor. Without one the allocator spent long games down to where its hard budget met
# the increment, 1.6 s of clock (0.35 of the usable clock is 500 ms there): the platform's
# round 42 (v14, 73 moves) went from 39.6 s at move 40 to 9.2 s at move 63 with six moves
# of 3 to 6 s, and a 66-game match at the platform's control read a clock minimum of 1.4 s
# on both sides. 1.5 s here is a flag on a slow container. At the floor a move gets
# FLOOR_SOFT_MS and FLOOR_HARD_MS, under the increment by the runner's own charge (2 ms
# median, 4 ms worst in the dashboard logs), the poll stride's blind window at the
# platform's node rate and get_move's work around the search, so a game of shuffles at
# the floor holds the clock where it is; under the floor both shrink with the clock.
# The floor was 10 s until 7 September: rounds 47 and 48 ended with 13.6 and 21.5 s unused after
# 54 s at move 40, and the platform's own charge is 2 ms median and 4 ms worst a move, so 5 s
# covers a thousand moves of overhead and the rest is spent.
FLOOR_MS = int(5_000 * CLOCK_SCALE)
FLOOR_SOFT_MS = int(400 * CLOCK_SCALE)
FLOOR_HARD_MS = int(450 * CLOCK_SCALE)
# The move horizon is the shortest of three: the schedule, MTG_START moves to go at move one
# and one fewer every two moves; the material's, MTG_MIN plus the tapered phase (24 with
# every piece on, 0 with none), so a game that trades down spends faster; and the moves
# left to SPEND_BY_MOVE. On a flat horizon of 25 the platform's round 35 game (v13,
# 83 moves) reached move 46 with 22 s of 120 and played its last 25 moves under a second
# each. The 20 s held back until move 40 that came with the decay was spent after it; the
# floor replaces that hold-back.
MTG_START = 44
MTG_MIN = 20
# The moves left to SPEND_BY_MOVE are fewer than the schedule's from move 29, so the pool
# above the floor is spent by then instead of settling where the soft budget meets the
# increment: rounds 60 to 72 ended with 20 to 23 s unused at 2.1 s a move, round 72 with
# 22.9 s. At 58 the pool was gone by move 60 and every later move got the increment alone
# (round 74 ran 109 moves at 0.43 s a move from move 84); a third of the rated games reach
# move 80, so the pool runs to 80 and the horizon stops at MTG_END, where the fifth-of-the-pool
# cap would bind: from 120 s that is 1.8 s a move at 50, 1.7 s at 60, 0.9 s at 80 and 0.6 s
# at 100 against 3.0, 1.4, 0.5 and 0.5 before.
SPEND_BY_MOVE = 80
MTG_END = 10
SOFT_SHARE = 0.73
SOFT_CAP = 0.20
HARD_RATIO = 4.0
HARD_CAP = 0.35
ABSOLUTE_CAP = 0.60
GARBAGE_CLOCK_MS = 1_000
# Below this much clock, return the fallback without searching.
PANIC_MS = RESERVE_MS + 50
# A forced move near the clock floor gets only a brief child search to warm the table.
FORCED_ROOT_PRESSURE_MS = 2 * FLOOR_MS
FORCED_ROOT_WARM_MS = 5
# A node limit a move for `tools/sprt.py`'s fixed-node match mode: the search stops at the
# first poll past it and the deepening ignores the clock. Unset on the platform, so 0.
NODE_LIMIT = int(os.environ.get("AICHESSATHON_NODE_LIMIT", "0"))

# The init guard (notes/london-day.md, a shorter init budget). The warm-up below compiles
# every search kernel, 23 s at the platform's median and 80 s on its slowest core over
# 124 rated logs, against a 90 s budget the final may shorten. It runs in a thread, and
# the import returns once INIT_GUARD_S seconds have passed since it began, compiled or
# not, so the runner's ready line goes out inside the budget; the first get_move then
# waits for the rest and pays that wait from its own clock. A host that compiles inside
# the guard sees no change. Set it about ten seconds under the budget; 0 waits in full.
# London, 12 September: the budget is 30 s. The platform host spends seconds on torch, numpy
# and numba before this file's first line and the ready line must land inside the budget;
# v80 to v83 held 20 and their validation ready lines left room. 25 hands the first move five
# more seconds of compiled search and still clears a loading gap of 4.5 s, which only three of
# 234 platform starts crossed (27 would miss the ten starts near 3 s); a ready line past 29 s
# reverts it.
INIT_GUARD_S = float(os.environ.get("AICHESSATHON_INIT_GUARD_S", "25"))

# After an iteration, the soft budget is scaled by how much of the search went to the best
# move: a move that took most of the nodes is settled and the next iteration is skipped
# sooner, a contested root gets longer (Alexandria's node-fraction scaling).
NODE_FRACTION_BASE = 1.5
NODE_FRACTION_SCALE = 1.35
# The root's node counts include the PVS probes and their re-searches. The scale is
# bounded so a change in their share cannot drive the soft budget beyond these limits.
NODE_FRACTION_MIN = 0.75
NODE_FRACTION_MAX = 1.25
# Best-move stability scales the soft budget too: a move that just changed gets longer, one
# that has held for four iterations gets three quarters (Viridithas's lookup after Stash;
# Russell and Wefald: search that does not change the preferred move has had no value).
# The product of the scalers never passes the hard budget (Alexandria's clamp).
STABILITY_SCALE = (2.5, 1.2, 0.9, 0.8, 0.75)
# The product of the two scalers has a floor of its own: each floor alone is 0.75 and the
# two together were 0.5625, half the soft budget, which the iteration predictor then ended
# the move on. Round 72 (v31 as Black): 43...Ka7 at depth 20 in 1.4 s with 57 s on the
# clock, one iteration short of Rb5, and the game lost by mate with 22.9 s unused.
SOFT_SCALE_MIN = 0.75
# A falling score extends the soft budget: when an iteration's root score is under the
# last searched root score of this game, the budget grows by the fall over FALLING_DIVISOR
# centipawns, never past FALLING_MAX (Stockfish's falling-eval term). Round 72: the score
# fell from -48 to -141 over three moves of 1.5 s each before the losing move, and a
# falling score is the one a deeper search is most likely to change.
FALLING_DIVISOR = 200.0
FALLING_MAX = 1.5

# Contempt. A repetition, a fifty-move draw or a stalemate is not worth nothing: once the
# last iteration's root score says we stand better by more than CONTEMPT_DEAD_ZONE, a draw
# scores CONTEMPT_MG below zero for us in the middlegame and CONTEMPT_EG in the endgame,
# tapered by the phase, and the same above zero when we stand worse, so the search plays
# on when ahead and takes the draw when behind. Our matches at the platform's control
# ended a fifth of the games we led by 300 cp as repetition draws, and the field's top bots
# draw by repetition; the margin is the old Stockfish default, halved for the endgame as
# it was there.
CONTEMPT_MG = 30
CONTEMPT_EG = 15
CONTEMPT_DEAD_ZONE = 20
PHASE_WEIGHTS = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}

# The clock is read inside the search once every stride nodes, a power of two. get_move
# sets the stride per move from the node budget over 100 (Berserk's rule), within these.
POLL_STRIDE_MIN = 64
POLL_STRIDE_MAX = 4096
DEFAULT_NPS = 500_000

# The agent's own clock, for the soft budget and the iteration times. time.monotonic()
# ticks every 16 ms on Windows, which is the development machine; perf_counter() is
# fine-grained on every platform. The search reads the OS clock through ctypes itself.
clock = time.perf_counter

# The arrays the search works in, kept for the whole game. Every one reaches the kernels
# as an argument, behind the one `search.Search` object built below, which holds
# references to them; see kernel.py for why none may be read as a global.
BB, SQ, ST, ML, UNDO = kernel.new_arrays()
MS = np.zeros_like(ML)
PATH = np.zeros(kernel.MAX_PLY, dtype=np.int64)
GAME = np.zeros(search.GAME_MAX, dtype=np.int64)
KILLERS = np.zeros(kernel.MAX_PLY + 2, dtype=np.int64)
QUIET_HISTORY = np.zeros(search.QUIET_HISTORY_SIZE, dtype=np.int64)
# #504: the butterfly table is scaled to ROOT_HISTORY_KEEP_NUM / ROOT_HISTORY_KEEP_DEN at the
# start of every iterative deepening loop, so the ordering partly forgets the previous
# move's tree (Stockfish, "Adjust main history with every new root position", 2025-12-29,
# +1.3 Elo at its short control; larger here, since this process keeps its tables for a
# whole game). Equal numerator and denominator turn the arm off.
ROOT_HISTORY_KEEP_NUM = 3
ROOT_HISTORY_KEEP_DEN = 4
CAPTURE_HISTORY = np.zeros(search.CAPTURE_HISTORY_SIZE, dtype=np.int64)
# The continuation history: a quiet move's worth after the move that led to the node,
# keyed by both moves' piece and target; read beside the butterfly history in the quiet
# stage and updated with it on a cutoff. int16, since the gravity update bounds it.
CONTINUATION_HISTORY = np.zeros(search.CONT_HISTORY_SIZE, dtype=np.int16)
# 0 leaves the continuation history unread and unwritten: the tree is then the butterfly
# history's alone, which a probe compares against.
CONT_HISTORY = 1
# The pawn-structure history (#499): a quiet move's worth under the pawn structure,
# read and moved beside the two tables above; `search.PAWN_HISTORY` switches it.
PAWN_STRUCTURE_HISTORY = np.zeros(search.PAWN_HISTORY_SIZE, dtype=np.int16)
LMR = search.lmr_table()
# Two words an entry: the packed word and the score word. The count is derived here
# rather than read as `search.TT_WORDS`, so a tool that sets `search.TT_ENTRIES`
# before this import still allocates a table of the size it asked for.
TT = np.zeros(search.TT_ENTRIES * search.TT_STRIDE, dtype=np.int64)
ROOT = np.zeros(kernel.MOVES_PER_PLY, dtype=np.int64)
ROOT_NODES = np.zeros(kernel.MOVES_PER_PLY, dtype=np.int64)
SS = np.zeros(search.SS_SIZE, dtype=np.int64)
CLK = np.zeros(2, dtype=np.int64)
# The correction history: the pawn structure's shift of the static evaluation, learned
# through the game and kept between moves, like the move histories.
CORRECTION = np.zeros(2 * search.CORR_SIZE, dtype=np.int32)
# Its twin keyed by the pieces under attack (tcheran's threat correction history).
THREAT_CORRECTION = np.zeros(2 * search.CORR_SIZE, dtype=np.int32)
# The move picker's state, one slice a ply.
PICKER = np.zeros((kernel.MAX_PLY + 2) * search.PK_WORDS, dtype=np.int64)
# The previous iteration's line as position keys, walked from the table after each
# completed iteration (#503); `search.FOLLOW_PV` switches it.
PV_LINE = np.zeros(search.PV_SIZE, dtype=np.int64)
EV = kernel_tables()
# The net, proven against its check file at import; without it the tables score the leaves.
# The same recipe fitted to sigmoid(score / 250) instead of 400: the best rank correlation
# with the label on every held-out set (notes/measurements/2026-09-05-first-net.md, the
# target scale); the loader reads the scale from the metadata, so the search still sees
# centipawns, and games against the four-bucket net price it.
# For a reviewer: the four-bucket net this team trained, the shipped lw2 net warm-restarted
# on our own self-play games (#505, 2026-09-12): +34.3 +/- 15.8 Elo over 600 games at 10+0.1.
NET_NAME = "scale250-net-l256-b4-lw2-mmap-w-ft01-lam1to075-sp50m"
NET = nnue.load(
    Path(__file__).resolve().parent / "weights" / f"{NET_NAME}.safetensors", require_check=True
)
# The head is bound in Python from the net's metadata before the warm-up compiles it in,
# so a process compiles one head and a plain net never compiles the dual tail.
nnue.choose_head(
    nnue.DUAL_HEAD if NET is not None and NET.dual else nnue.PLAIN_HEAD,
    NET is not None and NET.stacks > 1,
)
FTW, HD, ACC, XS, HB = NET.arrays if NET is not None else nnue.dummy_arrays()
# The king-move refresh's cache, one accumulator a colour and king key with the
# board it was built from. It is state within the game and is never reset: the
# update is taken against the board the entry holds, so an entry from an earlier
# move is as correct as a fresh one and only costs the pieces that have moved.
KACC, KBB = NET.cache if NET is not None else nnue.dummy_cache()
if NET is None:
    diag.note("net not loaded: piece-square tables in use")
TAB = kernel.TAB
KEYS = kernel.KEYS
# The king and pawn against king bitbase the search probes at its leaves; without the
# file the search keeps the tables and the mop-up term there.
_KPK_PATH = Path(__file__).resolve().parent / "weights" / "kpk.bin"
KPK = (
    np.frombuffer(_KPK_PATH.read_bytes(), dtype=np.uint8).copy()
    if _KPK_PATH.exists()
    else np.zeros(0, dtype=np.uint8)
)
S = search.Search(
    BB,
    SQ,
    ST,
    TAB,
    KEYS,
    EV,
    FTW,
    HD,
    ACC,
    XS,
    HB,
    KACC,
    KBB,
    ML,
    MS,
    UNDO,
    PATH,
    GAME,
    KILLERS,
    QUIET_HISTORY,
    CAPTURE_HISTORY,
    LMR,
    TT,
    ROOT,
    ROOT_NODES,
    SS,
    CLK,
    KPK,
    PICKER,
    CORRECTION,
    THREAT_CORRECTION,
    CONTINUATION_HISTORY,
    PAWN_STRUCTURE_HISTORY,
    PV_LINE,
)

# A second position, for hashing game positions without touching the search's.
_HASH_BB, _HASH_SQ, _HASH_ST, _, _ = kernel.new_arrays()


def position_hash(board: chess.Board) -> int:
    """The kernel's Zobrist hash of a python-chess position."""
    kernel.load(board, _HASH_BB, _HASH_SQ, _HASH_ST, KEYS)
    return int(_HASH_ST[kernel.HASH])


def position_key(board: chess.Board) -> object:
    """python-chess's transposition key: pieces, side to move, castling rights, en passant.

    It is what the referee's threefold claim compares, and it is private to python-chess,
    which the platform pins at 1.11.2.
    """
    return board._transposition_key()


class GameHistory:
    """The positions this game has been through, from the first FEN the platform sent.

    The platform shows us only the positions we move in. The position after our own move
    is recorded when we choose it, and the opponent's reply is recovered by finding the one
    legal move from there that reaches the next FEN. A capture or pawn move ends the run
    of positions that can still repeat, so the list restarts whenever the halfmove clock
    is zero. An unexpected FEN (a new game, or a position out of sequence) restarts the
    list from that position. The list holds the kernel's hashes, which the search reads.
    The last searched root score of the game is kept beside it, for the time manager's
    falling-score extension, and forgotten with the list on a new game.
    """

    def __init__(self) -> None:
        self.keys: list[int] = []
        self.ours: list[tuple[int, ...]] = []
        self.after_ours: chess.Board | None = None
        self.score: int | None = None  # the last searched root score this game

    def observe(self, board: chess.Board) -> list[int]:
        """Record the position we are asked to move in; return the positions before it."""
        reached = self._reached(board)
        if not reached:
            self.score = None  # a new game, or one out of sequence: nothing to fall from
        if not reached or board.halfmove_clock == 0:
            self.keys = []
            self.ours = []
        before = list(self.keys)
        self.keys.append(position_hash(board))
        return before

    def record(self, board: chess.Board) -> None:
        """Record the position after our move; the board is kept to read the reply from."""
        key = position_hash(board)
        placement = self.placement(board)
        if board.halfmove_clock == 0:
            self.keys = [key]
            self.ours = [placement]
        else:
            self.keys.append(key)
            self.ours.append(placement)
        self.after_ours = board

    @staticmethod
    def placement(after: chess.Board) -> tuple[int, ...]:
        """Where the side that just moved has its pieces: its bitboard by piece type."""
        us = after.occupied_co[not after.turn]
        kinds = (after.pawns, after.knights, after.bishops, after.rooks, after.queens, after.kings)
        return tuple(int(bb & us) for bb in kinds)

    def shuffles(self, after: chess.Board) -> bool:
        """Whether our pieces, after this move of ours, stand where they stood after one of
        our earlier moves since the last capture or pawn move, whatever the opponent did
        meanwhile.

        Round 67 (2026-09-08): the book played Qh4, Qg3, Qh4 against a bishop alternating
        f6 and e5, so no full position repeated until the ninth move and the game was drawn
        by repetition in ten. A return to an earlier placement is a shuffle whether or not
        the opponent shuffled too, and the search, which scores repetitions, decides it.
        A capture or a pawn move of ours ends the run before its placement is compared, so
        a recapture that puts a piece back where it stood is never a shuffle: the placements
        it would match belong to the run it ends. Castling and a lost castling right do not
        end the run; the placement ignores rights, since the run is about the pieces.
        """
        if after.halfmove_clock == 0:
            return False
        return self.placement(after) in self.ours

    def _reached(self, board: chess.Board) -> bool:
        previous = self.after_ours
        if previous is None:
            return False
        key = position_key(board)
        for move in previous.legal_moves:
            previous.push(move)
            same = position_key(previous) == key
            previous.pop()
            if same:
                return True
        return False


HISTORY = GameHistory()


class Searcher:
    """One search from one root position, bounded by a hard deadline in the kernel's ticks."""

    def __init__(
        self,
        board: chess.Board,
        hard_s: float,
        history: list[int],
        previous_score: int | None = None,
    ) -> None:
        kernel.load(board, BB, SQ, ST, KEYS)
        search.set_tapered(BB, ST, EV)
        recent = history[-search.GAME_MAX :]
        GAME[: len(recent)] = recent
        SS[:] = 0
        SS[search.SS_GAME_LENGTH] = len(recent)
        SS[search.SS_GENERATION] = self.next_generation()
        SS[search.SS_LMP_DEPTH] = search.LMP_MAX_DEPTH
        SS[search.SS_SEE_PRUNE] = 1
        SS[search.SS_SEE_QUIET_DEPTH] = search.SEE_QUIET_MAX_DEPTH
        SS[search.SS_BAD_CAPTURES] = 1
        SS[search.SS_CONT_HISTORY] = CONT_HISTORY
        SS[search.SS_PV_LEN] = 0  # the line is the iteration's, never the last move's
        SS[search.SS_STRIDE_MASK] = stride_mask(poll_stride(LAST_NPS, int(hard_s * 1000)))
        SS[search.SS_DEADLINE] = search.now(CLK) + int(hard_s * search.TICKS_PER_SECOND)
        if NODE_LIMIT:
            SS[search.SS_NODE_LIMIT] = NODE_LIMIT
            SS[search.SS_STRIDE_MASK] = 63  # a fine poll, so the stop lands near the limit
        SS[search.SS_ROOT_COUNT] = search.root_moves(S)
        self.deadline = clock() + hard_s
        self.hard_s = hard_s
        self.depth = 0
        self.score = 0
        self.previous_score = previous_score  # the last searched root score this game
        self.phase = game_phase(board)
        self.stable = 0  # iterations the best move has held
        self.windows_failed = 0
        # For the diag line: the last three completed iterations in ms, and the
        # product of the soft scalers in force when the move ended.
        self.iteration_ms: list[int] = []
        self.soft_scale = 0.0

    generation = 0

    def restrict(self, ucis: set[str]) -> None:
        """Keep only the root moves in `ucis`, the tablebase's best WDL class, when any remain."""
        count = int(SS[search.SS_ROOT_COUNT])
        kept = [int(ROOT[i]) for i in range(count) if kernel.move_to_uci(int(ROOT[i])) in ucis]
        if kept and len(kept) < count:
            ROOT[: len(kept)] = kept
            SS[search.SS_ROOT_COUNT] = len(kept)

    @classmethod
    def next_generation(cls) -> int:
        cls.generation = (cls.generation + 1) & search.GENERATION_MASK
        return cls.generation

    @property
    def nodes(self) -> int:
        return int(SS[search.SS_NODES])

    def iterate(self, started: float, soft_s: float) -> int:
        """Deepen one ply at a time until the soft budget is spent or the hard one aborts.

        From ASPIRATION_DEPTH each iteration starts on a window around the last score and
        is searched again on a wider one while its score lands outside. Returns the
        kernel's move: the best of the last finished iteration, or the best of the
        iteration in progress when the abort came after a root move had beaten it.
        """
        best_move = int(ROOT[0])
        iteration_times: list[float] = []
        if ROOT_HISTORY_KEEP_NUM < ROOT_HISTORY_KEEP_DEN:
            np.multiply(QUIET_HISTORY, ROOT_HISTORY_KEEP_NUM, out=QUIET_HISTORY)
            np.floor_divide(QUIET_HISTORY, ROOT_HISTORY_KEEP_DEN, out=QUIET_HISTORY)
        for depth in range(1, MAX_DEPTH + 1):
            iteration_started = clock()
            alpha, beta, delta = -INFINITY, INFINITY, ASPIRATION_DELTA
            if depth >= ASPIRATION_DEPTH and abs(self.score) < MATE_BOUND:
                alpha, beta = self.score - delta, self.score + delta
            first = best_move
            fails = 0
            searched = depth
            while True:
                nnue.refresh(BB, FTW, HD, ACC, int(ST[kernel.PLY]))
                move, score = search.search_root(searched, first, alpha, beta, S)
                if SS[search.SS_STOP] or alpha < score < beta:
                    break
                fails += 1
                self.windows_failed += 1
                if score >= beta:
                    first = int(move)  # the move that failed high leads the wider search
                    searched = fail_high_depth(searched, self.depth)
                alpha, beta, delta = widen_window(alpha, beta, delta, int(score), fails)
            if SS[search.SS_STOP]:
                if SS[search.SS_ITERATION_BEST]:
                    best_move = int(SS[search.SS_ITERATION_BEST])
                break
            self.stable = self.stable + 1 if int(move) == first else 0
            best_move = int(move)
            if search.FOLLOW_PV > 0:
                search.pv_walk(S, best_move)  # the line the next iteration follows (#503)
            now = clock()
            iteration_times.append(now - iteration_started)
            self.iteration_ms = [int(t * 1000) for t in iteration_times[-3:]]
            self.depth = depth
            self.score = int(score)
            SS[search.SS_CONTEMPT] = contempt_for(self.score, self.phase)
            if score >= MATE - MAX_DEPTH and depth >= MATE_STOP_DEPTH:
                # A mate for us: the move is in hand. A mate against us does not end the
                # deepening: it keeps the move's ordinary budget looking for the longest
                # defence, which the root scores higher. The platform's round 41 returned at
                # depth 1 to 4 in under 20 ms for the last five moves once a mate against us
                # was found; spending the hard budget instead measured -16 Elo at the fast
                # control, the clock drained on mate scores the deeper search overturned.
                break
            scale = soft_scale(self.node_fraction_scale(best_move), stability_scale(self.stable))
            scale *= falling_scale(self.previous_score, self.score)
            self.soft_scale = scale
            budget = soft_s * scale
            if NODE_LIMIT == 0 and now - started >= min(self.hard_s, budget):
                break
            # An iteration takes several times the last one; one that cannot finish before
            # the hard budget would be aborted with little to show, so it is not started.
            if NODE_LIMIT == 0 and now + predicted_iteration(iteration_times) > self.deadline:
                break
        return best_move

    def node_fraction_scale(self, best_move: int) -> float:
        """Stretch or shrink the soft budget by how contested the root was."""
        count = int(SS[search.SS_ROOT_COUNT])
        total = int(ROOT_NODES[:count].sum())
        if total == 0:
            return 1.0
        fraction = 0.0
        for i in range(count):
            if ROOT[i] == best_move:
                fraction = int(ROOT_NODES[i]) / total
                break
        scale = (NODE_FRACTION_BASE - fraction) * NODE_FRACTION_SCALE
        return max(NODE_FRACTION_MIN, min(NODE_FRACTION_MAX, scale))


# The upper end of the kernel's measured 2 to 3.3 growth per ply sets the floor.
# A larger live ratio of iteration times can raise the prediction, up to the cap.
ITERATION_GROWTH_DEFAULT = 3.3
ITERATION_GROWTH_MAX = 10.0


def predicted_iteration(iteration_times: list[float]) -> float:
    """Seconds the next iteration is expected to take.

    Use the larger of the kernel growth floor and the last two iterations' time ratio,
    capped at ITERATION_GROWTH_MAX. Ratios whose denominator is at most a millisecond
    are too sensitive to timing noise, so those predictions use the floor alone.
    """
    if not iteration_times:
        return 0.0
    growth = ITERATION_GROWTH_DEFAULT
    if len(iteration_times) >= 2 and iteration_times[-2] > 0.001:
        growth = max(growth, min(ITERATION_GROWTH_MAX, iteration_times[-1] / iteration_times[-2]))
    return iteration_times[-1] * growth


def stability_scale(stable: int) -> float:
    """The soft budget's factor for a best move that has held `stable` iterations."""
    return STABILITY_SCALE[min(stable, len(STABILITY_SCALE) - 1)]


def soft_scale(node_fraction: float, stability: float) -> float:
    """The two scalers' product, never under SOFT_SCALE_MIN."""
    return max(SOFT_SCALE_MIN, node_fraction * stability)


def falling_scale(previous: int | None, current: int) -> float:
    """The soft budget's factor for a root score that has fallen since the last move.

    One with no previous score, or one that held or rose, gets 1.0. A mate score or a
    tablebase-decided score passes through unchanged: this sizes a budget and never an
    evaluation, and the cap bounds what any score can ask for.
    """
    if previous is None or current >= previous:
        return 1.0
    return min(FALLING_MAX, 1.0 + (previous - current) / FALLING_DIVISOR)


def moves_to_go(move_number: int, phase: int = search.PHASE_MAX) -> int:
    """The horizon the clock is spread over: the shortest of the move schedule, the moves
    left to SPEND_BY_MOVE and the material, never under MTG_END."""
    return max(
        MTG_END,
        min(MTG_START - move_number // 2, SPEND_BY_MOVE - move_number, MTG_MIN + phase),
    )


def allocate(
    time_left_ms: int,
    move_number: int = 1,
    phase: int = search.PHASE_MAX,
    increment_ms: int = INCREMENT_MS,
) -> tuple[int, int]:
    """Soft and hard budgets for this move, in milliseconds.

    The reserve and the floor come off first (Ethereal subtracts its reserve before
    allocating), the increment, the contract's unless the clock has shown another
    (ClockTrace), is credited over the horizon less one move (Stockfish),
    the soft budget is a share of the projected pool capped at a fifth of the clock above
    the floor (Viridithas, Weiss), the hard budget four times the soft one (Ethereal)
    under two ceilings (Ethereal, Viridithas). The horizon decays with the move number
    and runs out at SPEND_BY_MOVE.
    Both budgets are at least the floor's, which the increment refills, so the clock is
    never planned below the floor; under the floor the floor's budgets shrink with the
    clock, and a clock at or below zero is read as one second (Alexandria). Neither budget
    passes the usable clock, so a floor edited for a shorter game cannot plan a flag.
    Neither falls under INCREMENT_SLICE of the increment either, above the floor or under
    it, and the hard budget never under the soft, so no clock the increment refills is
    searched at depth one for want of a budget (the London-day guard).
    """
    if time_left_ms <= 0:
        time_left_ms = GARBAGE_CLOCK_MS
    usable = max(1, time_left_ms - RESERVE_MS)
    slice_ms = int(INCREMENT_SLICE * increment_ms)
    if usable <= FLOOR_MS:
        scale = usable / FLOOR_MS
        soft_ms = max(1, int(FLOOR_SOFT_MS * scale), slice_ms)
        hard_ms = max(1, int(FLOOR_HARD_MS * scale), soft_ms)
    else:
        above = usable - FLOOR_MS
        horizon = moves_to_go(move_number, phase)
        projected = above + 0.9 * increment_ms * (horizon - 1)
        soft = min(SOFT_SHARE * projected / horizon, SOFT_CAP * above)
        hard = min(HARD_RATIO * soft, HARD_CAP * above, ABSOLUTE_CAP * above)
        soft_ms = int(max(soft, FLOOR_SOFT_MS, slice_ms))
        hard_ms = int(max(hard, FLOOR_HARD_MS, soft_ms))
    # Never past the usable clock, whatever the floor's budgets are set to.
    return min(soft_ms, usable), min(hard_ms, usable)


class ClockTrace:
    """The increment read from the clocks get_move is handed.

    Each call records the clock it was given and the time it spent; the next call's clock
    for the same colour, less what was left, is the increment the runner credited. Two
    consecutive readings that agree within OBSERVED_INCREMENT_AGREE_MS set `increment_ms`
    to their mean when it is more than OBSERVED_INCREMENT_SWITCH_MS from INCREMENT_MS, and
    back to INCREMENT_MS when it is not; a lone reading changes nothing. A reading past
    OBSERVED_INCREMENT_MAX_MS or under zero by more than the agreement is a new game in
    the same process (the smoke plays three) and is not a reading. The clocks are kept a
    colour each because the smoke and the local drivers play both sides from one process;
    the platform gives a process one side. `observe` is handed the clock exactly as
    get_move was, before anything is spent on the call, and `record` is charged with the
    whole call from its first line, so work done before the search (a warm-up joined on
    the first move, #488) is a spend and never comes off the reading: a clock with the
    join already subtracted would inflate the next reading by the join, and a join past
    OBSERVED_INCREMENT_MAX_MS would read as a new game.
    """

    def __init__(self) -> None:
        self.left_ms: dict[bool, int] = {}
        self.spent_ms: dict[bool, int] = {}
        self.readings: list[int] = []
        self.increment_ms = INCREMENT_MS

    def observe(self, time_left_ms: int, colour: bool) -> None:
        """Read this call's clock against what `colour`'s previous call left."""
        previous = self.left_ms.get(colour)
        if previous is not None:
            reading = time_left_ms - (previous - self.spent_ms.get(colour, 0))
            if -OBSERVED_INCREMENT_AGREE_MS <= reading <= OBSERVED_INCREMENT_MAX_MS:
                self.readings.append(max(0, reading))
                self._settle()
            else:
                self.readings.clear()
        self.left_ms[colour] = time_left_ms

    def record(self, spent_ms: int, colour: bool) -> None:
        """What this call spent, for `colour`'s next reading."""
        self.spent_ms[colour] = max(0, spent_ms)

    def _settle(self) -> None:
        if len(self.readings) < 2:
            return
        last, previous = self.readings[-1], self.readings[-2]
        if abs(last - previous) > OBSERVED_INCREMENT_AGREE_MS:
            return
        agreed = (last + previous) // 2
        far = abs(agreed - INCREMENT_MS) > OBSERVED_INCREMENT_SWITCH_MS
        settled = agreed if far else INCREMENT_MS
        if abs(settled - self.increment_ms) <= OBSERVED_INCREMENT_AGREE_MS and far:
            return  # the same clock read again within the noise of the runner's charge
        if settled != self.increment_ms:
            diag.note(f"increment observed {agreed} ms, {INCREMENT_MS} assumed: using {settled}")
        self.increment_ms = settled


CLOCK_TRACE = ClockTrace()


def game_phase(board: chess.Board) -> int:
    """The tapered evaluation's phase of `board`: PHASE_MAX with every piece on, 0 with none."""
    phase = sum(
        weight * len(board.pieces(piece, colour))
        for piece, weight in PHASE_WEIGHTS.items()
        for colour in (chess.WHITE, chess.BLACK)
    )
    return min(search.PHASE_MAX, phase)


def contempt_for(score: int, phase: int) -> int:
    """The draw margin against us for a root `score` at `phase`: zero near level."""
    if abs(score) <= CONTEMPT_DEAD_ZONE:
        return 0
    margin = (CONTEMPT_MG * phase + CONTEMPT_EG * (search.PHASE_MAX - phase)) // search.PHASE_MAX
    return margin if score > 0 else -margin


def poll_stride(nps: float, budget_ms: int) -> int:
    """How many nodes between clock reads: the expected node budget over 100, bounded."""
    expected_nodes = nps * budget_ms / 1000.0
    return max(POLL_STRIDE_MIN, min(POLL_STRIDE_MAX, int(expected_nodes / 100)))


def stride_mask(stride: int) -> int:
    """The stride rounded down to a power of two, as the mask the kernel tests nodes with."""
    return (1 << (max(1, stride).bit_length() - 1)) - 1


# Nodes per second of the previous move, for the poll stride of the next one.
LAST_NPS = float(DEFAULT_NPS)

# Aspiration: from ASPIRATION_DEPTH an iteration starts on a window of ASPIRATION_DELTA
# either side of the last score, and after ASPIRATION_FAILS failed windows it searches the
# open window. Below that depth the windows fail too often to pay, and a net's root score
# moves more between iterations than a table score, so the half-window is 50 rather than
# the 15 to 25 of the tuned C engines (part 2 rank 15, Buijs on TalkChess t=76115).
ASPIRATION_DEPTH = 5
# A mate for us ends the deepening, but not before MATE_STOP_DEPTH: at depth one the root's
# children answer from the table with mate scores of stale distance, and a move picked
# on those shuffles a won ending toward the fifty-move draw. A mate against us never ends
# it: the move's ordinary budget goes to the longest defence (see `iterate`).
MATE_STOP_DEPTH = 10
ASPIRATION_DELTA = 50
# Three failures rather than two, so a second widened window is tried before the open one
# (Codex's screen of 2026-09-07, `tmp/codex/2026-09-07-tune-aspiration-schedule.md`). At
# depth 10 the opening trees are unchanged and the suites take 8.0 percent fewer nodes; at
# depth 12 total nodes fall 2.4 percent. It is paid for in Win at Chess, 263 solved against
# 266, and repaid on the platform's own misses, 18 against 17, which is the set drawn from
# the games we actually got wrong. The match decides.
ASPIRATION_FAILS = 3


def fail_high_depth(searched: int, completed: int) -> int:
    """The depth of the re-search after a window failed high.

    A fail high means the move is better than we assumed, not that the tree was searched
    wrongly, so the re-search costs a whole iteration to confirm a bound the last one
    already crossed. Stockfish charges one ply for it (`adjustedDepth = std::max(1,
    rootDepth - failedHighCnt ...)` in `Search::Worker::search`) and tcheran pairs the
    same reduction with re-centring the window for +14.95 +/- 6.96 (tcheran 10.0,
    `sota-low-compute-chess-part-2.md`, search techniques, aspiration windows). Here the
    floor is the depth the last completed iteration reached as well as one, so the
    iteration never re-searches ground already covered at full depth and repeated fail
    highs cost one ply in total, not one each. A fail LOW keeps the full depth: there the
    move we have is worse than we assumed and the tree has to be seen properly.
    """
    return max(1, completed, searched - 1)


def widen_window(alpha: int, beta: int, delta: int, score: int, fails: int) -> tuple[int, int, int]:
    """The next window after a search whose score fell outside (alpha, beta).

    Only the bound that failed moves, to twice the last delta beyond the score that failed
    it; the other bound stays, so a fail high after a fail low still searches a narrow
    window. The ASPIRATION_FAILS-th failure, or a mate score, opens the window.
    """
    if fails >= ASPIRATION_FAILS or abs(score) >= MATE_BOUND:
        return -INFINITY, INFINITY, delta
    delta *= 2
    if score <= alpha:
        return max(-INFINITY, score - delta), beta, delta
    return alpha, min(INFINITY, score + delta), delta


# Piece worth for ordering captures, indexed by python-chess piece type (0 unused, 6 king).
ORDER_VALUE = (0, 1, 3, 3, 5, 9, 20)


def capture_order(board: chess.Board, move: chess.Move) -> int:
    """Sort key for captures: the most valuable victim first, the least valuable attacker."""
    victim = board.piece_type_at(move.to_square) or chess.PAWN  # en passant lands on a gap
    attacker = board.piece_type_at(move.from_square) or chess.PAWN
    return ORDER_VALUE[attacker] - 16 * ORDER_VALUE[victim]


def fallback_move(board: chess.Board, moves: list[chess.Move]) -> chess.Move:
    """The move to play when the search gives none: the best capture by victim, else the first."""
    captures = [move for move in moves if board.is_capture(move)]
    if captures:
        return min(captures, key=lambda move: capture_order(board, move))
    return moves[0]


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation.

    fen           the position to move in; your colour is the side to move
    time_left_ms  your clock before this move, in milliseconds
    returns       "e2e4", or "e7e8q" for a promotion

    Every return is checked against the legal moves of a fresh board built from the FEN.
    The ladder is the book's move, then the tablebase's move when five pieces or fewer
    are on the board, then the search's move, then the best capture, then the first legal
    move, so a search that raises or returns nonsense costs the quality of one move and
    never the game. Under PANIC_MS of clock the search is skipped and the ladder goes
    from the book and the tables to the capture.
    """
    started = time.monotonic()  # what diag.record_move reads, and what CLOCK_TRACE.record charges
    join_ms = join_warm_up()
    began = clock()
    board = chess.Board(fen)
    CLOCK_TRACE.observe(time_left_ms, board.turn)
    if join_ms:
        # The observer saw the raw clock; the search gets it less the join paid on the
        # way in, which the referee has already taken by the time the move is returned.
        usable = max(1, time_left_ms - join_ms)
        diag.note(f"init_guard join_ms={join_ms} clock_raw={time_left_ms} clock_usable={usable}")
        time_left_ms = usable
    moves = list(board.legal_moves)
    if not moves:
        return "0000"
    move: chess.Move | None = None
    depth = nodes = score = windows = 0
    iterations: list[int] = []
    scale = 0.0
    history: list[int] = []
    book_move: chess.Move | None = None
    # The `k` letters of the move line: what the book and the tablebase did with this move.
    flags = ""
    try:
        history = HISTORY.observe(board)
        book_move = BOOK.lookup(board)
        if book_move is not None:
            # A book move that repeats a position this game has seen, or returns our
            # pieces to a placement they held earlier in the run since the last capture or
            # pawn move, is left to the search, which scores repetitions; the book knows
            # nothing about the game so far (round 67: three book moves of a queen shuffle,
            # then a draw). A capture or pawn move ends the run, so it is never a return.
            # The probe is on a copy so a raise can never leave the search's board mid-line.
            after = board.copy(stack=False)
            after.push(book_move)
            if position_hash(after) in history or HISTORY.shuffles(after):
                flags += "r"
                book_move = None
    except Exception as error:
        # The tracker and the book are conveniences: a raise here costs their help on
        # this move, never the game.
        diag.note(f"history or book failed: {type(error).__name__}: {error}")
        history, book_move = [], None
    table_move: chess.Move | None = None
    # The verdict's two numbers for the move line's `w` and `z` tokens: what the table
    # said, which the log cannot recover from the score of a move that never searched.
    table_wdl: int | None = None
    table_dtz: int | None = None
    allowed: list[chess.Move] = []
    if book_move is None:
        try:
            verdict = TABLEBASE.best(board, history, position_hash)
            if verdict is not None:
                table_move = verdict.move
                table_wdl, table_dtz = verdict.wdl, verdict.dtz
            elif TABLEBASE.covers(board):
                allowed = TABLEBASE.best_moves(board)  # search supplies missing conversion detail
            skipped = TABLEBASE.unreported()  # once a material a game, not once a move
            if skipped:
                diag.note(f"tb missing {' '.join(skipped)}: searched")
            if TABLEBASE.open_ms is not None and not TABLEBASE.open_noted:
                TABLEBASE.open_noted = True  # the deferred open landed on this move's clock
                diag.note(f"tb opened on the clock: {TABLEBASE.open_ms} ms")
        except Exception as error:
            diag.note(f"tablebase failed: {type(error).__name__}: {error}")
            table_move, allowed = None, []
            table_wdl = table_dtz = None
        if table_move is not None or allowed:
            flags += "t"
    forced_warm = (
        NODE_LIMIT == 0 and time_left_ms <= FORCED_ROOT_PRESSURE_MS and len(allowed or moves) == 1
    )
    if book_move is not None:
        move = book_move
        flags += "b"
    elif table_move is not None:
        move = table_move
    elif time_left_ms >= PANIC_MS:
        try:
            global LAST_NPS
            soft_ms, hard_ms = allocate(
                time_left_ms, board.fullmove_number, game_phase(board), CLOCK_TRACE.increment_ms
            )
            if forced_warm:
                soft_ms = min(soft_ms, FORCED_ROOT_WARM_MS)
                hard_ms = min(hard_ms, FORCED_ROOT_WARM_MS)
            searcher = Searcher(board, hard_ms / 1000.0, history, HISTORY.score)
            if allowed:
                searcher.restrict({m.uci() for m in allowed})
            move = chess.Move.from_uci(
                kernel.move_to_uci(searcher.iterate(began, soft_ms / 1000.0))
            )
            depth, nodes, score = searcher.depth, searcher.nodes, searcher.score
            windows = searcher.windows_failed
            iterations, scale = searcher.iteration_ms, searcher.soft_scale
            # A forced-move warm-up supplies table entries, not the next turn's score.
            if depth > 0 and not forced_warm:
                HISTORY.score = score
            elapsed = clock() - began
            if elapsed > 0.05 and nodes > 0:
                LAST_NPS = nodes / elapsed
        except Exception as error:
            diag.note(f"search failed: {type(error).__name__}: {error}")
            move = None
    fresh = chess.Board(fen)
    fallback = (
        move is None or move not in fresh.legal_moves or (bool(allowed) and move not in allowed)
    )
    chosen = fallback_move(fresh, allowed or moves) if fallback or move is None else move
    fresh.push(chosen)
    try:
        HISTORY.record(fresh)
    except Exception as error:
        diag.note(f"history failed: {type(error).__name__}: {error}")
    diag.record_move(
        fen=fen,
        move=chosen.uci(),
        depth=depth,
        nodes=nodes,
        score_cp=score,
        started=started,
        time_left_ms=time_left_ms,
        windows_failed=windows,
        fallback=fallback,
        iterations=iterations,
        soft_scale=scale,
        flags=flags,
        wdl=table_wdl,
        dtz=table_dtz,
    )
    CLOCK_TRACE.record(int((time.monotonic() - started) * 1000), board.turn)
    return chosen.uci()


def warm_up() -> None:
    """Compile every search kernel on the real argument types, and check the stop flag.

    A short search from the start position reaches every kernel. Then a search asked to
    stop before it starts must search nothing: that proves the flag the kernels read is
    the array Python writes, which a global read would silently not be. Last, a perft of
    2 from the start position, driven from Python through the same entries the search
    calls, checks move generation, `make` and `unmake` against a known count.

    Before any of that, the four arrays the net's hand-written loads read are checked to
    start a cache line and the net's width to be whole lines of it. Those loads were
    compiled claiming 32-byte alignment (`nnue.VECTOR_ALIGN`), and an aligned load of a
    misaligned address faults the process, so a build whose arrays came from anywhere but
    `kernel.aligned` stops here, inside the init budget and in the log, rather than
    mid-game (#531).
    """
    for name, array in (("FTW", FTW), ("ACC", ACC), ("XS", XS), ("HB", HB)):
        if array.ctypes.data % kernel.ALIGNMENT != 0:
            raise RuntimeError(f"agent warm-up: {name} does not start a cache line")
    if NET is not None and (NET.l1 * ACC.itemsize) % kernel.ALIGNMENT != 0:
        raise RuntimeError("agent warm-up: one accumulator is not whole cache lines")
    # The kernels compile inside this first search, 11 to 15 s here and longer on a loaded
    # machine, and the deadline is set before they do: a hard budget of 60 s keeps a slow
    # compile from stopping the search at depth 0, which would end the process before
    # its ready line. The 20 ms soft budget still stops it after depth 1.
    # The two recursive searches first, on their declared signatures: the declaration is
    # what lets numba answer a recursive call's return type without re-inferring it.
    search.compile_recursive()
    searcher = Searcher(chess.Board(), 60.0, [])
    searcher.iterate(clock(), 0.02)
    if searcher.depth < 1 or searcher.nodes == 0:
        raise RuntimeError("agent warm-up: the search did not run")
    SS[search.SS_STOP] = 1
    SS[search.SS_NODES] = 0
    nnue.refresh(BB, FTW, HD, ACC, int(ST[kernel.PLY]))
    search.search_root(3, 0, -INFINITY, INFINITY, S)
    if SS[search.SS_NODES] > SS[search.SS_ROOT_COUNT] or SS[search.SS_ITERATION_BEST] != 0:
        raise RuntimeError("agent warm-up: the search does not honour the stop flag")
    SS[search.SS_STOP] = 0
    search.pv_walk(S, 0)  # compiled here; the root key alone, since no move is given
    # Perft 2 through `root_moves`, `make_s` and `unmake_s`, the entries the search runs:
    # the array entries of the same kernels are the tests' and would compile a second
    # copy of each here for nothing. `root_moves` writes its moves to `ROOT`, so a level's
    # moves are copied out before the next call overwrites them.
    first = search.root_moves(S)
    moves = [int(ROOT[i]) for i in range(first)]
    nodes = 0
    for move in moves:
        search.make_s(move, S)
        nodes += search.root_moves(S)
        search.unmake_s(move, S)
    if first != 20 or nodes != 400:
        raise RuntimeError("agent warm-up: perft 2 from the start is not 400")
    for name, fn in {**search.WARMED, **nnue.WARMED}.items():
        signatures = len(fn.signatures)
        # A callee compiled inside a cached caller has no signature of its own, so under
        # the cache the count is at most one; without it, exactly one.
        if signatures > 1 or (signatures == 0 and not kernel.CACHE):
            raise RuntimeError(f"agent warm-up: {name} has {signatures} signatures, expected 1")
    TT[:] = 0
    QUIET_HISTORY[:] = 0
    CAPTURE_HISTORY[:] = 0
    CONTINUATION_HISTORY[:] = 0
    PAWN_STRUCTURE_HISTORY[:] = 0
    KILLERS[:] = 0


# The warm-up in its thread, joined here until the guard and again by the first
# get_move. numba compiles under its own lock from any thread; the search itself runs
# only on the main thread, after the join, so the one-thread rule holds.
WARM_ERROR: list[BaseException] = []


def warm_in_thread() -> None:
    try:
        warm_up()
    except BaseException as error:  # reported by the first get_move
        WARM_ERROR.append(error)


WARM_THREAD: threading.Thread | None = None
if INIT_GUARD_S > 0:
    WARM_THREAD = threading.Thread(target=warm_in_thread, name="warm-up", daemon=True)
    WARM_THREAD.start()
    diag.note(f"init_guard s={INIT_GUARD_S:.0f} thread_at={time.monotonic() - IMPORT_STARTED:.1f}s")
    WARM_THREAD.join(max(0.0, INIT_GUARD_S - (time.monotonic() - IMPORT_STARTED)))
    WARM_DEFERRED = WARM_THREAD.is_alive()
    if WARM_DEFERRED:
        diag.note(f"warm-up still compiling at the {INIT_GUARD_S:.0f} s guard; move one joins it")
else:
    warm_up()
    WARM_DEFERRED = False


def join_warm_up() -> int:
    """Wait for a warm-up the import left compiling; returns the wait in milliseconds.

    The caller charges the wait to the clock after the clock observer has seen the raw
    reading, so the increment the observer detects between calls stays the referee's.
    Raises the warm-up's own error if it had one, since a search on uncompiled or
    unchecked kernels is a crash later.
    """
    waited_ms = 0
    if WARM_THREAD is not None and WARM_THREAD.is_alive():
        waited = time.monotonic()
        WARM_THREAD.join()
        waited_ms = int((time.monotonic() - waited) * 1000)
        diag.note(f"warm-up joined on the clock: {waited_ms} ms")
    if WARM_ERROR:
        raise WARM_ERROR[0]
    return waited_ms


# The tablebase's open finds its files and builds a table object for each; python-chess maps
# a table's data at that table's first probe, so the open costs milliseconds and not the two
# seconds an earlier reading of two games gave it (Codex 100, #464). It runs here when the
# import has been quick, and otherwise on the first root with five pieces or fewer, on that
# move's clock. An error here would abort the import and lose the game, so it is caught the
# way the deferred open is caught on the clock: the tables stay shut and the search plays.
TABLEBASE_EAGER = eager_at(time.monotonic() - IMPORT_STARTED)
if TABLEBASE_EAGER:
    try:
        TABLEBASE.open_tables()
    except Exception as error:
        TABLEBASE_EAGER = False
        diag.note(f"tablebase open failed at import: {type(error).__name__}: {error}")

diag.note(
    f"init_guard s={INIT_GUARD_S:.0f} at_init_line={time.monotonic() - IMPORT_STARTED:.1f}s "
    f"state={'compiling' if WARM_THREAD is not None and WARM_THREAD.is_alive() else 'done'}"
)

# Last in the import, after every table above is built: the init line carries the import
# time, the core and the benchmarks that calibrate local numbers against the platform's.
diag.init_done(
    net=NET_NAME if NET is not None else "none",
    isa=nnue.host_isa(),
    slider=kernel.SLIDER_INDEX,
    warm="deferred" if WARM_DEFERRED else "done",
    head=nnue.HEAD,
    first_layer=nnue.FIRST_LAYER,
    first_layer_check=nnue.FIRST_LAYER_CHECK,
    second_layer=nnue.SECOND_LAYER,
    second_layer_check=nnue.SECOND_LAYER_CHECK,
    clip=nnue.CLIPPED_PRODUCT,
    clip_check=nnue.CLIPPED_PRODUCT_CHECK,
    hidden=nnue.HIDDEN_CLIP,
    hidden_check=nnue.HIDDEN_CLIP_CHECK,
    acc=nnue.ACCUMULATE,
    acc_check=nnue.ACCUMULATE_CHECK,
    book=BOOK.entries,
    tb=TABLEBASE.tables,
    tbo="eager" if TABLEBASE_EAGER else "lazy",  # which open path the import took
)
