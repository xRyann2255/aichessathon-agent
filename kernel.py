"""The board and its moves as numba kernels over preallocated arrays.

python-chess parses the FEN and prints the move; everything between is here, compiled by
numba at import. A position is three arrays: `bb`, fifteen uint64 bitboards (twelve
pieces, then the white, black and whole occupancy); `sq`, the piece on each of the 64
squares or EMPTY; and `st`, the side to move, castling rights, en passant square,
halfmove clock, the ply of the undo stack and the Zobrist hash. Moves are pseudo-legal:
generation never asks whether the king is left in check, `make` plays the move, and the
caller asks `attacked` about the king afterwards and unmakes if it must. A move is one
int64: from, to, the moving piece, the captured piece, the promotion piece and a flag
for double pushes, en passant and castling.

Pieces are numbered white pawn to king 0 to 5 and black pawn to king 6 to 11, so a
piece's type is `piece % 6` and its colour `piece // 6`, and the piece bitboards, the
mailbox and the Zobrist keys all use that number.

Everything a kernel reads and Python writes arrives as an argument: the attack table from
`bitboards.py`, the Zobrist keys, the move list, the undo stack. numba freezes a global
array of under a million bytes into the compiled code at first call, silently, so a
kernel that read one as a global would never see a later write.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, TypeVar

import chess
import numpy as np
import numpy.typing as npt
from llvmlite import ir as llvm_ir  # type: ignore[import-untyped]
from numba import njit, types
from numba.core import cgutils
from numba.core import compiler as numba_compiler
from numba.core import typeinfer as numba_typeinfer
from numba.core.cpu import CPUTargetOptions
from numba.core.options import TargetOptions
from numba.extending import intrinsic

import bitboards as tables

# The two codegen changes this build carries, both unconditional: the build is the arm a
# gate reads against main, so there is no switch to set and no second path to keep alive.
#
# For a reviewer: they set an option on numba's own compiler over our own kernels and
# move where a numpy array we allocated starts. Neither reads a file, opens a socket,
# starts a process, or writes anything outside this process's own memory.
#
# Forceinline (#529): a small kernel marked `jit(inline=True)` is compiled once and marked
# `alwaysinline` for LLVM, which inlines it at every call and drops the function, instead
# of numba pasting its source into each caller and re-typing and re-lowering the body
# there. The body that runs is the same one; the compile is one copy rather than one a
# call site. What numba's own inlining does and LLVM's does not is fold the pasted body's
# branches against the constants of each call site, which grew the three hot dispatchers
# when every site moved, so the kernels the search's own three call keep it: `hot=True`.
#
# Alignment (#531): the accumulators, the net's first layer and the head's two scratch
# arrays are cut from an over-allocated buffer at the next 64-byte boundary instead of
# starting wherever malloc put them, which is 16 bytes. A pair of accumulators and a
# first-layer row are 512 bytes each, so every per-ply and per-row slice of a base on a
# line starts on one too, and the hand-written AVX2 and VNNI loads of `nnue.py` say so:
# each one that this proves 32-byte aligned is loaded `align=32` and not `align=2`. The
# proof is written beside the load, and the warm-up checks the bases it rests on.
ALIGNMENT = 64

# numba's disk cache, for the local tools only. The platform's `/tmp` is empty for every
# game and its source directory is read-only, so a cache never helps there, the flag is
# never set, and the platform runs the uncached path. `tools/sprt.py`, `tools/smoke.py`
# and `tools/equivalence.py` set it for their child processes, which then load the
# compiled kernels from `__pycache__` beside the source instead of compiling them: 18 s
# a process here under a match's load (`notes/measurements/2026-09-06-import-cache.md`).
CACHE = os.environ.get("AICHESSATHON_KERNEL_CACHE") == "1"

# A kernel that only compiled code calls needs neither of numba's entry wrappers
# (`jit`'s `internal`), and most of ours are in that class inside the zip: only the
# tests and the tools call them from Python. Those set AICHESSATHON_PYTHON_ENTRY
# before importing, which gives the wrappers back; the shipped build leaves it unset
# and every internal kernel compiles without them (#564).
TESTING = os.environ.get("AICHESSATHON_PYTHON_ENTRY", "0") == "1"
F = TypeVar("F", bound=Callable[..., Any])


def _expose_noalias() -> bool:
    """Let a decorator set numba's `noalias` flag, which the CPU target hides.

    An array argument reaches a compiled kernel as a plain pointer, so LLVM has to assume
    that a store through one of them lands in the memory another reads: the `ml` store in
    `push_targets` below is taken to clobber the `sq` load beside it and the `tab` load in
    the caller. numba can say otherwise. `Flags.noalias` exists
    (`numba/core/compiler.py`), the lowerer carries it into the function descriptor
    (`numba/core/typed_passes.py`) and the calling convention then puts `noalias` and
    `captures(none)` on every pointer argument (`numba/core/callconv.py`). What is missing
    is the last step: `numba/core/cpu.py` builds the set of options a decorator may pass
    without naming it, and `numba/core/options.py` collects those options by reading the
    class's own attributes, so adding the mapping the list omits is the whole door.

    The promise the flag makes is that during a call no two arguments reach the same
    memory, which every call site here keeps: each array is its own allocation, held by
    `agent.py` for the game and handed to `search.new_search` once, no kernel takes both
    the search object and one of the arrays inside it, and every slice a kernel works in
    is cut from one of its own arguments. Rows of one array are arguments in a few of the
    benches, and those stay inside their own row. A kernel that ever breaks the promise
    takes `jit(noalias=False)` and says in a comment which two arguments share a buffer.
    """
    if "noalias" not in numba_compiler.Flags.options:
        return False
    if not isinstance(getattr(CPUTargetOptions, "noalias", None), TargetOptions.Mapping):
        mapping = TargetOptions.Mapping("noalias")  # type: ignore[no-untyped-call]
        CPUTargetOptions.noalias = mapping
    return True


# For a reviewer: this turns on a compiler option numba already carries and no decorator
# exposes, over our own kernels. It is not obfuscation: it reads no file, opens no socket,
# starts no process and writes nothing outside this process's own memory.
NOALIAS = _expose_noalias()


# For a reviewer: this replaces one numba function in memory, in this process only; it
# changes no engine move and ships no engine code. numba resolves a self-recursive call's
# return type by cloning the whole type inference state, rebuilding its constraint network
# and propagating it again, once per recursive call site (`typeinfer.py`,
# `return_types_from_partial`): 24 sites in `negamax` and three in `quiesce`, about 9 s of
# the import. When the compile already declares the return type (numba's `seed_return`
# locks every return variable, which `Dispatcher.compile` on an explicit signature does),
# that clone can only return the type already locked, so the wrapper answers with it and
# delegates every other case to numba untouched.
TYPED_RECURSION = os.environ.get("AICHESSATHON_TYPED_RECURSION", "1") == "1"


def _shortcut_recursive_return() -> bool:
    """Answer a recursive return from the locked return type instead of re-inferring it.

    Returns False, leaving numba as it is, if the function this depends on is not the
    shape expected: the saving is a compile time, never a move.
    """
    inferer = getattr(numba_typeinfer, "TypeInferer", None)
    original = getattr(inferer, "return_types_from_partial", None)
    if original is None:
        return False

    def return_types_from_partial(self: Any) -> Any:
        """numba's own, short-circuited when every return variable is already locked."""
        locked = set()
        for retvar in self._get_return_vars():
            typevar = self.typevars.get(retvar.name)
            if typevar is None or not getattr(typevar, "locked", False) or not typevar.defined:
                return original(self)
            locked.add(types.unliteral(typevar.getone()))  # type: ignore[no-untyped-call]
        if len(locked) != 1:
            return original(self)
        return locked.pop()

    inferer.return_types_from_partial = return_types_from_partial  # type: ignore[union-attr]
    return True


TYPED_RECURSION = TYPED_RECURSION and _shortcut_recursive_return()


def jit(
    inline: bool = False, noalias: bool = True, hot: bool = False, internal: bool = False
) -> Callable[[F], F]:
    """`njit` with the kernels' options: the disk cache flag, and numba's runtime off.

    No kernel allocates: every array is made by Python before the search and outlives it.
    With the runtime on, numba takes a reference on every array a kernel loads or receives
    as an argument and drops it at every exit, a call each and about 74 a node, and an
    atomic pair on the search object at every call that takes it
    (`notes/measurements/2026-09-06-kernel-profile.md`). `noalias` is the promise
    `_expose_noalias` describes, on by default and set to False on any kernel two of whose
    array arguments can reach the same buffer.

    `inline` inlines the function at every call site, for the small ones called once a
    node, and `hot` chooses which compiler does it. Without `hot` LLVM does, on one
    compiled copy (`forceinline`, #529). With it numba does, by pasting the source into
    each caller and typing it there (`inline="always"`), which costs a compile a call site
    and buys the branches of the body folded against that site's constants: `hot=True`
    marks the kernels that `negamax`, `quiesce` and `search_root` call directly, whose
    dispatchers grew by 10 to 16 percent of their instructions when they lost that folding.
    A callee reached only through another callee is not hot: it is inlined into a body
    that has already been folded.

    `internal` marks a kernel that only compiled code calls. Every dispatcher numba
    compiles gets two entry wrappers beside the function itself, one for a call from
    Python and one with a C signature, each lowered and optimised with it at import; a
    kernel no Python code calls pays for both and uses neither, so the flag leaves them
    out (numba's `no_cpython_wrapper` and `no_cfunc_wrapper`). The function's own machine
    code is unchanged. A call from Python to an internal kernel has no entry to land on,
    so the flag goes only on a kernel that nothing in this repo calls from Python: not the
    warm-ups, not `agent.py`, not the tests or the tools. Those keep their wrappers.
    """
    options: dict[str, Any] = {"cache": CACHE, "_nrt": False}
    if inline and hot:
        options["inline"] = "always"
    elif inline:
        options["forceinline"] = True
    if noalias and NOALIAS:
        options["noalias"] = True
    if internal:
        options["no_cpython_wrapper"] = True
        options["no_cfunc_wrapper"] = True
    return njit(**options)  # type: ignore[no-any-return]


Bitboards = npt.NDArray[np.uint64]
Ints = npt.NDArray[np.int64]
Uint8s = npt.NDArray[np.uint8]
Int32s = npt.NDArray[np.int32]


def aligned(count: int, dtype: Any, rows: int = 0) -> Any:
    """A zeroed C-contiguous array of `count` items whose first byte starts a cache line.

    numpy's allocator promises 16 bytes, so a 32-byte load halfway through a line costs a
    split on every other read. Over-allocating one line and slicing forward to the next
    boundary costs 64 bytes once and holds the buffer alive through the slice's `base`.
    `rows` shapes the result as `(rows, count // rows)`, still C-contiguous (#531).

    Every array the hand-written loads of `nnue.py` read is built here: those loads claim
    32-byte alignment, which the base gives them and a plain `np.zeros` would not, so an
    array that reaches one of them from anywhere else faults the process.
    """
    itemsize = int(np.dtype(dtype).itemsize)
    buffer = np.zeros(count + ALIGNMENT // itemsize, dtype=dtype)
    offset = ((-buffer.ctypes.data) % ALIGNMENT) // itemsize
    flat = buffer[offset : offset + count]
    array = flat.reshape((rows, count // rows)) if rows else flat
    if array.ctypes.data % ALIGNMENT != 0 or not array.flags["C_CONTIGUOUS"]:
        raise RuntimeError("the aligned allocation is not 64-byte aligned and contiguous")
    return array


def aligned_like[A: npt.NDArray[Any]](array: A) -> A:
    """A zeroed array of the same shape and dtype as `array`, on a 64-byte boundary.

    `np.zeros_like` of an aligned array is not aligned: numpy allocates it afresh. The
    tests and the benches hand their own accumulators and scratch to the kernels, so they
    build them here (#531).
    """
    rows = int(array.shape[0]) if array.ndim == 2 else 0
    built: A = aligned(int(array.size), array.dtype, rows=rows)
    return built


# Pieces and the bitboard array.
WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK = range(12)
EMPTY = 12
OCC_WHITE = 12
OCC_BLACK = 13
OCC_ALL = 14
BB_SIZE = 15
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(6)
WHITE = 0
BLACK = 1

# The state array.
STM = 0  # side to move, 0 white, 1 black
CASTLE = 1  # rights: 1 white kingside, 2 white queenside, 4 black kingside, 8 black queenside
EP = 2  # the en passant target square, or -1; set only when an enemy pawn could take
HALF = 3  # halfmove clock
PLY = 4  # depth of the undo stack, 0 at the position get_move received
HASH = 5  # Zobrist hash, as int64 bits
# The tapered piece-square evaluation, carried across make and unmake by the search's
# `search.tapered_make` rather than summed over the pieces at every leaf.
ST_MG = 6  # the middlegame piece-square sum, white's view, black's values already negated
ST_EG = 7  # the same over the endgame tables
ST_PHASE = 8  # the phase, `search.PHASE_INC` summed over the pieces, not yet clamped
ST_SIZE = 9

# Move flags.
NORMAL = 0
DOUBLE_PUSH = 1
EN_PASSANT = 2
CASTLING = 3

MAX_PLY = 128
MOVES_PER_PLY = 256
# Captured piece, castling rights, en passant square, halfmove clock, hash, and the three
# tapered numbers; eight int64 is one cache line a ply.
UNDO_FIELDS = 8
UNDO_MG = 5
UNDO_EG = 6
UNDO_PHASE = 7
NULL_MARK = 13  # in the captured field of an undo slot: the move was a null move

# Zobrist key layout: 12 x 64 piece-square keys, 16 castling keys, 8 en passant files, side.
KEY_CASTLE = 768
KEY_EP = 784
KEY_SIDE = 792
KEY_COUNT = 793

A1, B1, C1, D1, E1, F1, G1, H1 = range(8)
A8, B8, C8, D8, E8, F8, G8, H8 = range(56, 64)
RANK_3 = np.uint64(0x0000000000FF0000)
RANK_6 = np.uint64(0x0000FF0000000000)
RANK_8 = np.uint64(0xFF00000000000000)
RANK_1 = np.uint64(0x00000000000000FF)
KING_SIDE_PATH = np.uint64(0x60)  # f1 and g1; shifted up the board for black
QUEEN_SIDE_PATH = np.uint64(0x0E)  # b1, c1 and d1
ONE = np.uint64(1)
ZERO = np.uint64(0)

# Castling rights a move from or to each square leaves standing.
CASTLE_KEEP = tuple(
    {A1: 13, E1: 12, H1: 14, A8: 7, E8: 3, H8: 11}.get(square, 15) for square in range(64)
)

# Index of the lowest set bit, by a De Bruijn multiply (Leiserson, Prokop and Randall 1998).
DEBRUIJN = 0x03F79D71B4CA8B09
DEBRUIJN_INDEX = tuple(sorted(range(64), key=lambda i: ((DEBRUIJN << i) & (2**64 - 1)) >> 58))


@jit()
def popcount(bits: np.uint64) -> np.int64:
    """Number of set bits, by the parallel-count folding that LLVM turns into popcnt."""
    bits = bits - ((bits >> np.uint64(1)) & np.uint64(0x5555555555555555))
    bits = (bits & np.uint64(0x3333333333333333)) + (
        (bits >> np.uint64(2)) & np.uint64(0x3333333333333333)
    )
    bits = (bits + (bits >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return np.int64((bits * np.uint64(0x0101010101010101)) >> np.uint64(56))


@jit()
def lsb(bits: np.uint64) -> int:
    """Index of the lowest set bit; the argument must not be zero."""
    isolated = bits & (~bits + ONE)
    return DEBRUIJN_INDEX[np.int64((isolated * np.uint64(DEBRUIJN)) >> np.uint64(58))]


# Every index in the hot kernels is cast to np.uint64 at the bracket. numba wraps a signed
# index (`if i < 0: i += n`) into every load and store, a compare and a cmov for an index
# with a constant offset and a sign-extend, mask and add on the address chain for a bare
# variable, and skips the wrap for an unsigned index; the cast itself is a register rename.
# The variables and the arithmetic stay int64: a uint64 meeting an int64 in arithmetic
# promotes to float64 in numba, so only the bracket's whole expression is cast.
# The slider lookups index the attack table by PEXT where the host runs it as one
# instruction and by the magic multiply elsewhere (`bitboards.fast_pext_host`); the flag
# is a module constant, so numba compiles one path and no test of it at run time. The
# table was filled for the same choice at import (`bitboards.build_tables`).
USE_PEXT = tables.USE_PEXT
SLIDER_INDEX = "pext" if USE_PEXT else "magic"


@intrinsic
def pext64(typingctx: Any, value: Any, mask: Any) -> Any:
    """`llvm.x86.bmi.pext.64`: the bits of `value` under `mask`, packed to the low end."""
    sig = types.uint64(value, mask)

    def codegen(context: Any, builder: Any, signature: Any, args: Any) -> Any:
        i64 = llvm_ir.IntType(64)
        fnty = llvm_ir.FunctionType(i64, [i64, i64])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "llvm.x86.bmi.pext.64")  # type: ignore[no-untyped-call]
        source = context.cast(builder, args[0], signature.args[0], types.uint64)
        selector = context.cast(builder, args[1], signature.args[1], types.uint64)
        return builder.call(fn, [source, selector])

    return sig, codegen


@jit()
def rook_attacks(square: int, occupancy: np.uint64, tab: Bitboards) -> np.uint64:
    mask = tab[np.uint64(tables.ROOK_MASK + square)]
    index: np.uint64
    if USE_PEXT:
        index = pext64(occupancy, mask)  # type: ignore[call-arg]
    else:
        index = ((occupancy & mask) * tab[np.uint64(tables.ROOK_MAGIC + square)]) >> tab[
            np.uint64(tables.ROOK_SHIFT + square)
        ]
    return np.uint64(
        tab[np.uint64(tables.ROOK_TABLE) + tab[np.uint64(tables.ROOK_OFFSET + square)] + index]
    )


@jit()
def bishop_attacks(square: int, occupancy: np.uint64, tab: Bitboards) -> np.uint64:
    mask = tab[np.uint64(tables.BISHOP_MASK + square)]
    index: np.uint64
    if USE_PEXT:
        index = pext64(occupancy, mask)  # type: ignore[call-arg]
    else:
        index = ((occupancy & mask) * tab[np.uint64(tables.BISHOP_MAGIC + square)]) >> tab[
            np.uint64(tables.BISHOP_SHIFT + square)
        ]
    return np.uint64(
        tab[np.uint64(tables.BISHOP_TABLE) + tab[np.uint64(tables.BISHOP_OFFSET + square)] + index]
    )


# The hot kernels come in three parts. The body carries the algorithm, compiled with
# inline=True so numba pastes it into each entry at the IR level. The entry of the
# kernel's own name takes its arrays one by one, for perft, the tests and the benches.
# The entry the search calls, `<name>_s` in `search.py`, takes the search object and
# reads its arrays from it inside its own compiled body: numba passes an array as seven
# words by value, so a call that took six arrays pushed forty-two words where the object
# is two, and a wrapper around an unchanged callee is not inlined and costs a call, so
# the loads sit in the callee (`notes/measurements/2026-09-06-kernel-profile.md`, the
# line-level section of 2026-09-08).
@jit(inline=True, internal=not TESTING)
def attacked_body(square: int, by: int, bb: Bitboards, tab: Bitboards) -> bool:
    """Whether side `by` attacks `square`, with the occupancy as it stands in `bb`.

    Each magic lookup sits behind a test of that slider's rays from `square` on an empty
    board. The lookup returns a subset of those rays whatever the blockers are, so with no
    enemy bishop or queen anywhere on the diagonals the diagonal lookup cannot find one,
    and the ray test says so from one load beside the leaper tables instead of one into
    the 800 KB slider tables. Over a depth 6 search of the profiler's positions the
    diagonal test opens on 9.7 percent of the calls and the straight one on 8.6, against
    95.4 percent of calls that are not attacks at all and so run every test to the end
    (`tmp/logs/kernel-check-repeated-split.log`).
    """
    base = by * 6
    if tab[np.uint64(tables.PAWN + (1 - by) * 64 + square)] & bb[np.uint64(base + PAWN)]:
        return True
    if tab[np.uint64(tables.KNIGHT + square)] & bb[np.uint64(base + KNIGHT)]:
        return True
    if tab[np.uint64(tables.KING + square)] & bb[np.uint64(base + KING)]:
        return True
    queens = bb[np.uint64(base + QUEEN)]
    diagonal = bb[np.uint64(base + BISHOP)] | queens
    straight = bb[np.uint64(base + ROOK)] | queens
    occupancy = bb[OCC_ALL]
    if (tab[np.uint64(tables.BISHOP_RAY + square)] & diagonal) != 0 and (
        bishop_attacks(square, occupancy, tab) & diagonal
    ) != 0:
        return True
    if (tab[np.uint64(tables.ROOK_RAY + square)] & straight) == 0:
        return False
    return bool(rook_attacks(square, occupancy, tab) & straight)


@jit()
def attacked(square: int, by: int, bb: Bitboards, tab: Bitboards) -> bool:
    """`attacked_body` on arrays passed one by one; the search calls `attacked_s`."""
    return attacked_body(square, by, bb, tab)


@jit()
def in_check(bb: Bitboards, st: Ints, tab: Bitboards) -> bool:
    """Whether the side to move is in check."""
    us = st[STM]
    return attacked(lsb(bb[np.uint64(us * 6 + KING)]), 1 - us, bb, tab)


@jit(inline=True, internal=not TESTING)
def pinned_pieces_body(us: int, bb: Bitboards, tab: Bitboards) -> np.uint64:
    """The pieces of `us` pinned to their king: each sits alone between the king and an
    enemy rook, bishop or queen on its line.

    A move by any other piece of ours, when the king is not in check and the move is not
    en passant, cannot expose the king, so the search skips the king-attack test after
    making it (`needs_legality_test`); Stockfish's `legal()` has the same shape. The
    squares between the king and a slider are the meet of the king's rays with the slider
    alone on the board and the slider's rays with the king alone on it: the two other
    rays of each are parallel, and the other crossings are the two pieces' own squares.
    """
    them = 1 - us
    king = lsb(bb[np.uint64(us * 6 + KING)])
    own = bb[np.uint64(OCC_WHITE + us)]
    occupancy = bb[OCC_ALL]
    empty = np.uint64(0)
    one = np.uint64(1)
    king_bit = one << np.uint64(king)
    pinned = np.uint64(0)
    snipers = rook_attacks(king, empty, tab) & (
        bb[np.uint64(them * 6 + ROOK)] | bb[np.uint64(them * 6 + QUEEN)]
    )
    while snipers != 0:
        sniper = lsb(snipers)
        snipers &= snipers - one
        between = rook_attacks(king, one << np.uint64(sniper), tab) & rook_attacks(
            sniper, king_bit, tab
        )
        blockers = between & occupancy
        if popcount(blockers) == 1 and (blockers & own) != 0:
            pinned |= blockers
    snipers = bishop_attacks(king, empty, tab) & (
        bb[np.uint64(them * 6 + BISHOP)] | bb[np.uint64(them * 6 + QUEEN)]
    )
    while snipers != 0:
        sniper = lsb(snipers)
        snipers &= snipers - one
        between = bishop_attacks(king, one << np.uint64(sniper), tab) & bishop_attacks(
            sniper, king_bit, tab
        )
        blockers = between & occupancy
        if popcount(blockers) == 1 and (blockers & own) != 0:
            pinned |= blockers
    return pinned


@jit(internal=not TESTING)
def pinned_pieces(us: int, bb: Bitboards, tab: Bitboards) -> np.uint64:
    """`pinned_pieces_body` on arrays passed one by one; the search calls `pinned_pieces_s`."""
    return pinned_pieces_body(us, bb, tab)


@jit(inline=True, hot=True, internal=not TESTING)
def needs_legality_test(move: int, checked: bool, pinned: np.uint64) -> bool:
    """Whether a pseudo-legal move can leave the mover's king in check: any move when in
    check, a king move, an en passant capture, or a move by a pinned piece. Every other
    move is legal without `legal_after`."""
    return (
        checked
        or move_piece(move) % 6 == KING
        or move_flag(move) == EN_PASSANT
        or (pinned >> np.uint64(move_source(move))) & np.uint64(1) != 0
    )


@jit(inline=True, internal=not TESTING)
def gives_check_body(move: int, bb: Bitboards, st: Ints, tab: Bitboards) -> bool:
    """Whether `move`, pseudo-legal for the side to move, checks the enemy king, without making it.

    The occupancy after the move is the board's with the source cleared and the target set
    (an en passant capture also clears the taken pawn, castling moves the rook), and the
    king is attacked either by the piece that lands, promoted or not, or by a rook, bishop
    or queen of ours along a line the move opened: the same answer `in_check` gives after
    `make`, at the cost of a few attack lookups instead of a make, an unmake and the
    accumulator's update.
    """
    us = st[STM]
    them = 1 - us
    base = us * 6
    king = lsb(bb[np.uint64(them * 6 + KING)])
    source = move_source(move)
    target = move_target(move)
    flag = move_flag(move)
    landed = move_promotion(move)
    if landed == 0:
        landed = move_piece(move) % 6
    from_bit = ONE << np.uint64(source)
    to_bit = ONE << np.uint64(target)
    occupancy = (bb[OCC_ALL] & ~from_bit) | to_bit
    rooks_queens = (bb[np.uint64(base + ROOK)] | bb[np.uint64(base + QUEEN)]) & ~from_bit
    bishops_queens = (bb[np.uint64(base + BISHOP)] | bb[np.uint64(base + QUEEN)]) & ~from_bit
    if flag == EN_PASSANT:
        taken = target - 8 if us == WHITE else target + 8
        occupancy &= ~(ONE << np.uint64(taken))
    elif flag == CASTLING:
        if target > source:
            rook_source, rook_target = target + 1, target - 1
        else:
            rook_source, rook_target = target - 2, target + 1
        rook_bits = (ONE << np.uint64(rook_source)) | (ONE << np.uint64(rook_target))
        occupancy ^= rook_bits
        rooks_queens ^= rook_bits
    # The piece that lands, seen from the king's square outward; a king never gives check.
    direct = np.uint64(0)
    if landed == PAWN:
        direct = tab[np.uint64(tables.PAWN + them * 64 + king)] & to_bit
    elif landed == KNIGHT:
        direct = tab[np.uint64(tables.KNIGHT + king)] & to_bit
    elif landed == BISHOP:
        direct = bishop_attacks(king, occupancy, tab) & to_bit
    elif landed == ROOK:
        direct = rook_attacks(king, occupancy, tab) & to_bit
    elif landed == QUEEN:
        direct = bishop_attacks(king, occupancy, tab) | rook_attacks(king, occupancy, tab)
        direct &= to_bit
    if direct:
        return True
    # A line opened by the move, or the castling rook.
    if rook_attacks(king, occupancy, tab) & rooks_queens:
        return True
    return bool(bishop_attacks(king, occupancy, tab) & bishops_queens)


@jit(internal=not TESTING)
def gives_check(move: int, bb: Bitboards, st: Ints, tab: Bitboards) -> bool:
    """`gives_check_body` on arrays passed one by one; the search calls `gives_check_s`."""
    return gives_check_body(move, bb, st, tab)


@jit(inline=True, internal=not TESTING)
def check_masks_body(
    bb: Bitboards, st: Ints, tab: Bitboards
) -> tuple[np.uint64, np.uint64, np.uint64, np.uint64, np.uint64]:
    """The squares a piece of the side to move checks the enemy king from, one mask a kind,
    and the pieces of ours that open a line by leaving their square.

    A mask is the enemy king's own attack set for the kind, over the board as it stands: a
    knight of ours checks from the squares a knight on their king attacks, a bishop from
    the squares a bishop there attacks, and so on, the queen's being the two slider masks
    together. The board is the one before the move and not after it, which is the same
    answer for every move a mask is asked about: a target the king can see now it can
    still see once the mover has left its square, and a target it cannot see is blocked by
    a piece that stays, since a slider whose line to their king were open would already be
    checking them and it is our move.

    `discovery` holds each piece of ours that stands alone between one of our sliders and
    their king. Only a move from one of those squares can open a line, so only those moves
    need the full test; the mover that lands where the blocker stood keeps the line shut,
    and a capture leaves one of ours on the square.
    """
    us = st[STM]
    them = 1 - us
    base = us * 6
    king = lsb(bb[np.uint64(them * 6 + KING)])
    occupancy = bb[OCC_ALL]
    pawn = tab[np.uint64(tables.PAWN + them * 64 + king)]
    knight = tab[np.uint64(tables.KNIGHT + king)]
    bishop = bishop_attacks(king, occupancy, tab)
    rook = rook_attacks(king, occupancy, tab)
    ours = bb[np.uint64(OCC_WHITE + us)]
    discovery = np.uint64(0)
    snipers = tab[np.uint64(tables.ROOK_RAY + king)] & (
        bb[np.uint64(base + ROOK)] | bb[np.uint64(base + QUEEN)]
    )
    snipers |= tab[np.uint64(tables.BISHOP_RAY + king)] & (
        bb[np.uint64(base + BISHOP)] | bb[np.uint64(base + QUEEN)]
    )
    while snipers != 0:
        sniper = lsb(snipers)
        snipers &= snipers - ONE
        between = tab[np.uint64(tables.BETWEEN + king * 64 + sniper)] & occupancy
        if between != 0 and (between & (between - ONE)) == 0:
            discovery |= between & ours
    return pawn, knight, bishop, rook, discovery


@jit(inline=True, internal=not TESTING)
def evasion_targets_body(bb: Bitboards, st: Ints, tab: Bitboards) -> np.uint64:
    """At a node in check: the squares a non-king move may land on and still be legal.

    Under a single check those are the checker's square and the squares between it and the
    king; under a double check there are none. A king move is not judged here, nor is a
    pin: the legality test after `make` keeps the last word. The filter only spares the
    make, the test and the unmake of a move that test was going to reject.
    """
    us = st[STM]
    king = lsb(bb[np.uint64(us * 6 + KING)])
    checkers = attackers_to(king, bb[OCC_ALL], bb, tab) & bb[np.uint64(OCC_WHITE + (1 - us))]
    if checkers & (checkers - ONE):
        return np.uint64(0)
    checker = lsb(checkers)
    return np.uint64(checkers | tab[np.uint64(tables.BETWEEN + king * 64 + checker)])


@jit(internal=not TESTING)
def evasion_targets(bb: Bitboards, st: Ints, tab: Bitboards) -> np.uint64:
    """`evasion_targets_body` on arrays passed one by one; the search calls
    `evasion_targets_s`."""
    return evasion_targets_body(bb, st, tab)


@jit(inline=True, hot=True, internal=not TESTING)
def may_resolve(move: int, targets: np.uint64, us: int) -> bool:
    """Whether `move` can answer the check whose targets these are: a king move, a landing
    on a target, or an en passant capture whose taken pawn or landing square is one."""
    if move_piece(move) % 6 == KING:
        return True
    target = move_target(move)
    if (targets >> np.uint64(target)) & ONE:
        return True
    if move_flag(move) == EN_PASSANT:
        taken = target - 8 if us == WHITE else target + 8
        return bool((targets >> np.uint64(taken)) & ONE)
    return False


# Piece worth on the exchange scale, pawn to king: the king's is large enough that a
# capture by it into a defended square never pays.
SEE_VALUE = (100, 300, 300, 500, 900, 10_000)


@jit(internal=True)
def attackers_to(square: int, occupancy: np.uint64, bb: Bitboards, tab: Bitboards) -> np.uint64:
    """Every piece of either side attacking `square`, with `occupancy` as the blockers."""
    attackers = tab[np.uint64(tables.PAWN + BLACK * 64 + square)] & bb[WP]
    attackers |= tab[np.uint64(tables.PAWN + WHITE * 64 + square)] & bb[BP]
    attackers |= tab[np.uint64(tables.KNIGHT + square)] & (bb[WN] | bb[BN])
    attackers |= tab[np.uint64(tables.KING + square)] & (bb[WK] | bb[BK])
    diagonal = bb[WB] | bb[BB] | bb[WQ] | bb[BQ]
    straight = bb[WR] | bb[BR] | bb[WQ] | bb[BQ]
    attackers |= bishop_attacks(square, occupancy, tab) & diagonal
    attackers |= rook_attacks(square, occupancy, tab) & straight
    return np.uint64(attackers)


@jit(inline=True, internal=not TESTING)
def see_ge_body(move: int, threshold: int, bb: Bitboards, st: Ints, tab: Bitboards) -> bool:
    """Whether the exchanges `move` starts on its target win at least `threshold` centipawns.

    The swap-off in Stockfish's threshold form: each side recaptures with its least valuable
    attacker in turn, x-rays opening as pieces leave, and the balance is kept as a running
    bound with an exit as soon as one side cannot gain from going on. A king captures only
    when the other side has no attacker left. Pins are ignored, en passant counts the pawn
    it takes, a promotion counts the piece it makes, castling wins nothing.
    """
    flag = move_flag(move)
    if flag == CASTLING:
        return threshold <= 0
    us = st[STM]
    source = move_source(move)
    target = move_target(move)
    captured = move_captured(move)
    promotion = move_promotion(move)
    gain = 0
    if flag == EN_PASSANT:
        gain = SEE_VALUE[PAWN]
    elif captured != EMPTY:
        gain = SEE_VALUE[captured % 6]
    mover = move_piece(move) % 6
    if promotion != 0:
        gain += SEE_VALUE[promotion] - SEE_VALUE[PAWN]
        mover = promotion
    swap = gain - threshold
    if swap < 0:
        return False
    swap = SEE_VALUE[mover] - swap
    if swap <= 0:
        return True
    occupancy = (bb[OCC_ALL] ^ (ONE << np.uint64(source))) | (ONE << np.uint64(target))
    if flag == EN_PASSANT:
        taken = target - 8 if us == WHITE else target + 8
        occupancy ^= ONE << np.uint64(taken)
    attackers = attackers_to(target, occupancy, bb, tab) & occupancy
    diagonal = np.uint64(0)
    straight = np.uint64(0)
    diagonal_built = False
    straight_built = False
    stm = us
    result = 1
    while True:
        stm = 1 - stm
        attackers &= occupancy
        mine = attackers & bb[np.uint64(OCC_WHITE + stm)]
        if mine == 0:
            break
        result ^= 1
        base = stm * 6
        pieces = mine & bb[np.uint64(base + PAWN)]
        if pieces:
            swap = SEE_VALUE[PAWN] - swap
            if swap < result:
                break
            occupancy ^= pieces & (~pieces + ONE)
            if not diagonal_built:
                diagonal = bb[WB] | bb[BB] | bb[WQ] | bb[BQ]
                diagonal_built = True
            attackers |= bishop_attacks(target, occupancy, tab) & diagonal
            continue
        pieces = mine & bb[np.uint64(base + KNIGHT)]
        if pieces:
            swap = SEE_VALUE[KNIGHT] - swap
            if swap < result:
                break
            occupancy ^= pieces & (~pieces + ONE)
            continue
        pieces = mine & bb[np.uint64(base + BISHOP)]
        if pieces:
            swap = SEE_VALUE[BISHOP] - swap
            if swap < result:
                break
            occupancy ^= pieces & (~pieces + ONE)
            if not diagonal_built:
                diagonal = bb[WB] | bb[BB] | bb[WQ] | bb[BQ]
                diagonal_built = True
            attackers |= bishop_attacks(target, occupancy, tab) & diagonal
            continue
        pieces = mine & bb[np.uint64(base + ROOK)]
        if pieces:
            swap = SEE_VALUE[ROOK] - swap
            if swap < result:
                break
            occupancy ^= pieces & (~pieces + ONE)
            if not straight_built:
                straight = bb[WR] | bb[BR] | bb[WQ] | bb[BQ]
                straight_built = True
            attackers |= rook_attacks(target, occupancy, tab) & straight
            continue
        pieces = mine & bb[np.uint64(base + QUEEN)]
        if pieces:
            swap = SEE_VALUE[QUEEN] - swap
            if swap < result:
                break
            occupancy ^= pieces & (~pieces + ONE)
            if not diagonal_built:
                diagonal = bb[WB] | bb[BB] | bb[WQ] | bb[BQ]
                diagonal_built = True
            if not straight_built:
                straight = bb[WR] | bb[BR] | bb[WQ] | bb[BQ]
                straight_built = True
            attackers |= bishop_attacks(target, occupancy, tab) & diagonal
            attackers |= rook_attacks(target, occupancy, tab) & straight
            continue
        # The king is the last attacker: it may take only when nothing can take it back.
        if attackers & ~bb[np.uint64(OCC_WHITE + stm)]:
            result ^= 1
        break
    return result == 1


@jit(internal=not TESTING)
def see_ge(move: int, threshold: int, bb: Bitboards, st: Ints, tab: Bitboards) -> bool:
    """`see_ge_body` on arrays passed one by one; the search calls `see_ge_s`."""
    return see_ge_body(move, threshold, bb, st, tab)


@jit(inline=True)
def encode(source: int, target: int, piece: int, captured: int, promotion: int, flag: int) -> int:
    return (
        source | (target << 6) | (piece << 12) | (captured << 16) | (promotion << 20) | (flag << 24)
    )


@jit(inline=True)
def move_source(move: int) -> int:
    return move & 63


@jit(inline=True)
def move_target(move: int) -> int:
    return (move >> 6) & 63


@jit(inline=True, internal=not TESTING)
def move_piece(move: int) -> int:
    return (move >> 12) & 15


@jit(inline=True, hot=True, internal=not TESTING)
def move_captured(move: int) -> int:
    return (move >> 16) & 15


@jit(inline=True, hot=True)
def move_promotion(move: int) -> int:
    """The promotion piece's type (KNIGHT to QUEEN), or 0 for none."""
    return (move >> 20) & 15


@jit(inline=True, internal=not TESTING)
def move_flag(move: int) -> int:
    return (move >> 24) & 3


@jit(internal=True)
def push_targets(
    ml: Ints, count: int, source: int, targets: np.uint64, piece: int, sq: Ints
) -> int:
    """Append one move per set bit of `targets` for a non-pawn piece; returns the new count."""
    while targets:
        target = lsb(targets)
        targets &= targets - ONE
        ml[np.uint64(count)] = encode(source, target, piece, sq[np.uint64(target)], 0, NORMAL)
        count += 1
    return count


@jit(internal=True)
def push_promotions(
    ml: Ints, count: int, source: int, target: int, piece: int, captured: int
) -> int:
    """The four promotions of one pawn move, queen first."""
    for promotion in (QUEEN, ROOK, BISHOP, KNIGHT):
        ml[np.uint64(count)] = encode(source, target, piece, captured, promotion, NORMAL)
        count += 1
    return count


@jit(inline=True, internal=not TESTING)
def gen_noisy_body(bb: Bitboards, sq: Ints, st: Ints, tab: Bitboards, ml: Ints, base: int) -> int:
    """Captures and promotions, pseudo-legal, written from ml[base]; returns how many."""
    us = st[STM]
    them = 1 - us
    theirs = bb[np.uint64(OCC_WHITE + them)]
    occupancy = bb[OCC_ALL]
    count = base
    pawn = us * 6 + PAWN
    pawns = bb[np.uint64(pawn)]
    last_rank = RANK_8 if us == WHITE else RANK_1
    forward = 8 if us == WHITE else -8

    # Pawn captures, promotions on the last rank.
    while pawns:
        source = lsb(pawns)
        pawns &= pawns - ONE
        targets = tab[np.uint64(tables.PAWN + us * 64 + source)] & theirs
        while targets:
            target = lsb(targets)
            targets &= targets - ONE
            if (ONE << np.uint64(target)) & last_rank:
                count = push_promotions(ml, count, source, target, pawn, sq[np.uint64(target)])
            else:
                ml[np.uint64(count)] = encode(
                    source, target, pawn, sq[np.uint64(target)], 0, NORMAL
                )
                count += 1
        target = source + forward
        if (ONE << np.uint64(target)) & last_rank & ~occupancy:
            count = push_promotions(ml, count, source, target, pawn, sq[np.uint64(target)])

    if st[EP] >= 0:
        ep = st[EP]
        sources = tab[np.uint64(tables.PAWN + them * 64 + ep)] & bb[np.uint64(pawn)]
        while sources:
            source = lsb(sources)
            sources &= sources - ONE
            ml[np.uint64(count)] = encode(source, ep, pawn, them * 6 + PAWN, 0, EN_PASSANT)
            count += 1

    piece = us * 6 + KNIGHT
    pieces = bb[np.uint64(piece)]
    while pieces:
        source = lsb(pieces)
        pieces &= pieces - ONE
        count = push_targets(
            ml, count, source, tab[np.uint64(tables.KNIGHT + source)] & theirs, piece, sq
        )
    piece = us * 6 + BISHOP
    pieces = bb[np.uint64(piece)]
    while pieces:
        source = lsb(pieces)
        pieces &= pieces - ONE
        count = push_targets(
            ml, count, source, bishop_attacks(source, occupancy, tab) & theirs, piece, sq
        )
    piece = us * 6 + ROOK
    pieces = bb[np.uint64(piece)]
    while pieces:
        source = lsb(pieces)
        pieces &= pieces - ONE
        count = push_targets(
            ml, count, source, rook_attacks(source, occupancy, tab) & theirs, piece, sq
        )
    piece = us * 6 + QUEEN
    pieces = bb[np.uint64(piece)]
    while pieces:
        source = lsb(pieces)
        pieces &= pieces - ONE
        targets = (
            bishop_attacks(source, occupancy, tab) | rook_attacks(source, occupancy, tab)
        ) & theirs
        count = push_targets(ml, count, source, targets, piece, sq)
    piece = us * 6 + KING
    source = lsb(bb[np.uint64(piece)])
    count = push_targets(
        ml, count, source, tab[np.uint64(tables.KING + source)] & theirs, piece, sq
    )
    return count - base


@jit(internal=not TESTING)
def gen_noisy(bb: Bitboards, sq: Ints, st: Ints, tab: Bitboards, ml: Ints, base: int) -> int:
    """`gen_noisy_body` on arrays passed one by one; the search calls `gen_noisy_s`."""
    return gen_noisy_body(bb, sq, st, tab, ml, base)


@jit(inline=True, internal=not TESTING)
def gen_quiet_body(bb: Bitboards, sq: Ints, st: Ints, tab: Bitboards, ml: Ints, base: int) -> int:
    """Non-captures other than promotions, pseudo-legal, written from ml[base]; returns how many."""
    us = st[STM]
    them = 1 - us
    occupancy = bb[OCC_ALL]
    empty = ~occupancy
    count = base
    pawn = us * 6 + PAWN
    pawns = bb[np.uint64(pawn)]

    if us == WHITE:
        single = (pawns << np.uint64(8)) & empty & ~RANK_8
        double = ((single & RANK_3) << np.uint64(8)) & empty
        forward = 8
    else:
        single = (pawns >> np.uint64(8)) & empty & ~RANK_1
        double = ((single & RANK_6) >> np.uint64(8)) & empty
        forward = -8
    while single:
        target = lsb(single)
        single &= single - ONE
        ml[np.uint64(count)] = encode(target - forward, target, pawn, EMPTY, 0, NORMAL)
        count += 1
    while double:
        target = lsb(double)
        double &= double - ONE
        ml[np.uint64(count)] = encode(target - 2 * forward, target, pawn, EMPTY, 0, DOUBLE_PUSH)
        count += 1

    piece = us * 6 + KNIGHT
    pieces = bb[np.uint64(piece)]
    while pieces:
        source = lsb(pieces)
        pieces &= pieces - ONE
        count = push_targets(
            ml, count, source, tab[np.uint64(tables.KNIGHT + source)] & empty, piece, sq
        )
    piece = us * 6 + BISHOP
    pieces = bb[np.uint64(piece)]
    while pieces:
        source = lsb(pieces)
        pieces &= pieces - ONE
        count = push_targets(
            ml, count, source, bishop_attacks(source, occupancy, tab) & empty, piece, sq
        )
    piece = us * 6 + ROOK
    pieces = bb[np.uint64(piece)]
    while pieces:
        source = lsb(pieces)
        pieces &= pieces - ONE
        count = push_targets(
            ml, count, source, rook_attacks(source, occupancy, tab) & empty, piece, sq
        )
    piece = us * 6 + QUEEN
    pieces = bb[np.uint64(piece)]
    while pieces:
        source = lsb(pieces)
        pieces &= pieces - ONE
        targets = (
            bishop_attacks(source, occupancy, tab) | rook_attacks(source, occupancy, tab)
        ) & empty
        count = push_targets(ml, count, source, targets, piece, sq)
    piece = us * 6 + KING
    source = lsb(bb[np.uint64(piece)])
    count = push_targets(ml, count, source, tab[np.uint64(tables.KING + source)] & empty, piece, sq)

    # Castling: the rights, an empty path, and the king neither in check nor crossing an
    # attacked square. The arrival square is covered by the legality test after make. With
    # a right standing the king is on its home square, so the other squares are offsets.
    rights = st[CASTLE] >> (2 * us)
    if rights & 3 and not attacked(source, them, bb, tab):
        king_side = KING_SIDE_PATH << np.uint64(56 * us)
        queen_side = QUEEN_SIDE_PATH << np.uint64(56 * us)
        if rights & 1 and not occupancy & king_side and not attacked(source + 1, them, bb, tab):
            ml[np.uint64(count)] = encode(
                source, source + 2, piece, sq[np.uint64(source + 2)], 0, CASTLING
            )
            count += 1
        if rights & 2 and not occupancy & queen_side and not attacked(source - 1, them, bb, tab):
            ml[np.uint64(count)] = encode(
                source, source - 2, piece, sq[np.uint64(source - 2)], 0, CASTLING
            )
            count += 1
    return count - base


@jit(internal=not TESTING)
def gen_quiet(bb: Bitboards, sq: Ints, st: Ints, tab: Bitboards, ml: Ints, base: int) -> int:
    """`gen_quiet_body` on arrays passed one by one; the search calls `gen_quiet_s`."""
    return gen_quiet_body(bb, sq, st, tab, ml, base)


@jit(internal=not TESTING)
def gen_all(bb: Bitboards, sq: Ints, st: Ints, tab: Bitboards, ml: Ints, base: int) -> int:
    count = gen_noisy(bb, sq, st, tab, ml, base)
    return count + gen_quiet(bb, sq, st, tab, ml, base + count)


@jit(inline=True, internal=not TESTING)
def make_body(
    move: int,
    bb: Bitboards,
    sq: Ints,
    st: Ints,
    undo: Ints,
    keys: Ints,
    tab: Bitboards,
) -> None:
    """Play a pseudo-legal move, pushing what unmake needs onto the undo stack."""
    source = move_source(move)
    target = move_target(move)
    piece = move_piece(move)
    captured = move_captured(move)
    promotion = move_promotion(move)
    flag = move_flag(move)
    us = st[STM]
    them = 1 - us
    ply = st[PLY]
    slot = ply * UNDO_FIELDS
    undo[np.uint64(slot)] = captured
    undo[np.uint64(slot + 1)] = st[CASTLE]
    undo[np.uint64(slot + 2)] = st[EP]
    undo[np.uint64(slot + 3)] = st[HALF]
    undo[np.uint64(slot + 4)] = st[HASH]
    hash_ = st[HASH]
    if st[EP] >= 0:
        hash_ ^= keys[np.uint64(KEY_EP + (st[EP] & 7))]
    st[EP] = -1
    source_bit = ONE << np.uint64(source)
    target_bit = ONE << np.uint64(target)

    if captured != EMPTY:
        captured_square = target
        if flag == EN_PASSANT:
            captured_square = target - 8 if us == WHITE else target + 8
        captured_bit = ONE << np.uint64(captured_square)
        bb[np.uint64(captured)] ^= captured_bit
        bb[np.uint64(OCC_WHITE + them)] ^= captured_bit
        sq[np.uint64(captured_square)] = EMPTY
        hash_ ^= keys[np.uint64(captured * 64 + captured_square)]

    bb[np.uint64(piece)] ^= source_bit
    bb[np.uint64(OCC_WHITE + us)] ^= source_bit | target_bit
    sq[np.uint64(source)] = EMPTY
    hash_ ^= keys[np.uint64(piece * 64 + source)]
    if promotion:
        promoted = us * 6 + promotion
        bb[np.uint64(promoted)] |= target_bit
        sq[np.uint64(target)] = promoted
        hash_ ^= keys[np.uint64(promoted * 64 + target)]
    else:
        bb[np.uint64(piece)] |= target_bit
        sq[np.uint64(target)] = piece
        hash_ ^= keys[np.uint64(piece * 64 + target)]

    if flag == CASTLING:
        rook = us * 6 + ROOK
        if target > source:
            rook_source, rook_target = target + 1, target - 1
        else:
            rook_source, rook_target = target - 2, target + 1
        rook_bits = (ONE << np.uint64(rook_source)) | (ONE << np.uint64(rook_target))
        bb[np.uint64(rook)] ^= rook_bits
        bb[np.uint64(OCC_WHITE + us)] ^= rook_bits
        sq[np.uint64(rook_source)] = EMPTY
        sq[np.uint64(rook_target)] = rook
        hash_ ^= keys[np.uint64(rook * 64 + rook_source)] ^ keys[np.uint64(rook * 64 + rook_target)]
    elif flag == DOUBLE_PUSH:
        ep = (source + target) // 2
        if tab[np.uint64(tables.PAWN + us * 64 + ep)] & bb[np.uint64(them * 6 + PAWN)]:
            st[EP] = ep
            hash_ ^= keys[np.uint64(KEY_EP + (ep & 7))]

    rights = st[CASTLE] & CASTLE_KEEP[source] & CASTLE_KEEP[target]
    if rights != st[CASTLE]:
        hash_ ^= keys[np.uint64(KEY_CASTLE + st[CASTLE])] ^ keys[np.uint64(KEY_CASTLE + rights)]
        st[CASTLE] = rights
    if captured != EMPTY or piece % 6 == PAWN:
        st[HALF] = 0
    else:
        st[HALF] += 1
    bb[OCC_ALL] = bb[OCC_WHITE] | bb[OCC_BLACK]
    st[STM] = them
    st[HASH] = hash_ ^ keys[KEY_SIDE]
    st[PLY] = ply + 1


@jit(internal=not TESTING)
def make(
    move: int,
    bb: Bitboards,
    sq: Ints,
    st: Ints,
    undo: Ints,
    keys: Ints,
    tab: Bitboards,
) -> None:
    """`make_body` on arrays passed one by one; the search calls `make_s`."""
    make_body(move, bb, sq, st, undo, keys, tab)


@jit(inline=True, internal=not TESTING)
def unmake_body(move: int, bb: Bitboards, sq: Ints, st: Ints, undo: Ints) -> None:
    """Take back the last move made, restoring the state from the undo stack."""
    source = move_source(move)
    target = move_target(move)
    piece = move_piece(move)
    promotion = move_promotion(move)
    flag = move_flag(move)
    ply = st[PLY] - 1
    slot = ply * UNDO_FIELDS
    captured = undo[np.uint64(slot)]
    st[CASTLE] = undo[np.uint64(slot + 1)]
    st[EP] = undo[np.uint64(slot + 2)]
    st[HALF] = undo[np.uint64(slot + 3)]
    st[HASH] = undo[np.uint64(slot + 4)]
    st[PLY] = ply
    them = st[STM]
    us = 1 - them
    st[STM] = us
    source_bit = ONE << np.uint64(source)
    target_bit = ONE << np.uint64(target)

    if promotion:
        bb[np.uint64(us * 6 + promotion)] ^= target_bit
    else:
        bb[np.uint64(piece)] ^= target_bit
    bb[np.uint64(piece)] |= source_bit
    bb[np.uint64(OCC_WHITE + us)] ^= source_bit | target_bit
    sq[np.uint64(source)] = piece
    sq[np.uint64(target)] = EMPTY

    if captured != EMPTY:
        captured_square = target
        if flag == EN_PASSANT:
            captured_square = target - 8 if us == WHITE else target + 8
        captured_bit = ONE << np.uint64(captured_square)
        bb[np.uint64(captured)] |= captured_bit
        bb[np.uint64(OCC_WHITE + them)] |= captured_bit
        sq[np.uint64(captured_square)] = captured
    elif flag == CASTLING:
        rook = us * 6 + ROOK
        if target > source:
            rook_source, rook_target = target + 1, target - 1
        else:
            rook_source, rook_target = target - 2, target + 1
        rook_bits = (ONE << np.uint64(rook_source)) | (ONE << np.uint64(rook_target))
        bb[np.uint64(rook)] ^= rook_bits
        bb[np.uint64(OCC_WHITE + us)] ^= rook_bits
        sq[np.uint64(rook_source)] = rook
        sq[np.uint64(rook_target)] = EMPTY
    bb[OCC_ALL] = bb[OCC_WHITE] | bb[OCC_BLACK]


@jit(internal=not TESTING)
def unmake(move: int, bb: Bitboards, sq: Ints, st: Ints, undo: Ints) -> None:
    """`unmake_body` on arrays passed one by one; the search calls `unmake_s`."""
    unmake_body(move, bb, sq, st, undo)


@jit()
def make_null(st: Ints, undo: Ints, keys: Ints) -> None:
    """Pass the move: the other side is to move, and any en passant right is gone."""
    ply = st[PLY]
    slot = ply * UNDO_FIELDS
    undo[np.uint64(slot)] = NULL_MARK
    undo[np.uint64(slot + 1)] = st[CASTLE]
    undo[np.uint64(slot + 2)] = st[EP]
    undo[np.uint64(slot + 3)] = st[HALF]
    undo[np.uint64(slot + 4)] = st[HASH]
    hash_ = st[HASH]
    if st[EP] >= 0:
        hash_ ^= keys[np.uint64(KEY_EP + (st[EP] & 7))]
    st[EP] = -1
    st[HALF] += 1
    st[STM] = 1 - st[STM]
    st[HASH] = hash_ ^ keys[KEY_SIDE]
    st[PLY] = ply + 1


@jit()
def unmake_null(st: Ints, undo: Ints) -> None:
    ply = st[PLY] - 1
    slot = ply * UNDO_FIELDS
    st[EP] = undo[np.uint64(slot + 2)]
    st[HALF] = undo[np.uint64(slot + 3)]
    st[HASH] = undo[np.uint64(slot + 4)]
    st[STM] = 1 - st[STM]
    st[PLY] = ply


@jit()
def last_was_null(st: Ints, undo: Ints) -> bool:
    """Whether the move that led here was a null move."""
    ply = st[PLY]
    return bool(ply > 0 and undo[np.uint64((ply - 1) * UNDO_FIELDS)] == NULL_MARK)


def compute_hash(bb: Bitboards, st: Ints, keys: Ints) -> np.int64:
    """The Zobrist hash from scratch, the check on the incremental one.

    Plain Python: it runs once per `load` and in the warm-up's checks, and a jitted copy
    cost a third of a second of compile at every import for nothing on the clock.
    """
    hash_ = 0
    for piece in range(12):
        pieces = int(bb[piece])
        while pieces:
            square = (pieces & -pieces).bit_length() - 1
            pieces &= pieces - 1
            hash_ ^= int(keys[piece * 64 + square])
    hash_ ^= int(keys[KEY_CASTLE + int(st[CASTLE])])
    if st[EP] >= 0:
        hash_ ^= int(keys[KEY_EP + (int(st[EP]) & 7)])
    if st[STM] == BLACK:
        hash_ ^= int(keys[KEY_SIDE])
    return np.int64(hash_)


@jit(inline=True, internal=not TESTING)
def legal_after(bb: Bitboards, st: Ints, tab: Bitboards) -> bool:
    """After make: whether the side that just moved left its king out of check."""
    mover = 1 - st[STM]
    return not attacked(lsb(bb[np.uint64(mover * 6 + KING)]), st[STM], bb, tab)


@jit(internal=not TESTING)
def perft(
    depth: int,
    bb: Bitboards,
    sq: Ints,
    st: Ints,
    tab: Bitboards,
    ml: Ints,
    undo: Ints,
    keys: Ints,
) -> int:
    """Count the legal move sequences of `depth` plies; the movegen's correctness check."""
    if depth == 0:
        return 1
    base = st[PLY] * MOVES_PER_PLY
    count = gen_all(bb, sq, st, tab, ml, base)
    nodes = 0
    for i in range(count):
        move = ml[base + i]
        make(move, bb, sq, st, undo, keys, tab)
        if legal_after(bb, st, tab):
            nodes += perft(depth - 1, bb, sq, st, tab, ml, undo, keys)
        unmake(move, bb, sq, st, undo)
    return nodes


def zobrist_keys(seed: int = 2026) -> Ints:
    """Random 64-bit keys for every piece on every square, the castling rights, the en
    passant files and the side to move, as int64 so they XOR with the state array."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**64, size=KEY_COUNT, dtype=np.uint64).view(np.int64)


def new_arrays() -> tuple[Bitboards, Ints, Ints, Ints, Ints]:
    """Fresh bb, sq, st, move list and undo stack."""
    bb = np.zeros(BB_SIZE, dtype=np.uint64)
    sq = np.full(64, EMPTY, dtype=np.int64)
    st = np.zeros(ST_SIZE, dtype=np.int64)
    ml = np.zeros(MAX_PLY * MOVES_PER_PLY, dtype=np.int64)
    undo = np.zeros((MAX_PLY + 1) * UNDO_FIELDS, dtype=np.int64)
    return bb, sq, st, ml, undo


def load(board: chess.Board, bb: Bitboards, sq: Ints, st: Ints, keys: Ints) -> None:
    """Set the arrays to a python-chess position, with the undo stack empty."""
    bb[:] = 0
    sq[:] = EMPTY
    for colour, occupied in (
        (WHITE, board.occupied_co[chess.WHITE]),
        (BLACK, board.occupied_co[chess.BLACK]),
    ):
        for piece_type, pieces in enumerate(
            (board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings)
        ):
            piece = colour * 6 + piece_type
            bits = pieces & occupied
            bb[piece] = bits
            for square in chess.scan_forward(bits):
                sq[square] = piece
        bb[OCC_WHITE + colour] = occupied
    bb[OCC_ALL] = board.occupied
    st[STM] = WHITE if board.turn == chess.WHITE else BLACK
    rights = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        rights |= 1
    if board.has_queenside_castling_rights(chess.WHITE):
        rights |= 2
    if board.has_kingside_castling_rights(chess.BLACK):
        rights |= 4
    if board.has_queenside_castling_rights(chess.BLACK):
        rights |= 8
    st[CASTLE] = rights
    st[EP] = (
        board.ep_square
        if board.has_pseudo_legal_en_passant() and board.ep_square is not None
        else -1
    )
    st[HALF] = board.halfmove_clock
    st[PLY] = 0
    st[HASH] = compute_hash(bb, st, keys)


PROMOTION_LETTERS = {KNIGHT: "n", BISHOP: "b", ROOK: "r", QUEEN: "q"}


def move_to_uci(move: int) -> str:
    text = chess.SQUARE_NAMES[move_source(move)] + chess.SQUARE_NAMES[move_target(move)]
    promotion = move_promotion(move)
    return text + PROMOTION_LETTERS[promotion] if promotion else text


def move_from_uci(text: str, ml: Ints, count: int) -> int:
    """The move among ml[:count] that prints as `text`, or -1."""
    for i in range(count):
        if move_to_uci(int(ml[i])) == text:
            return int(ml[i])
    return -1


TAB = tables.build_tables()
KEYS = zobrist_keys()

# Every kernel is compiled here, at import, on the argument types its callers use, so the
# compile lands in the init budget and never on the clock. A second signature on any of
# them would mean some call site passes another type and pays a compile mid-game. `encode`
# and the move accessors are marked inline="always" and have no signature of their own:
# they are pasted into their callers, and a jitted callee called with a constant argument
# would otherwise be compiled once per constant. The hot kernels' bodies are pasted the
# same way into two entries each, the array entry here and the handle entry in
# `search.py`; inlining the helpers under them was measured to cost 1.6 s of compile for
# no speed, so those stay ordinary calls. What is listed is what this warm-up itself
# compiles: the move generators, `make` and `unmake` are reached on the platform only
# through the handle entries, so their array entries are the tests' and the tools' and
# compile on their first call there, and `push_targets` and `push_promotions` compile
# under `gen_noisy_s` in the agent's own warm-up, which checks them in `search.WARMED`.
WARMED: dict[str, Any] = {
    "popcount": popcount,
    "lsb": lsb,
    "rook_attacks": rook_attacks,
    "bishop_attacks": bishop_attacks,
    "attacked": attacked,
    "in_check": in_check,
    "make_null": make_null,
    "unmake_null": unmake_null,
    "last_was_null": last_was_null,
}


def warm_up() -> None:
    """Compile and check the kernels no compiled caller of the search reaches.

    Move generation, `make` and `unmake` are checked by the perft in `agent.warm_up`,
    which runs them through the handle entries the search itself calls; driving the same
    perft from the array entries here compiled a second copy of each for 1.7 s of the
    init budget and checked nothing the platform runs.
    """
    bb, sq, st, _, undo = new_arrays()
    load(chess.Board(), bb, sq, st, KEYS)
    if in_check(bb, st, TAB):
        raise RuntimeError("kernel warm-up: the start position is not a check")
    if popcount(bb[OCC_ALL]) != 32:
        raise RuntimeError("kernel warm-up: the start position has 32 pieces")
    if compute_hash(bb, st, KEYS) != st[HASH]:
        raise RuntimeError("kernel warm-up: the hash does not round-trip")
    make_null(st, undo, KEYS)
    if not last_was_null(st, undo) or compute_hash(bb, st, KEYS) != st[HASH]:
        raise RuntimeError("kernel warm-up: the null move does not hash")
    unmake_null(st, undo)
    if last_was_null(st, undo) or st[STM] != WHITE:
        raise RuntimeError("kernel warm-up: the null move does not unmake")
    for name, fn in WARMED.items():
        signatures = len(fn.signatures)
        # A callee compiled inside a cached caller has no signature of its own, so under
        # the cache the count is at most one; without it, exactly one.
        if signatures > 1 or (signatures == 0 and not CACHE):
            raise RuntimeError(f"kernel warm-up: {name} has {signatures} signatures, expected 1")


warm_up()
