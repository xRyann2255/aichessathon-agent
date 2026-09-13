"""The attack tables the kernel reads, built at import into one flat uint64 array.

Squares are numbered as python-chess numbers them: a1 is 0, h1 is 7, a8 is 56, h8 is 63.
Knight, king and pawn attacks are one lookup per square. Rook and bishop attacks use
magic bitboards: the blockers on a slider's rays, masked and multiplied by a per-square
magic number, shift down to an index into a table of attack sets, so a sliding attack is
one multiply, one shift and one load. Beside them are each slider's rays from a square on
an empty board, which contain every square that square's magic lookup can return, so one
load proves a lookup empty before it is made. The magic numbers below came out of
`tools/find_magics.py`, our own search, and the tables are filled here from scratch.

Everything lives in one array so a kernel takes one argument for all of it, at the offsets
this module names. The array is read by the kernels and never written after import, and it
is passed as an argument rather than read as a global, because numba freezes a global
array of under a million bytes into the compiled code silently.
"""

from __future__ import annotations

import os

import numpy as np
from llvmlite import binding as llvm  # type: ignore[import-untyped]
from numba.core import config
from numba.core.codegen import get_host_cpu_features

# Offsets into the table array, in uint64 entries.
KNIGHT = 0  # 64: squares a knight on s attacks
KING = 64  # 64: squares a king on s attacks
PAWN = 128  # 2 x 64: squares a pawn of colour c (0 white, 1 black) on s attacks
# The two ray tables hold the whole ray, edge squares included, where ROOK_MASK stops
# short of the edge; they sit here beside the leaper tables, which the attack test reads
# in the same breath, rather than out by the slider tables 800 KB away.
ROOK_RAY = 256  # 64: squares a rook on s attacks with the board otherwise empty
BISHOP_RAY = 320  # 64: the same for a bishop
ROOK_MASK = 384  # 64: the rook's rays from s without the board edge
ROOK_MAGIC = 448  # 64
ROOK_SHIFT = 512  # 64: 64 minus the bits in the mask
ROOK_OFFSET = 576  # 64: where the square's attack sets start in ROOK_TABLE
BISHOP_MASK = 640
BISHOP_MAGIC = 704
BISHOP_SHIFT = 768
BISHOP_OFFSET = 832
BETWEEN = 896  # 64 x 64: the squares strictly between two squares on a line, else 0
ROOK_TABLE = 4992  # 102400 attack sets
BISHOP_TABLE = 107392  # 5248 attack sets
TABLE_SIZE = 112640

ROOK_TABLE_SIZE = 102400
BISHOP_TABLE_SIZE = 5248

# From tools/find_magics.py, seed 2026.
ROOK_MAGICS = (
    0x0480002480144000,
    0x0040200040001001,
    0x218010018020008A,
    0x0480080004100080,
    0x8A00020008112004,
    0x2200100401020008,
    0x0400208128100204,
    0x408000228000C900,
    0x0800800384664000,
    0x0445004000810026,
    0x2002004020108200,
    0x010A000840120220,
    0x0002800800810400,
    0x0010800200040080,
    0x5004801100020080,
    0x0401000100004082,
    0x0280004040002000,
    0x0800434010046000,
    0x0010008010200081,
    0x0080828008005001,
    0x1402020008200410,
    0x0000808004000200,
    0x4004240090010228,
    0x0008020007288044,
    0x0400802080004000,
    0x0080200040005004,
    0x0220001100450021,
    0x0020100100090020,
    0x008800808004000A,
    0x0106000404001020,
    0x0002000200080104,
    0x00000082000D4411,
    0x0020904000800024,
    0x0160100020C00148,
    0x0010081080802000,
    0x0000401202000820,
    0x2180080080800400,
    0x0002000402001009,
    0x8000011004008208,
    0xA4002121860000CC,
    0xC820804000308000,
    0xA620003000414004,
    0x8018100020008080,
    0x0022100100090020,
    0x1203008488010010,
    0x0002001020040400,
    0x0400080182040030,
    0x0410008041020024,
    0x0862044102802600,
    0x8028842010400080,
    0x0801002000144100,
    0x8008008030006A80,
    0x0248010410090100,
    0x0800800200040080,
    0x0000021001080400,
    0x2040041045208200,
    0x0002420010822102,
    0x2023241182010042,
    0x040C0A0012804022,
    0x0800040900201001,
    0x9142002008041002,
    0x0201000400020801,
    0x0000121000914804,
    0x0000010080240042,
)
BISHOP_MAGICS = (
    0x2002100A51840280,
    0xA850300A0CA02000,
    0x10080800408A0200,
    0x0008085100042204,
    0x2104030800080401,
    0x110201500A040800,
    0x000C110108A0000A,
    0x421904008A017020,
    0x808018085044C200,
    0x0008090D03040504,
    0x0102080809002189,
    0x1000090411010000,
    0x28C0042420000440,
    0x0208120802091004,
    0x841C004804042010,
    0x0010104414010890,
    0x209120848408080A,
    0x0004028204082A00,
    0x5404020204040008,
    0x000895280200C000,
    0x0881001820080000,
    0x8409018020A01000,
    0x01C5002448088400,
    0x0000210201040244,
    0x8210112140046101,
    0x0009104848122800,
    0x000802400C040282,
    0x040C040008009090,
    0x0021010000104008,
    0x2010210000804100,
    0x0044990084050820,
    0x0104008280320104,
    0x08042004A0091000,
    0x0880829040081000,
    0x0020223000880081,
    0x8000600800058820,
    0x04181008208C0020,
    0xC00A280200010884,
    0x0024080080084452,
    0x4202020208004044,
    0x000C422004A99010,
    0x0182129008C21420,
    0x0080084410028200,
    0x000002A124080801,
    0x0180084104000040,
    0x4222202040801301,
    0x0103440800810208,
    0xC010110040890100,
    0x004084300804000C,
    0x0004220802480000,
    0x010020D200B00008,
    0x4040004420A80000,
    0x0090004051450000,
    0x008020030222004A,
    0x5440480144009081,
    0x0009480814802002,
    0x92410022012028CC,
    0x08C0084210900800,
    0x2CEC420320A41002,
    0x0004001001208840,
    0x8100000040450101,
    0x109144601441C200,
    0x8214202042408100,
    0x02400186008D0102,
)


def square_bit(square: int) -> int:
    return 1 << square


def ray(square: int, step_rank: int, step_file: int, occupancy: int, stop_at_edge: bool) -> int:
    """Squares along one direction from `square`, stopping at a blocker.

    With `stop_at_edge` the last square before the board edge is left out, which is the
    shape of a magic mask: a blocker on the edge changes nothing about the attack set.
    """
    result = 0
    rank, file = divmod(square, 8)
    while True:
        rank += step_rank
        file += step_file
        if not (0 <= rank < 8 and 0 <= file < 8):
            break
        if stop_at_edge:
            next_rank, next_file = rank + step_rank, file + step_file
            if not (0 <= next_rank < 8 and 0 <= next_file < 8):
                break
        bit = square_bit(rank * 8 + file)
        result |= bit
        if occupancy & bit:
            break
    return result


ROOK_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))
BISHOP_STEPS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def slider_attacks(square: int, occupancy: int, rook: bool) -> int:
    """Attack set of a rook or bishop on `square` with the given blockers, by walking rays."""
    steps = ROOK_STEPS if rook else BISHOP_STEPS
    result = 0
    for step_rank, step_file in steps:
        result |= ray(square, step_rank, step_file, occupancy, False)
    return result


def slider_mask(square: int, rook: bool) -> int:
    """The relevant occupancy of a slider on `square`: its rays without the edge squares."""
    steps = ROOK_STEPS if rook else BISHOP_STEPS
    result = 0
    for step_rank, step_file in steps:
        result |= ray(square, step_rank, step_file, 0, True)
    return result


# The two ways from a slider's blockers to its slot in the attack table. The magic way
# multiplies the masked blockers by the square's magic and shifts; the PEXT way packs the
# blockers' bits under the mask to the low end with one BMI2 instruction (`pext`, the
# parallel bit extract, a published alternative to magics: the Chess Programming Wiki's
# "BMI2 - PEXT Bitboards"). Both index a table of the same shape, filled here from scratch
# by `fill_magic`; which one the kernels compile is decided once at import from the host.
# `pext` is one instruction at 3 cycles on Zen 4 (the contract's EPYC 9V74) and on the
# Intel cores below, and microcoded at about 19 cycles on Zen 3 and slower before it (the
# ladder ran on both znver3 and znver4 hosts). Only a core named here compiles the PEXT
# path; any other name, a generic one, or no BMI2 keeps the magic path, since the loss
# from `pext` on a core that microcodes it is many times the gain on one that does not.
# The names are LLVM's, matched as prefixes so "skylake-avx512" and "icelake-server"
# follow their families.
FAST_PEXT_CORES = (
    "haswell",
    "broadwell",
    "skylake",
    "cannonlake",
    "cascadelake",
    "cooperlake",
    "icelake",
    "tigerlake",
    "rocketlake",
    "alderlake",
    "raptorlake",
    "meteorlake",
    "arrowlake",
    "lunarlake",
    "pantherlake",
    "sapphirerapids",
    "emeraldrapids",
    "graniterapids",
    "diamondrapids",
    "znver4",
    "znver5",
)


def fast_pext_core(name: str, features: str) -> bool:
    """Whether a core LLVM calls `name` with the feature string `features` (numba's form,
    `+bmi2,-avx512f,...`) runs `pext` as one fast instruction: BMI2 present and the name
    in `FAST_PEXT_CORES`. Anything else, an unlisted name included, answers False."""
    if "+bmi2" not in features.split(","):
        return False
    return any(name.startswith(fast) for fast in FAST_PEXT_CORES)


def fast_pext_host() -> bool:
    """`fast_pext_core` on the host as numba sees it (`NUMBA_CPU_NAME` and
    `NUMBA_CPU_FEATURES` first, then LLVM's own reading), so the intrinsic is only chosen
    where LLVM can select it. `AICHESSATHON_PEXT=0` or `1` forces the answer, for the
    equivalence gate's arms and the tests."""
    forced = os.environ.get("AICHESSATHON_PEXT")
    if forced in ("0", "1"):
        return forced == "1"
    features: str | None = config.CPU_FEATURES  # type: ignore[attr-defined]
    if features is None:
        features = get_host_cpu_features()  # type: ignore[no-untyped-call]
    name: str = config.CPU_NAME or llvm.get_host_cpu_name()  # type: ignore[attr-defined]
    return fast_pext_core(str(name), str(features))


USE_PEXT = fast_pext_host()


def pext(value: int, mask: int) -> int:
    """The bits of `value` under `mask`, packed to the low end in ascending order: the
    reference for the intrinsic and for the fill's index."""
    result = 0
    position = 0
    while mask:
        low = mask & -mask
        if value & low:
            result |= 1 << position
        position += 1
        mask ^= low
    return result


def lookup(table: np.ndarray, square: int, occupancy: int, rook: bool, pext_index: bool) -> int:
    """The kernel's slider lookup in Python, on a table filled the same way, for the tests."""
    mask_at, magic_at, shift_at, offset_at, table_at = (
        (ROOK_MASK, ROOK_MAGIC, ROOK_SHIFT, ROOK_OFFSET, ROOK_TABLE)
        if rook
        else (BISHOP_MASK, BISHOP_MAGIC, BISHOP_SHIFT, BISHOP_OFFSET, BISHOP_TABLE)
    )
    mask = int(table[mask_at + square])
    if pext_index:
        index = pext(occupancy, mask)
    else:
        blockers = occupancy & mask
        index = ((blockers * int(table[magic_at + square])) & 0xFFFFFFFFFFFFFFFF) >> int(
            table[shift_at + square]
        )
    return int(table[table_at + int(table[offset_at + square]) + index])


def mask_subsets(mask: int) -> np.ndarray:
    """Every subset of the set bits of `mask`, as a uint64 array of 2 to the popcount."""
    bits = [i for i in range(64) if mask >> i & 1]
    count = 1 << len(bits)
    index = np.arange(count, dtype=np.uint64)
    subsets = np.zeros(count, dtype=np.uint64)
    for position, bit in enumerate(bits):
        subsets |= ((index >> np.uint64(position)) & np.uint64(1)) << np.uint64(bit)
    return subsets


def leaper_attacks(square: int, steps: tuple[tuple[int, int], ...]) -> int:
    rank, file = divmod(square, 8)
    result = 0
    for step_rank, step_file in steps:
        to_rank, to_file = rank + step_rank, file + step_file
        if 0 <= to_rank < 8 and 0 <= to_file < 8:
            result |= square_bit(to_rank * 8 + to_file)
    return result


KNIGHT_STEPS = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
KING_STEPS = ROOK_STEPS + BISHOP_STEPS
WHITE_PAWN_STEPS = ((1, 1), (1, -1))
BLACK_PAWN_STEPS = ((-1, 1), (-1, -1))

ROOK_MASKS = tuple(slider_mask(square, True) for square in range(64))
BISHOP_MASKS = tuple(slider_mask(square, False) for square in range(64))


def between(a: int, b: int) -> int:
    """Squares strictly between a and b when they share a rank, file or diagonal, else 0."""
    rank_a, file_a = divmod(a, 8)
    rank_b, file_b = divmod(b, 8)
    aligned = rank_a == rank_b or file_a == file_b or abs(rank_a - rank_b) == abs(file_a - file_b)
    if a == b or not aligned:
        return 0
    step_rank = (rank_b > rank_a) - (rank_b < rank_a)
    step_file = (file_b > file_a) - (file_b < file_a)
    return ray(a, step_rank, step_file, square_bit(b), False) & ~square_bit(b)


def fill_magic(
    table: np.ndarray,
    masks: tuple[int, ...],
    magics: tuple[int, ...],
    rook: bool,
    pext_index: bool = False,
) -> None:
    """Write one slider's masks, magics, shifts, offsets and attack sets into the table.

    With `pext_index` the attack sets sit at `pext(blockers, mask)` instead of the magic
    index. `mask_subsets` lists subset `i` as the mask's bits taken in ascending order from
    the bits of `i`, which is `pext`'s own definition, so `pext(subsets[i], mask) == i` and
    the sets go in as listed; `tests/test_pext_tables.py` holds the identity to that.
    """
    mask_at, magic_at, shift_at, offset_at, table_at, size = (
        (ROOK_MASK, ROOK_MAGIC, ROOK_SHIFT, ROOK_OFFSET, ROOK_TABLE, ROOK_TABLE_SIZE)
        if rook
        else (
            BISHOP_MASK,
            BISHOP_MAGIC,
            BISHOP_SHIFT,
            BISHOP_OFFSET,
            BISHOP_TABLE,
            BISHOP_TABLE_SIZE,
        )
    )
    offset = 0
    for square in range(64):
        mask = masks[square]
        magic = magics[square]
        bits = mask.bit_count()
        shift = 64 - bits
        table[mask_at + square] = mask
        table[magic_at + square] = magic
        table[shift_at + square] = shift
        table[offset_at + square] = offset
        subsets = mask_subsets(mask)
        attacks = np.array([slider_attacks(square, int(s), rook) for s in subsets], dtype=np.uint64)
        if pext_index:
            index = np.arange(1 << bits, dtype=np.uint64)
        else:
            index = (subsets * np.uint64(magic)) >> np.uint64(shift)
        slots = table[table_at + offset : table_at + offset + (1 << bits)]
        slots[index] = attacks
        # Two occupancies with different attack sets in one slot would leave the last
        # write standing, so any subset that reads back wrong is a bad magic.
        if np.any(slots[index] != attacks):
            raise RuntimeError(f"magic collision on square {square}, rook={rook}")
        offset += 1 << bits
    if offset != size:
        raise RuntimeError(f"slider table size {offset}, expected {size}")


def build_tables(pext_index: bool | None = None) -> np.ndarray:
    """The whole attack table, freshly computed; about 0.9 MB, under a second to build.

    The slider sets are laid out for the lookup the kernels compile (`USE_PEXT`) unless
    `pext_index` says otherwise, which only the tests do.
    """
    if pext_index is None:
        pext_index = USE_PEXT
    table = np.zeros(TABLE_SIZE, dtype=np.uint64)
    for square in range(64):
        table[KNIGHT + square] = leaper_attacks(square, KNIGHT_STEPS)
        table[KING + square] = leaper_attacks(square, KING_STEPS)
        table[PAWN + square] = leaper_attacks(square, WHITE_PAWN_STEPS)
        table[PAWN + 64 + square] = leaper_attacks(square, BLACK_PAWN_STEPS)
        table[ROOK_RAY + square] = slider_attacks(square, 0, True)
        table[BISHOP_RAY + square] = slider_attacks(square, 0, False)
        for other in range(64):
            table[BETWEEN + square * 64 + other] = between(square, other)
    fill_magic(table, ROOK_MASKS, ROOK_MAGICS, True, pext_index)
    fill_magic(table, BISHOP_MASKS, BISHOP_MAGICS, False, pext_index)
    return table
