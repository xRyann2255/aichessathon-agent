"""The net inside the kernel: accumulators kept per ply, the head run at the leaves.

The net is the train lane's export in `weights/`: HalfKA inputs with the king planes
merged, the net's king buckets (1, 4, 8 or 16, from its metadata) chosen by each
perspective's own king square, the board mirrored so that king sits on files a to d at
every count but eight, which keeps both wings, a feature transformer of L1 lanes in int16, a
pairwise product, two 32-lane layers and one output, every number an integer and every
rescale a shift, as `docs/weights/README.md` lays out. A net whose metadata says `head` is
`pairwise-dual-32-32` carries the dual-activation head instead: a squared clipped ReLU beside the
clipped ReLU after each hidden layer, so each layer hands the next 64 bytes rather than
32, the output layer reads both layers' bytes, and two raw first-layer sums reach the
score, halved and clamped to the net's `skip_clamp_raw` metadata in raw units when it
carries one nonzero, unbounded otherwise. Which head runs is chosen once in Python:
`choose_head` binds `head` from the
net's metadata before anything compiles, so a process compiles one head and a plain
net never compiles the dual tail; a net without the flag is the head above, byte for
byte. A net whose metadata says `output_buckets` is N carries N copies of the whole head,
and a position reads the copy of its piece-count band, so an ending and a full board are
scored by heads trained apart; a net without that flag carries one copy and every offset
below is the single head's. The raw output over 8192 is the side
to move's score in sigmoid units, times 400 its score in centipawns.

Each ply of the search owns a pair of accumulators, one per colour, in `acc`. A move made
by the search writes each lane of the child's pair once, as the parent's lane less and plus
the rows of the features it changed (the moved piece, the captured piece, a promotion, the
castling rook); a king move
that changes its bucket or mirror rebuilds that king's perspective from the board, and one
that keeps them is a piece like any other; a null move copies. Nothing is undone: the
parent's pair is intact when the move is taken back. `evaluate_net` runs the head over the
pair of the position's ply.

Arrays the kernels read are arguments, never module names: `ftw` is the feature
transformer's weight matrix, `hd` the biases, the head's layers, the king bucket
table and the net's centipawn scale packed as one int32 array behind the width in its
first slot, `acc` the
accumulators of every ply, `xs` the head's int32 scratch, `hb` its byte half (the
clipped products, the first layer's weights packed for `first_layer`, then the first
layer's clipped outputs and the second layer's weights packed for `second_layer`). A net that
fails to load or to reproduce the check file leaves the agent with the piece-square
tables: `dummy_arrays` gives arrays the kernels treat as no net.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import chess
import numpy as np
import numpy.typing as npt
from llvmlite import binding as llvm  # type: ignore[import-untyped]
from llvmlite import ir as llvm_ir
from numba import types
from numba.core import cgutils, config
from numba.core.codegen import get_host_cpu_features
from numba.extending import intrinsic

import kernel
from kernel import (
    BLACK,
    CASTLING,
    EMPTY,
    EN_PASSANT,
    KING,
    OCC_ALL,
    PLY,
    ROOK,
    STM,
    TESTING,
    WHITE,
    Bitboards,
    Ints,
    aligned,
    jit,
    lsb,
    move_captured,
    move_flag,
    move_piece,
    move_promotion,
    move_source,
    move_target,
)

Int8s = npt.NDArray[np.int8]
Int16s = npt.NDArray[np.int16]
Int32s = npt.NDArray[np.int32]

CONTRACT = "nnue-int16-int8-v1"
FEATURES_PER_BUCKET = 704
KING_PLANE = 10
FT_MAX = 255
HIDDEN_MAX = 127
HIDDEN_SHIFT = 7  # the first layer's sums to the bytes the second layer reads
OUTPUT_SCALE = 8192
CENTIPAWNS = 400  # the scale a net trained without one in its metadata was fitted to
L2 = 32
L3 = 32
SECOND_SHIFT = 6  # the second layer's sums to the bytes the output layer reads
SQR_SHIFT = 7  # the squared clipped ReLU: a clipped byte squared, back into the byte range
SKIP_SHIFT = 1  # a first-layer sum carries 16,384 to the unit against the raw output's 8,192
DUAL_INPUTS = 2 * L2 + 2 * L3  # the dual output layer's inputs: both activations of both layers
STACKS_MAX = 8  # the most head stacks `output_buckets` may ask for
# The tensors a stack owns: everything past the transformer and the pairwise product.
HEAD_TENSORS = ("fc0.weight", "fc0.bias", "fc1.weight", "fc1.bias", "out.weight", "out.bias")
PLAIN_HEAD = "pairwise-32-32"  # the metadata's `head`: the head `docs/weights/README.md` opens with
DUAL_HEAD = "pairwise-dual-32-32"  # the metadata's `head`: the dual-activation head

DTYPES = {"I16": np.int16, "I8": np.int8, "I32": np.int32}

# The king bucket by the king's rank and file, for the bucket counts the trainer exports
# (`training/records.py`, `bucket_table`). Sixteen, four and one mirror the board onto
# files a to d first: sixteen buckets are the table below; four split the mirrored half
# by files a and b against c and d and ranks 1 and 2 against the rest; one is a single
# bank. Eight never mirror: the file pairs ab, cd, ef and gh, and ranks 1 and 2 against
# the rest, so a king on either wing keeps its own side of the board. Both tables the
# net needs ride in `hd`, the feature base and the mirror mask per king square, so the
# kernels compile once and never ask the count.
_BUCKET_ROWS = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (8, 8, 9, 9),
    (10, 10, 11, 11),
    (12, 12, 13, 13),
    (12, 12, 13, 13),
    (14, 14, 15, 15),
    (14, 14, 15, 15),
)


def king_bucket_table(buckets: int) -> tuple[int, ...]:
    """The bucket of each king square in the perspective's own view, for 1, 4, 8 or 16.

    The square is the king's after the colour flip and before any mirror: the tables
    that mirror (1, 4 and 16) are symmetric about the centre file, so a king on files e
    to h reads the bucket of its mirror image, and the eight-bucket table is not.
    """
    if buckets == 16:
        return tuple(
            _BUCKET_ROWS[square // 8][min(square % 8, 7 - square % 8)] for square in range(64)
        )
    if buckets == 8:
        return tuple((square % 8) // 2 + (4 if square // 8 >= 2 else 0) for square in range(64))
    if buckets == 4:
        return tuple(
            (2 if square // 8 >= 2 else 0) + (1 if min(square % 8, 7 - square % 8) >= 2 else 0)
            for square in range(64)
        )
    if buckets == 1:
        return (0,) * 64
    raise ValueError(f"no king bucket table for {buckets} buckets")


def king_mirror_table(buckets: int) -> tuple[int, ...]:
    """The mask every square is xored with when the king stands on a square, by count.

    7 mirrors the board horizontally onto files a to d, which the counts that share one
    table for both wings apply to a king on e to h; the eight-bucket net keeps both
    wings and mirrors nothing. The mask is a table so the kernels read it beside the
    bucket base and no count is a compile-time branch.
    """
    if buckets == 8:
        return (0,) * 64
    if buckets in (1, 4, 16):
        return tuple(7 if square % 8 >= 4 else 0 for square in range(64))
    raise ValueError(f"no king mirror table for {buckets} buckets")


KING_BUCKET = king_bucket_table(16)

# Layout of `hd`: the width, the head stacks the net carries and the slots one stack
# spans, the dual head's skip clamp, then the transformer bias, then the stacks
# themselves, then the feature base of each of the 64 king squares (its bucket times the
# features a bucket), the scale, the head's flag and the mirror mask of each of the 64
# king squares, all int32. The count and the stride sit beside the width because
# that is the one place in `hd` that does not move with the count, and every kernel that
# reads them reads the width too; the skip clamp sits beside them for the same reason,
# a single value that never moves with l1 or the stack count either.
HD_L1 = 0
HD_STACKS = 1  # the head stacks the net carries, 1 to STACKS_MAX
HD_STRIDE = 2  # the int32 slots from one stack's head to the next
HD_BASES = 3  # where the 64 king bucket bases start, which moves with the stack count
HD_SKIP_CLAMP = 4  # the dual head's linear skip clamp, raw units; 0 is unbounded
HD_FTB = 5  # the transformer bias, the slot after them
# The bucket region, from the slot `HD_BASES` names: the 64 bases, the scale, the flag,
# then the 64 mirror masks, so a net's mirror rides beside its bases and `mapping`
# reads both by the king square.
BASES_MIRRORS = 66  # the 64 mirror masks, after the bases, the scale and the flag
BASES_SPAN = 130  # the slots the bucket region owns; the dual output weights follow


def stack_stride(l1: int) -> int:
    """The int32 slots one head stack owns in `hd`: `fc0.weight` through `out.bias`.

    Those parts are one run of the single head's layout, so a net of one stack lays out
    exactly as it did before stacks existed and a net of N repeats the run N times.
    """
    return L2 * l1 + L2 + L3 * L2 + L3 + L3 + 1


def stack_slots(l1: int, stack: int) -> tuple[int, int, int, int, int]:
    """One stack's `fc0.bias`, `fc1.weight`, `fc1.bias`, `out.weight` and `out.bias` in `hd`."""
    fc0b = HD_FTB + l1 + stack * stack_stride(l1) + L2 * l1
    fc1w = fc0b + L2
    fc1b = fc1w + L3 * L2
    outw = fc1b + L3
    return fc0b, fc1w, fc1b, outw, outw + L3


def head_offsets(l1: int, stacks: int = 1) -> tuple[int, int, int, int, int, int, int, int, int]:
    """Where each part of the head and the bucket table start in `hd`, and its length.

    The head parts returned are the first stack's; a net of N stacks holds N runs of them
    at `stack_stride` apart, which `stack_slots` indexes, and the bucket table follows the
    last run. The slot after the 64 bucket bases holds the net's centipawn scale, the
    trainer's fit of `sigmoid(output)` to `sigmoid(score / scale)`, so a net trained at
    another scale still reaches the search in centipawns, the slot after that the
    head's flag: 0 for the head this file's first contract describes, 1 for the
    dual-activation head, whose own output weights follow the region, and the 64 slots
    after the flag the mirror mask of each king square (`BASES_MIRRORS`). The flag is
    for the loader and the tests; the kernel never reads it, since `choose_head` binds
    the head in Python before the first compile. Both heads share these offsets, so the
    bucket bases, the scale and the masks sit at the same slots whichever head a net
    carries and `mapping` never asks which; the count of stacks and the stride between
    them are the two slots after the width, which `HD_STACKS` and `HD_STRIDE` name.
    """
    ftb = HD_FTB
    fc0w = ftb + l1
    fc0b = fc0w + L2 * l1
    fc1w = fc0b + L2
    fc1b = fc1w + L3 * L2
    outw = fc1b + L3
    outb = outw + L3
    bases = fc0w + stacks * stack_stride(l1)
    return ftb, fc0w, fc0b, fc1w, fc1b, outw, outb, bases, bases + BASES_SPAN


def dual_offsets(l1: int, stacks: int = 1) -> tuple[int, int, int]:
    """A dual net's `hd`: the head's flag, its output weights, and the array's length.

    The dual head's `fc1` weights are twice the plain region and reach the kernel packed
    in `hb` alone, so nothing before the flag moves; its 128 output weights a stack are
    the only part of it `hd` carries, and the N runs of them ride after the bucket
    region's mirror masks, at the end of the array.
    """
    *_, bases, size = head_offsets(l1, stacks)
    return bases + 65, bases + BASES_SPAN, size + stacks * DUAL_INPUTS


def dual_byte_offsets(l1: int, stacks: int = 1) -> tuple[int, int, int]:
    """A dual net's `hb`: the two hidden layers' byte regions, and the array's length.

    The first region holds the first layer's 64 bytes and then `fc1`'s packed weights,
    which is the one shape `_packed_layer` compiles; the second holds the second layer's
    64 bytes, which only the output layer reads, so no weights follow them. The offsets
    returned are the first stack's: the N first-layer regions come first, then the N
    regions this pair points into, each at its own fixed stride.
    """
    h2 = stacks * (l1 + L2 * l1)
    h3 = h2 + 2 * L2 + L3 * 2 * L2
    return h2, h3, h2 + stacks * (2 * L2 + L3 * 2 * L2 + 2 * L3)


def byte_offsets(l1: int, stacks: int = 1) -> tuple[int, int]:
    """Where the second layer's byte region starts in `hb`, and the array's length.

    `hb` holds the l1 clipped products, the first layer's packed weights, then the L2
    clipped first-layer outputs and the second layer's packed weights. Each layer reads
    its inputs from the front of its own region and its weights straight after them,
    which is the one shape `_packed_layer` compiles. A net of N stacks holds N of each
    region at a fixed stride, the first-layer ones first, and the offset returned is the
    first stack's second region.
    """
    h2 = stacks * (l1 + L2 * l1)
    return h2, h2 + stacks * (L2 + L3 * L2)


def read_safetensors(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """The tensors and the metadata of a safetensors file, through numpy alone."""
    data = path.read_bytes()
    (header_size,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + header_size])
    metadata = header.pop("__metadata__", {})
    base = 8 + header_size
    arrays = {}
    for name, info in header.items():
        start, end = info["data_offsets"]
        raw = np.frombuffer(data[base + start : base + end], dtype=DTYPES[info["dtype"]])
        # A copy: frombuffer's view is read-only, and numba types a read-only array apart.
        arrays[name] = np.array(raw.reshape(info["shape"]), copy=True)
    return arrays, metadata


# Pasted into their callers: a separately compiled function costs 40 ms of import.
@jit(inline=True)
def mapping(king_square: int, colour: int, hd: Int32s) -> tuple[int, int, int]:
    """The flip, the mirror and the bucket's feature base for a king of `colour` on a square."""
    flip = 56 if colour == BLACK else 0
    square = king_square ^ flip
    # The bucket bases follow the last head stack, so where they start moves with the
    # count; the loader writes it into `HD_BASES` and this reads it, one load and no
    # arithmetic. Every caller is behind the `hd.shape[0] < 2` guard that means a net.
    # The mirror is the net's own table beside the bases, 7 on files e to h for a net
    # that folds the wings together and 0 everywhere for one that keeps both, and the
    # bucket tables that mirror are symmetric, so the base is read by the king's own
    # square whichever the net does.
    bases = int(hd[np.uint64(HD_BASES)])
    mirror = int(hd[np.uint64(bases + BASES_MIRRORS + square)])
    return flip, mirror, hd[np.uint64(bases + square)]


@jit(inline=True, internal=not TESTING)
def perspective(bb: Bitboards, colour: int, hd: Int32s) -> tuple[int, int, int]:
    """The flip, the mirror and the bucket's feature base for one colour's king."""
    return mapping(lsb(bb[np.uint64(colour * 6 + KING)]), colour, hd)


@jit(inline=True, internal=not TESTING)
def feature(piece: int, square: int, colour: int, flip: int, mirror: int, base: int) -> int:
    """The input index of a kernel piece on a square, seen by `colour`."""
    kind = piece % 6
    if kind == KING:
        plane = KING_PLANE
    elif piece // 6 == colour:
        plane = kind
    else:
        plane = kind + 5
    return base + plane * 64 + (square ^ flip ^ mirror)


@jit()
def refresh_one(bb: Bitboards, colour: int, ftw: Int16s, hd: Int32s, acc: Int16s, ply: int) -> None:
    """One colour's accumulator at `ply` from the board: the bias plus every piece's row."""
    l1 = hd[HD_L1]
    offset = (ply * 2 + colour) * l1
    lanes = acc[offset : offset + l1]
    bias = hd[HD_FTB : HD_FTB + l1]
    for i in range(l1):
        lanes[np.uint64(i)] = np.int16(bias[np.uint64(i)])
    flip, mirror, base = perspective(bb, colour, hd)
    for piece in range(12):
        pieces = bb[piece]
        while pieces:
            square = lsb(pieces)
            pieces &= pieces - np.uint64(1)
            weights = ftw[np.uint64(feature(piece, square, colour, flip, mirror, base))]
            for i in range(l1):
                lanes[np.uint64(i)] = np.int16(lanes[np.uint64(i)] + weights[np.uint64(i)])


@jit()
def refresh(bb: Bitboards, ftw: Int16s, hd: Int32s, acc: Int16s, ply: int) -> None:
    """Both accumulators at `ply` from the board; a no-op without a net."""
    if hd.shape[0] < 2:
        return
    for colour in range(2):  # a loop, so both calls type the colour alike
        refresh_one(bb, colour, ftw, hd, acc, ply)


# The cache the king-move refresh reads: one accumulator per colour and king key, with
# the twelve bitboards it was built from. The key is the colour, the bucket of the king
# square and its mirror, so a net of any bucket count keys into the same 64 slots, of
# which the shipped net uses 8 a colour. Its size is `KEY_SLOTS * l1` lanes and
# `KEY_SLOTS * KEY_WORDS` bitboards; `agent.py` holds both for the game.
KEY_SLOTS = 64
KEY_WORDS = 13  # the twelve bitboards the entry was built from, then whether it is built


@jit()
def refresh_cached(
    bb: Bitboards,
    colour: int,
    ftw: Int16s,
    hd: Int32s,
    acc: Int16s,
    ply: int,
    kacc: Int16s,
    kbb: Bitboards,
) -> None:
    """One colour's accumulator at `ply`, from the cache of its king key where there is one.

    The entry holds the accumulator last built for the key and the board it was built
    from. Its lanes are copied and then walked over the pieces that differ: the rows of
    the pieces that have left their squares are taken off first, so no partial sum holds
    more rows than the 32 the loader's int16 bound covers, and the rows of the pieces that
    arrived are added after. With no entry yet the accumulator is built from the board as
    before. How old the entry is changes the cost and never the result, because the
    difference is taken against the board the entry was built from, so the cache needs no
    reset between moves; the counter measured 5.4 differing pieces against the 29 on the
    board (`tmp/logs/kernel-king-cache-2-counter.log`).
    """
    l1 = hd[HD_L1]
    flip, mirror, base = perspective(bb, colour, hd)
    key = colour * 32 + (base // FEATURES_PER_BUCKET) * 2 + (1 if mirror else 0)
    entry = key * KEY_WORDS
    offset = (ply * 2 + colour) * l1
    lanes = acc[offset : offset + l1]
    if kbb[np.uint64(entry + 12)] == np.uint64(0):
        refresh_one(bb, colour, ftw, hd, acc, ply)
    else:
        cached = kacc[key * l1 : key * l1 + l1]
        for i in range(l1):
            lanes[np.uint64(i)] = cached[np.uint64(i)]
        for piece in range(12):
            left = kbb[np.uint64(entry + piece)] & ~bb[np.uint64(piece)]
            while left:
                square = lsb(left)
                left &= left - np.uint64(1)
                weights = ftw[np.uint64(feature(piece, square, colour, flip, mirror, base))]
                for i in range(l1):
                    lanes[np.uint64(i)] = np.int16(lanes[np.uint64(i)] - weights[np.uint64(i)])
        for piece in range(12):
            arrived = bb[np.uint64(piece)] & ~kbb[np.uint64(entry + piece)]
            while arrived:
                square = lsb(arrived)
                arrived &= arrived - np.uint64(1)
                weights = ftw[np.uint64(feature(piece, square, colour, flip, mirror, base))]
                for i in range(l1):
                    lanes[np.uint64(i)] = np.int16(lanes[np.uint64(i)] + weights[np.uint64(i)])
    store = kacc[key * l1 : key * l1 + l1]
    for i in range(l1):
        store[np.uint64(i)] = lanes[np.uint64(i)]
    for piece in range(12):
        kbb[np.uint64(entry + piece)] = bb[np.uint64(piece)]
    kbb[np.uint64(entry + 12)] = np.uint64(1)


# `nn_make_body` is pasted into `nn_make` here and into `nn_make_s` in `search.py`,
# which reads the arrays from the search object (the note above `attacked_body` in
# `kernel.py`).
@jit(inline=True, internal=not TESTING)
def nn_make_body(
    move: int,
    bb: Bitboards,
    st: Ints,
    ftw: Int16s,
    hd: Int32s,
    acc: Int16s,
    kacc: Int16s,
    kbb: Bitboards,
) -> None:
    """Bring the accumulators of the ply just entered up to date, after `make`.

    Each lane is written once: the parent's lane, less the row of the piece that left its
    square, plus the row of the piece that landed, less the captured piece's row; castling
    moves the rook's rows too. One pass over the lanes in place of a copy and two to four
    row passes, each of which was a call with its own slices. The pass itself is
    `accumulate_two`, `accumulate_three` or `accumulate_four`, resolved at import to the
    form the host proved (`accumulate_form` below).
    """
    if hd.shape[0] < 2:
        return
    l1 = hd[HD_L1]
    ply = st[PLY]
    us = 1 - st[STM]  # the side that just moved
    source = move_source(move)
    target = move_target(move)
    piece = move_piece(move)
    captured = move_captured(move)
    promotion = move_promotion(move)
    flag = move_flag(move)
    landed = us * 6 + promotion if promotion else piece
    captured_square = target
    if flag == EN_PASSANT:
        captured_square = target - 8 if us == WHITE else target + 8
    rook = us * 6 + ROOK
    if target > source:
        rook_source, rook_target = target + 1, target - 1
    else:
        rook_source, rook_target = target - 2, target + 1
    for colour in range(2):
        if piece == colour * 6 + KING:
            # This king moved. Its perspective is rebuilt only when its bucket or mirror
            # changed with the square; otherwise the king is a piece like any other below,
            # its feature the king plane at its square.
            _, was_mirror, was_base = mapping(source, colour, hd)
            flip, mirror, base = mapping(target, colour, hd)
            if was_mirror != mirror or was_base != base:
                refresh_cached(bb, colour, ftw, hd, acc, ply, kacc, kbb)
                continue
        else:
            flip, mirror, base = perspective(bb, colour, hd)
        offset = (ply * 2 + colour) * l1
        parent = ((ply - 1) * 2 + colour) * l1
        gone = feature(piece, source, colour, flip, mirror, base)
        came = feature(landed, target, colour, flip, mirror, base)
        if flag == CASTLING:
            rook_gone = feature(rook, rook_source, colour, flip, mirror, base)
            rook_came = feature(rook, rook_target, colour, flip, mirror, base)
            accumulate_four(acc, offset, parent, ftw, gone, came, rook_gone, rook_came, l1)
        elif captured != EMPTY:
            taken = feature(captured, captured_square, colour, flip, mirror, base)
            accumulate_three(acc, offset, parent, ftw, gone, came, taken, l1)
        else:
            accumulate_two(acc, offset, parent, ftw, gone, came, l1)


@jit(internal=not TESTING)
def nn_make(
    move: int,
    bb: Bitboards,
    st: Ints,
    ftw: Int16s,
    hd: Int32s,
    acc: Int16s,
    kacc: Int16s,
    kbb: Bitboards,
) -> None:
    """`nn_make_body` on arrays passed one by one; the search calls `nn_make_s`."""
    nn_make_body(move, bb, st, ftw, hd, acc, kacc, kbb)


@jit(inline=True, internal=not TESTING)
def nn_apply_body(move: int, ply: int, bb: Bitboards, ftw: Int16s, hd: Int32s, acc: Int16s) -> None:
    """The accumulator update of a move that is not a king move, applied at `ply`.

    `nn_make_body` writes the ply it is standing on and reads the board it is standing on.
    This one is handed the ply, because it runs after the search has gone deeper, and it
    reads the board only for the two kings, whose squares are the same as they were: a
    king move is applied where it is made and never deferred, so no king has moved between
    the ply this update belongs to and the board it is applied on. The rest of the update
    is the move's own, and the parent's lanes at `ply - 1` are up to date because
    `nn_flush_s` applies a line in the order it was played.
    """
    l1 = hd[HD_L1]
    piece = move_piece(move)
    us = piece // 6
    source = move_source(move)
    target = move_target(move)
    captured = move_captured(move)
    promotion = move_promotion(move)
    landed = us * 6 + promotion if promotion else piece
    captured_square = target
    if move_flag(move) == EN_PASSANT:
        captured_square = target - 8 if us == WHITE else target + 8
    for colour in range(2):
        flip, mirror, base = perspective(bb, colour, hd)
        offset = (ply * 2 + colour) * l1
        parent = ((ply - 1) * 2 + colour) * l1
        gone = feature(piece, source, colour, flip, mirror, base)
        came = feature(landed, target, colour, flip, mirror, base)
        if captured != EMPTY:
            taken = feature(captured, captured_square, colour, flip, mirror, base)
            accumulate_three(acc, offset, parent, ftw, gone, came, taken, l1)
        else:
            accumulate_two(acc, offset, parent, ftw, gone, came, l1)


@jit()
def nn_null(st: Ints, hd: Int32s, acc: Int16s) -> None:
    """After a null move the position's pieces are the parent's: copy both accumulators."""
    if hd.shape[0] < 2:
        return
    l1 = hd[HD_L1]
    ply = st[PLY]
    for colour in range(2):
        offset = (ply * 2 + colour) * l1
        parent = ((ply - 1) * 2 + colour) * l1
        lanes = acc[offset : offset + l1]
        above = acc[parent : parent + l1]
        for i in range(l1):
            lanes[np.uint64(i)] = above[np.uint64(i)]


# The first layer's weights in `hb`, after the l1 input bytes: by block of four inputs,
# then by output, then the block's four weights, so 32 consecutive bytes hold the
# weights of 8 outputs for one block and 64 hold 16, one vector load either way.
WEIGHT_BLOCK = 4
# Bytes a hand-written vector load of this file claims when it is allowed to claim more
# than the item's own width. Every array those loads read comes from `kernel.aligned`, so
# it starts a 64-byte line, and `Net` below refuses a net whose accumulator is not a whole
# number of lines. The widths the kernels then step by are all multiples of 32 bytes: an
# accumulator (`l1` int16) and half of one (`l1` bytes), a first-layer row (`l1` int16),
# a byte region (`l1 + L2 * l1`, then `L2 + L3 * L2` after it), a block of packed weights
# (`WEIGHT_BLOCK * L2` bytes) and a vector of lanes. Each load below names the base and
# the offsets its own address is built from; one whose offsets this does not cover keeps
# the item's alignment, because an aligned load of a misaligned address faults the process
# and loses the game rather than costing a split (#531).
VECTOR_ALIGN = 32


def host_features() -> set[str]:
    """The CPU features numba compiles for: the host's unless `NUMBA_CPU_FEATURES` names others.

    Every form below reads this same string, so a form chosen here is one the target can
    select.
    """
    features: str | None = config.CPU_FEATURES  # type: ignore[attr-defined]
    if features is None:
        features = get_host_cpu_features()  # type: ignore[no-untyped-call]
    return {name[1:] for name in features.split(",") if name.startswith("+")}


# The features the forms above choose on, in the order a reader expects; `avxvnni` is
# the 256-bit VNNI a later core or a hypervisor may offer without AVX-512.
ISA_FEATURES = ("avx2", "avx512f", "avx512vnni", "avxvnni")


def host_isa() -> str:
    """The host as LLVM names it and the vector features it reports, for the init line.

    Rounds 73 and 77 ran on an EPYC 9V74, whose Zen 4 core has AVX-512 VNNI, and both
    init lines read `first_layer=avx2`: either the hypervisor hides AVX-512 from cpuid
    or LLVM names the core without it. This field says which, one rated game at a time.
    """
    on = host_features()
    present = ",".join(name for name in ISA_FEATURES if name in on) or "none"
    return f"{llvm.get_host_cpu_name()}:{present}"


def first_layer_form() -> str:
    """Which first layer this process compiles, from the features numba compiles for.

    `vnni` is one AVX-512 instruction a step (`vpdpbusd`, the platform's EPYC 9V74),
    `avx2` the pair `vpmaddubsw` then `vpmaddwd` (this machine, any EPYC 7763), `plain`
    a loop over the same bytes for a host with neither and the control the two are
    measured against.
    """
    on = host_features()
    if {"avx512f", "avx512vnni"} <= on:
        return "vnni"
    if "avx2" in on:
        return "avx2"
    return "plain"


def clipped_product_form() -> str:
    """Which clipped product this process compiles: `avx2` needs nothing above AVX2."""
    return "avx2" if "avx2" in host_features() else "plain"


def hidden_clip_form() -> str:
    """Which hidden clip this process compiles: `avx2` needs nothing above AVX2."""
    return "avx2" if "avx2" in host_features() else "plain"


def _packed_layer(lanes: int, step: Any, outputs: int) -> Any:
    """The IR of one packed layer: `lanes` int32 outputs a vector, `step` the multiply-add.

    Each of the `outputs // lanes` output vectors starts at its biases, gains one block
    of four inputs (broadcast across the vector as one int32) times the block's weights
    at every step, and ends in its slots of `xs` after the width. The vectors are
    independent accumulators, so the adds of one never wait on another's. The first layer
    is `outputs = L2` over 256 inputs, the second `outputs = L3` over 32.
    """
    groups = outputs // lanes
    span = WEIGHT_BLOCK * lanes  # bytes of weights a vector reads in one block

    def codegen(context: Any, builder: Any, signature: Any, args: Any) -> Any:
        bytes_ = context.make_array(signature.args[0])(context, builder, value=args[0]).data
        head_ = context.make_array(signature.args[1])(context, builder, value=args[1]).data
        scratch = context.make_array(signature.args[3])(context, builder, value=args[3]).data
        bias_at = context.cast(builder, args[2], signature.args[2], types.int64)
        width = context.cast(builder, args[4], signature.args[4], types.int64)
        i8, i32, i64 = llvm_ir.IntType(8), llvm_ir.IntType(32), llvm_ir.IntType(64)
        vbytes = llvm_ir.VectorType(i8, span)
        vints = llvm_ir.VectorType(i32, lanes)
        weights = builder.gep(bytes_, [width])
        blocks = builder.sdiv(width, i64(WEIGHT_BLOCK))
        starts = []
        for group in range(groups):
            bias = builder.gep(head_, [builder.add(bias_at, i64(group * lanes))])
            # `hd` is not cut to a line and `bias_at` is a head offset with no whole
            # number of them in it, so this one keeps the int32's own four bytes.
            starts.append(builder.load(builder.bitcast(bias, vints.as_pointer()), align=4))
        entry = builder.block
        loop = builder.append_basic_block("first_layer")
        body = builder.append_basic_block("first_layer_block")
        done = builder.append_basic_block("first_layer_done")
        builder.branch(loop)
        builder.position_at_end(loop)
        block = builder.phi(i64)
        block.add_incoming(i64(0), entry)
        sums = [builder.phi(vints) for _ in range(groups)]
        for phi, start in zip(sums, starts, strict=True):
            phi.add_incoming(start, entry)
        builder.cbranch(builder.icmp_signed("<", block, blocks), body, done)
        builder.position_at_end(body)
        inputs = builder.gep(bytes_, [builder.mul(block, i64(WEIGHT_BLOCK))])
        # `bytes_` starts a line and `block * WEIGHT_BLOCK` is a multiple of four bytes.
        word = builder.load(builder.bitcast(inputs, i32.as_pointer()), align=4)
        spread = builder.shuffle_vector(
            builder.insert_element(llvm_ir.Constant(vints, llvm_ir.Undefined), word, i32(0)),
            llvm_ir.Constant(vints, llvm_ir.Undefined),
            llvm_ir.Constant(vints, [0] * lanes),
        )
        spread = builder.bitcast(spread, vbytes)
        row = builder.mul(block, i64(WEIGHT_BLOCK * outputs))
        added = []
        for group, phi in enumerate(sums):
            at = builder.gep(weights, [builder.add(row, i64(group * span))])
            # `bytes_` starts a line, `width` is the layer's inputs in bytes and `row` is
            # `block * WEIGHT_BLOCK * outputs`, both whole numbers of 32 bytes, and
            # `group * span` is whole vectors.
            weight = builder.load(builder.bitcast(at, vbytes.as_pointer()), align=VECTOR_ALIGN)
            added.append(step(builder, phi, spread, weight))
        following = builder.add(block, i64(1))
        builder.branch(loop)
        block.add_incoming(following, body)
        for phi, value in zip(sums, added, strict=True):
            phi.add_incoming(value, body)
        builder.position_at_end(done)
        for group, phi in enumerate(sums):
            at = builder.gep(scratch, [builder.add(width, i64(group * lanes))])
            # `scratch` starts a line, `width` int32 is a whole number of them in, and
            # `group * lanes` int32 is whole vectors.
            builder.store(phi, builder.bitcast(at, vints.as_pointer()), align=VECTOR_ALIGN)
        return context.get_dummy_value()

    return codegen


def _layer_signature(hb: Any, hd: Any, fc0b: Any, xs: Any, l1: Any) -> Any:
    """The one signature a packed first layer accepts: the head's arrays and two ints."""
    arrays = ((hb, types.int8), (hd, types.int32), (xs, types.int32))
    if not all(
        isinstance(array, types.Array)
        and array.dtype == dtype
        and array.ndim == 1
        and array.layout == "C"
        for array, dtype in arrays
    ) or not (
        isinstance(fc0b, types.Integer)  # type: ignore[attr-defined]
        and isinstance(l1, types.Integer)  # type: ignore[attr-defined]
    ):
        raise TypeError("a first layer takes int8[::1], int32[::1], int, int32[::1], int")
    return types.none(hb, hd, fc0b, xs, l1)


def _avx2_step(builder: Any, total: Any, inputs: Any, weights: Any) -> Any:
    """`total + inputs . weights` over 32 bytes into 8 int32 lanes: `vpmaddubsw`, `vpmaddwd`.

    `vpmaddubsw` multiplies unsigned input bytes by signed weight bytes and sums each
    adjacent pair into int16, saturating; `vpmaddwd` against ones sums each adjacent
    pair of those into int32. The loader's bound keeps every pair inside int16, so the
    sums are exact.
    """
    bytes32 = llvm_ir.VectorType(llvm_ir.IntType(8), 32)
    words16 = llvm_ir.VectorType(llvm_ir.IntType(16), 16)
    ints8 = llvm_ir.VectorType(llvm_ir.IntType(32), 8)
    pmaddubsw = cgutils.get_or_insert_function(  # type: ignore[no-untyped-call]
        builder.module,
        llvm_ir.FunctionType(words16, [bytes32, bytes32]),
        "llvm.x86.avx2.pmadd.ub.sw",
    )
    pmaddwd = cgutils.get_or_insert_function(  # type: ignore[no-untyped-call]
        builder.module, llvm_ir.FunctionType(ints8, [words16, words16]), "llvm.x86.avx2.pmadd.wd"
    )
    pairs = builder.call(pmaddubsw, [inputs, weights])
    quads = builder.call(pmaddwd, [pairs, llvm_ir.Constant(words16, [1] * 16)])
    return builder.add(total, quads)


def _vnni_step(builder: Any, total: Any, inputs: Any, weights: Any) -> Any:
    """`total + inputs . weights` over 64 bytes into 16 int32 lanes: one `vpdpbusd`.

    The AVX-512 VNNI instruction multiplies unsigned bytes by signed bytes and adds each
    four products into the int32 lane below them with no int16 stage, so nothing can
    saturate.
    """
    ints16 = llvm_ir.VectorType(llvm_ir.IntType(32), 16)
    vpdpbusd = cgutils.get_or_insert_function(  # type: ignore[no-untyped-call]
        builder.module,
        llvm_ir.FunctionType(ints16, [ints16, ints16, ints16]),
        "llvm.x86.avx512.vpdpbusd.512",
    )
    return builder.call(
        vpdpbusd, [total, builder.bitcast(inputs, ints16), builder.bitcast(weights, ints16)]
    )


# The three first layers share one contract: `xs[l1 + j] = hd[fc0b + j] + sum(w[j, i] *
# x0[i])` for the 32 outputs, the inputs the first `l1` bytes of `hb` (0 to 127), the
# weights the packed bytes after them. The two packed ones are team-written LLVM IR
# through numba's intrinsic door, the pattern of `prefetch_line` in `search.py`, because
# a plain loop over bytes widens them back to words before multiplying; nothing here
# comes from another engine.


@intrinsic
def first_layer_avx2(typingctx: Any, hb: Any, hd: Any, fc0b: Any, xs: Any, l1: Any) -> Any:
    """The first layer by `vpmaddubsw` then `vpmaddwd`, eight outputs a vector."""
    return _layer_signature(hb, hd, fc0b, xs, l1), _packed_layer(8, _avx2_step, L2)


@intrinsic
def first_layer_vnni(typingctx: Any, hb: Any, hd: Any, fc0b: Any, xs: Any, l1: Any) -> Any:
    """The first layer by `vpdpbusd`, sixteen outputs a vector."""
    return _layer_signature(hb, hd, fc0b, xs, l1), _packed_layer(16, _vnni_step, L2)


@jit(inline=True)
def first_layer_plain(hb: Int8s, hd: Int32s, fc0b: int, xs: Int32s, l1: int) -> None:
    """The first layer as a plain loop over the packed bytes: the fallback and the control.

    The inputs are read as int8, which for 0 to 127 is the byte's unsigned value too.
    """
    blocks = l1 // WEIGHT_BLOCK
    for j in range(L2):
        t = np.int32(hd[np.uint64(fc0b + j)])
        for block in range(blocks):
            at = l1 + block * WEIGHT_BLOCK * L2 + j * WEIGHT_BLOCK
            for k in range(WEIGHT_BLOCK):
                t = np.int32(
                    t
                    + np.int32(hb[np.uint64(at + k)])
                    * np.int32(hb[np.uint64(block * WEIGHT_BLOCK + k)])
                )
        xs[np.uint64(l1 + j)] = t


# The packed forms can only be called from compiled code, so each has a compiled entry
# of its own for the check below; `head` never calls these.
@jit()
def run_avx2(hb: Int8s, hd: Int32s, fc0b: int, xs: Int32s, l1: int) -> None:
    first_layer_avx2(hb, hd, fc0b, xs, l1)  # type: ignore[call-arg]


@jit()
def run_vnni(hb: Int8s, hd: Int32s, fc0b: int, xs: Int32s, l1: int) -> None:
    first_layer_vnni(hb, hd, fc0b, xs, l1)  # type: ignore[call-arg]


@jit()
def run_plain(hb: Int8s, hd: Int32s, fc0b: int, xs: Int32s, l1: int) -> None:
    first_layer_plain(hb, hd, fc0b, xs, l1)


RUNNERS: dict[str, Any] = {"vnni": run_vnni, "avx2": run_avx2, "plain": run_plain}


# The second layer is the first layer's shape at width 32: L3 outputs over the L2 clipped
# first-layer values, which the head writes as bytes after the first layer's weights in
# `hb`, and int8 weights the loader packs the same way. So the three forms above take it
# unchanged, with `outputs = L3` and the byte region and scratch passed as slices; only
# the plain loop is written out again, because its output count is a compiled constant.


@jit(inline=True)
def second_layer_plain(hb: Int8s, hd: Int32s, fc1b: int, xs: Int32s, l2: int) -> None:
    """The second layer as a plain loop over the packed bytes: the fallback and the control."""
    blocks = l2 // WEIGHT_BLOCK
    for j in range(L3):
        t = np.int32(hd[np.uint64(fc1b + j)])
        for block in range(blocks):
            at = l2 + block * WEIGHT_BLOCK * L3 + j * WEIGHT_BLOCK
            for k in range(WEIGHT_BLOCK):
                t = np.int32(
                    t
                    + np.int32(hb[np.uint64(at + k)])
                    * np.int32(hb[np.uint64(block * WEIGHT_BLOCK + k)])
                )
        xs[np.uint64(l2 + j)] = t


@intrinsic
def second_layer_avx2(typingctx: Any, hb: Any, hd: Any, fc1b: Any, xs: Any, l2: Any) -> Any:
    """The second layer by `vpmaddubsw` then `vpmaddwd`, eight outputs a vector."""
    return _layer_signature(hb, hd, fc1b, xs, l2), _packed_layer(8, _avx2_step, L3)


@intrinsic
def second_layer_vnni(typingctx: Any, hb: Any, hd: Any, fc1b: Any, xs: Any, l2: Any) -> Any:
    """The second layer by `vpdpbusd`, sixteen outputs a vector."""
    return _layer_signature(hb, hd, fc1b, xs, l2), _packed_layer(16, _vnni_step, L3)


@jit()
def run_second_avx2(hb: Int8s, hd: Int32s, fc1b: int, xs: Int32s, l2: int) -> None:
    second_layer_avx2(hb, hd, fc1b, xs, l2)  # type: ignore[call-arg]


@jit()
def run_second_vnni(hb: Int8s, hd: Int32s, fc1b: int, xs: Int32s, l2: int) -> None:
    second_layer_vnni(hb, hd, fc1b, xs, l2)  # type: ignore[call-arg]


@jit()
def run_second_plain(hb: Int8s, hd: Int32s, fc1b: int, xs: Int32s, l2: int) -> None:
    second_layer_plain(hb, hd, fc1b, xs, l2)


SECOND_RUNNERS: dict[str, Any] = {
    "vnni": run_second_vnni,
    "avx2": run_second_avx2,
    "plain": run_second_plain,
}
CLIP_LANES = 16  # int16 lanes a vector: 256 bits, what AVX2 holds


def _clipped_product() -> Any:
    """The IR of the clipped product: 16 accumulator lanes a vector, into 16 bytes.

    One perspective's `count` lanes, read as the pair `acc[at + i]` and `acc[at + half +
    i]`, clipped to 0 to 255 by `vpmaxsw` and `vpminsw`, multiplied by `vpmullw` and
    shifted by `vpsrlw`, and stored as bytes at `hb[out + i]`. `vpmullw` keeps the low
    16 bits, which is the whole product because 255 * 255 is 65,025 and fits an unsigned
    word; the shift by nine is logical, so the byte is the same 0 to 127 the loop below
    computes. `count` is a multiple of the lane count; the caller does any remainder.

    The caller passes `acc` from `kernel.aligned`, `at` a whole number of accumulators
    into it and `half` half of one, which is what lets the pair of loads claim
    VECTOR_ALIGN; `out` is free, and the store below claims nothing.
    """

    def codegen(context: Any, builder: Any, signature: Any, args: Any) -> Any:
        lanes = context.make_array(signature.args[0])(context, builder, value=args[0]).data
        bytes_ = context.make_array(signature.args[2])(context, builder, value=args[2]).data
        as_int = [
            context.cast(builder, args[i], signature.args[i], types.int64) for i in (1, 3, 4, 5)
        ]
        at, out, half, count = as_int
        i16, i8, i64 = llvm_ir.IntType(16), llvm_ir.IntType(8), llvm_ir.IntType(64)
        words = llvm_ir.VectorType(i16, CLIP_LANES)
        octets = llvm_ir.VectorType(i8, CLIP_LANES)
        zero = llvm_ir.Constant(words, [0] * CLIP_LANES)
        top = llvm_ir.Constant(words, [FT_MAX] * CLIP_LANES)
        nine = llvm_ir.Constant(words, [9] * CLIP_LANES)
        lo_at = builder.gep(lanes, [at])
        hi_at = builder.gep(lanes, [builder.add(at, half)])
        out_at = builder.gep(bytes_, [out])
        entry = builder.block
        loop = builder.append_basic_block("clipped_product")
        body = builder.append_basic_block("clipped_product_block")
        done = builder.append_basic_block("clipped_product_done")
        builder.branch(loop)
        builder.position_at_end(loop)
        index = builder.phi(i64)
        index.add_incoming(i64(0), entry)
        builder.cbranch(builder.icmp_signed("<", index, count), body, done)
        builder.position_at_end(body)
        pair = []
        for base in (lo_at, hi_at):
            at_i = builder.bitcast(builder.gep(base, [index]), words.as_pointer())
            # `acc` starts a line, `at` is whole accumulators and `half` half of one into
            # it (`l1` bytes), and `index` steps CLIP_LANES int16, a whole vector.
            value = builder.load(at_i, align=VECTOR_ALIGN)
            # The clip: select is what the backend turns into vpmaxsw and vpminsw.
            value = builder.select(builder.icmp_signed(">", value, zero), value, zero)
            value = builder.select(builder.icmp_signed("<", value, top), value, top)
            pair.append(value)
        product = builder.lshr(builder.mul(pair[0], pair[1]), nine)
        store_at = builder.bitcast(builder.gep(out_at, [index]), octets.as_pointer())
        # `out` is `front + half` at the second perspective, and half an accumulator in
        # bytes is not a whole vector at every width, so this store claims one byte.
        builder.store(builder.trunc(product, octets), store_at, align=1)
        index.add_incoming(builder.add(index, i64(CLIP_LANES)), body)
        builder.branch(loop)
        builder.position_at_end(done)
        return context.get_dummy_value()

    return codegen


def _clip_signature(acc: Any, at: Any, hb: Any, out: Any, half: Any, count: Any) -> Any:
    """The one signature the clipped product accepts: the two arrays and four indices."""
    arrays = ((acc, types.int16), (hb, types.int8))
    if not all(
        isinstance(array, types.Array)
        and array.dtype == dtype
        and array.ndim == 1
        and array.layout == "C"
        for array, dtype in arrays
    ) or not all(
        isinstance(value, types.Integer)  # type: ignore[attr-defined]
        for value in (at, out, half, count)
    ):
        raise TypeError("a clipped product takes int16[::1], int, int8[::1], int, int, int")
    return types.none(acc, at, hb, out, half, count)


@intrinsic
def clipped_product_avx2(
    typingctx: Any, acc: Any, at: Any, hb: Any, out: Any, half: Any, count: Any
) -> Any:
    """The clipped product over 16 accumulator lanes a vector."""
    return _clip_signature(acc, at, hb, out, half, count), _clipped_product()


@jit(inline=True)
def clipped_product_plain(acc: Int16s, at: int, hb: Int8s, out: int, half: int, count: int) -> None:
    """The clipped product as a loop over the lanes: the fallback, the control and the tail."""
    for i in range(count):
        a = min(max(np.int32(acc[np.uint64(at + i)]), 0), FT_MAX)
        b = min(max(np.int32(acc[np.uint64(at + half + i)]), 0), FT_MAX)
        hb[np.uint64(out + i)] = (a * b) >> 9  # type: ignore[operator]  # at most 127: 255 * 255 >> 9


@jit()
def run_clip_avx2(acc: Int16s, at: int, hb: Int8s, out: int, half: int, count: int) -> None:
    clipped_product_avx2(acc, at, hb, out, half, count)  # type: ignore[call-arg]


@jit()
def run_clip_plain(acc: Int16s, at: int, hb: Int8s, out: int, half: int, count: int) -> None:
    clipped_product_plain(acc, at, hb, out, half, count)


CLIP_RUNNERS: dict[str, Any] = {"avx2": run_clip_avx2, "plain": run_clip_plain}


# The first layer's 32 sums, shifted and clipped to bytes: the second layer's inputs.
# A loop over the sums stores one byte a step; the form below is team-written LLVM IR
# through the same intrinsic door as the clipped product above, four vectors of sums
# to one vector of bytes. Nothing here comes from another engine.
HIDDEN_LANES = 8  # int32 lanes a vector: 256 bits, what AVX2 holds


def _hidden_clip() -> Any:
    """The IR of the hidden clip: the L2 sums at `xs[l1]` to the L2 bytes at `hb[h2]`.

    Four vectors of eight sums, each shifted right by HIDDEN_SHIFT (`vpsrad`), clipped
    to 0 to HIDDEN_MAX (`vpmaxsd`, `vpminsd`), packed to words (`vpackssdw`) and to
    bytes (`vpacksswb`), and stored as one vector of 32 bytes. The packs cannot
    saturate: every lane is already 0 to 127. Each pack works within its two 128-bit
    halves, so the bytes come out in blocks of four, the first half of every input
    vector before any second half, and one cross-lane permute of the eight blocks
    (`vpermd`) restores the order. L2 is four vectors; the pack tree assumes it.
    """

    def codegen(context: Any, builder: Any, signature: Any, args: Any) -> Any:
        bytes_ = context.make_array(signature.args[0])(context, builder, value=args[0]).data
        sums = context.make_array(signature.args[2])(context, builder, value=args[2]).data
        h2 = context.cast(builder, args[1], signature.args[1], types.int64)
        l1 = context.cast(builder, args[3], signature.args[3], types.int64)
        i8, i16, i32, i64 = (llvm_ir.IntType(bits) for bits in (8, 16, 32, 64))
        ints = llvm_ir.VectorType(i32, HIDDEN_LANES)
        words = llvm_ir.VectorType(i16, 2 * HIDDEN_LANES)
        octets = llvm_ir.VectorType(i8, 4 * HIDDEN_LANES)
        zero = llvm_ir.Constant(ints, [0] * HIDDEN_LANES)
        top = llvm_ir.Constant(ints, [HIDDEN_MAX] * HIDDEN_LANES)
        shift = llvm_ir.Constant(ints, [HIDDEN_SHIFT] * HIDDEN_LANES)
        packssdw = cgutils.get_or_insert_function(  # type: ignore[no-untyped-call]
            builder.module, llvm_ir.FunctionType(words, [ints, ints]), "llvm.x86.avx2.packssdw"
        )
        packsswb = cgutils.get_or_insert_function(  # type: ignore[no-untyped-call]
            builder.module, llvm_ir.FunctionType(octets, [words, words]), "llvm.x86.avx2.packsswb"
        )
        at = builder.gep(sums, [l1])
        clipped = []
        for group in range(L2 // HIDDEN_LANES):
            src = builder.gep(at, [i64(group * HIDDEN_LANES)])
            # `xs` starts a line, `l1` int32 is a whole number of them in, and
            # `group * HIDDEN_LANES` int32 is a whole vector.
            value = builder.load(builder.bitcast(src, ints.as_pointer()), align=VECTOR_ALIGN)
            value = builder.ashr(value, shift)
            # The clip: select is what the backend turns into vpmaxsd and vpminsd.
            value = builder.select(builder.icmp_signed(">", value, zero), value, zero)
            value = builder.select(builder.icmp_signed("<", value, top), value, top)
            clipped.append(value)
        low = builder.call(packssdw, [clipped[0], clipped[1]])
        high = builder.call(packssdw, [clipped[2], clipped[3]])
        blocks = builder.bitcast(builder.call(packsswb, [low, high]), ints)
        ordered = builder.shuffle_vector(
            blocks,
            llvm_ir.Constant(ints, llvm_ir.Undefined),
            llvm_ir.Constant(ints, [0, 4, 1, 5, 2, 6, 3, 7]),
        )
        store_at = builder.bitcast(builder.gep(bytes_, [h2]), octets.as_pointer())
        # `hidden_clip_agrees` stores at an odd `h2` on purpose, to check the bytes
        # around it, so this store claims one byte whatever the head's own offset is.
        builder.store(builder.bitcast(ordered, octets), store_at, align=1)
        return context.get_dummy_value()

    return codegen


def _hidden_signature(hb: Any, h2: Any, xs: Any, l1: Any) -> Any:
    """The one signature the hidden clip accepts: the two arrays and their offsets."""
    arrays = ((hb, types.int8), (xs, types.int32))
    if not all(
        isinstance(array, types.Array)
        and array.dtype == dtype
        and array.ndim == 1
        and array.layout == "C"
        for array, dtype in arrays
    ) or not all(
        isinstance(value, types.Integer)  # type: ignore[attr-defined]
        for value in (h2, l1)
    ):
        raise TypeError("a hidden clip takes int8[::1], int, int32[::1], int")
    return types.none(hb, h2, xs, l1)


@intrinsic
def hidden_clip_avx2(typingctx: Any, hb: Any, h2: Any, xs: Any, l1: Any) -> Any:
    """The hidden clip over four vectors of sums into one vector of bytes."""
    return _hidden_signature(hb, h2, xs, l1), _hidden_clip()


@jit(inline=True)
def hidden_clip_plain(hb: Int8s, h2: int, xs: Int32s, l1: int) -> None:
    """The hidden clip as a loop over the sums: the fallback and the control."""
    for j in range(L2):
        hb[np.uint64(h2 + j)] = min(max(xs[np.uint64(l1 + j)] >> HIDDEN_SHIFT, 0), HIDDEN_MAX)


@jit()
def run_hidden_avx2(hb: Int8s, h2: int, xs: Int32s, l1: int) -> None:
    hidden_clip_avx2(hb, h2, xs, l1)  # type: ignore[call-arg]


@jit()
def run_hidden_plain(hb: Int8s, h2: int, xs: Int32s, l1: int) -> None:
    hidden_clip_plain(hb, h2, xs, l1)


HIDDEN_RUNNERS: dict[str, Any] = {"avx2": run_hidden_avx2, "plain": run_hidden_plain}
CHECK_WIDTH = 256
CHECK_TRIALS = 3000
SECOND_CHECK_TRIALS = 3000
CLIP_CHECK_TRIALS = 2000
HIDDEN_CHECK_TRIALS = 2000


def first_layer_agrees(form: str, trials: int = CHECK_TRIALS, seed: int = 2026) -> bool:
    """Whether `form` gives the plain loop's 32 sums bit for bit over random inputs.

    Every trial draws fresh inputs (0 to 127, a quarter of them pinned at 127) and fresh
    int8 weights (-128 to 127, with -128 and 127 sown in), and biases across int32's
    middle. Three fixed trials sit at the edges: every input 127 against every weight
    -128 (the pair sum at -32,512, the bound), against every weight 127 (32,258), and
    against weights alternating between the two.
    """
    l1 = CHECK_WIDTH
    fc0b = 1 + l1 + L2 * l1
    rng = np.random.default_rng(seed)
    # `hb` and `xs` reach a load and a store that claim VECTOR_ALIGN, so they are cut to
    # a line here as the net's own are; `hd` reaches the bias load, which claims four.
    hb = aligned(l1 + L2 * l1, np.int8)
    hd = np.zeros(fc0b + L2, dtype=np.int32)
    xs = aligned(l1 + L2 + L3, np.int32)
    want = kernel.aligned_like(xs)
    inputs = rng.integers(0, 128, size=(trials, l1), dtype=np.int8)
    inputs[rng.integers(0, 4, size=inputs.shape, dtype=np.uint8) == 0] = 127
    weights = rng.integers(-128, 128, size=(trials, L2 * l1), dtype=np.int8)
    sow = rng.integers(0, 50, size=weights.shape, dtype=np.uint8)
    weights[sow == 0] = -128
    weights[sow == 1] = 127
    biases = rng.integers(-(1 << 24), 1 << 24, size=(trials, L2), dtype=np.int32)
    edges = np.array(
        [[-128] * (L2 * l1), [127] * (L2 * l1), [-128, 127] * (L2 * l1 // 2)], dtype=np.int8
    )
    for trial in range(trials + len(edges)):
        if trial < trials:
            hb[:l1] = inputs[trial]
            hb[l1:] = weights[trial]
            hd[fc0b:] = biases[trial]
        else:
            hb[:l1] = 127
            hb[l1:] = edges[trial - trials]
            hd[fc0b:] = 0
        RUNNERS["plain"](hb, hd, fc0b, want, l1)
        RUNNERS[form](hb, hd, fc0b, xs, l1)
        if not np.array_equal(xs[l1 : l1 + L2], want[l1 : l1 + L2]):
            return False
    return True


def second_layer_agrees(form: str, trials: int = SECOND_CHECK_TRIALS, seed: int = 2027) -> bool:
    """Whether `form` gives the plain loop's L3 sums bit for bit over random inputs.

    The same procedure as `first_layer_agrees` at the second layer's width: inputs 0 to
    127, a quarter of them pinned at the largest the first layer's clip can produce,
    int8 weights with both ends sown in, and three edge trials at the pair bound.
    """
    l2 = L2
    fc1b = 1 + l2 + L3 * l2
    rng = np.random.default_rng(seed)
    hb = aligned(l2 + L3 * l2, np.int8)
    hd = np.zeros(fc1b + L3, dtype=np.int32)
    xs = aligned(l2 + L3, np.int32)
    want = kernel.aligned_like(xs)
    inputs = rng.integers(0, 128, size=(trials, l2), dtype=np.int8)
    inputs[rng.integers(0, 4, size=inputs.shape, dtype=np.uint8) == 0] = HIDDEN_MAX
    weights = rng.integers(-128, 128, size=(trials, L3 * l2), dtype=np.int8)
    sow = rng.integers(0, 50, size=weights.shape, dtype=np.uint8)
    weights[sow == 0] = -128
    weights[sow == 1] = 127
    biases = rng.integers(-(1 << 24), 1 << 24, size=(trials, L3), dtype=np.int32)
    edges = np.array(
        [[-128] * (L3 * l2), [127] * (L3 * l2), [-128, 127] * (L3 * l2 // 2)], dtype=np.int8
    )
    for trial in range(trials + len(edges)):
        if trial < trials:
            hb[:l2] = inputs[trial]
            hb[l2:] = weights[trial]
            hd[fc1b:] = biases[trial]
        else:
            hb[:l2] = HIDDEN_MAX
            hb[l2:] = edges[trial - trials]
            hd[fc1b:] = 0
        SECOND_RUNNERS["plain"](hb, hd, fc1b, want, l2)
        SECOND_RUNNERS[form](hb, hd, fc1b, xs, l2)
        if not np.array_equal(xs[l2 : l2 + L3], want[l2 : l2 + L3]):
            return False
    return True


def checked_layer(chosen: str, agrees: Any) -> tuple[str, str]:
    """The form to compile into `head`, and the outcome for the init line.

    The chosen form is held to the plain loop by `agrees`; on a mismatch a VNNI host
    falls back to the AVX2 form, checked the same way, and anything still wrong falls
    back to the plain loop, which needs no check. The AVX-512 form has run on no machine
    of ours, so this is what proves it on the platform, before `head` compiles.
    """
    fallbacks = {"vnni": "avx2", "avx2": "plain", "plain": "plain"}
    form = chosen
    while form != "plain":
        if agrees(form):
            return form, "passed" if form == chosen else f"fell_back_to_{form}"
        form = fallbacks[form]
    return form, "passed" if chosen == "plain" else "fell_back_to_plain"


def clipped_product_agrees(form: str, trials: int = CLIP_CHECK_TRIALS, seed: int = 2028) -> bool:
    """Whether `form` gives the plain loop's bytes over random accumulator pairs.

    Every trial draws a whole pair of accumulators across int16, a quarter of the lanes
    pinned outside the clip at each end so the vector's max and min are exercised, and
    checks both perspectives. Four fixed trials sit at the ends: every lane at 0, at
    FT_MAX, at int16's largest and at its smallest.
    """
    half = CHECK_WIDTH // 2
    rng = np.random.default_rng(seed)
    # Cut to a line, and `at` is 0 or a whole accumulator into it: the shape the head
    # calls in, which is what the pair of loads claims.
    acc = aligned(2 * CHECK_WIDTH, np.int16)
    got = np.zeros(CHECK_WIDTH, dtype=np.int8)
    want = np.zeros_like(got)
    limit = np.iinfo(np.int16)
    banks = rng.integers(-1000, 1001, size=(trials, 2 * CHECK_WIDTH), dtype=np.int16)
    marks = rng.integers(0, 4, size=banks.shape, dtype=np.uint8)
    banks[marks == 0] = rng.integers(-limit.max - 1, 0, size=int((marks == 0).sum()))
    banks[marks == 1] = rng.integers(FT_MAX + 1, limit.max + 1, size=int((marks == 1).sum()))
    ends = (0, FT_MAX, limit.max, limit.min)
    for trial in range(trials + len(ends)):
        acc[:] = banks[trial] if trial < trials else ends[trial - trials]
        for at, out in ((0, 0), (CHECK_WIDTH, half)):
            CLIP_RUNNERS["plain"](acc, at, want, out, half, half)
            CLIP_RUNNERS[form](acc, at, got, out, half, half)
        if not np.array_equal(got, want):
            return False
    return True


def checked_clipped_product(chosen: str) -> tuple[str, str]:
    """The clipped product's form and outcome; there is no VNNI form to fall back from."""
    if chosen != "plain" and clipped_product_agrees(chosen):
        return chosen, "passed"
    return "plain", "passed" if chosen == "plain" else "fell_back_to_plain"


def hidden_clip_agrees(form: str, trials: int = HIDDEN_CHECK_TRIALS, seed: int = 2030) -> bool:
    """Whether `form` gives the plain loop's bytes over random first-layer sums.

    Every trial draws the L2 sums around the clip's two edges (300 below zero to 300
    above HIDDEN_MAX << HIDDEN_SHIFT), a quarter of them from int32's negative half and
    a quarter from above the upper edge to int32's largest, so the vector's shift, max
    and min all run on most trials. Seven fixed trials pin every sum at an edge: 0, -1,
    the upper edge itself, the last sum the shift leaves at HIDDEN_MAX, the first it
    does not, and int32's largest and smallest. The bytes land at an odd offset, so
    the store is checked unaligned, and the guard bytes around them must stay.
    """
    rng = np.random.default_rng(seed)
    limit = np.iinfo(np.int32)
    edge = HIDDEN_MAX << HIDDEN_SHIFT
    at = 3
    xs = aligned(CHECK_WIDTH + L2, np.int32)
    got = np.full(at + L2 + 3, -7, dtype=np.int8)
    want = got.copy()
    banks = rng.integers(-300, edge + 301, size=(trials, L2), dtype=np.int32)
    marks = rng.integers(0, 4, size=banks.shape, dtype=np.uint8)
    banks[marks == 0] = rng.integers(limit.min, 0, size=int((marks == 0).sum()))
    banks[marks == 1] = rng.integers(edge + 1, limit.max + 1, size=int((marks == 1).sum()))
    step = 1 << HIDDEN_SHIFT
    ends = (0, -1, edge, edge + step - 1, edge + step, limit.max, limit.min)
    for trial in range(trials + len(ends)):
        xs[CHECK_WIDTH:] = banks[trial] if trial < trials else ends[trial - trials]
        HIDDEN_RUNNERS["plain"](want, at, xs, CHECK_WIDTH)
        HIDDEN_RUNNERS[form](got, at, xs, CHECK_WIDTH)
        if not np.array_equal(got, want):
            return False
    return True


def checked_hidden_clip(chosen: str) -> tuple[str, str]:
    """The hidden clip's form and outcome; there is no VNNI form to fall back from."""
    if chosen != "plain" and hidden_clip_agrees(chosen):
        return chosen, "passed"
    return "plain", "passed" if chosen == "plain" else "fell_back_to_plain"


def checked_first_layer(chosen: str) -> tuple[str, str]:
    """The first layer's form and outcome."""
    return checked_layer(chosen, first_layer_agrees)


def checked_second_layer(chosen: str) -> tuple[str, str]:
    """The second layer's form and outcome, chosen and proven on its own."""
    return checked_layer(chosen, second_layer_agrees)


FIRST_LAYER, FIRST_LAYER_CHECK = checked_first_layer(first_layer_form())
SECOND_LAYER, SECOND_LAYER_CHECK = checked_second_layer(first_layer_form())
CLIPPED_PRODUCT, CLIPPED_PRODUCT_CHECK = checked_clipped_product(clipped_product_form())
HIDDEN_CLIP, HIDDEN_CLIP_CHECK = checked_hidden_clip(hidden_clip_form())

# Resolved into `head` when it compiles: one form a process each, chosen and checked
# above at import, before anything compiles `head`.
first_layer: Any = {
    "vnni": first_layer_vnni,
    "avx2": first_layer_avx2,
    "plain": first_layer_plain,
}[FIRST_LAYER]
second_layer: Any = {
    "vnni": second_layer_vnni,
    "avx2": second_layer_avx2,
    "plain": second_layer_plain,
}[SECOND_LAYER]
clipped_product: Any = {
    "avx2": clipped_product_avx2,
    "plain": clipped_product_plain,
}[CLIPPED_PRODUCT]
hidden_clip: Any = {"avx2": hidden_clip_avx2, "plain": hidden_clip_plain}[HIDDEN_CLIP]


# The accumulator update, the one pass `nn_make_body` above makes over the lanes. That
# loop already writes each lane once and numba vectorises it, but around it the backend
# leaves a pointer-overlap test, its trip guards and a scalar copy of the loop for the
# shapes it cannot prove disjoint, and it unrolls the castling shape to two vectors where
# the registers hold eight. The forms below are team-written LLVM IR through
# numba's intrinsic door, the pattern of the clipped product above; nothing here comes
# from another engine.
ACC_LANES = 16  # int16 lanes a vector: 256 bits, what AVX2 holds
# Vectors in flight, so no lane's add waits on the one before it. Eight is what the bench
# chose over four and sixteen, x1.082 against x1.062 and x1.054 of the loop the backend
# writes, on an A/A floor of x0.999: four leaves the loop's own arithmetic a larger share
# and sixteen spills on the shapes that carry four rows.
ACC_VECTORS = 8
# The stages of one update, widest first: 128 lanes an iteration, then 16, then one. Any
# width is covered, so the caller passes the whole of `l1` and keeps no tail of its own.
ACC_STAGES = ((ACC_LANES, ACC_VECTORS), (ACC_LANES, 1), (1, 1))
ACC_ROWS = 5  # feature rows the check draws
ACC_CHECK_TRIALS = 200


def _accumulate(signs: tuple[int, ...]) -> Any:
    """The IR of one fused update: `acc[dst + i] = acc[src + i] + sum(sign * ftw[row, i])`.

    Every pending row is applied to the lane in registers, so the parent's lane is loaded
    once and the ply's lane stored once however many rows the move moves. The adds wrap
    at int16 exactly as the plain loop's cast does, since truncation and two's complement
    addition commute, so the lanes are the same bits either way.

    The caller passes `acc` and `ftw` from `kernel.aligned` and `dst` and `src` a whole
    number of accumulators into `acc`, which is what lets the vector stages claim
    VECTOR_ALIGN; `accumulate_agrees` lays its own regions out at a pitch of whole
    vectors for the same reason.
    """

    def codegen(context: Any, builder: Any, signature: Any, args: Any) -> Any:
        lanes = context.make_array(signature.args[0])(context, builder, value=args[0]).data
        rows_array = context.make_array(signature.args[3])(context, builder, value=args[3])
        stride = cgutils.unpack_tuple(builder, rows_array.shape, 2)[1]  # type: ignore[no-untyped-call]
        values = [
            context.cast(builder, args[i], signature.args[i], types.int64)
            for i in (1, 2, *range(4, len(args)))
        ]
        dst, src, count = values[0], values[1], values[-1]
        i16, i64 = llvm_ir.IntType(16), llvm_ir.IntType(64)
        src_at = builder.gep(lanes, [src])
        dst_at = builder.gep(lanes, [dst])
        row_at = [builder.gep(rows_array.data, [builder.mul(row, stride)]) for row in values[2:-1]]

        def slot(base: Any, offset: Any, words: Any) -> Any:
            """The pointer to `words` at `offset` lanes into `base`."""
            at = builder.gep(base, [offset])
            return at if words is i16 else builder.bitcast(at, words.as_pointer())

        index: Any = i64(0)
        for width, vectors in ACC_STAGES:
            block = width * vectors
            words = i16 if width == 1 else llvm_ir.VectorType(i16, width)
            # A vector stage's `step` is a multiple of ACC_LANES int16 from a base that is
            # whole accumulators into `acc` or a whole row into `ftw`, both cut to a line;
            # the one-lane stage walks a lane at a time and keeps the int16's two bytes.
            align = VECTOR_ALIGN if width > 1 else 2
            entry = builder.block
            loop = builder.append_basic_block("accumulate")
            body = builder.append_basic_block("accumulate_block")
            done = builder.append_basic_block("accumulate_done")
            builder.branch(loop)
            builder.position_at_end(loop)
            walk = builder.phi(i64)
            walk.add_incoming(index, entry)
            builder.cbranch(
                builder.icmp_signed("<=", builder.add(walk, i64(block)), count), body, done
            )
            builder.position_at_end(body)
            steps = [builder.add(walk, i64(vector * width)) for vector in range(vectors)]
            totals = [builder.load(slot(src_at, step, words), align=align) for step in steps]
            for base, sign in zip(row_at, signs, strict=True):
                for vector, step in enumerate(steps):
                    weight = builder.load(slot(base, step, words), align=align)
                    totals[vector] = (
                        builder.add(totals[vector], weight)
                        if sign > 0
                        else builder.sub(totals[vector], weight)
                    )
            for total, step in zip(totals, steps, strict=True):
                builder.store(total, slot(dst_at, step, words), align=align)
            walk.add_incoming(builder.add(walk, i64(block)), body)
            builder.branch(loop)
            builder.position_at_end(done)
            index = walk
        return context.get_dummy_value()

    return codegen


def _accumulate_signature(acc: Any, ftw: Any, *values: Any) -> Any:
    """The one signature a fused update accepts: the lanes, the rows and the indices."""
    if (
        not isinstance(acc, types.Array)
        or acc.dtype != types.int16
        or acc.ndim != 1
        or acc.layout != "C"
        or not isinstance(ftw, types.Array)
        or ftw.dtype != types.int16
        or ftw.ndim != 2
        or ftw.layout != "C"
        or not all(isinstance(value, types.Integer) for value in values)  # type: ignore[attr-defined]
    ):
        raise TypeError("a fused update takes int16[::1], int, int, int16[:, ::1] and ints")
    dst, src, *rest = values
    return types.none(acc, dst, src, ftw, *rest)


@intrinsic
def accumulate_two_avx2(
    typingctx: Any, acc: Any, dst: Any, src: Any, ftw: Any, gone: Any, came: Any, count: Any
) -> Any:
    """The quiet shape in one pass: the parent's lane, less `gone`, plus `came`."""
    return _accumulate_signature(acc, ftw, dst, src, gone, came, count), _accumulate((-1, 1))


@intrinsic
def accumulate_three_avx2(
    typingctx: Any,
    acc: Any,
    dst: Any,
    src: Any,
    ftw: Any,
    gone: Any,
    came: Any,
    taken: Any,
    count: Any,
) -> Any:
    """The capture shape in one pass: the quiet shape less the captured piece's row."""
    return (
        _accumulate_signature(acc, ftw, dst, src, gone, came, taken, count),
        _accumulate((-1, 1, -1)),
    )


@intrinsic
def accumulate_four_avx2(
    typingctx: Any,
    acc: Any,
    dst: Any,
    src: Any,
    ftw: Any,
    gone: Any,
    came: Any,
    rook_gone: Any,
    rook_came: Any,
    count: Any,
) -> Any:
    """The castling shape in one pass: the king's two rows and the rook's two."""
    return (
        _accumulate_signature(acc, ftw, dst, src, gone, came, rook_gone, rook_came, count),
        _accumulate((-1, 1, -1, 1)),
    )


@jit(inline=True)
def accumulate_two_plain(
    acc: Int16s, dst: int, src: int, ftw: Int16s, gone: int, came: int, count: int
) -> None:
    """The quiet shape as a loop over the lanes: the fallback and the control.

    The two slices are what let the backend vectorise the loop: indexed off `acc`
    directly it cannot rule out an overlap and falls back to a lane at a time.
    """
    lanes = acc[dst : dst + count]
    above = acc[src : src + count]
    for i in range(count):
        lanes[np.uint64(i)] = np.int16(
            above[np.uint64(i)]
            - ftw[np.uint64(gone), np.uint64(i)]
            + ftw[np.uint64(came), np.uint64(i)]
        )


@jit(inline=True)
def accumulate_three_plain(
    acc: Int16s, dst: int, src: int, ftw: Int16s, gone: int, came: int, taken: int, count: int
) -> None:
    """The capture shape as a loop over the lanes."""
    lanes = acc[dst : dst + count]
    above = acc[src : src + count]
    for i in range(count):
        lanes[np.uint64(i)] = np.int16(
            above[np.uint64(i)]
            - ftw[np.uint64(gone), np.uint64(i)]
            + ftw[np.uint64(came), np.uint64(i)]
            - ftw[np.uint64(taken), np.uint64(i)]
        )


@jit(inline=True)
def accumulate_four_plain(
    acc: Int16s,
    dst: int,
    src: int,
    ftw: Int16s,
    gone: int,
    came: int,
    rook_gone: int,
    rook_came: int,
    count: int,
) -> None:
    """The castling shape as a loop over the lanes."""
    lanes = acc[dst : dst + count]
    above = acc[src : src + count]
    for i in range(count):
        lanes[np.uint64(i)] = np.int16(
            above[np.uint64(i)]
            - ftw[np.uint64(gone), np.uint64(i)]
            + ftw[np.uint64(came), np.uint64(i)]
            - ftw[np.uint64(rook_gone), np.uint64(i)]
            + ftw[np.uint64(rook_came), np.uint64(i)]
        )


# The packed forms can only be called from compiled code, so the check below has one
# compiled entry that runs all three; `nn_make_body` never calls it. One entry and not
# three keeps the import's share of this to a single compile.
@jit()
def run_accumulate(acc: Int16s, ftw: Int16s, rows: Ints, count: int, pitch: int) -> None:
    """The three shapes into the three regions of `acc` below the parent's, the check's entry.

    `pitch` is the lanes between one region and the next, `count` rounded up to whole
    vectors, so every region starts where an accumulator would: the kernel's own callers
    pass whole accumulators and the loads claim it (#531).
    """
    accumulate_two_avx2(acc, 0, 3 * pitch, ftw, rows[0], rows[1], count)  # type: ignore[call-arg]
    accumulate_three_avx2(acc, pitch, 3 * pitch, ftw, rows[0], rows[1], rows[2], count)
    accumulate_four_avx2(acc, 2 * pitch, 3 * pitch, ftw, rows[0], rows[1], rows[2], rows[3], count)


def accumulate_form() -> str:
    """Which fused update this process compiles: `avx2` needs nothing above AVX2."""
    return "avx2" if "avx2" in host_features() else "plain"


def accumulate_agrees(trials: int = ACC_CHECK_TRIALS, seed: int = 2029) -> bool:
    """Whether the fused update gives numpy's lanes bit for bit over random rows.

    Every trial draws a parent accumulator and its feature rows across the whole of
    int16, so the wrap the plain loop's cast performs is reached, and runs all three
    shapes. The widths rotate between the net's, one that leaves the first two stages a
    remainder, and one below a single vector, so every stage of the IR is entered. The
    regions sit at a pitch of whole vectors rather than of the width, because the loads
    claim the alignment the shipped arrays give them; the lanes each call covers are the
    width, so the stages the widths were chosen for are the ones that run. The plain loop
    needs no check of its own and is not compiled on a host that passes here.
    """
    rng = np.random.default_rng(seed)
    limit = np.iinfo(np.int16)
    widths = (CHECK_WIDTH, ACC_LANES * (ACC_VECTORS + 1) + 3, 7)
    signs = ((-1, 1), (-1, 1, -1), (-1, 1, -1, 1))
    for trial in range(trials):
        width = widths[trial % len(widths)]
        pitch = -(-width // ACC_LANES) * ACC_LANES
        ftw = aligned(ACC_ROWS * pitch, np.int16, rows=ACC_ROWS)
        ftw[:, :width] = rng.integers(
            limit.min, limit.max + 1, size=(ACC_ROWS, width), dtype=np.int16
        )
        acc = aligned(4 * pitch, np.int16)
        acc[3 * pitch : 3 * pitch + width] = rng.integers(
            limit.min, limit.max + 1, size=width, dtype=np.int16
        )
        rows = rng.integers(0, ACC_ROWS, size=4).astype(np.int64)
        parent = acc[3 * pitch : 3 * pitch + width].astype(np.int64)
        run_accumulate(acc, ftw, rows, width, pitch)
        for shape, shape_signs in enumerate(signs):
            want = parent.copy()
            for sign, row in zip(shape_signs, rows, strict=False):
                want += sign * ftw[row, :width].astype(np.int64)
            got = acc[shape * pitch : shape * pitch + width]
            if not np.array_equal(got, want.astype(np.int16)):
                return False
    return True


def checked_accumulate(chosen: str) -> tuple[str, str]:
    """The fused update's form and outcome; there is no VNNI form to fall back from."""
    if chosen != "plain" and accumulate_agrees():
        return chosen, "passed"
    return "plain", "passed" if chosen == "plain" else "fell_back_to_plain"


ACCUMULATE, ACCUMULATE_CHECK = checked_accumulate(accumulate_form())

# Resolved into `nn_make_body` when it compiles, which is after this import.
accumulate_two: Any = {"avx2": accumulate_two_avx2, "plain": accumulate_two_plain}[ACCUMULATE]
accumulate_three: Any = {"avx2": accumulate_three_avx2, "plain": accumulate_three_plain}[ACCUMULATE]
accumulate_four: Any = {"avx2": accumulate_four_avx2, "plain": accumulate_four_plain}[ACCUMULATE]


@jit(inline=True, internal=not TESTING)
def dual_clip(hb: Int8s, at: int, xs: Int32s, base: int, shift: int, count: int) -> None:
    """One hidden layer's 64 bytes: the squared activation at `at`, the clipped after it.

    `xs[base + j]` are the layer's `count` int32 sums. The clipped ReLU is the plain
    head's, `clip(sum >> shift, 0, 127)`; the squared clipped ReLU squares that byte and
    shifts it back into the byte range, `(a * a) >> 7`, so a saturated lane reads 126 and
    no upper clip is needed. The squared bytes come first, as `docs/weights/README.md` says.
    """
    for j in range(count):
        a = min(max(xs[np.uint64(base + j)] >> shift, 0), HIDDEN_MAX)
        hb[np.uint64(at + count + j)] = a
        hb[np.uint64(at + j)] = (a * a) >> SQR_SHIFT  # at most 126: 127 * 127 >> 7


@jit(inline=True, internal=not TESTING)
def stack_of(hd: Int32s, men: int) -> int:
    """The head stack a position of `men` pieces reads: its piece-count band.

    `min(N - 1, (men - 1) * N // 32)`, the PSQT lanes' rule, so a net that carries both
    computes one index: four stacks cover 1 to 8 men, 9 to 16, 17 to 24 and 25 to 32. The
    count is the slot after the width, which is where it stays whatever N is, and a net
    of one stack lands on stack 0 from any piece count. `men` is the popcount of the
    occupancy bitboard, taken once at the head's call.
    """
    stacks = int(hd[np.uint64(HD_STACKS)])
    band = ((men - 1) * stacks) // 32
    return int(min(max(band, 0), stacks - 1))


@jit(internal=True)
def dual_tail(hd: Int32s, xs: Int32s, hb: Int8s, l1: int, stack: int, stacks: int) -> int:
    """The dual head past the first layer: both activations twice, then the output.

    `xs[l1 + j]` holds the first layer's 32 sums, which `head` has just computed. The
    first layer's 64 bytes go at `h2` and `fc1`'s packed weights follow them, so the
    second layer is the plain head's packed form at 64 inputs; its 32 sums land at
    `xs[l1 + 2 * L2 + j]` and become the 64 bytes at `h3` the same way. The output is 128
    multiply-adds over both layers' bytes, plus the net's linear skip: the difference of
    the last two raw first-layer sums, halved, since a first-layer sum carries 16,384 to
    the unit and the raw output 8,192, then clamped to plus or minus `HD_SKIP_CLAMP` raw
    units when the net's metadata set one; 0 there is unbounded, the trainer's own
    default. `stacks` comes from the caller, where a one-stack build has it folded to
    the constant 1, so no build reads the count at the leaf.
    """
    stride = L2 * l1 + L2 + L3 * L2 + L3 + L3 + 1  # `stack_stride`, from the width
    fc0b = HD_FTB + l1 + stack * stride + L2 * l1
    fc1b = fc0b + L2 + L3 * L2
    outb = fc1b + L3 + L3
    # The output weights follow the bucket region (the bases, the scale, the flag and
    # the mirror masks), one run of 128 a stack. At one stack this is the single head's
    # `outb + 1 + BASES_SPAN`.
    douw = int(hd[np.uint64(HD_BASES)]) + BASES_SPAN + stack * DUAL_INPUTS
    h2 = stacks * (l1 + L2 * l1) + stack * (2 * L2 + L3 * 2 * L2 + 2 * L3)
    h3 = h2 + 2 * L2 + L3 * 2 * L2
    dual_clip(hb, h2, xs, l1, HIDDEN_SHIFT, L2)
    second_layer(hb[h2:], hd, fc1b, xs[l1:], 2 * L2)
    dual_clip(hb, h3, xs, l1 + 2 * L2, SECOND_SHIFT, L3)
    total = np.int32(hd[np.uint64(outb)])
    for i in range(2 * L2):
        total += np.int32(hd[np.uint64(douw + i)]) * np.int32(hb[np.uint64(h2 + i)])
    for i in range(2 * L3):
        total += np.int32(hd[np.uint64(douw + 2 * L2 + i)]) * np.int32(hb[np.uint64(h3 + i)])
    skip = np.int32(xs[np.uint64(l1 + L2 - 2)]) - np.int32(xs[np.uint64(l1 + L2 - 1)])
    halved = skip >> SKIP_SHIFT
    clamp = hd[np.uint64(HD_SKIP_CLAMP)]
    if clamp != 0:
        halved = min(max(halved, -clamp), clamp)
    return int(total + halved)


@jit(inline=True, internal=not TESTING)
def head_front(
    stm: int, ply: int, stack: int, hd: Int32s, acc: Int16s, xs: Int32s, hb: Int8s
) -> int:
    """The part both heads share: the clipped products into `hb`, the first layer into `xs`.

    `stack` is the head stack the position reads, which the head chose before calling.
    The clipped products are the first `l1` bytes of that stack's byte region and its
    first layer's weights follow them; `first_layer` sums the region into the int32 sums
    at `xs[0:L2]` with the packed instructions its docstring names. In a one-stack build
    the stack is the literal 0, so every offset here folds back to the single head's.
    Returns the net's `l1`, which the tails index from. Inlined into each head, so a head
    is one compiled function.
    """
    l1 = hd[HD_L1]
    half = l1 // 2
    stride = L2 * l1 + L2 + L3 * L2 + L3 + L3 + 1  # `stack_stride`, from the width
    fc0b = HD_FTB + l1 + stack * stride + L2 * l1
    front = stack * (l1 + L2 * l1)  # this stack's products and packed first layer
    own = (ply * 2 + stm) * l1
    other = (ply * 2 + (1 - stm)) * l1
    # The clipped products: `clipped_product` takes whole vectors of lanes and the loop
    # after it any remainder, which is none at the widths the loader accepts.
    whole = (half // CLIP_LANES) * CLIP_LANES
    clipped_product(acc, own, hb, front, half, whole)
    clipped_product(acc, other, hb, front + half, half, whole)
    for i in range(whole, half):
        a = min(max(np.int32(acc[np.uint64(own + i)]), 0), FT_MAX)
        b = min(max(np.int32(acc[np.uint64(own + half + i)]), 0), FT_MAX)
        hb[np.uint64(front + i)] = (a * b) >> 9  # type: ignore[operator]  # at most 127
        a = min(max(np.int32(acc[np.uint64(other + i)]), 0), FT_MAX)
        b = min(max(np.int32(acc[np.uint64(other + half + i)]), 0), FT_MAX)
        hb[np.uint64(front + half + i)] = (a * b) >> 9  # type: ignore[operator]
    first_layer(hb[front:], hd, fc0b, xs, l1)  # one of the three forms above
    return int(l1)


def make_heads(stacked: bool) -> tuple[Any, Any]:
    """The plain head and the dual head, for a net of one stack or of several.

    `stacked` is a closure constant, so the choice is made when the head compiles and not
    when it runs: the one-stack build folds the stack to 0 and lays out the arithmetic the
    single head always had, and the stacked build reads the count and the band. One pair
    compiles a process, whichever `choose_head` binds, so the init budget never carries
    the other and neither does the disk cache, which hashes a closure's cells into its key.
    """

    @jit()
    def head_plain(
        stm: int, ply: int, men: int, hd: Int32s, acc: Int16s, xs: Int32s, hb: Int8s
    ) -> int:
        """The raw output at OUTPUT_SCALE from the accumulators of `ply`, side to move's view.

        The head `docs/weights/README.md` opens with. Both matrix layers run in bytes: after
        `head_front`, `hidden_clip` turns the first layer's 32 sums, shifted and clipped to
        0 to 127, into the 32 bytes at `h2`, and the second layer's weights follow those;
        `second_layer` sums that region into int32 the same way. The output layer is 32
        multiply-adds over the second layer's sums, each shifted and clipped as it is read,
        and stays a loop.
        """
        stack = stack_of(hd, men) if stacked else 0
        stacks = int(hd[np.uint64(HD_STACKS)]) if stacked else 1
        l1 = head_front(stm, ply, stack, hd, acc, xs, hb)
        stride = L2 * l1 + L2 + L3 * L2 + L3 + L3 + 1  # `stack_stride`, from the width
        fc1b = HD_FTB + l1 + stack * stride + L2 * l1 + L2 + L3 * L2
        outw = fc1b + L3
        outb = outw + L3
        # This stack's second-layer region: the N first-layer regions, then its own.
        h2 = stacks * (l1 + L2 * l1) + stack * (L2 + L3 * L2)
        hidden_clip(hb, h2, xs, l1)  # the 32 sums to the 32 bytes at `h2`, one of two forms
        # The same call shape at width L2: the byte region at `h2` and the scratch from
        # `l1`, so the sums land in `xs[l1 + L2 + j]`, the slots `x2` reads.
        second_layer(hb[h2:], hd, fc1b, xs[l1:], L2)
        x2 = xs[l1 + L2 : l1 + L2 + L3]
        out = hd[outw : outw + L3]
        # Each sum is shifted and clipped to 0 to HIDDEN_MAX as the output reads it, so no
        # byte is stored back. The total fits int32 at any width the compiler picks: L3
        # products of HIDDEN_MAX by the largest output weight, plus the bias, is the
        # loader's `output_bound`, 519,148 for the shipped net against 2,147,483,647.
        total = np.int32(hd[np.uint64(outb)])
        for i in range(L3):
            total += out[np.uint64(i)] * np.int32(min(max(x2[np.uint64(i)] >> 6, 0), HIDDEN_MAX))
        return int(total)

    @jit()
    def head_dual(
        stm: int, ply: int, men: int, hd: Int32s, acc: Int16s, xs: Int32s, hb: Int8s
    ) -> int:
        """The dual-activation head's raw output: the shared front, then `dual_tail`."""
        stack = stack_of(hd, men) if stacked else 0
        stacks = int(hd[np.uint64(HD_STACKS)]) if stacked else 1
        l1 = head_front(stm, ply, stack, hd, acc, xs, hb)
        return int(dual_tail(hd, xs, hb, l1, stack, stacks))

    return head_plain, head_dual


head_plain, head_dual = make_heads(False)
head_plain_stacked, head_dual_stacked = make_heads(True)
HEADS: dict[tuple[str, bool], Any] = {
    (PLAIN_HEAD, False): head_plain,
    (DUAL_HEAD, False): head_dual,
    (PLAIN_HEAD, True): head_plain_stacked,
    (DUAL_HEAD, True): head_dual_stacked,
}
HEAD = PLAIN_HEAD  # the head this process runs, for the init line
STACKED = False  # and whether it is the build that reads a net of several stacks
# Resolved into `evaluate_net` when it compiles, like the layer forms above: one head a
# process, bound by `choose_head` from the net's metadata before the warm-up. A branch on
# the net's flag inside one `head` would compile `dual_tail` as its callee in every
# process, 1.3 s of init a plain net never uses (kernel lane, 2026-09-09).
head: Any = head_plain


@jit(internal=True)
def evaluate_net(st: Ints, men: int, hd: Int32s, acc: Int16s, xs: Int32s, hb: Int8s) -> int:
    """Centipawns from the side to move's view, from the accumulators of the position's ply.

    `men` is the pieces on the board, which chooses the head stack the position reads: the
    search takes the popcount of the occupancy bitboard, and a net of one stack reads its
    one stack from any count. An int and not the bitboards, because numba passes an array
    as its pointer, shape and strides and this call is made at every leaf.
    """
    raw = int(head(st[STM], st[PLY], men, hd, acc, xs, hb))
    # The scale is the slot after the 64 bucket bases, wherever the stacks leave them.
    scale = int(hd[np.uint64(int(hd[np.uint64(HD_BASES)]) + 64)])
    return (raw * scale + OUTPUT_SCALE // 2) >> 13


WARMED: dict[str, Any] = {
    "nnue.refresh": refresh,
    "nnue.refresh_one": refresh_one,
    "nnue.refresh_cached": refresh_cached,
    # `nn_make` is the tests' and the benches' entry and compiles on its first call; the
    # search's `nn_make_s` is in `search.WARMED`.
    "nnue.nn_null": nn_null,
    "nnue.head": head,
    "nnue.evaluate_net": evaluate_net,
}


def choose_head(kind: str, stacked: bool = False) -> None:
    """Bind `head` to the net's head and stack count, before anything compiles it.

    `agent.py` calls this right after the net loads and before the warm-up, which is
    where `evaluate_net` and the search compile `head` in. numba resolves a module
    global at compile time, so a change after that would leave the compiled search on
    the old head: the call refuses it. numba's disk cache (`kernel.CACHE`, the local
    tools only) keys on the source, not on this binding; the tools snapshot each side
    into its own copy, so a snapshot's cache is built with its own net, and the platform
    compiles fresh every game.
    """
    global HEAD, STACKED, head
    if (kind, stacked) not in HEADS:
        raise ValueError(f"no head named {kind}")
    if (kind, stacked) == (HEAD, STACKED):
        return
    compiled: Any = evaluate_net  # the dispatcher; its signatures say if it has
    if compiled.signatures or head.signatures:
        raise RuntimeError("the head was chosen after it compiled; choose it before the warm-up")
    HEAD = kind
    STACKED = stacked
    head = HEADS[(kind, stacked)]
    WARMED["nnue.head"] = head


def dummy_cache() -> tuple[Int16s, Bitboards]:
    """The cache's arrays where there is no net: one slot each, never read."""
    return np.zeros(1, dtype=np.int16), np.zeros(KEY_WORDS, dtype=np.uint64)


def dummy_arrays() -> tuple[Int16s, Int32s, Int16s, Int32s, Int8s]:
    """Arrays of the kernels' types that mean no net: `hd` has one slot.

    Cut to a cache line like the net's own, so the four arrays the hand-written loads
    take hold their contract even in a build that loaded no net and never reaches them.
    """
    return (
        aligned(1, np.int16, rows=1),
        np.zeros(1, dtype=np.int32),
        aligned(1, np.int16),
        aligned(1, np.int32),
        aligned(1, np.int8),
    )


class Net:
    """A loaded net: the kernels' arrays and the metadata of the export."""

    def __init__(self, arrays: dict[str, np.ndarray], metadata: dict[str, str]) -> None:
        self.metadata = metadata
        l1 = int(metadata["l1"])
        # One accumulator has to be a whole number of cache lines. The arrays below start
        # on one, and the offsets the head and the fused update take into them are all
        # multiples of the width, so this is what carries the base's alignment to the
        # loads that claim VECTOR_ALIGN; a net of another width would fault them, and
        # `load` reports it as a net that did not load (#531).
        if (l1 * np.dtype(np.int16).itemsize) % kernel.ALIGNMENT != 0:
            raise ValueError(f"a first layer of {l1} lanes is not whole cache lines")
        if int(metadata["l2"]) != L2 or int(metadata["l3"]) != L3:
            raise ValueError("the head's widths are not the kernel's")
        # The head the file carries: the nets before the flag existed name none.
        head_kind = metadata.get("head", PLAIN_HEAD)
        if head_kind not in (PLAIN_HEAD, DUAL_HEAD):
            raise ValueError(f"no head named {head_kind}")
        dual = head_kind == DUAL_HEAD
        self.dual = dual
        # `skip_clamp_raw`: the trainer's clamp on the dual head's halved linear skip, in
        # raw units; absent or 0 is unbounded, which is every net shipped before this key.
        skip_clamp = int(metadata.get("skip_clamp_raw", "0"))
        if skip_clamp < 0:
            raise ValueError(f"skip_clamp_raw of {skip_clamp} is negative")
        if skip_clamp != 0 and not dual:
            raise ValueError("skip_clamp_raw is a dual head key; this net's head is not dual")
        # The head stacks the file carries: absent or 1 is the single head, in the shapes
        # and at the offsets it had before stacks existed.
        stacks = int(metadata.get("output_buckets", 1))
        if not 1 <= stacks <= STACKS_MAX:
            raise ValueError(f"{stacks} head stacks is not 1 to {STACKS_MAX}")
        self.stacks = stacks
        buckets = int(metadata["king_buckets"])
        table = king_bucket_table(buckets)
        mirrors = king_mirror_table(buckets)
        # Every tensor's exact shape, before anything is read from it: a transformer with
        # one column would broadcast through the lane bound below and send refresh_one
        # past the end of its rows (Codex's int16 review, 2026-09-07).
        # A net of N stacks carries a leading axis of N on every head tensor and
        # nothing else changes, so a flag and shapes that disagree are refused here.
        lead: tuple[int, ...] = () if stacks == 1 else (stacks,)
        expected: dict[str, tuple[int, ...]] = {
            "ft.weight": (buckets * FEATURES_PER_BUCKET, l1),
            "ft.bias": (l1,),
            "fc0.weight": (*lead, L2, l1),
            "fc0.bias": (*lead, L2),
            "fc1.weight": (*lead, L3, 2 * L2) if dual else (*lead, L3, L2),
            "fc1.bias": (*lead, L3),
            "out.weight": (*lead, DUAL_INPUTS) if dual else (*lead, L3),
            "out.bias": (*lead, 1),
        }
        for name, shape in expected.items():
            if tuple(arrays[name].shape) != shape:
                raise ValueError(f"{name} is {tuple(arrays[name].shape)}, not {shape}")
        # Each head tensor with its stack axis in front, so one loop writes every stack and
        # a one-stack net takes the same path with a single stack in it.
        by_stack = {
            name: arrays[name].reshape((stacks, *expected[name][len(lead) :]))
            for name in HEAD_TENSORS
        }
        ftb, fc0w, _, _, _, _, _, bases, size = head_offsets(l1, stacks)
        flag, douw, dual_size = dual_offsets(l1, stacks)
        stride = stack_stride(l1)
        hd = np.zeros(dual_size if dual else size, dtype=np.int32)
        hd[HD_L1] = l1
        hd[HD_STACKS] = stacks
        hd[HD_STRIDE] = stride
        hd[HD_BASES] = bases
        hd[HD_SKIP_CLAMP] = skip_clamp
        hd[ftb : ftb + l1] = arrays["ft.bias"]
        for s in range(stacks):
            fc0b, fc1w, fc1b, outw, outb = stack_slots(l1, s)
            at = fc0w + s * stride
            hd[at : at + L2 * l1] = by_stack["fc0.weight"][s].reshape(-1)
            hd[fc0b : fc0b + L2] = by_stack["fc0.bias"][s]
            hd[fc1b : fc1b + L3] = by_stack["fc1.bias"][s]
            if dual:
                # The dual head's `fc1` weights are twice the plain region and the kernel
                # reads them packed in `hb` alone, so they are left out of `hd`; its output
                # weights are 128 a stack and ride after the flag, where nothing else moves.
                hd[douw + s * DUAL_INPUTS : douw + (s + 1) * DUAL_INPUTS] = by_stack["out.weight"][
                    s
                ]
            else:
                hd[fc1w : fc1w + L3 * L2] = by_stack["fc1.weight"][s].reshape(-1)
                hd[outw : outw + L3] = by_stack["out.weight"][s]
            hd[outb] = by_stack["out.bias"][s][0]
        if dual:
            hd[flag] = 1
        for square in range(64):
            hd[bases + square] = table[square] * FEATURES_PER_BUCKET
            hd[bases + BASES_MIRRORS + square] = mirrors[square]
        scale = round(float(metadata.get("scale", str(CENTIPAWNS))))
        if not 50 <= scale <= 2000:
            raise ValueError(f"a centipawn scale of {scale} is not one the trainer fits")
        hd[bases + 64] = scale
        self.scale = scale
        self.l1 = l1
        self.king_buckets = buckets
        # The first layer, cut to a cache line: a row is `l1` int16, 512 bytes for the
        # shipped net, so once the base starts a line every row and every 32-byte load
        # the accumulate intrinsic makes starts one too (#531).
        weights = np.ascontiguousarray(arrays["ft.weight"], dtype=np.int16)
        self.ftw: Int16s = aligned(weights.size, np.int16, rows=int(weights.shape[0]))
        self.ftw[:] = weights
        # No position holds more than 32 pieces, so a lane's magnitude is at most the bias
        # plus its 32 largest weights over every feature; int16 lanes are exact when that
        # bound fits. The shipped net's bound is 12,969 against 32,767.
        magnitudes = np.abs(self.ftw.astype(np.int64))
        largest = -np.sort(-magnitudes, axis=0)[:32].sum(axis=0)
        self.lane_bound = int((np.abs(hd[ftb : ftb + l1].astype(np.int64)) + largest).max())
        if self.lane_bound > np.iinfo(np.int16).max:
            raise ValueError("an accumulator lane could exceed int16")
        self.hd: Int32s = hd
        # The accumulators are int16: the loader proves below that no position can push a
        # lane past int16, so the lanes, their update and the head's clip move half the bytes.
        self.acc: Int16s = aligned((kernel.MAX_PLY + 2) * 2 * l1, np.int16)
        # The king-move refresh's cache: one accumulator a colour and king key,
        # and the board each was built from.
        self.kacc: Int16s = np.zeros(KEY_SLOTS * l1, dtype=np.int16)
        self.kbb: Bitboards = np.zeros(KEY_SLOTS * KEY_WORDS, dtype=np.uint64)
        # The scratch: the head's inputs, then each layer's sums. The dual head's
        # second layer reads 64 bytes, so its sums sit that much further along. Cut to a
        # cache line: the packed layers store their sums here and the hidden clip loads
        # them back, both a whole vector at a time (#531).
        self.xs: Int32s = aligned(l1 + (2 * L2 if dual else L2) + L3, np.int32)
        # The head's byte half: l1 bytes for the clipped products, then the first layer's
        # int8 weights in `first_layer`'s packed order. Its packed instruction saturates
        # the sum of two products at int16, so the largest input (255 * 255 >> 9 = 127)
        # times the largest weight, twice, has to fit; the shipped net's bound is 32,258
        # against 32,767, and a weight of -128 would give 32,512.
        fc0 = arrays["fc0.weight"]
        if l1 % WEIGHT_BLOCK != 0:
            raise ValueError("the first layer's width is not a multiple of four")
        if fc0.min() < np.iinfo(np.int8).min or fc0.max() > np.iinfo(np.int8).max:
            raise ValueError("the first layer's weights do not fit int8")
        largest_input = (FT_MAX * FT_MAX) >> 9
        self.pair_bound = 2 * largest_input * int(np.abs(fc0.astype(np.int64)).max())
        if self.pair_bound > np.iinfo(np.int16).max:
            raise ValueError("a pair of first-layer products could saturate int16")
        blocks = l1 // WEIGHT_BLOCK
        packed = np.stack(
            [
                by_stack["fc0.weight"][s].reshape(L2, blocks, WEIGHT_BLOCK).transpose(1, 0, 2)
                for s in range(stacks)
            ]
        )
        # The second layer's region: its L2 inputs are the first layer's sums after the
        # shift and clip, so 0 to 127 like the first layer's, and its weights are int8
        # in every net the contract describes. Its pair bound is read the same way.
        fc1 = arrays["fc1.weight"]
        second_inputs = 2 * L2 if dual else L2
        if second_inputs % WEIGHT_BLOCK != 0:
            raise ValueError("the second layer's width is not a multiple of four")
        if fc1.min() < np.iinfo(np.int8).min or fc1.max() > np.iinfo(np.int8).max:
            raise ValueError("the second layer's weights do not fit int8")
        self.second_pair_bound = 2 * HIDDEN_MAX * int(np.abs(fc1.astype(np.int64)).max())
        if self.second_pair_bound > np.iinfo(np.int16).max:
            raise ValueError("a pair of second-layer products could saturate int16")
        second_blocks = second_inputs // WEIGHT_BLOCK
        packed1 = np.stack(
            [
                by_stack["fc1.weight"][s]
                .reshape(L3, second_blocks, WEIGHT_BLOCK)
                .transpose(1, 0, 2)
                for s in range(stacks)
            ]
        )
        # The output layer's total in `head`: L3 products of an input at most HIDDEN_MAX
        # by an output weight, plus the bias, has to fit int32. The shipped net's bound
        # is 519,148 against 2,147,483,647.
        largest_output = int(np.abs(arrays["out.weight"].astype(np.int64)).max())
        inputs = DUAL_INPUTS if dual else L3
        largest_bias = int(np.abs(arrays["out.bias"].astype(np.int64)).max())
        self.output_bound = inputs * HIDDEN_MAX * largest_output + largest_bias
        if dual:
            # The linear skip adds half the difference of two raw first-layer sums, so
            # the largest a first-layer sum can reach joins the bound.
            self.output_bound += l1 * largest_input * int(np.abs(fc0.astype(np.int64)).max())
            self.output_bound += int(np.abs(arrays["fc0.bias"].astype(np.int64)).max())
        if self.output_bound > np.iinfo(np.int32).max:
            raise ValueError("the output layer's total could exceed int32")
        # `hb` holds the N first-layer regions and then the N second-layer regions, each
        # at its own fixed stride, a region being the layer's inputs and its packed weights
        # straight after them: the one shape `_packed_layer` compiles. The head writes its
        # stack's inputs into its stack's region, and one stack is the layout above.
        first_stride = l1 + L2 * l1
        if dual:
            h2, _h3, size = dual_byte_offsets(l1, stacks)
            second_stride = 2 * L2 + L3 * 2 * L2 + 2 * L3
        else:
            h2, size = byte_offsets(l1, stacks)
            second_stride = L2 + L3 * L2
        # Cut to a cache line, like the scratch above: a packed layer reads its weights
        # from here a whole vector at a time (#531).
        self.hb: Int8s = aligned(size, np.int8)
        for s in range(stacks):
            at = s * first_stride
            self.hb[at + l1 : at + first_stride] = packed[s].reshape(-1)
            at = h2 + s * second_stride + second_inputs
            self.hb[at : at + L3 * second_inputs] = packed1[s].reshape(-1)

    @property
    def arrays(self) -> tuple[Int16s, Int32s, Int16s, Int32s, Int8s]:
        return self.ftw, self.hd, self.acc, self.xs, self.hb

    @property
    def cache(self) -> tuple[Int16s, Bitboards]:
        """The king-move refresh cache's arrays: the accumulators and their boards."""
        return self.kacc, self.kbb

    def raw(self, board: chess.Board) -> int:
        """The raw output for a python-chess position, through a full refresh at ply 0."""
        bb, sq, st, _, _ = kernel.new_arrays()
        kernel.load(board, bb, sq, st, kernel.KEYS)
        refresh(bb, self.ftw, self.hd, self.acc, 0)
        # The pieces on the board choose the head stack, `evaluate_net`'s rule in Python.
        men = int(bb[OCC_ALL]).bit_count()
        own_head = HEADS[(DUAL_HEAD if self.dual else PLAIN_HEAD, self.stacks > 1)]
        raw: int = own_head(int(st[STM]), 0, men, self.hd, self.acc, self.xs, self.hb)
        return int(raw)

    def evaluate(self, board: chess.Board) -> int:
        """Centipawns from the side to move's view, for a python-chess position."""
        return int((self.raw(board) * self.scale + OUTPUT_SCALE // 2) >> 13)


def check_positions(check_path: Path) -> list[dict[str, Any]]:
    """The check file's positions: a FEN and the integer output this net must reproduce."""
    rows: list[dict[str, Any]] = json.loads(check_path.read_text(encoding="utf-8"))["positions"]
    return rows


def check(net: Net, check_path: Path) -> int:
    """How many of the check file's positions the net gets wrong; 0 proves the loader.

    Zero is a proof only when there are positions to get wrong: a file holding an empty
    list also sums to zero, which is why `load` counts them before it trusts this (#464).
    """
    positions = check_positions(check_path)
    return sum(1 for row in positions if net.raw(chess.Board(row["fen"])) != int(row["raw_int"]))


def check_path_for(weights_path: Path) -> Path:
    """The check file beside a net, by the net's full name: `a.b.safetensors` keeps `a.b`.

    `Path.with_suffix` cuts a stem at its last dot, so a name holding one (the train
    lane's `ft0.1` exports) pointed at a file that does not exist and the net loaded
    unchecked (#436, 2026-09-11).
    """
    return weights_path.parent / f"{weights_path.stem}.check.json"


def load(weights_path: Path, require_check: bool = False) -> Net | None:
    """The net at the path, proven against its check file and compiled; None on any failure.

    A failure is reported through the returned None only: the agent then keeps its
    piece-square evaluation, which costs strength and never the game. The one exception
    is a check file with `require_check` that is missing, unreadable or empty, which
    raises: the shipped net loads that way, so a net that would skip its 250-position
    check fails the build's import where the validation log shows it, rather than playing
    unverified.
    """
    check_path = check_path_for(weights_path)
    if require_check:
        if not check_path.exists():
            raise FileNotFoundError(
                f"{weights_path.name} has no {check_path.name} beside it: a net ships "
                "checked (#436)"
            )
        # `check` counts the positions the net gets wrong, so an empty list of them counts
        # zero and the net would load with nothing proven (Codex 101, #464).
        try:
            positions = check_positions(check_path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{check_path.name} does not read as a check file: {error}") from error
        if not positions:
            raise ValueError(f"{check_path.name} holds no positions: a net ships checked (#464)")
    try:
        arrays, metadata = read_safetensors(weights_path)
        if metadata.get("contract") != CONTRACT:
            return None
        net = Net(arrays, metadata)
        net.raw(chess.Board())  # compiles refresh and the head, inside the init budget
        if check_path.exists() and check(net, check_path) != 0:
            return None
        return net
    except Exception:
        return None
