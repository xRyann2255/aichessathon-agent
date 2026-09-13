"""Diagnostics written to stdout for the platform's per-game log.

The platform keeps the first 4 KB and the last 4 KB of everything the agent prints and
shows them beside the PGN after every rated game, with its own record of init time and
time per move. Only our team can read that log, so it is the one channel that reports what
the real container does: which core, what the clock charged against what we measured, how
much memory, and how deep the search got in the time it had.

This module prints one `diag init` line once the agent has finished importing, one
`diag m` line per move, a `diag sum` line and a `diag clk` line every eight moves and one
`diag end` line when the process exits. The init, sum, clk and end lines are
`diag <kind> key=value ...`. The move line is compact, around 60 bytes, so the 8 KB the
platform keeps holds a whole game: one token a field, a single letter naming it and no
`=`, as in

    diag m n37w v37 me2e4 d19 c+14 t1828 l60000 g2210 f1 i173,510,561 x0.94 kb

The `g` token and the `clk` line are the opponent-clock observer (#248): the platform
suspends the process while the opponent thinks and time.monotonic() runs on through the
suspension, so the gap from our return to the next call is their charge plus the runner's
turnaround. `g` is that gap in ms, and the `clk` line is what the gaps add up to: a point
estimate and an interval of the opponent's clock. Nothing reads either inside the agent;
they exist so the rated logs can calibrate the turnaround against the PGN's clocks before
any policy is allowed to read the estimate.

`parse_line` reads either shape back under the same field names, so `tools/platform_logs.py`
and `tools/postmortem.py` read a log of either. The init line carries three fixed benchmarks
(a Python loop, python-chess perft, a numpy matmul) so the same code run here gives the
speed ratio between this machine and the platform's core. What the move line no longer
carries a move at a time, the sum and end lines carry as running totals: the nodes, the
rate, the resident size and what the clock charged beyond what we measured.

Nothing here may raise into the agent: every probe is wrapped, and a probe that fails
leaves its field out.
"""

from __future__ import annotations

import atexit
import contextlib
import importlib
import os
import platform
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import chess

VERSION = 1
TAG = "diag"
MOVE_KIND = "m"

# The `k` field's letters, in the order they are printed, and what each says. Each one
# stands for a note that used to cost the log a line of its own on most moves.
MOVE_FLAGS = {
    "b": "book move played",
    "r": "book move refused",
    "t": "tablebase at the root",
    "f": "fallback move played",
    "e": "move probe failed",
}
FLAG_ORDER = "brtfe"
PROBE_FAILED = "move probe failed"

# The move line's one-letter prefixes and the field name each parses back to. The names
# are the ones the older `diag mv` line printed, so one reader reads both shapes.
MOVE_PREFIX = {
    "m": "bm",
    "v": "fm",
    "d": "d",
    "c": "cp",
    "t": "ms",
    "l": "left",
    "g": "opp_ms",
    "f": "fw",
    "i": "it",
    "x": "sc",
    "k": "k",
    "w": "wdl",
    "z": "dtz",
}

# perft 3 from the start position: 8,902 leaves through 421 generator calls, about 10 ms
# here. Depth 4 (197,281 leaves, about 200 ms) was the steadier timer, but under the
# 30 s init budget of 12 September the bench shares the core with the warm-up thread
# still compiling at the guard, and its tail read 0.48 s on a box and 1.1 s on the
# slowest platform host (kernel-lane probe 5); the rate is a probe, not a measurement.
PERFT_DEPTH = 3
PYLOOP_ITERATIONS = 200_000
MATMUL_SIZE = 256
MATMUL_ROUNDS = 10

_FIELD = re.compile(r'(\w+)=("[^"]*"|\S+)')
# A field's value is held to one line and to this many characters. An exception's text can
# run to dozens of lines (a numba TypingError is around 1 KB over 31 of them) and the
# platform keeps only the first 4 KB and the last 4 KB of the log, so three such notes
# filled the head and four the tail and evicted every move line; the reader fared no
# better, since an unterminated quote falls through `_FIELD`'s quoted branch and the
# `\S+` branch's quote strip eats the last real character (#443).
VALUE_MAX = 200
_NEWLINES = str.maketrans({"\n": " ", "\r": " ", "\t": " "})

_import_started = time.perf_counter()

# The opponent-clock observer's constants (#248). get_move is told neither the base clock
# nor the increment, so the contract's are assumed; a local match at another control reads
# a wrong clock here and a right gap, which is the field the calibration uses. The turnaround
# is what v33's gap probe read over the PGN's charge for the same move: a median of 163 ms,
# 136 to 202 ms from the 5th to the 95th percentile, over 922 moves in 18 rated games
# (#248's log, 2026-09-10). That probe stamped before its print and this one stamps after,
# so the first rated logs of this build recalibrate all three; until then the point estimate
# carries the old median and the interval a band widened a little past the old percentiles.
OPP_BASE_MS = 120_000
OPP_INCREMENT_MS = 500
OPP_OVERHEAD_MS = 163
OPP_OVERHEAD_LO_MS = 120
OPP_OVERHEAD_HI_MS = 250


@dataclass
class Tally:
    """What the move lines add up to, for the end line."""

    moves: int = 0
    nodes: int = 0
    ms_sum: int = 0
    ms_max: int = 0
    depth_sum: int = 0
    fallbacks: int = 0
    windows_failed: int = 0
    left_min: int | None = None
    left_prev: int | None = None
    ms_prev: int | None = None
    # What the clock charged beyond what we measured, less the control's increment, which
    # only the reader knows: the sum, the worst and the count of the moves behind them.
    ov_sum: int = 0
    ov_max: int = 0
    ov_n: int = 0
    rss_max: float = 0.0
    ended: bool = False
    lines: list[str] = field(default_factory=list)
    # The opponent-clock observer (#248): the stamp taken last before the previous return,
    # the one taken as the init line was printed, and what the gaps between them add up to.
    returned_prev: float | None = None
    ready_at: float | None = None
    first_ms: int | None = None
    gap_n: int = 0
    gap_sum: int = 0
    gap_max: int = 0
    gap_fast: int = 0
    gap_bad: int = 0
    opp_est: int = OPP_BASE_MS
    opp_lo: int = 0
    opp_hi: int = OPP_BASE_MS + OPP_INCREMENT_MS


_tally = Tally()


def single_line(text: str, limit: int = VALUE_MAX) -> str:
    """`text` on one line and no longer than `limit`, with a marker where it was cut.

    A newline, a carriage return or a tab becomes a space, so one record is one physical
    line whatever a caught exception's text held, and the tail past `limit` is dropped.
    """
    text = text.translate(_NEWLINES)
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def format_line(kind: str, fields: dict[str, object]) -> str:
    """`diag <kind> key=value ...`; a value with a space or a quote is double-quoted.

    Every value is held to one line and to VALUE_MAX characters first, so no field can
    split the record across physical lines or crowd the move lines out of the log (#443).
    """
    parts = [TAG, kind]
    for key, value in fields.items():
        if value is None:
            continue
        text = f"{value:.1f}" if isinstance(value, float) else str(value)
        text = single_line(text)
        if " " in text or '"' in text or text == "":
            text = '"' + text.replace('"', "'") + '"'
        parts.append(f"{key}={text}")
    return " ".join(parts)


def format_move_line(fields: dict[str, str]) -> str:
    """`diag m` and one token a field: its letter and its value, with no separator.

    No value a move line carries holds a space, so the line needs neither `=` nor quotes.
    """
    return " ".join([TAG, MOVE_KIND, *(f"{key}{value}" for key, value in fields.items())])


def parse_move_body(body: str) -> dict[str, str]:
    """A compact move line's tokens under the field names the older `mv` line printed.

    `n37w` splits back into the move number and the side to move, and the `k` letters give
    back the `fb` and `error` fields the reader used to take from the line itself.
    """
    fields: dict[str, str] = {}
    for token in body.split():
        prefix, value = token[:1], token[1:]
        if prefix == "n" and value:
            fields["n"] = value[:-1] if value[-1].isalpha() else value
            if value[-1].isalpha():
                fields["stm"] = value[-1]
        elif prefix in MOVE_PREFIX:
            fields[MOVE_PREFIX[prefix]] = value
    flags = fields.setdefault("k", "")
    if "f" in flags:
        fields["fb"] = "1"
    if "e" in flags:
        fields["error"] = PROBE_FAILED
    return fields


def parse_line(line: str) -> tuple[str, dict[str, str]] | None:
    """The kind and fields of a diag line, or None for any other line.

    A compact move line comes back as `mv` under the older line's field names, so a reader
    written against either shape reads both.
    """
    stripped = line.strip()
    if not stripped.startswith(TAG + " "):
        return None
    rest = stripped[len(TAG) + 1 :]
    kind, _, body = rest.partition(" ")
    if not kind:
        return None
    if kind == MOVE_KIND:
        return "mv", parse_move_body(body)
    fields = {}
    for key, value in _FIELD.findall(body):
        fields[key] = value[1:-1] if value.startswith('"') else value
    if kind == "mv":
        # The older line said fallback in `fb` and everything else in notes beside it.
        fields["k"] = "f" if fields.get("fb", "0") not in ("0", "") else ""
    return kind, fields


def flag_notes(flags: str, skip: str = "", wdl: str = "", dtz: str = "") -> list[str]:
    """What the `k` letters of a move line say, in the words the notes used to use.

    The tablebase's letter takes the line's verdict with it, as the note it replaced did:
    `tablebase at the root wdl=2 dtz=-13`. A WDL class is a verdict and not an estimate,
    so it is quoted and never averaged with anything.
    """
    notes: list[str] = []
    for letter in flags:
        if letter not in MOVE_FLAGS or letter in skip:
            continue
        text = MOVE_FLAGS[letter]
        if letter == "t" and wdl != "":
            text += f" wdl={wdl}" + (f" dtz={dtz}" if dtz != "" else "")
        notes.append(text)
    return notes


def _emit_raw(line: str) -> None:
    try:
        _tally.lines.append(line)
        print(line, flush=True)
    except Exception:
        pass


def _emit(kind: str, fields: dict[str, object]) -> None:
    with contextlib.suppress(Exception):
        _emit_raw(format_line(kind, fields))


def cpu_info() -> dict[str, object]:
    """The core, named without reading a file outside the agent's own directory and /tmp.

    `platform.machine()` is a `uname` field the interpreter already holds, so it starts no
    process; `platform.processor()` is the call to avoid, because on a Linux host it falls
    back to running `uname -p` in a subprocess. The vector units are on the init line
    already, under `isa=` and `first_layer=`, which `nnue.py` reads from LLVM's own view of
    the host inside this process, so the core's name is all that is wanted here; the model
    string, the clock speed, the cgroup limits and the huge-page settings went with the
    files they came from (#464).
    """
    info: dict[str, object] = {"cores": os.cpu_count(), "cpu": platform.machine() or "?"}
    getaffinity = getattr(os, "sched_getaffinity", None)
    if getaffinity is not None:
        with contextlib.suppress(OSError):
            info["aff"] = len(getaffinity(0))
    try:
        import shutil

        info["tmp_free_mb"] = shutil.disk_usage("/tmp").free // 1_000_000
    except OSError:
        pass
    return info


def codegen_key() -> tuple[str, str]:
    """numba's cache key for this host, as `codegen.magic_tuple()` sees it, and its features.

    The key is the target triple, LLVM's name for the host CPU and the first twelve hex
    digits of the SHA-1 of the feature string, joined by `|`: two hosts compile the same
    cache when the key matches, so a box whose key equals the platform's could build a
    shipped compile cache (#125), whose trigger (platform init at 75 s) is met at 80.6 s
    (#470, 2026-09-11). Beside it the features reported present, without the absent ones,
    for reading; the line's value cap may cut that list, and the digest is the check.
    """
    import hashlib

    from numba.core.registry import cpu_target

    triple, cpu, features = cpu_target.target_context.codegen().magic_tuple()
    digest = hashlib.sha1(str(features).encode()).hexdigest()[:12]
    present = ",".join(f[1:] for f in str(features).split(",") if f.startswith("+"))
    return f"{triple}|{cpu}|{digest}", present


def rss_mb() -> float | None:
    """Peak resident set size in megabytes, from the kernel's accounting for this process.

    `resource.getrusage` is a system call and reads no file, where the older reader took
    `VmRSS` from /proc/self/status, outside the directories this agent may read (#464).
    `ru_maxrss` is a high-water mark rather than the size right now, which is what both
    readers of this number want: the init line asks how close the import came to the 2 GB
    limit, and the move line keeps a maximum over the game anyway. It is kilobytes on
    Linux, where the platform runs, and bytes on macOS; `resource` is absent on Windows.
    """
    try:
        # Imported by name: the module is absent on Windows, where the loop's own machine
        # runs, and typed for Linux alone.
        resource: Any = importlib.import_module("resource")
        peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, AttributeError, OSError, ValueError):
        return None
    return peak / 1e6 if sys.platform == "darwin" else peak / 1000.0


def _perft(board: chess.Board, depth: int) -> int:
    if depth == 1:
        return board.legal_moves.count()
    nodes = 0
    for move in board.legal_moves:
        board.push(move)
        nodes += _perft(board, depth - 1)
        board.pop()
    return nodes


def bench() -> dict[str, float]:
    """Three fixed workloads, timed; the same code run locally gives the speed ratio."""
    out: dict[str, float] = {}
    started = time.perf_counter()
    total = 0
    for i in range(PYLOOP_ITERATIONS):
        total += i * i
    out["pyloop_ms"] = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    nodes = _perft(chess.Board(), PERFT_DEPTH)
    elapsed = time.perf_counter() - started
    out["perft_nps"] = nodes / elapsed if elapsed > 0 else 0.0

    try:
        import numpy as np

        matrix = np.full((MATMUL_SIZE, MATMUL_SIZE), 0.001, dtype=np.float32)
        matrix @ matrix
        started = time.perf_counter()
        for _ in range(MATMUL_ROUNDS):
            matrix = matrix @ matrix * 0.5
        elapsed = time.perf_counter() - started
        flops = 2.0 * MATMUL_SIZE**3 * MATMUL_ROUNDS
        out["np_gflops"] = flops / elapsed / 1e9 if elapsed > 0 else 0.0
    except Exception:
        pass
    return out


def _versions() -> str:
    names = ("chess", "numpy", "numba", "torch", "onnxruntime")
    loaded = []
    for name in names:
        module = sys.modules.get(name)
        if module is not None:
            loaded.append(f"{name}:{getattr(module, '__version__', '?')}")
    return ",".join(loaded)


def init_done(**extra: object) -> None:
    """Print the init line. Call it last in the agent's import, after every table is built."""
    fields: dict[str, object] = {"v": VERSION}
    # Registered here and not at import, so only the agent process prints an end line;
    # a tool that imports this module for its parser stays silent at exit.
    atexit.register(end)
    try:
        fields["import_ms"] = int((time.perf_counter() - _import_started) * 1000)
        fields["cpu_ms"] = int(time.process_time() * 1000)
        fields["py"] = platform.python_version()
        fields["os"] = sys.platform
        fields.update(cpu_info())
        with contextlib.suppress(Exception):
            fields["magic"], fields["feat"] = codegen_key()
        fields.update(bench())
        fields["rss_mb"] = rss_mb()
        fields["mods"] = _versions()
        fields.update(extra)
    except Exception:
        fields["error"] = "init probe failed"
    _emit("init", fields)
    _tally.ready_at = time.monotonic()


def observe_gap(started: float) -> int | None:
    """The gap from the previous return to this call in ms, folded into the opponent's clock.

    None on the first call: the opponent's first move, if there was one, was never
    bracketed by our stamps, so no gap is invented for it. What that call does know is the
    time from the init line to itself, which bounds their first charge from above and so
    their clock from below. The point estimate starts from the full base clock: exact when
    we moved first, and high by their first charge less the increment when they did.

    Each later gap moves the three readings by the increment less the charge it implies:
    the point estimate under the median turnaround, the upper bound under the largest and
    the lower bound under the smallest, so the interval widens by their difference each
    move. A gap under the smallest turnaround counts as fast (the runner was quicker than
    the band, or the opponent replied from a book), and one that implies a charge over what
    the upper bound could have paid counts as bad and turns `ok` off: the model is
    contradicted, and a consumer must not read the interval for the rest of the game.

    The stamps are time.monotonic(), the clock get_move reads at entry: CLOCK_MONOTONIC at
    nanosecond resolution on the platform's Linux, GetTickCount64 at a 15.6 ms tick on this
    Windows machine, so a local game against a fast opponent reads `g0` on every move and
    only a rated log calibrates anything.
    """
    prev = _tally.returned_prev
    if prev is None:
        if _tally.ready_at is not None:
            _tally.first_ms = max(0, round((started - _tally.ready_at) * 1000))
            reply = OPP_BASE_MS + OPP_INCREMENT_MS - _tally.first_ms
            _tally.opp_lo = min(OPP_BASE_MS, max(0, reply))
        return None
    gap = max(0, round((started - prev) * 1000))
    _tally.gap_n += 1
    _tally.gap_sum += gap
    _tally.gap_max = max(_tally.gap_max, gap)
    if gap < OPP_OVERHEAD_LO_MS:
        _tally.gap_fast += 1
    least = max(0, gap - OPP_OVERHEAD_HI_MS)
    if least > _tally.opp_hi + OPP_INCREMENT_MS:
        _tally.gap_bad += 1
    _tally.opp_est += OPP_INCREMENT_MS - max(0, gap - OPP_OVERHEAD_MS)
    _tally.opp_hi = max(0, _tally.opp_hi + OPP_INCREMENT_MS - least)
    most = max(0, gap - OPP_OVERHEAD_LO_MS)
    _tally.opp_lo = max(0, _tally.opp_lo + OPP_INCREMENT_MS - most)
    return gap


def clock_fields() -> dict[str, object]:
    """The observer's readout: the gaps counted, summed and at their largest, the bracket
    on the opponent's first move, the point estimate and the interval of their clock in
    ms, the fast and bad gap counts and whether the interval may be read at all."""
    return {
        "n": _tally.gap_n,
        "first": _tally.first_ms,
        "sum": _tally.gap_sum,
        "max": _tally.gap_max,
        "est": _tally.opp_est,
        "lo": _tally.opp_lo,
        "hi": _tally.opp_hi,
        "fast": _tally.gap_fast,
        "bad": _tally.gap_bad,
        "ok": 0 if _tally.gap_bad else 1,
    }


def record_move(
    *,
    fen: str,
    move: str,
    depth: int,
    nodes: int,
    score_cp: int,
    started: float,
    time_left_ms: int,
    windows_failed: int = 0,
    fallback: bool = False,
    iterations: list[int] | None = None,
    soft_scale: float = 0.0,
    flags: str = "",
    wdl: int | None = None,
    dtz: int | None = None,
) -> None:
    """Print the move line; `started` is the time.monotonic() read at the top of get_move.

    One token a field: `n` the move number with the side to move on it, `v` the FEN's own
    full-move number, `m` the move, `d` the depth, `c` the score in centipawns, `t` the
    milliseconds we measured, `l` the clock left, `g` the gap from our previous return to
    this call (the opponent's charge plus the runner's turnaround; absent on the first
    call), `f` the aspiration windows that failed,
    `i` the ms of the last three completed iterations, `x` the product of the soft scalers
    in force when the move ended, `k` the letters of `flags` and MOVE_FLAGS, and on a
    tablebase move `w` the WDL class and `z` the DTZ. A field with nothing to say is left
    out, so a book move is `diag m n3w v3 mg1f3 d0 c+0 t1 l118500 kb`.

    `v` is not our move count: the clock's constants are keyed by the full-move number and
    a curated opening starts at one of its own, so a reader that counted our moves read the
    wrong end of the game. It parses back as the `fm` the older line printed.

    `i` and `x` together give the time manager's growth ratio per move, which nothing else
    in the log carries. The nodes, the resident size and what the clock charged are counted
    here and printed on the `sum` and `end` lines, which cost the log a line every eight
    moves instead of one on every move.

    The last statement stamps the return for the next call's `g`. get_move returns the
    move as soon as this returns, so nothing of ours but that return sits inside the gap.
    """
    try:
        ms = int((time.monotonic() - started) * 1000)
        gap = observe_gap(started)
        parts = fen.split()
        fields: dict[str, str] = {
            "n": f"{_tally.moves + 1}{parts[1] if len(parts) > 1 else '?'}",
        }
        # The FEN's own full-move number, four bytes that our move count cannot supply
        # from a curated root; it reads back as `fm`.
        if len(parts) > 5 and parts[5].isdigit():
            fields["v"] = parts[5]
        fields.update(
            {
                "m": move,
                "d": str(depth),
                "c": f"{score_cp:+d}",
                "t": str(ms),
                "l": str(time_left_ms),
            }
        )
        if gap is not None:
            fields["g"] = str(gap)
        # Aspiration windows this move that fell outside their bounds and were searched
        # again. A wasted re-search costs whole seconds of the clock, so it is the second
        # readout beside the depth; the field is left out when there were none.
        if windows_failed:
            fields["f"] = str(windows_failed)
            _tally.windows_failed += windows_failed
        if iterations:
            fields["i"] = ",".join(str(each) for each in iterations[-3:])
        if soft_scale > 0:
            fields["x"] = f"{soft_scale:.2f}"
        letters = flags + ("f" if fallback else "")
        letters = "".join(letter for letter in FLAG_ORDER if letter in letters)
        if letters:
            fields["k"] = letters
        # The tablebase's verdict, which no other field on the line can stand in for: the
        # score of a table-decided move says nothing about how the win is reached.
        if "t" in letters and wdl is not None:
            fields["w"] = str(wdl)
            if dtz is not None:
                fields["z"] = str(dtz)
        rss = rss_mb()
        if rss is not None:
            _tally.rss_max = max(_tally.rss_max, rss)
        # What the clock charged for the previous move beyond what we measured for it,
        # less the control's increment, which only the reader knows: the previous clock
        # less this one, less the previous move's own ms.
        if _tally.left_prev is not None and _tally.ms_prev is not None:
            over = _tally.left_prev - time_left_ms - _tally.ms_prev
            _tally.ov_max = over if _tally.ov_n == 0 else max(_tally.ov_max, over)
            _tally.ov_sum += over
            _tally.ov_n += 1
        if fallback:
            _tally.fallbacks += 1
        _tally.moves += 1
        _tally.nodes += nodes
        _tally.ms_sum += ms
        _tally.ms_max = max(_tally.ms_max, ms)
        _tally.depth_sum += depth
        _tally.left_min = (
            time_left_ms if _tally.left_min is None else min(_tally.left_min, time_left_ms)
        )
        _tally.left_prev = time_left_ms
        _tally.ms_prev = ms
    except Exception:
        fields = {"k": "e"}
    _emit_raw(format_move_line(fields))
    summary_line()
    _tally.returned_prev = time.monotonic()


def note(message: str) -> None:
    """A free-text line for anything unusual: a fallback taken, an exception caught."""
    _emit("note", {"msg": message})


SUM_EVERY = 8


def summary_fields() -> dict[str, object]:
    """The game's running totals: the end line's fields, less the two the exit adds."""
    moves = _tally.moves
    return {
        "moves": moves,
        "nodes": _tally.nodes,
        "nps": int(_tally.nodes * 1000 / _tally.ms_sum) if _tally.ms_sum > 0 else 0,
        "ms_sum": _tally.ms_sum,
        "ms_max": _tally.ms_max,
        "d_mean": _tally.depth_sum / moves if moves else 0.0,
        "left_min": _tally.left_min,
        "fb": _tally.fallbacks,
        "fw": _tally.windows_failed,
        "rss_max": _tally.rss_max or None,
        # The overhead the move lines no longer carry: the reader adds the control's
        # increment to `ov_sum / ov_n` and to `ov_max` to get what the runner charged.
        "ov_sum": _tally.ov_sum if _tally.ov_n else None,
        "ov_max": _tally.ov_max if _tally.ov_n else None,
        "ov_n": _tally.ov_n or None,
    }


def summary_line() -> None:
    """A `sum` line every SUM_EVERY moves: the platform kills the process at the game's
    end, so the atexit end line never reaches its log (round 75, 2026-09-08), and the
    tail's 4 KB holds the last such line instead."""
    if _tally.moves % SUM_EVERY == 0:
        _emit("clk", clock_fields())
        _emit("sum", summary_fields())


def end() -> None:
    """Print the end line once; registered with atexit and safe to call early."""
    if _tally.ended:
        return
    _tally.ended = True
    fields = summary_fields()
    fields["wall_s"] = int(time.perf_counter() - _import_started)
    # The observer's line goes ahead of the end line, which stays the last line printed.
    _emit("clk", clock_fields())
    _emit("end", fields)
