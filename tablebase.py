"""Root Syzygy probes for the full 3/4-man set and a selected 5-man subset.

WDL decides the allowed result class. Within that class, known distance to zeroing
ranks progress and game history breaks distance ties. Missing DTZ leaves a reversible
move's distance unknown, so search chooses within the WDL class when needed.

WDL assumes a fresh fifty-move clock; without DTZ it cannot certify conversion before
a later claim. Immediate claims and known distances are checked. Missing WDL, including
a promotion dependency outside the subset, leaves the root to search: the shipped
five-man set is 30 of the 145 WDL tables, 2 of them with DTZ, so a five-man root is probed
only when its own table and every child's are present, and a skipped material is recorded
once for the log.

Opening the set finds the files and builds a table object for each. It maps no data:
python-chess memory-maps a table at its first probe, so the open costs milliseconds and a
probe pays for the one or two files its own material needs. The earlier reading of about
two seconds was two games' move times, whose roots had six pieces or no shipped table and
whose searches reached different depths (Codex 100, #464). The module's `TABLEBASE` counts
its files at import and opens the set on the first root it may probe; `agent.py` opens it
during the import instead when the import has been quick (`eager_at`), which keeps even
those milliseconds off a move's clock.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.syzygy

# For a reviewer, the rules of the competition: "A table you ship and read during a game may
# answer the opening or the endgame." The endgame they define is at most seven pieces; these
# tables answer at most five, read from the FEN we are given.
MAX_PIECES = 5
FIFTY_MOVE_PLIES = 100
# The agent opens the set at import when the import has taken under this many seconds by
# the time the tables are set up (v66 read 74.6 s of the 90 on the slow validation host,
# against the 75 s hold), else on the first root probe.
EAGER_UNDER_S = 55.0
# "1" forces the deferred open whatever the import took, "0" the eager one; for tests.
LAZY_ENV = "AICHESSATHON_TABLEBASE_LAZY"


def eager_at(elapsed_s: float) -> bool:
    """Whether an import that has taken `elapsed_s` so far opens the set now."""
    forced = os.environ.get(LAZY_ENV)
    if forced == "1":
        return False
    if forced == "0":
        return True
    return elapsed_s < EAGER_UNDER_S


def transposition_key(board: chess.Board) -> object:
    """python-chess's key for repetition: pieces, side to move, castling, en passant."""
    return board._transposition_key()


@dataclass(frozen=True)
class Verdict:
    """One root move's result, optional child DTZ, root distance and game history."""

    move: chess.Move
    wdl: int
    dtz: int | None
    distance: int | None
    repeats: bool

    @property
    def rank(self) -> tuple[int, int, int]:
        """Prefer the WDL class, then progress, then a position the game has not seen."""
        if self.distance is None:
            raise ValueError("DTZ ranking requires a known root distance")
        progress = -self.distance if self.wdl > 0 else self.distance
        return self.wdl, progress, 0 if self.repeats else 1


class Tablebase:
    """The shipped Syzygy set, or an empty one when the files are absent.

    `eager` opens the files at construction; otherwise the first probe that passes the
    cheap checks opens them, once, and `open_ms` keeps what that probe paid for the log.
    """

    def __init__(self, directory: Path, eager: bool = True) -> None:
        self.directory = directory
        self.tables = len(list(directory.glob("*.rtbw"))) if directory.is_dir() else 0
        self._tablebase: chess.syzygy.Tablebase | None = None
        self.opened = False  # whether open_tables has run; an empty directory opens nothing
        self.open_ms: int | None = None  # the deferred open's cost, when it ran
        self.open_noted = False  # whether the diag log has the deferred open's line
        # Materials met this game whose WDL table is not shipped, in the order met;
        # `reported` is how many of them the diag log has already named.
        self.missing: list[str] = []
        self.reported = 0
        if eager:
            self.open_tables()

    def open_tables(self, deferred: bool = False) -> None:
        """Find the files and build their table objects, once; a later call is free.

        `opened` is set after the open returns, not before it: an error on the way in used
        to leave the flag true with no tables behind it, which retired the whole set for the
        game on one failure (Codex 100, #464). The caller that catches the error decides
        whether to try again.
        """
        if self.opened:
            return
        if self.tables == 0:
            self.opened = True
            return
        started = time.perf_counter()
        self._tablebase = chess.syzygy.open_tablebase(str(self.directory))
        self.opened = True
        if deferred:
            self.open_ms = int((time.perf_counter() - started) * 1000)

    def _ready(self) -> chess.syzygy.Tablebase | None:
        """The open set, opened here on the first probe of the deferred path."""
        if not self.opened:
            self.open_tables(deferred=True)
        return self._tablebase

    def has_table(self, board: chess.Board) -> bool:
        """Whether the WDL table for this material is shipped (bare kings need none)."""
        tables = self._ready()
        if tables is None:
            return False
        if board.kings == board.occupied:
            return True
        return chess.syzygy.calc_key(board) in tables.wdl

    def _skip(self, board: chess.Board) -> None:
        """Record a material left to the search because its table is not shipped."""
        key = chess.syzygy.calc_key(board)
        if key not in self.missing:
            self.missing.append(key)

    def unreported(self) -> list[str]:
        """The skipped materials not yet named in the log, marked as named."""
        fresh = self.missing[self.reported :]
        self.reported = len(self.missing)
        return fresh

    def covers(self, board: chess.Board) -> bool:
        """Whether this root is small enough to probe, uncastled, and its table shipped."""
        # The cheap checks first: a full board never opens the deferred set.
        if self.tables == 0 or len(board.piece_map()) > MAX_PIECES or board.castling_rights:
            return False
        if self._ready() is None:
            return False
        if self.has_table(board):
            return True
        self._skip(board)
        return False

    def verdicts(
        self,
        board: chess.Board,
        seen: Collection[object] = (),
        key: Callable[[chess.Board], object] = transposition_key,
    ) -> list[Verdict]:
        """Probe legal moves using the same key function that produced the game history.

        A child whose table is not shipped leaves the whole root to the search, since a
        move of unknown result may be the best one; the list is then empty.
        """
        tables = self._ready()
        assert tables is not None
        verdicts = []
        for move in board.legal_moves:
            zeroing = board.is_zeroing(move)
            after = board.copy(stack=False)
            after.push(move)
            repeats = key(after) in seen
            if after.is_checkmate():
                verdicts.append(Verdict(move, 2, 0, 0, repeats))
                continue
            if not self.has_table(after):
                self._skip(after)
                return []
            wdl = -tables.probe_wdl(after)
            dtz: int | None
            try:
                dtz = -tables.probe_dtz(after)
            except chess.syzygy.MissingTableError:
                dtz = None
            if wdl == 2 and (
                after.can_claim_fifty_moves()
                or (dtz is not None and after.halfmove_clock + abs(dtz) >= FIFTY_MOVE_PLIES)
            ):
                wdl = 1
            # A zeroing root move has known distance even if the child's DTZ is absent.
            distance = 1 if zeroing else None if dtz is None else abs(dtz) + 1
            verdicts.append(Verdict(move, wdl, dtz, distance, repeats))
        return verdicts

    def best(
        self,
        board: chess.Board,
        seen: Collection[object] = (),
        key: Callable[[chess.Board], object] = transposition_key,
    ) -> Verdict | None:
        """Rank known distances inside the best WDL class, with history breaking ties.

        A known winning distance can be used even when another winning move lacks DTZ.
        Unknown losing distances go to search instead of being guessed shorter.
        """
        if not self.covers(board):
            return None
        verdicts = self.verdicts(board, seen, key)
        if not verdicts:
            return None
        best_wdl = max(verdict.wdl for verdict in verdicts)
        best = [verdict for verdict in verdicts if verdict.wdl == best_wdl]
        known = [verdict for verdict in best if verdict.distance is not None]
        if best_wdl == 0 or not known or (best_wdl < 0 and len(known) != len(best)):
            return None
        return max(known, key=lambda verdict: verdict.rank)

    def best_moves(self, board: chess.Board) -> list[chess.Move]:
        """Restrict search and fallback to the highest available WDL class."""
        if not self.covers(board):
            return []
        verdicts = self.verdicts(board)
        if not verdicts:
            return []
        best_wdl = max(verdict.wdl for verdict in verdicts)
        return [verdict.move for verdict in verdicts if verdict.wdl == best_wdl]


# Deferred here; agent.py opens it at import when the import has been quick (eager_at).
TABLEBASE = Tablebase(Path(__file__).resolve().parent / "weights" / "syzygy", eager=False)
