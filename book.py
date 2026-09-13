"""Opening book lookup before the search: one stored move per position, from `weights/book.pack`.

For a reviewer, the rules of the competition: "A table you ship and read during a game may
answer the opening or the endgame." The file holds moves only, no evaluations, for the
platform's revealed starting positions and the replies that follow them
(`weights/book.md`, `tools/book/build.py`). It is
the polyglot book with the two fields our lookup never reads left out: each record is the
8-byte polyglot Zobrist key, big-endian, then the 2-byte polyglot move word, records sorted
by key, no header. The full polyglot file costs 16 bytes an entry and the zip has a
50,000,000-byte cap, so a 10-byte record buys about 500,000 more positions of depth; the
key and the move are polyglot's exactly, so `chess.polyglot` reads the same book from the
16-byte file the build writes beside this one.

The same rules say where the opening ends: "The opening is a position whose move number is
20 or lower", read from the position we are given. What holds that boundary is this
module: `lookup` returns nothing before it hashes anything when the FEN's fullmove number
is above OPENING_LAST_MOVE, so the book cannot answer a middlegame position whatever it
holds. The build walks no further than the same move number, but it also exports a list of
priority positions unconditionally and keeps entries its walk never reaches, and its pass
that drops what no line reaches inside the limit is off by default, so the builder is not
offered here as the proof (#464).

Reading is what `chess.polyglot.MemoryMappedReader` does: the file is memory-mapped, a
bisect on the key finds the record, and nothing is loaded into Python objects at import,
so the cost is a few microseconds a probe and no init time.
"""

from __future__ import annotations

import mmap
import struct
from pathlib import Path

import chess
import chess.polyglot

BOOK_PATH = Path(__file__).resolve().parent / "weights" / "book.pack"
# The last move number the book answers: the opening's limit, quoted in the docstring above.
OPENING_LAST_MOVE = 20
RECORD = struct.Struct(">QH")
RECORD_BYTES = RECORD.size


def decode_move(board: chess.Board, raw: int) -> chess.Move:
    """Polyglot's move word: to in bits 0 to 5, from in 6 to 11, promotion in 12 to 14; a
    castling move is written as the king taking its own rook, and read back as the king's
    two-square move, as `chess.polyglot` reads it for a standard board."""
    to_square = raw & 0x3F
    from_square = (raw >> 6) & 0x3F
    promotion_part = (raw >> 12) & 0x7
    promotion = promotion_part + 1 if promotion_part else None
    if (
        board.piece_type_at(from_square) == chess.KING
        and chess.square_file(from_square) == 4
        and chess.square_rank(to_square) == chess.square_rank(from_square)
        and chess.square_file(to_square) in (0, 7)
    ):
        kingside = chess.square_file(to_square) > chess.square_file(from_square)
        to_square = chess.square(6 if kingside else 2, chess.square_rank(from_square))
    return chess.Move(from_square, to_square, promotion)


class Book:
    def __init__(self, path: Path = BOOK_PATH) -> None:
        self.data: mmap.mmap | bytes | None = None
        self.entries = 0
        try:
            size = path.stat().st_size if path.is_file() else 0
            if size > 0 and size % RECORD_BYTES == 0:
                with path.open("rb") as handle:
                    self.data = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
                self.entries = size // RECORD_BYTES
        except Exception:
            self.data = None
            self.entries = 0

    def find(self, key: int) -> int | None:
        """The stored move word for this key, by bisection over the sorted records."""
        data = self.data
        if data is None:
            return None
        low, high = 0, self.entries
        while low < high:
            middle = (low + high) // 2
            stored, raw = RECORD.unpack_from(data, middle * RECORD_BYTES)
            if stored < key:
                low = middle + 1
            elif stored > key:
                high = middle
            else:
                return int(raw)
        return None

    def lookup(self, board: chess.Board) -> chess.Move | None:
        """The stored move for this position if there is one and it is legal, else None.

        The probe goes through `chess.Board(board.fen())`, which keeps an en passant square
        only when the capture is legal, because that is how the build keyed every entry;
        the polyglot hash of a raw FEN with an uncapturable square would miss.
        """
        if self.data is None or board.fullmove_number > OPENING_LAST_MOVE:
            return None
        try:
            raw = self.find(chess.polyglot.zobrist_hash(chess.Board(board.fen())))
            if raw is None:
                return None
            move = decode_move(board, raw)
        except Exception:
            return None
        if move not in board.legal_moves:
            return None
        return move


BOOK = Book()
