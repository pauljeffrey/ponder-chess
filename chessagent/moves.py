"""Move vocabulary shared by the policy head, the line decoder and the data pipeline.

Everything lives in the *root frame*: the side to move at the root position is always
white and moves "up" the board. Black-to-move positions are mirrored before encoding
(see board.py). A move is a (from_square, to_square, promotion) triple; castling is the
king's two-square move (e1g1), exactly like standard UCI.
"""
from __future__ import annotations

import chess

# promotion slots: 0 = none, 1..4 = knight, bishop, rook, queen
PROMO_PIECES = [None, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
PROMO_SLOT = {p: i for i, p in enumerate(PROMO_PIECES)}

_QUEEN_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
_KNIGHT_JUMPS = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]


def _targets(sq: int) -> list[int]:
    f, r = chess.square_file(sq), chess.square_rank(sq)
    out = []
    for df, dr in _QUEEN_DIRS:
        for k in range(1, 8):
            nf, nr = f + df * k, r + dr * k
            if 0 <= nf < 8 and 0 <= nr < 8:
                out.append(chess.square(nf, nr))
    for df, dr in _KNIGHT_JUMPS:
        nf, nr = f + df, r + dr
        if 0 <= nf < 8 and 0 <= nr < 8:
            out.append(chess.square(nf, nr))
    return out


def _build() -> list[tuple[int, int, int]]:
    vocab: list[tuple[int, int, int]] = []
    for frm in range(64):
        for to in _targets(frm):
            vocab.append((frm, to, 0))
    # promotions: white pawns 7th->8th rank, black (opponent, in the root frame) 2nd->1st
    for frm in range(64):
        r = chess.square_rank(frm)
        if r not in (1, 6):
            continue
        step = 1 if r == 6 else -1
        f = chess.square_file(frm)
        for df in (-1, 0, 1):
            nf = f + df
            if 0 <= nf < 8:
                to = chess.square(nf, r + step)
                for slot in range(1, 5):
                    vocab.append((frm, to, slot))
    return vocab


MOVE_VOCAB: list[tuple[int, int, int]] = _build()
VOCAB_SIZE = len(MOVE_VOCAB)
MOVE_TO_IDX: dict[tuple[int, int, int], int] = {m: i for i, m in enumerate(MOVE_VOCAB)}

# parallel arrays for gathering pointer logits
FROM_IDX = [m[0] for m in MOVE_VOCAB]
TO_IDX = [m[1] for m in MOVE_VOCAB]
PROMO_IDX = [m[2] for m in MOVE_VOCAB]


def move_to_index(move: chess.Move) -> int:
    """Index of a root-frame python-chess move (standard castling convention)."""
    slot = PROMO_SLOT[move.promotion]
    return MOVE_TO_IDX[(move.from_square, move.to_square, slot)]


def index_to_move(idx: int) -> chess.Move:
    frm, to, slot = MOVE_VOCAB[idx]
    return chess.Move(frm, to, promotion=PROMO_PIECES[slot])


def mirror_move(move: chess.Move) -> chess.Move:
    """Flip a move vertically (a1 <-> a8). Used to go between real and root frames."""
    return chess.Move(move.from_square ^ 56, move.to_square ^ 56, promotion=move.promotion)


def legal_mask_indices(board: chess.Board) -> list[int]:
    """Vocab indices of all legal moves of a root-frame board (any side to move)."""
    return [move_to_index(m) for m in board.legal_moves]
