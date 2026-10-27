"""Board <-> token encoding.

Token ids per square (13): 0 empty, 1..6 mover's P N B R Q K, 7..12 opponent's P N B R Q K.
"Mover" is the side to move at the root; black-to-move boards are mirrored so the mover is
always white. Extras: castling bitmask (mover K, mover Q, opp K, opp Q), en-passant file
(0..7, 8 = none) and the halfmove clock (clipped to 100).
"""
from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


@dataclass
class Encoded:
    squares: np.ndarray  # uint8 [64]
    castle: int
    ep: int
    halfmove: int


def to_root_frame(board: chess.Board) -> tuple[chess.Board, bool]:
    """Return (board with white to move, flipped?)."""
    if board.turn == chess.WHITE:
        return board, False
    return board.mirror(), True


def encode_root_board(board: chess.Board) -> Encoded:
    """Encode a board whose side to move is white (call to_root_frame first)."""
    sq = np.zeros(64, dtype=np.uint8)
    for s, piece in board.piece_map().items():
        base = PIECE_ORDER.index(piece.piece_type) + 1
        sq[s] = base if piece.color == chess.WHITE else base + 6
    castle = (
        int(board.has_kingside_castling_rights(chess.WHITE))
        | int(board.has_queenside_castling_rights(chess.WHITE)) << 1
        | int(board.has_kingside_castling_rights(chess.BLACK)) << 2
        | int(board.has_queenside_castling_rights(chess.BLACK)) << 3
    )
    ep = chess.square_file(board.ep_square) if board.ep_square is not None else 8
    return Encoded(sq, castle, ep, min(board.halfmove_clock, 100))


def encode_board(board: chess.Board) -> tuple[Encoded, bool]:
    root, flipped = to_root_frame(board)
    return encode_root_board(root), flipped


def board_from_tokens(squares, castle: int, ep: int, halfmove: int = 0) -> chess.Board:
    """Rebuild a white-to-move root-frame board from stored tokens (used by RL rewards)."""
    board = chess.Board(None)
    for s in range(64):
        t = int(squares[s])
        if t:
            color = chess.WHITE if t <= 6 else chess.BLACK
            board.set_piece_at(s, chess.Piece(PIECE_ORDER[(t - 1) % 6], color))
    board.turn = chess.WHITE
    rights = 0
    if castle & 1:
        rights |= chess.BB_H1
    if castle & 2:
        rights |= chess.BB_A1
    if castle & 4:
        rights |= chess.BB_H8
    if castle & 8:
        rights |= chess.BB_A8
    board.castling_rights = rights
    if ep < 8:
        board.ep_square = chess.square(ep, 5)  # white to move: ep target on rank 6
    board.halfmove_clock = halfmove
    return board
