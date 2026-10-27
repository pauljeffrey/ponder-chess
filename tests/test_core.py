import json
import random

import chess
import numpy as np
import torch

from chessagent.board import board_from_tokens, encode_root_board, to_root_frame
from chessagent.config import preset
from chessagent.data import parse_record
from chessagent.model import ChessNet
from chessagent.moves import VOCAB_SIZE, index_to_move, mirror_move, move_to_index


def test_every_legal_move_is_in_vocab():
    random.seed(1)
    for _ in range(300):
        b = chess.Board()
        for _ in range(random.randint(0, 120)):
            if b.is_game_over():
                break
            b.push(random.choice(list(b.legal_moves)))
        for m in b.legal_moves:               # root frame and the raw frame (opponent promotions etc.)
            assert index_to_move(move_to_index(m)) == m
        root, _ = to_root_frame(b)
        for m in root.legal_moves:
            assert 0 <= move_to_index(m) < VOCAB_SIZE


def test_mirror_roundtrip_and_token_roundtrip():
    b = chess.Board()
    for u in ["e2e4", "c7c5", "g1f3", "d7d6"]:
        b.push_uci(u)
    root, flipped = to_root_frame(b)
    assert not flipped and root is b
    b.push_uci("d2d4")                          # black to move now -> mirrored
    root, flipped = to_root_frame(b)
    assert flipped and root.turn == chess.WHITE
    enc = encode_root_board(root)
    rebuilt = board_from_tokens(enc.squares, enc.castle, enc.ep, enc.halfmove)
    assert rebuilt.board_fen() == root.board_fen()
    assert set(rebuilt.legal_moves) == set(root.legal_moves)
    m = list(b.legal_moves)[0]
    assert mirror_move(mirror_move(m)) == m


def test_parse_record_black_to_move_and_castling_conversion():
    fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1"
    rec = {"fen": fen, "evals": [{"depth": 20, "knodes": 1, "pvs": [
        {"cp": 50, "line": "e8h8 e1h1 e7e5"}, {"cp": 10, "line": "a7a6"}]}]}
    out = parse_record(json.dumps(rec), pov="white")
    assert out is not None
    # black to move with a white-POV score of +50  =>  -50 for the mover
    assert out["pv_cp"][0] == -50
    first = index_to_move(int(out["pv_moves"][0, 0]))
    assert first == chess.Move.from_uci("e1g1")          # mirrored + converted from e8h8
    assert out["pv_moves"][0, 1] >= 0 and out["pv_moves"][0, 2] >= 0


def test_model_shapes_and_backward():
    m = ChessNet(preset("tiny"))
    B = 3
    sq = torch.randint(0, 13, (B, 64))
    out = m(sq, torch.randint(0, 16, (B,)), torch.randint(0, 9, (B,)), torch.randint(0, 100, (B,)))
    assert out["policy"].shape == (B, VOCAB_SIZE) and out["value"].shape == (B, 3)
    dec = m.decode(out["memory"], torch.randint(0, VOCAB_SIZE, (B, 5)))
    assert dec.shape == (B, 5, VOCAB_SIZE)
    (out["policy"].sum() + out["value"].sum() + dec.sum()).backward()


def test_decoder_is_causal():
    m = ChessNet(preset("tiny")).eval()
    mem = torch.randn(1, 66, m.cfg.d_model)
    a = torch.tensor([[5, 9, 20, 30]])
    b = torch.tensor([[5, 9, 21, 31]])
    la, lb = m.decode(mem, a), m.decode(mem, b)
    assert torch.allclose(la[:, :2], lb[:, :2], atol=1e-5)
    assert not torch.allclose(la[:, 2:], lb[:, 2:], atol=1e-5)


def test_engine_returns_legal_move_for_both_colors():
    from chessagent.engine import Engine
    e = Engine(None)
    b = chess.Board()
    for _ in range(6):
        r = e.analyse(b, level=3)
        assert chess.Move.from_uci(r["move"]) in b.legal_moves
        assert 0.0 <= r["eval_white"] <= 1.0
        b.push_uci(r["move"])
