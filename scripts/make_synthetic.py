"""Generate a small synthetic file in the Lichess-eval JSONL format (no download needed).

Positions come from random games; the "engine" is a 2-ply material search, good enough to give
the network a learnable signal so the whole pipeline can be tested offline on a laptop.
"""
import argparse
import json
import random

import chess

VAL = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}


def material(b: chess.Board, color: bool) -> int:
    return sum(VAL[p.piece_type] * (1 if p.color == color else -1) for p in b.piece_map().values())


def two_ply(b: chess.Board, mv: chess.Move) -> tuple[int, list[chess.Move]]:
    """Score (mover POV) after mv and the opponent's best material reply; returns (score, line)."""
    me = b.turn
    b.push(mv)
    if b.is_checkmate():
        b.pop()
        return 3000, [mv]
    best, best_reply = None, None
    for r in b.legal_moves:
        b.push(r)
        s = material(b, me)
        b.pop()
        if best is None or s < best:
            best, best_reply = s, r
    b.pop()
    if best_reply is None:
        return 0, [mv]
    return best, [mv, best_reply]


def extend(b: chess.Board, line: list[chess.Move], plies: int) -> list[chess.Move]:
    b = b.copy()
    for m in line:
        b.push(m)
    while len(line) < plies and not b.is_game_over():
        color = b.turn
        sc = []
        for m in b.legal_moves:
            b.push(m)
            sc.append((material(b, color), random.random(), m))
            b.pop()
        m = max(sc)[2]
        b.push(m)
        line.append(m)
    return line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic/evals.jsonl")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--pov", choices=["stm", "white"], default="stm")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    n = 0
    with open(a.out, "w") as f:
        while n < a.n:
            b = chess.Board()
            for _ in range(random.randint(0, 70)):
                if b.is_game_over():
                    break
                b.push(random.choice(list(b.legal_moves)))
            if b.is_game_over():
                continue
            scored = [(two_ply(b, m), m) for m in b.legal_moves]
            scored.sort(key=lambda x: -x[0][0])
            pvs = []
            for (score, line), m in scored[:4]:
                line = extend(b, list(line), 8)
                cp = score if (a.pov == "stm" or b.turn == chess.WHITE) else -score
                pvs.append({"cp": cp, "line": " ".join(x.uci() for x in line)})
            f.write(json.dumps({"fen": b.fen().rsplit(" ", 2)[0], "evals": [{"knodes": 1, "depth": 8, "pvs": pvs}]}) + "\n")
            n += 1
    print(f"wrote {n} positions to {a.out}")


if __name__ == "__main__":
    main()
