"""Gauntlet: Ponder vs Stockfish at several nominal Elo levels. Writes a summary and the games.

    python scripts/play_stockfish.py --ckpt weights/ponder-rook.pt --elos 1500,1800,2100,2400 --games 20 --depth 6
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chessagent.arena import StockfishPool, elo_from_score, play_games  # noqa: E402
from chessagent.train import load_checkpoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--elos", default="1500,1800,2100,2400")
    ap.add_argument("--games", type=int, default=20, help="per level (colours alternate)")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--sf-seconds", type=float, default=0.1)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--sf", default="/usr/games/stockfish")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    model = load_checkpoint(a.ckpt, a.device)
    sf, judge = StockfishPool(a.sf, a.games), StockfishPool(a.sf, 8, hash_mb=64)
    summary, rng = {}, random.Random(0)
    try:
        for elo in [int(e) for e in a.elos.split(",")]:
            gs, _ = play_games(model, sf, [elo] * a.games, a.k, a.depth, 0.0, a.sf_seconds, a.max_plies, rng=rng, judge=judge)
            w, d, l = (sum(g.result == r for g in gs) for r in (1, 0.5, 0))
            s = (w + 0.5 * d) / len(gs)
            summary[elo] = dict(win=w, draw=d, loss=l, score=s, elo_estimate=elo_from_score(elo, s),
                                games=[" ".join(m.uci() for m in g.board.move_stack) for g in gs])
            print(f"vs Stockfish {elo}: +{w} ={d} -{l}  score {s:.2f}  -> ~{summary[elo]['elo_estimate']:.0f}", flush=True)
    finally:
        sf.close(); judge.close()
    if a.out:
        Path(a.out).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
