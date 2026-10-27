"""RL from game results vs Stockfish (needs a stockfish binary: apt install stockfish).

    python scripts/rl_games.py --ckpt runs/large_rl/last.pt --data data/all --out runs/large_games --steps 200
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chessagent.config import TrainConfig  # noqa: E402
from chessagent.rl_games import games_rl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/processed", help="only for the supervised anchor")
    ap.add_argument("--out", default="runs/rl_games")
    ap.add_argument("--steps", type=int, default=100, help="iterations (each plays --games games)")
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--elo", type=float, default=1500.0, help="starting Stockfish Elo (adapts towards a 50% score)")
    ap.add_argument("--no-adapt", action="store_true")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--w-dense", type=float, default=0.5, help="weight of the per-move centipawn-loss signal (0 = results only)")
    ap.add_argument("--temperature", type=float, default=0.3, help="Ponder's move-choice temperature while generating games")
    ap.add_argument("--sf-seconds", type=float, default=0.1)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--sf", default="/usr/games/stockfish")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    cfg = TrainConfig(data_dir=a.data, out_dir=a.out, steps=a.steps, lr=a.lr, device=a.device, num_workers=a.workers)
    games_rl(cfg, a.ckpt, sf_path=a.sf, games=a.games, k=a.k, depth=a.depth, elo=a.elo, adapt=not a.no_adapt, beta=a.beta,
             w_dense=a.w_dense, temperature=a.temperature, sf_seconds=a.sf_seconds, max_plies=a.max_plies)


if __name__ == "__main__":
    main()
