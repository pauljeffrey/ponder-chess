"""RL on the policy's own look-ahead lines (encoder-only models).

    python scripts/rl_lookahead.py --ckpt runs/large_pv/final.pt --data data/all --out runs/large_rl --steps 2500
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chessagent.config import TrainConfig  # noqa: E402
from chessagent.rl_lookahead import lookahead_rl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="runs/rl_lookahead")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=24, help="root positions per step")
    ap.add_argument("--group", type=int, default=2, help="sampled lines per candidate move")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--depth", type=int, default=5, help="max look-ahead depth (each step draws 2..depth)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--w-sup", type=float, default=0.2)
    ap.add_argument("--w-match", type=float, default=1.0)
    ap.add_argument("--w-leaf", type=float, default=1.0)
    ap.add_argument("--w-calc", type=float, default=1.0)
    ap.add_argument("--w-dec", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=250)
    a = ap.parse_args()
    cfg = TrainConfig(data_dir=a.data, out_dir=a.out, steps=a.steps, batch_size=a.batch_size, lr=a.lr, device=a.device,
                      num_workers=a.workers, log_every=a.log_every)
    lookahead_rl(cfg, a.ckpt, group=a.group, k=a.k, depth=a.depth, beta=a.beta, w_sup=a.w_sup, w_match=a.w_match,
                 w_leaf=a.w_leaf, w_calc=a.w_calc, w_dec=a.w_dec, temperature=a.temperature, save_every=a.save_every)


if __name__ == "__main__":
    main()
