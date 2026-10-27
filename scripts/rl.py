"""python scripts/rl.py --ckpt runs/pretrain/last.pt --data data/processed --steps 100"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chessagent.config import TrainConfig  # noqa: E402
from chessagent.rl import grpo  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="runs/rl")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--group", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--w-dec", type=float, default=1.0, help="weight of the search-decision term (0 = off)")
    ap.add_argument("--w-rank", type=float, default=0.0, help="weight of the decision-vs-Stockfish ranking term")
    ap.add_argument("--w-ground", type=float, default=0.0, help="weight of the leaf-value grounding terms")
    ap.add_argument("--line-reward", type=float, default=0.5, help="weight of the matched-prefix line reward")
    ap.add_argument("--leaf-reward", type=float, default=0.0, help="weight of the leaf-consistency line reward")
    ap.add_argument("--branch", type=int, default=1, help="decoder replies per candidate in the decision tree")
    ap.add_argument("--w-recall", type=float, default=0.0, help="policy pass@k term: mass on any near-best stored move")
    ap.add_argument("--w-branch", type=float, default=0.0, help="decoder term: explored replies matched against stored lines")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=5)
    a = ap.parse_args()
    cfg = TrainConfig(data_dir=a.data, out_dir=a.out, steps=a.steps, batch_size=a.batch_size, lr=a.lr,
                      device=a.device, num_workers=a.workers, log_every=a.log_every, warmup=1)
    grpo(cfg, a.ckpt, group=a.group, beta=a.beta, w_dec=a.w_dec, k=a.k, depth=a.depth,
         w_rank=a.w_rank, w_ground=a.w_ground, line_reward=a.line_reward,
         leaf_reward=a.leaf_reward, branch=a.branch, w_recall=a.w_recall, w_branch=a.w_branch)


if __name__ == "__main__":
    main()
