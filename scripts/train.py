"""python scripts/train.py --data data/processed --model tiny --steps 300 --batch-size 32"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chessagent.config import PRESETS, TrainConfig  # noqa: E402
from chessagent.train import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="runs/pretrain")
    ap.add_argument("--model", choices=list(PRESETS), default="tiny")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", default=None, help="checkpoint to continue from")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--pv-aug", type=float, default=0.0, help="share of positions advanced along their stored best line")
    ap.add_argument("--w-line", type=float, default=1.0)
    a = ap.parse_args()
    cfg = TrainConfig(data_dir=a.data, out_dir=a.out, model=a.model, steps=a.steps, batch_size=a.batch_size,
                      lr=a.lr, num_workers=a.workers, device=a.device, log_every=a.log_every,
                      eval_every=a.eval_every, save_every=a.save_every, warmup=a.warmup, pv_aug=a.pv_aug, w_line=a.w_line)
    train(cfg, init_ckpt=a.resume)


if __name__ == "__main__":
    main()
