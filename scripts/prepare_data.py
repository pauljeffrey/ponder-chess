"""Turn a Lichess eval dump (file or URL) into training shards.

Mac smoke test (first ~100 MB of the compressed file, ~1M positions available):
    python scripts/prepare_data.py --src https://database.lichess.org/lichess_db_eval.jsonl.zst \
        --max-bytes 100000000 --max-positions 50000 --pov white
GPU box (stream straight from the URL, nothing big touches the disk):
    python scripts/prepare_data.py --src https://database.lichess.org/lichess_db_eval.jsonl.zst \
        --max-positions 50000000 --workers 16 --shard-size 500000 --pov <white|stm>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chessagent.data import build_shards  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path or URL of .jsonl / .jsonl.zst")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--max-positions", type=int, default=50_000)
    ap.add_argument("--shard-size", type=int, default=50_000)
    ap.add_argument("--pov", choices=["stm", "white"], default="stm",
                    help="whose point of view the source cp values use (run scripts/check_pov.py on real data)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-bytes", type=int, default=None, help="for URLs: only fetch this many compressed bytes")
    ap.add_argument("--skip", type=int, default=0, help="skip this many input lines")
    ap.add_argument("--min-depth", type=int, default=0)
    a = ap.parse_args()
    n = build_shards(a.src, a.out, max_positions=a.max_positions, shard_size=a.shard_size, pov=a.pov,
                     workers=a.workers, max_bytes=a.max_bytes, skip=a.skip, min_depth=a.min_depth)
    print(f"done: {n} positions -> {a.out}")


if __name__ == "__main__":
    main()
