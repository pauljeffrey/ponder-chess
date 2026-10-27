"""Measure decision quality against the stored Stockfish analysis on held-out positions.

For each position the engine picks a move; we compare it with the multi-PV list shipped with the position:
  agree   - picked move == Stockfish's best move
  listed  - picked move is one of the (up to 4) analysed lines
  regret  - expected-score loss vs. the best line (unlisted legal moves count as worse than the worst listed one)
Compared configurations: policy only (k=1, depth=1) vs. the variation search at different budgets / loop counts.

    python scripts/evaluate.py --ckpt runs/pretrain/last.pt --data data/full --n 400 --device cpu
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chessagent.board import board_from_tokens  # noqa: E402
from chessagent.data import ShardDataset, cp_to_score  # noqa: E402
from chessagent.engine import Engine  # noqa: E402
from chessagent.moves import move_to_index  # noqa: E402

UNLISTED_MARGIN_CP = 200.0


def run(engine: Engine, ds: ShardDataset, idxs, **kw) -> dict:
    agree = listed = 0
    regrets, ms = [], []
    for i in idxs:
        r = ds[int(i)]
        board = board_from_tokens(r["squares"].tolist(), int(r["castle"]), int(r["ep"]), int(r["halfmove"]))
        if not any(board.legal_moves):
            continue
        pv_moves, pv_cp = r["pv_moves"][:, 0].tolist(), r["pv_cp"].tolist()
        valid = [j for j, m in enumerate(pv_moves) if m >= 0]
        res = engine.analyse(board, temperature=0.0, **kw)
        import chess
        pick = move_to_index(chess.Move.from_uci(res["move"]))
        best_cp = max(pv_cp[j] for j in valid)
        worst_cp = min(pv_cp[j] for j in valid) - UNLISTED_MARGIN_CP
        hit = [j for j in valid if pv_moves[j] == pick]
        cp = pv_cp[hit[0]] if hit else worst_cp
        agree += bool(hit and hit[0] == valid[0]) or (bool(hit) and pv_cp[hit[0]] == best_cp)
        listed += bool(hit)
        regrets.append(float(cp_to_score(torch.tensor(best_cp)) - cp_to_score(torch.tensor(cp))))
        ms.append(res["ms"])
    n = len(regrets)
    return dict(n=n, agree=agree / n, listed=listed / n, regret=float(np.mean(regrets)), ms=float(np.mean(ms)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/full")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rollout", default="decoder", choices=["decoder", "policy"], help="who generates the variation lines")
    ap.add_argument("--depths", default=None, help="comma list: evaluate k=--k at these depths instead of the default set")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--branch", type=int, default=1)
    a = ap.parse_args()
    ds = ShardDataset(a.data, "val")
    idxs = np.random.default_rng(0).choice(len(ds), a.n, replace=False)
    eng = Engine(a.ckpt, device=a.device, threads=a.threads)
    max_loops = eng.model.cfg.enc_loops
    configs = {
        "policy only (k=1,d=1)": dict(k=1, depth=1),
        "search k=4 d=4": dict(k=4, depth=4),
        "search k=4 d=8": dict(k=4, depth=8),
        "search k=8 d=6": dict(k=8, depth=6),
    }
    if max_loops > 1:
        configs["search k=4 d=6, 1 loop"] = dict(k=4, depth=6, loops=1)
    if a.depths:
        configs = {"policy only (k=1,d=1)": dict(k=1, depth=1)}
        configs.update({f"{a.rollout} lines k={a.k} d={d} b={a.branch}": dict(k=a.k, depth=int(d), branch=a.branch, rollout=a.rollout) for d in a.depths.split(",")})
    results = {}
    for name, kw in configs.items():
        t = time.time()
        results[name] = run(eng, ds, idxs, **kw)
        print(f"{name:26s} agree {results[name]['agree']:.3f}  listed {results[name]['listed']:.3f}  "
              f"regret {results[name]['regret']:.4f}  {results[name]['ms']:.0f} ms/move  ({time.time() - t:.0f}s)", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
