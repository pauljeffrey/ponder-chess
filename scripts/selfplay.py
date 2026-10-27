"""Self-play match: the same network playing with its policy head only (greedy) vs. with the look-ahead loop
(decoder variations -> python-chess board -> encoder re-reads the end positions -> decision).

Every opening is played twice with colours swapped, so the two sides see the same positions.

    python scripts/selfplay.py --ckpt runs/box/rl_v4.pt --k 4 --depth 4 --branch 1 --out runs/selfplay_k4d4.json
"""
import argparse
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import chess
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chessagent.engine import Engine  # noqa: E402

OPENINGS = [
    "e2e4 e7e5 g1f3 b8c6", "e2e4 c7c5 g1f3 d7d6", "e2e4 e7e6 d2d4 d7d5", "e2e4 c7c6 d2d4 d7d5",
    "d2d4 d7d5 c2c4 e7e6", "d2d4 g8f6 c2c4 g7g6", "d2d4 g8f6 c2c4 e7e6", "c2c4 e7e5 b1c3 g8f6",
    "g1f3 d7d5 g2g3 g8f6", "g1f3 g8f6 c2c4 b7b6", "e2e4 e7e5 g1f3 g8f6", "d2d4 d7d5 g1f3 g8f6",
    "e2e4 d7d6 d2d4 g8f6", "d2d4 f7f5 g2g3 g8f6", "e2e4 g7g6 d2d4 f8g7",
]
MAX_PLIES = 240
_eng: Engine | None = None
_args = None


def _init(ckpt, threads, args):
    global _eng, _args
    torch.set_num_threads(threads)
    _eng, _args = Engine(ckpt, device="cpu", threads=threads), args


def play(job):
    opening, search_is_white = job
    board = chess.Board()
    for u in OPENINGS[opening].split():
        board.push_uci(u)
    t = {"search": [0.0, 0], "policy": [0.0, 0]}
    while True:
        out = board.outcome(claim_draw=True)
        if out or board.ply() >= MAX_PLIES:
            break
        searcher = (board.turn == chess.WHITE) == search_is_white
        t0 = time.perf_counter()
        if searcher:
            r = _eng.analyse(board, k=_args.k, depth=_args.depth, branch=_args.branch, temperature=0.0, rollout=_args.rollout)
        else:
            r = _eng.analyse(board, k=1, depth=1, temperature=0.0, lookahead=False)
        key = "search" if searcher else "policy"
        t[key][0] += time.perf_counter() - t0
        t[key][1] += 1
        board.push_uci(r["move"])
    out = board.outcome(claim_draw=True)
    if out is None or out.winner is None:
        score = 0.5                                   # draw, or adjudicated at the ply cap
    else:
        score = 1.0 if (out.winner == chess.WHITE) == search_is_white else 0.0
    return dict(opening=opening, search_white=search_is_white, score=score, plies=board.ply(),
                result=board.result(claim_draw=True) if out else "cap", termination=out.termination.name if out else "ply cap",
                search_ms=1000 * t["search"][0] / max(1, t["search"][1]), policy_ms=1000 * t["policy"][0] / max(1, t["policy"][1]),
                pgn=" ".join(m.uci() for m in board.move_stack))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--branch", type=int, default=1)
    ap.add_argument("--rollout", default="policy", choices=["policy", "decoder"])
    ap.add_argument("--openings", type=int, default=15)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    jobs = [(o, w) for o in range(a.openings) for w in (True, False)]
    t0 = time.time()
    with Pool(a.procs, initializer=_init, initargs=(a.ckpt, a.threads, a)) as pool:
        games = []
        for g in pool.imap_unordered(play, jobs):
            games.append(g)
            print(f"[{len(games):2d}/{len(jobs)}] opening {g['opening']:2d} search={'W' if g['search_white'] else 'B'} "
                  f"-> {g['score']:.1f} ({g['termination']}, {g['plies']} plies)  {time.time() - t0:.0f}s", flush=True)
    w = sum(g["score"] == 1 for g in games); d = sum(g["score"] == 0.5 for g in games); l = sum(g["score"] == 0 for g in games)
    n = len(games); s = (w + 0.5 * d) / n
    var = sum((g["score"] - s) ** 2 for g in games) / n
    se = math.sqrt(var / n)
    elo = lambda p: -400 * math.log10(1 / min(max(p, 1e-3), 1 - 1e-3) - 1)
    res = dict(config=vars(a), games=n, win=w, draw=d, loss=l, score=s, elo=elo(s), elo_lo=elo(s - 1.96 * se), elo_hi=elo(s + 1.96 * se),
               search_ms=sum(g["search_ms"] for g in games) / n, policy_ms=sum(g["policy_ms"] for g in games) / n,
               mean_plies=sum(g["plies"] for g in games) / n, results=games)
    print(f"\nlook-ahead (k={a.k} d={a.depth} b={a.branch}) vs policy-only: +{w} ={d} -{l}  score {s:.3f}  "
          f"Elo {res['elo']:+.0f} [{res['elo_lo']:+.0f}, {res['elo_hi']:+.0f}]  "
          f"{res['search_ms']:.0f} vs {res['policy_ms']:.0f} ms/move")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
