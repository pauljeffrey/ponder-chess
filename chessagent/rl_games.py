"""RL from game results against Stockfish at set (nominal) Elo levels.

Everything in rl_lookahead.py is graded against Stockfish's *analysis*, so its ceiling is Stockfish's analysis. Here the
reward is what happens in real games: Ponder plays complete games against Stockfish (Ponder's moves come from its
look-ahead), and wins, draws and losses reinforce the moves that led to them. A move Stockfish's analysis rates poorly
but that wins games is now rewarded, which is the only way to learn something Stockfish's labels cannot teach.

Each iteration: play `games` games (the opponent's Elo is adapted so that Ponder scores about 50%: the most informative
level), then update
  policy  REINFORCE on the moves Ponder played (the look-ahead's decisions), advantage = game outcome (+1/0/-1)
          + w_dense * (-centipawn loss of the move as judged by full-strength Stockfish), normalised over the batch;
          KL leash to the starting policy. Reinforcing the search's decisions also distils the look-ahead into the policy.
  value   cross-entropy of the value head on Ponder's positions against the final result (label-smoothed)
  anchor  the supervised policy/value loss on Stockfish-line positions
"""
from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import chess

from .arena import StockfishPool, elo_from_score, play_games
from .config import TrainConfig
from .data import ShardDataset
from .rl_lookahead import _forever, sup_losses
from .train import load_checkpoint, pick_device, save_checkpoint, to_device


def games_rl(cfg: TrainConfig, init_ckpt: str, sf_path: str = "/usr/games/stockfish", games: int = 32, k: int = 4,
             depth: int = 4, elo: float = 1500.0, spread: int = 100, adapt: bool = True, beta: float = 0.02,
             w_sup: float = 0.3, w_val: float = 0.3, w_dense: float = 0.5, temperature: float = 0.3,
             sf_seconds: float = 0.1, max_plies: int = 200, dense_depth: int = 10, micro: int = 256,
             sup_bs: int = 64, pv_aug: float = 0.5, save_every: int = 5, elo_lo: int = 1320, elo_hi: int = 3190) -> Path:
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    device = pick_device(cfg.device)
    model = load_checkpoint(init_ckpt, "cpu").to(device).train()
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    sup = _forever(DataLoader(ShardDataset(cfg.data_dir, "train", aug_p=pv_aug), batch_size=sup_bs, shuffle=True,
                              drop_last=True, num_workers=max(1, cfg.num_workers)))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sf, an = StockfishPool(sf_path, games), StockfishPool(sf_path, 16, hash_mb=64)
    out_dir, t0 = Path(cfg.out_dir), time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "games.jsonl", "a")
    print(f"[games-rl] {model.num_params() / 1e6:.1f}M params | {games} games/iter vs Stockfish ~{elo:.0f} Elo | "
          f"k={k} depth={depth} T={temperature} w_dense={w_dense} device={device}", flush=True)
    try:
        for it in range(1, cfg.steps + 1):
            for g in opt.param_groups:
                g["lr"] = cfg.lr * min(1.0, it / 3)
            elos = [int(min(elo_hi, max(elo_lo, elo + spread * rng.choice([-1, 0, 1])))) for _ in range(games)]
            model.eval()
            gs, rec = play_games(model, sf, elos, k, depth, temperature, sf_seconds, max_plies, collect=True, rng=rng, judge=an)
            model.train()
            z = torch.tensor([g.result for g in gs])
            score = z.mean().item()
            w, d_, l = int((z == 1).sum()), int((z == 0.5).sum()), int((z == 0).sum())
            for g in gs:
                log.write(json.dumps(dict(it=it, elo=g.elo, ponder_white=g.ponder_white, result=g.result, how=g.how,
                                          moves=" ".join(m.uci() for m in g.board.move_stack))) + "\n")
            log.flush()
            R = len(rec["act"])
            outcome = (2 * z - 1)[rec["game"]]
            dense = torch.zeros(R)
            if w_dense > 0:                                         # cp loss of each Ponder move, full-strength Stockfish
                before = an.evals([chess.Board(f) for f in rec["fen"]], dense_depth)
                after_boards = []
                for f, m in zip(rec["fen"], rec["move"]):
                    b = chess.Board(f)
                    b.push_uci(m)
                    after_boards.append(b)
                after = an.evals(after_boards, dense_depth)
                loss_cp = torch.tensor([float(bf + af) for bf, af in zip(before, after)]).clamp(0, 300)
                dense = -loss_cp / 300
            A = outcome + w_dense * (dense - dense.mean())
            A = ((A - A.mean()) / (A.std() + 1e-3)).clamp(-4, 4)
            tgt = torch.where(z[rec["game"]] == 1, 0, torch.where(z[rec["game"]] == 0.5, 1, 2))

            opt.zero_grad(set_to_none=True)
            kl_sum = vl_sum = 0.0
            for s in range(0, R, micro):
                sl = slice(s, s + micro)
                sq, ca, ep, hm = (rec[x][sl].to(device) for x in ("sq", "castle", "ep", "hm"))
                legal = rec["legal"][sl].to(device)
                act = rec["act"][sl].to(device)[:, None]
                mem = model.encode(sq, ca, ep, hm)
                lp = F.log_softmax(model.policy_logits(mem).masked_fill(~legal, float("-inf")), -1).gather(1, act)[:, 0]
                with torch.no_grad():
                    rlp = F.log_softmax(ref.policy_logits(ref.encode(sq, ca, ep, hm)).masked_fill(~legal, float("-inf")), -1).gather(1, act)[:, 0]
                kl = (rlp - lp).exp() - (rlp - lp) - 1
                vl = F.cross_entropy(model.value_logits(mem), tgt[sl].to(device), label_smoothing=0.15, reduction="none")
                ((-(A[sl].to(device) * lp) + beta * kl + w_val * vl).sum() / R).backward()
                kl_sum += kl.sum().item(); vl_sum += vl.sum().item()
            pl, vl_s = sup_losses(model, to_device(next(sup), device), cfg.policy_temp_cp)
            (w_sup * (pl + vl_s)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            perf = elo_from_score(sum(elos) / len(elos), score)
            print(f"[games-rl] it {it:>4} vs ~{sum(elos) / len(elos):.0f} | +{w} ={d_} -{l} score {score:.2f} -> ~{perf:.0f} Elo | "
                  f"moves {R} plies/game {sum(g.board.ply() for g in gs) / games:.0f} | cp-loss {(-dense.mean() * 300):.0f} "
                  f"kl {kl_sum / R:.4f} val {vl_sum / R:.3f} sup {pl.item():.3f}/{vl_s.item():.3f} | "
                  f"{sum(g.ponder_ms for g in gs) / max(1, sum(g.ponder_moves for g in gs)):.0f} ms/move | {time.time() - t0:.0f}s", flush=True)
            if adapt:
                elo = min(elo_hi, max(elo_lo, elo + 250 * (score - 0.5)))
            if it % save_every == 0:
                save_checkpoint(out_dir / "last.pt", model, it)
    finally:
        sf.close(); an.close(); log.close()
    path = out_dir / "last.pt"
    save_checkpoint(path, model, cfg.steps)
    print(f"[games-rl] saved {path}", flush=True)
    return path
