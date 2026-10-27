"""RL on the look-ahead itself, for the encoder-only model (no decoder): the policy writes its own variations.

Inference (engine.analyse, rollout="policy"): for each of the top-k root moves, replay the policy head on a real
board for `depth` plies (re-encoding the board after every move, both colours), let the encoder read the final position
(value head), and pick the move whose line ends best. Every move of every line is therefore a policy decision, and the
whole line is generated recursively: ply t is sampled on the board that plies 0..t-1 produced.

Training: sample those lines with temperature, grade the whole line against the Stockfish analysis stored with the
position, and give every move in the line credit for what came later (return-to-go, group-normalised, plain
policy gradient; this is on-policy so there is no ratio to clip) with a KL leash to the starting policy.

Rewards (per ply unless noted):
  match   +1/depth for each ply that still equals the stored Stockfish line of that root move (prefix match, so a
          line is rewarded for its length *and* order, for both colours)
  leaf    (terminal) -|leaf score - stored score| when the whole line stayed on the stored line: the encoder must
          judge the end of a Stockfish line the way Stockfish does
  calc    (terminal, every line of a candidate) the candidate's soft-min leaf value should equal the stored score of
          that move; for a move Stockfish did not list it must not look better than Stockfish's worst listed line
  decide  (terminal, lines of the chosen candidate) -regret of the move the look-ahead decision picks, i.e. the final
          decision is graded against Stockfish too
A supervised anchor (policy + value on positions advanced along their stored lines) keeps the heads calibrated.
"""
from __future__ import annotations

import copy
import math
import random
import time
from pathlib import Path

import chess
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import TrainConfig
from .data import ShardDataset, cp_to_score, cp_to_wdl
from .engine import _tensors
from .moves import index_to_move, mirror_move, move_to_index
from .rl import _mask, _root_boards, eval_boards
from .train import load_checkpoint, pick_device, save_checkpoint, to_device

UNLISTED_MARGIN_CP = 200.0


def _forever(dl):
    while True:
        yield from dl


@torch.no_grad()
def sample_rollouts(model, boards, first, depth: int, temperature: float):
    """Play each root move in `first`, then sample the policy on the real board for `depth - 1` more plies.

    boards: root-frame boards (white to move); first: list of root move ids. Returns (final boards, lines as absolute
    ids, records) where records hold, for every sampled ply, what the policy saw: encoded board, legal mask, the move
    it chose (in the frame it was shown) and which line / ply it belongs to."""
    dev = next(model.parameters()).device
    bs = [b.copy(stack=False) for b in boards]
    lines = [[int(f)] for f in first]
    for b, f in zip(bs, lines):
        b.push(index_to_move(f[0]))
    rec = dict(sq=[], castle=[], ep=[], hm=[], legal=[], act=[], line=[], ply=[])
    for ply in range(1, depth):
        alive = [i for i, b in enumerate(bs) if not b.is_game_over()]
        if not alive:
            break
        frames = [(b.mirror(), True) if b.turn == chess.BLACK else (b, False) for b in (bs[i] for i in alive)]
        roots = [f[0] for f in frames]
        enc = _tensors(roots, dev)
        legal = _mask(roots, dev)
        logits = model.policy_logits(model.encode(*enc)).masked_fill(~legal, float("-inf"))
        act = torch.multinomial(torch.softmax(logits / temperature, -1), 1)[:, 0]
        for n, i in enumerate(alive):
            mv = index_to_move(int(act[n]))
            mv = mirror_move(mv) if frames[n][1] else mv
            bs[i].push(mv)
            lines[i].append(move_to_index(mv))
        rec["sq"].append(enc[0].cpu()); rec["castle"].append(enc[1].cpu()); rec["ep"].append(enc[2].cpu())
        rec["hm"].append(enc[3].cpu()); rec["legal"].append(legal.cpu()); rec["act"].append(act.cpu())
        rec["line"] += alive
        rec["ply"] += [ply] * len(alive)
    return bs, lines, {k: (torch.cat(v) if k not in ("line", "ply") else torch.tensor(v, dtype=torch.long))
                       for k, v in rec.items() if v}


def _softmin(s: torch.Tensor, tau: float) -> float:
    return float(-tau * (torch.logsumexp(-s / tau, 0) - math.log(len(s))))


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def sup_losses(model, batch, temp_cp: float):
    """Policy + value cross-entropy (the encoder-only part of losses.compute_losses)."""
    from .losses import policy_targets
    mem = model.encode(batch["squares"], batch["castle"], batch["ep"], batch["halfmove"])
    pol = model.policy_logits(mem)
    tgt = policy_targets(batch["pv_moves"], batch["pv_cp"], pol.shape[-1], temp_cp)
    pl = -(tgt * F.log_softmax(pol, -1)).sum(-1).mean()
    vl = -(cp_to_wdl(batch["cp"]) * F.log_softmax(model.value_logits(mem), -1)).sum(-1).mean()
    return pl, vl


def lookahead_rl(cfg: TrainConfig, init_ckpt: str, group: int = 2, k: int = 4, depth: int = 5, beta: float = 0.02,
                 w_sup: float = 0.2, w_match: float = 1.0, w_leaf: float = 1.0, w_calc: float = 1.0, w_dec: float = 1.0,
                 temperature: float = 1.0, tau: float = 0.1, line_weight: float = 0.25, micro: int = 256,
                 sup_bs: int = 64, pv_aug: float = 0.5, warmup: int = 30, save_every: int = 250) -> Path:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    device = pick_device(cfg.device)
    model = load_checkpoint(init_ckpt, "cpu").to(device).train()
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    ds = ShardDataset(cfg.data_dir, "train")
    roots = _forever(DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers))
    sup = _forever(DataLoader(ShardDataset(cfg.data_dir, "train", aug_p=pv_aug), batch_size=sup_bs, shuffle=True,
                              drop_last=True, num_workers=max(1, cfg.num_workers // 2)))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    out_dir, t0 = Path(cfg.out_dir), time.time()
    print(f"[la-rl] {len(ds)} positions | {model.num_params() / 1e6:.1f}M params | k={k} group={group} depth<={depth} "
          f"beta={beta} device={device}", flush=True)

    for step in range(1, cfg.steps + 1):
        for g in opt.param_groups:
            g["lr"] = cfg.lr * min(1.0, step / warmup)
        batch = to_device(next(roots), device)
        boards = _root_boards(batch)
        B = len(boards)
        d = random.randint(2, depth)
        pvm, pvc = batch["pv_moves"].cpu(), batch["pv_cp"].cpu()
        S = cp_to_score(pvc)                                                         # [B,K] stored score, mover's view

        with torch.no_grad():
            out = model(batch["squares"], batch["castle"], batch["ep"], batch["halfmove"])
            legal = _mask(boards, device)
            logp = F.log_softmax(out["policy"].masked_fill(~legal, float("-inf")), -1)
            top = logp.topk(k, -1)
            cand, cand_lp, cvalid = top.indices.cpu(), top.values.cpu(), torch.isfinite(top.values.cpu())
            meta = [(b, c) for b in range(B) for c in range(k) if cvalid[b, c] for _ in range(group)]
            fb, lines, rec = sample_rollouts(model, [boards[b] for b, _ in meta], [int(cand[b, c]) for b, c in meta],
                                             d, temperature)
            leaf = eval_boards(model, fb, device)[0].cpu()

        # ---------------- grade the lines against the stored Stockfish analysis ----------------
        N = len(meta)
        tok_r = [[0.0] * len(l) for l in lines]
        info = {}                                                                    # (b,c) -> candidate bookkeeping
        stats = dict(match=[], leaf_err=[], calc_err=[], regret=[], regret_pol=[], agree=[], full=[])
        for i, (b, c) in enumerate(meta):
            valid_pv = [j for j in range(pvm.shape[1]) if pvm[b, j, 0] >= 0]
            j = next((j for j in valid_pv if int(pvm[b, j, 0]) == int(cand[b, c])), None)
            ent = info.setdefault((b, c), dict(lines=[], j=j))
            ent["lines"].append(i)
            if j is None:
                continue
            P = [int(x) for x in pvm[b, j] if x >= 0]
            n = 0
            for t, mv in enumerate(lines[i]):
                if t < len(P) and mv == P[t] and n == t:
                    n += 1
                    tok_r[i][t] += w_match / d
                else:
                    break
            stats["match"].append(n / len(lines[i]))
            if n == len(lines[i]):                                                    # whole line is the stored line
                err = abs(float(leaf[i]) - float(S[b, j]))
                tok_r[i][-1] -= w_leaf * err
                stats["leaf_err"].append(err)
        term_dec = {}
        for b in range(B):
            vc, ok = {}, []
            valid_pv = [j for j in range(pvm.shape[1]) if pvm[b, j, 0] >= 0]
            if not valid_pv:
                continue
            s_best, s_low = float(S[b, valid_pv].max()), float(S[b, valid_pv].min())
            s_unl = float(cp_to_score(pvc[b, valid_pv].min() - UNLISTED_MARGIN_CP))
            for c in range(k):
                if (b, c) not in info:
                    continue
                ent = info[(b, c)]
                V = _softmin(leaf[ent["lines"]], tau)
                ent["V"], ent["Q"] = V, (float(S[b, ent["j"]]) if ent["j"] is not None else s_unl)
                calc = abs(V - ent["Q"]) if ent["j"] is not None else max(0.0, V - s_low - 0.03)
                stats["calc_err"].append(calc)
                for i in ent["lines"]:
                    tok_r[i][-1] -= w_calc * calc
                ok.append(c)
            if not ok:
                continue
            z = {c: _logit(info[(b, c)]["V"]) + line_weight * float(cand_lp[b, c]) for c in ok}
            pick = max(z, key=z.get)
            regret = s_best - info[(b, pick)]["Q"]
            stats["regret"].append(regret)
            stats["regret_pol"].append(s_best - info[(b, ok[0])]["Q"])
            stats["agree"].append(float(info[(b, pick)]["j"] is not None and float(S[b, info[(b, pick)]["j"]]) >= s_best - 1e-6))
            for i in info[(b, pick)]["lines"]:
                tok_r[i][-1] -= w_dec * regret

        # return-to-go per ply, baseline = group mean at the same ply, scale = group std of the full return
        by_root: dict[int, list[int]] = {}
        for i, (b, _) in enumerate(meta):
            by_root.setdefault(b, []).append(i)
        ret = [torch.tensor(tok_r[i]).flip(0).cumsum(0).flip(0) for i in range(N)]
        adv = [None] * N
        for b, idx in by_root.items():
            g0 = torch.stack([ret[i][0] for i in idx])
            sd = g0.std().item() if len(idx) > 1 else 1.0
            for i in idx:
                a = ret[i].clone()
                for t in range(len(a)):
                    base = torch.stack([ret[o][t] for o in idx if len(ret[o]) > t]).mean()
                    a[t] = (a[t] - base) / (sd + 1e-2)
                adv[i] = a.clamp(-4, 4)

        # ---------------- policy-gradient pass over every move of every line ----------------
        # ply 0: the root move, shown the root board; plies >= 1: the sampled records
        r0_idx = torch.tensor([b for b, _ in meta])
        R0 = dict(sq=batch["squares"].cpu()[r0_idx], castle=batch["castle"].cpu()[r0_idx], ep=batch["ep"].cpu()[r0_idx],
                  hm=batch["halfmove"].cpu()[r0_idx], legal=legal.cpu()[r0_idx],
                  act=torch.tensor([int(cand[b, c]) for b, c in meta]))
        allr = {key: torch.cat([R0[key], rec[key]]) if key in rec else R0[key] for key in R0}
        a_all = torch.cat([torch.stack([adv[i][0] for i in range(N)]),
                           torch.stack([adv[int(li)][int(p)] for li, p in zip(rec["line"], rec["ply"])])]
                          if "line" in rec else [torch.stack([adv[i][0] for i in range(N)])])
        R = len(a_all)
        opt.zero_grad(set_to_none=True)
        kl_sum = pg_sum = 0.0
        for s in range(0, R, micro):
            sl = slice(s, s + micro)
            sq, ca, ep, hm = (allr[x][sl].to(device) for x in ("sq", "castle", "ep", "hm"))
            lg = model.policy_logits(model.encode(sq, ca, ep, hm)).masked_fill(~allr["legal"][sl].to(device), float("-inf"))
            lp = F.log_softmax(lg, -1).gather(1, allr["act"][sl].to(device)[:, None])[:, 0]
            with torch.no_grad():
                rg = ref.policy_logits(ref.encode(sq, ca, ep, hm)).masked_fill(~allr["legal"][sl].to(device), float("-inf"))
                rlp = F.log_softmax(rg, -1).gather(1, allr["act"][sl].to(device)[:, None])[:, 0]
            kl = (rlp - lp).exp() - (rlp - lp) - 1
            pg = -(a_all[sl].to(device) * lp)
            ((pg + beta * kl).sum() / R).backward()
            kl_sum += kl.sum().item(); pg_sum += pg.sum().item()
        pl, vl = sup_losses(model, to_device(next(sup), device), cfg.policy_temp_cp)
        (w_sup * (pl + vl)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % cfg.log_every == 0 or step == 1:
            m = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
            print(f"[la-rl] step {step:>5} d={d} lines {N} recs {R} | match {m(stats['match']):.2f} leaf_err {m(stats['leaf_err']):.3f} "
                  f"calc_err {m(stats['calc_err']):.3f} | regret look-ahead {m(stats['regret']):.4f} vs policy {m(stats['regret_pol']):.4f} "
                  f"agree {m(stats['agree']):.2f} | kl {kl_sum / R:.4f} sup {pl.item():.3f}/{vl.item():.3f} | {time.time() - t0:.0f}s", flush=True)
        if step % save_every == 0:
            save_checkpoint(out_dir / "last.pt", model, step)
    path = out_dir / "last.pt"
    save_checkpoint(path, model, cfg.steps)
    print(f"[la-rl] saved {path}", flush=True)
    return path
