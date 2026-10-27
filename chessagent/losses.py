"""Supervised multi-task losses. Positions/labels with missing fields are masked, not dropped."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import cp_to_wdl


def policy_targets(pv_moves: torch.Tensor, pv_cp: torch.Tensor, vocab: int, temp_cp: float) -> torch.Tensor:
    """Soft policy target [B,V]: softmax over the multi-PV first moves of (score / temp)."""
    valid = pv_moves[:, :, 0] >= 0                                   # [B,K]
    w = torch.softmax((pv_cp / temp_cp).masked_fill(~valid, float("-inf")), dim=1)
    w = torch.nan_to_num(w)
    tgt = torch.zeros(pv_moves.shape[0], vocab, device=pv_moves.device)
    tgt.scatter_add_(1, pv_moves[:, :, 0].clamp_min(0), w * valid)
    return tgt


def compute_losses(model, batch: dict, temp_cp: float = 100.0) -> dict:
    out = model(batch["squares"], batch["castle"], batch["ep"], batch["halfmove"])
    B, K, L = batch["pv_moves"].shape

    # 1. policy: cross-entropy against the soft multi-PV distribution
    tgt = policy_targets(batch["pv_moves"], batch["pv_cp"], out["policy"].shape[-1], temp_cp)
    logp = F.log_softmax(out["policy"], dim=-1)
    policy_loss = -(tgt * logp).sum(-1).mean()

    # 2. value: cross-entropy against the heuristic WDL of the best line
    wdl = cp_to_wdl(batch["cp"])
    value_loss = -(wdl * F.log_softmax(out["value"], dim=-1)).sum(-1).mean()

    # 3. line decoder: teacher-forced next-move prediction on every available PV
    pv = batch["pv_moves"].reshape(B * K, L)
    has_line = pv[:, 1] >= 0                                          # need >= 2 plies to learn anything
    if has_line.any():
        mem = out["memory"].repeat_interleave(K, dim=0)[has_line]
        pv = pv[has_line]
        logits = model.decode(mem, pv[:, :-1].clamp_min(0))          # [N,L-1,V]
        line_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), pv[:, 1:].reshape(-1), ignore_index=-1)
    else:
        line_loss = out["policy"].sum() * 0.0

    with torch.no_grad():
        best = batch["pv_moves"][:, 0, 0]
        top1 = (out["policy"].argmax(-1) == best).float().mean()
        top4 = (out["policy"].topk(4, dim=-1).indices == best[:, None]).any(-1).float().mean()
    return dict(policy=policy_loss, value=value_loss, line=line_loss, top1=top1, top4=top4)
