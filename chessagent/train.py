"""Supervised pretraining loop."""
from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import ModelConfig, TrainConfig, preset
from .data import ShardDataset
from .losses import compute_losses
from .model import ChessNet


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path: Path, model: ChessNet, step: int, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")       # write then rename, so a reader never sees a half-written checkpoint
    torch.save(dict(cfg=model.cfg.to_dict(), state=model.state_dict(), step=step, extra=extra or {}), tmp)
    tmp.replace(path)


def _fold_legacy_loop_emb(state: dict) -> dict:
    """Checkpoints trained with enc_loops = dec_loops = 1 still added a learned per-pass vector to every token at
    the start of the stack. That constant is folded into the embeddings (piece table, CLS, state-token bias for the
    encoder; ply table for the decoder) so the parameter-free model computes exactly the same function."""
    state = dict(state)
    ve = state.pop("enc_loop_emb.weight")[0]
    vd = state.pop("dec_loop_emb.weight")[0]
    state["piece_emb.weight"] = state["piece_emb.weight"] + ve
    state["cls"] = state["cls"] + ve
    state["castle_proj.bias"] = state["castle_proj.bias"] + ve
    state["ply_emb.weight"] = state["ply_emb.weight"] + vd
    return state


def load_checkpoint(path: str, device: torch.device | str = "cpu") -> ChessNet:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ck["cfg"])
    state = ck["state"]
    legacy = "enc_loop_emb.weight" in state and "loop_embed" not in ck["cfg"]   # trained before the flag existed
    if legacy and cfg.enc_loops == 1 and cfg.dec_loops == 1:
        state = _fold_legacy_loop_emb(state)
    elif legacy:
        cfg.loop_embed = True                      # multi-pass legacy models genuinely need their per-pass vectors
    model = ChessNet(cfg)
    if not cfg.loop_embed:
        state = {k: v for k, v in state.items() if "loop_emb" not in k}
    elif not any("loop_emb" in k for k in state):
        raise ValueError("model wants loop embeddings but the checkpoint has none")
    model.load_state_dict(state)
    model.ckpt_step = ck.get("step")
    return model.to(device).eval()


@torch.no_grad()
def evaluate(model, ds, device, batch_size: int, temp_cp: float, n_batches: int = 20) -> dict:
    model.eval()
    g = torch.Generator().manual_seed(123)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g, drop_last=True)
    tot: dict = {}
    for i, batch in enumerate(dl):
        if i >= n_batches:
            break
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            L = compute_losses(model, to_device(batch, device), temp_cp)
        for k, v in L.items():
            tot[k] = tot.get(k, 0.0) + v.item()
    model.train()
    return {k: v / max(1, min(n_batches, i)) for k, v in tot.items()}


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def train(cfg: TrainConfig, init_ckpt: str | None = None) -> Path:
    torch.manual_seed(cfg.seed)
    device = pick_device(cfg.device)
    ds = ShardDataset(cfg.data_dir, "train", aug_p=cfg.pv_aug)
    val_ds = ShardDataset(cfg.data_dir, "val")
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                    drop_last=True, persistent_workers=cfg.num_workers > 0)
    model = load_checkpoint(init_ckpt, "cpu").train() if init_ckpt else ChessNet(preset(cfg.model))
    model.to(device).train()
    print(f"[train] {len(ds)} positions | {model.num_params() / 1e6:.2f}M params | device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    def lr_at(step: int) -> float:
        if step < cfg.warmup:
            return cfg.lr * (step + 1) / cfg.warmup
        prog = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
        return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

    out_dir = Path(cfg.out_dir)
    step, t0 = 0, time.time()
    use_amp = device.type == "cuda"
    while step < cfg.steps:
        for batch in dl:
            batch = to_device(batch, device)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                L = compute_losses(model, batch, cfg.policy_temp_cp)
            loss = cfg.w_policy * L["policy"] + cfg.w_value * L["value"] + cfg.w_line * L["line"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % cfg.log_every == 0 or step == 1:
                print(f"[train] step {step:>6} loss {loss.item():.3f} | pol {L['policy'].item():.3f} "
                      f"val {L['value'].item():.3f} line {L['line'].item():.3f} | "
                      f"top1 {L['top1'].item():.2f} top4 {L['top4'].item():.2f} | {time.time() - t0:.0f}s", flush=True)
            if step % cfg.eval_every == 0:
                v = evaluate(model, val_ds, device, cfg.batch_size, cfg.policy_temp_cp)
                print(f"[val ] step {step:>6} pol {v['policy']:.3f} val {v['value']:.3f} line {v['line']:.3f} | "
                      f"top1 {v['top1']:.3f} top4 {v['top4']:.3f}", flush=True)
            if step % cfg.save_every == 0:
                save_checkpoint(out_dir / "last.pt", model, step)
            if step >= cfg.steps:
                break
    path = out_dir / "last.pt"
    save_checkpoint(path, model, step)
    print(f"[train] saved {path}")
    return path
