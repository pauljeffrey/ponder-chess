"""Slim a training checkpoint into a deployable weights file (fp16 on disk, loaded back as fp32 by load_checkpoint).

    python scripts/export_weights.py runs/box/large_final.pt weights/ponder-rook.pt [--strip-decoder]

--strip-decoder drops the line decoder (about a third of the parameters): the deployed look-ahead only needs the
encoder, because it replays the policy head on a real board.
"""
import sys
from pathlib import Path

import torch

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
strip = "--strip-decoder" in sys.argv
ck = torch.load(src, map_location="cpu", weights_only=False)
state = {k: (v.half() if v.is_floating_point() else v) for k, v in ck["state"].items()}
DEC = ("move_emb.", "ply_emb.", "dec_layers.", "dec_norm.", "dec_bias", "dec_loop_emb.")
if strip:
    state = {k: v for k, v in state.items() if not k.startswith(DEC)}
    ck["cfg"] = dict(ck["cfg"], has_decoder=False)
dst.parent.mkdir(parents=True, exist_ok=True)
torch.save(dict(cfg=ck["cfg"], state=state, step=ck.get("step"), extra={}), dst)
print(f"{src} ({src.stat().st_size / 1e6:.0f} MB) -> {dst} ({dst.stat().st_size / 1e6:.0f} MB)")
