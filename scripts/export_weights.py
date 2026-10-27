"""Slim a training checkpoint into a deployable weights file (fp16 on disk, loaded back as fp32 by load_checkpoint).

    python scripts/export_weights.py runs/box/large_final.pt weights/ponder-64m.pt
"""
import sys
from pathlib import Path

import torch

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
ck = torch.load(src, map_location="cpu", weights_only=False)
state = {k: (v.half() if v.is_floating_point() else v) for k, v in ck["state"].items()}
dst.parent.mkdir(parents=True, exist_ok=True)
torch.save(dict(cfg=ck["cfg"], state=state, step=ck.get("step"), extra={}), dst)
print(f"{src} ({src.stat().st_size / 1e6:.0f} MB) -> {dst} ({dst.stat().st_size / 1e6:.0f} MB)")
