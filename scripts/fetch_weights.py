"""Download any model file listed in models.json that is missing locally.

Set PONDER_WEIGHTS_BASE to a URL prefix that serves the files by name, e.g. a Hugging Face repo:
    https://huggingface.co/<user>/<repo>/resolve/main
(HF_TOKEN is sent as a bearer token when set, for private repos.) With no base URL it does nothing.
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
base = os.environ.get("PONDER_WEIGHTS_BASE", "").rstrip("/")
for m in json.loads((ROOT / "models.json").read_text()):
    dst = ROOT / m["path"] if m.get("path") else None
    if dst is None or dst.exists():
        continue
    if not base:
        print(f"[weights] {dst.name} missing and PONDER_WEIGHTS_BASE is not set", file=sys.stderr)
        continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(f"{base}/{dst.name}")
    if os.environ.get("HF_TOKEN"):
        req.add_header("Authorization", "Bearer " + os.environ["HF_TOKEN"])
    print(f"[weights] downloading {dst.name} ...", flush=True)
    with urllib.request.urlopen(req) as r, open(dst.with_suffix(".part"), "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    dst.with_suffix(".part").replace(dst)
