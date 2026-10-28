"""FastAPI app: serves the web UI and the move endpoint.

    uvicorn server.app:app --host 0.0.0.0 --port 8000

Models offered in the UI are listed in models.json (id, label, desc, path); a model whose file does not exist yet is
shown as unavailable and picked up automatically once it appears. Without models.json, CHESS_CKPT names one checkpoint.
Only one model stays in memory: a request for another model loads it and drops the old one, unless someone is
still using the old one (then both are held until that search ends). Env: CHESS_CKPT (fallback checkpoint), CHESS_THREADS (torch CPU threads), CHESS_CORS (comma list of origins).
"""
from __future__ import annotations

import ctypes
import gc
import json
import os
import queue
import threading
from pathlib import Path

import chess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from chessagent.engine import Engine

ROOT = Path(__file__).resolve().parent.parent
THREADS = int(os.environ.get("CHESS_THREADS", "0")) or None
MAX_COST = int(os.environ.get("PONDER_MAX_COST", "4000"))   # cap on candidates x replies x depth per move (public servers)
lock = threading.RLock()  # one search at a time: the CPU is the bottleneck, not the API


def _registry() -> list[dict]:
    """Models offered in the UI: models.json (id, label, desc, path), else the single CHESS_CKPT, else untrained."""
    f = ROOT / "models.json"
    if f.exists():
        return json.loads(f.read_text())
    ckpt = os.environ.get("CHESS_CKPT")
    return [dict(id="default", label="Chess Agent", desc="", path=ckpt or "")]


_engines: dict[str, tuple[float, Engine]] = {}    # loaded models: id -> (checkpoint mtime, engine); normally one entry
_users: dict[str, int] = {}                       # id -> requests waiting for or running on that model
_info: dict[str, tuple[float, dict]] = {}         # id -> (mtime, params/step/trained): read without loading the weights
_last: str | None = None                          # the model most recently served; kept resident for the next request


def _path(m: dict) -> Path | None:
    p = Path(m["path"]) if m.get("path") else None
    p = p if p is None or p.is_absolute() else ROOT / p
    return p if p is not None and p.exists() else None


def _resolve(model_id: str | None) -> tuple[dict, Path | None]:
    reg = _registry()
    m = next((x for x in reg if x["id"] == model_id), None) if model_id else None
    if m is None:
        m = next((x for x in reg if _path(x)), reg[0])     # default: first model whose file exists
    p = _path(m)
    if p is None and m["path"]:
        raise HTTPException(409, f"model '{m['label']}' is not available yet")
    return m, p


def _evict():
    """Drop every loaded model that nobody is using, other than the one served last. Caller holds the lock."""
    dropped = [i for i in _engines if i != _last and _users.get(i, 0) == 0]
    for i in dropped:
        del _engines[i]
    if dropped:
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)        # hand the freed weights back to the OS (glibc)
        except Exception:
            pass


def get_engine(model_id: str | None) -> tuple[dict, Engine]:
    global _last
    m, p = _resolve(model_id)
    mt = p.stat().st_mtime if p else 0.0
    with lock:
        if m["id"] not in _engines or _engines[m["id"]][0] != mt:
            try:
                _engines[m["id"]] = (mt, Engine(str(p) if p else None, device="cpu", threads=THREADS))
            except Exception as e:          # e.g. the file is still being written: keep serving the previous load
                if m["id"] not in _engines:
                    raise HTTPException(409, f"model '{m['label']}' could not be loaded yet ({type(e).__name__})")
        _last = m["id"]
        _evict()
        return m, _engines[m["id"]][1]


def _model_info(m: dict, p: Path) -> dict:
    """Parameter count etc. from the checkpoint header alone (memory-mapped, weights never materialised)."""
    mt = p.stat().st_mtime
    if m["id"] not in _info or _info[m["id"]][0] != mt:
        import torch
        from chessagent.config import ModelConfig
        from chessagent.model import ChessNet
        ck = torch.load(p, map_location="cpu", mmap=True, weights_only=False)
        with torch.device("meta"):
            n = ChessNet(ModelConfig(**ck["cfg"])).num_params()
        _info[m["id"]] = (mt, dict(params=n, step=ck.get("step"), trained=True))
        del ck
    return _info[m["id"]][1]


app = FastAPI(title="Chess Agent")
if os.environ.get("CHESS_CORS"):
    app.add_middleware(CORSMiddleware, allow_origins=os.environ["CHESS_CORS"].split(","),
                       allow_methods=["*"], allow_headers=["*"])


class MoveRequest(BaseModel):
    fen: str
    model: str | None = None
    level: int = Field(4, ge=1, le=5)
    k: int | None = Field(None, ge=1, le=25)       # candidate root moves; None = the level's preset
    depth: int | None = Field(None, ge=1, le=32)    # plies per variation; None = the level's preset
    branch: int = Field(1, ge=1, le=10)             # decoder replies explored per candidate move
    lookahead: bool = True                          # False = the policy head's first instinct only


@app.get("/api/models")
def models():
    out = []
    for m in _registry():
        p = _path(m)
        info = dict(id=m["id"], label=m["label"], desc=m.get("desc", ""), available=p is not None or not m["path"])
        if p is not None:
            info.update(_model_info(m, p))
        out.append(info)
    return out


@app.get("/api/health")
def health():
    return dict(ok=True, loaded=sorted(_engines))      # never loads a model: the health check must not swap the resident one


def _checked_board(req: MoveRequest) -> chess.Board:
    try:
        board = chess.Board(req.fen)
    except ValueError:
        raise HTTPException(400, "invalid FEN")
    if board.is_game_over():
        raise HTTPException(400, "game is over")
    cost = (req.k or 4) * req.branch * (req.depth or 8)
    if req.lookahead and cost > MAX_COST:
        raise HTTPException(400, f"that search is too large for this server ({cost} imagined moves, limit {MAX_COST}): lower the candidates, depth or replies")
    return board


def _search(req: MoveRequest, board: chess.Board, progress=None) -> dict:
    m, _ = _resolve(req.model)
    with lock:
        _users[m["id"]] = _users.get(m["id"], 0) + 1       # counted before queueing, so a model with waiting users is never dropped
    try:
        with lock:
            m, eng = get_engine(m["id"])
            try:
                res = eng.analyse(board, level=req.level, k=req.k, depth=req.depth, branch=req.branch,
                                  lookahead=req.lookahead, progress=progress)
            except ValueError as e:
                raise HTTPException(400, str(e))
    finally:
        with lock:
            _users[m["id"]] -= 1
            _evict()
    res["model"] = m["id"]
    res["model_label"] = m["label"]
    return res


@app.post("/api/move")
def move(req: MoveRequest):
    return _search(req, _checked_board(req))


@app.post("/api/move/stream")
def move_stream(req: MoveRequest):
    """Same search as /api/move, as newline-delimited JSON: one {"partial": ...} event per ply of look-ahead
    (the variations as they grow), then {"done": true, "result": ...} (or {"error": ...})."""
    board = _checked_board(req)
    q: queue.Queue = queue.Queue()

    def work():
        try:
            q.put(dict(done=True, result=_search(req, board, progress=q.put)))
        except HTTPException as e:
            q.put(dict(error=e.detail))
        except Exception as e:
            q.put(dict(error=f"{type(e).__name__}: {e}"))

    threading.Thread(target=work, daemon=True).start()

    def gen():
        while True:
            ev = q.get()
            yield json.dumps(ev) + "\n"
            if ev.get("done") or ev.get("error"):
                return

    return StreamingResponse(gen(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
