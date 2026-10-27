"""Data pipeline: Lichess evaluation JSONL -> compact numpy shards -> torch Dataset.

One lichess eval record:
    {"fen": "...", "evals": [{"knodes":..., "depth":..., "pvs": [{"cp":..., "line": "e2e4 e7e5 ..."}]}]}
We keep the deepest eval, its first `n_pv` principal variations and its first `max_line` plies each.
Everything is stored in the root frame (mover = white, see board.py).

Shard layout (one directory per shard, one .npy per field so training can np.load(mmap_mode="r")):
    squares  uint8 [N,64]     castle uint8 [N]    ep uint8 [N]    halfmove uint8 [N]
    cp       int16 [N]        best-line score, mover's point of view, clipped to +-MATE_CP
    depth    int16 [N]
    pv_moves int16 [N,K,L]    vocab ids, -1 = padding
    pv_cp    int16 [N,K]      score of each line (mover's point of view)
"""
from __future__ import annotations

import io
import json
import multiprocessing as mp
import os
import random
import urllib.request
from pathlib import Path
from typing import Iterable, Iterator

import chess
import numpy as np
import torch
from torch.utils.data import Dataset

from .board import board_from_tokens, encode_root_board, to_root_frame
from .moves import index_to_move, mirror_move, move_to_index

MATE_CP = 3000
FIELDS = ["squares", "castle", "ep", "halfmove", "cp", "depth", "pv_moves", "pv_cp"]


# --------------------------------------------------------------------------- parsing
def _pv_score(pv: dict, flip_pov: bool) -> int | None:
    if "mate" in pv and pv["mate"] is not None:
        m = int(pv["mate"])
        if m == 0:
            return None
        cp = (MATE_CP - min(abs(m), 50)) * (1 if m > 0 else -1)
    elif "cp" in pv and pv["cp"] is not None:
        cp = max(-MATE_CP, min(MATE_CP, int(pv["cp"])))
    else:
        return None
    return -cp if flip_pov else cp


def _parse_line(board: chess.Board, line: str, flipped: bool, max_len: int) -> list[int]:
    """Replay a UCI(_Chess960) line on `board`; return root-frame vocab ids (stops at first problem)."""
    b = board.copy(stack=False)
    out: list[int] = []
    for u in line.split()[:max_len]:
        try:
            mv = chess.Move.from_uci(u)
        except ValueError:
            break
        # Chess960 castling is "king takes own rook": convert to the standard king 2-square move
        if b.piece_type_at(mv.from_square) == chess.KING and b.color_at(mv.to_square) == b.turn:
            rank = chess.square_rank(mv.from_square)
            to_file = 6 if chess.square_file(mv.to_square) > chess.square_file(mv.from_square) else 2
            mv = chess.Move(mv.from_square, chess.square(to_file, rank))
        if mv not in b.legal_moves:
            break
        b.push(mv)
        out.append(move_to_index(mirror_move(mv) if flipped else mv))
    return out


def parse_record(raw: str, pov: str = "stm", n_pv: int = 4, max_line: int = 8) -> dict | None:
    """Convert one JSONL line into one training record (or None if unusable).

    pov: "stm"   -> cp values are from the side-to-move's point of view
         "white" -> cp values are from white's point of view
    """
    try:
        rec = json.loads(raw)
        board = chess.Board(rec["fen"])
    except (ValueError, KeyError):
        return None
    evals = rec.get("evals") or []
    if not evals:
        return None
    # prefer the analysis with the most lines (multi-PV is what the policy targets need), then depth
    ev = max(evals, key=lambda e: (min(len(e.get("pvs", [])), n_pv), e.get("depth", 0)))
    root, flipped = to_root_frame(board)
    # cp sign: convert to mover's point of view
    flip_pov = (pov == "white" and board.turn == chess.BLACK)

    pv_moves = np.full((n_pv, max_line), -1, dtype=np.int16)
    pv_cp = np.zeros(n_pv, dtype=np.int16)
    k = 0
    for pv in ev.get("pvs", []):
        if k >= n_pv:
            break
        score = _pv_score(pv, flip_pov)
        line = _parse_line(board, pv.get("line", ""), flipped, max_line)
        if score is None or not line:
            continue
        pv_moves[k, : len(line)] = line
        pv_cp[k] = score
        k += 1
    if k == 0:
        return None
    enc = encode_root_board(root)
    return dict(
        squares=enc.squares, castle=enc.castle, ep=enc.ep, halfmove=enc.halfmove,
        cp=pv_cp[0], depth=min(int(ev.get("depth", 0)), 32767), pv_moves=pv_moves, pv_cp=pv_cp,
    )


def _parse_chunk(args) -> list[dict]:
    lines, pov, n_pv, max_line = args
    return [r for r in (parse_record(x, pov, n_pv, max_line) for x in lines) if r is not None]


# ---------------------------------------------------------------------------- reading
def open_lines(src: str, max_bytes: int | None = None) -> Iterator[str]:
    """Yield text lines from a .jsonl / .jsonl.zst file or http(s) URL, streaming (never fully on disk).

    max_bytes: for URLs, only request that many compressed bytes (a truncated stream is fine).
    """
    import zstandard

    if src.startswith("http"):
        req = urllib.request.Request(src)
        if max_bytes:
            req.add_header("Range", f"bytes=0-{max_bytes - 1}")
        raw = urllib.request.urlopen(req)
    else:
        raw = open(src, "rb")
    stream = zstandard.ZstdDecompressor().stream_reader(raw) if src.endswith(".zst") else raw
    text = io.TextIOWrapper(stream, encoding="utf-8", errors="ignore")
    try:
        for line in text:
            yield line
    except zstandard.ZstdError:
        pass  # truncated download: ignore the cut-off tail
    finally:
        raw.close()


def _chunks(it: Iterable[str], size: int) -> Iterator[list[str]]:
    buf: list[str] = []
    for x in it:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def _write_shard(path: Path, recs: list[dict]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    dtypes = dict(squares=np.uint8, castle=np.uint8, ep=np.uint8, halfmove=np.uint8,
                  cp=np.int16, depth=np.int16, pv_moves=np.int16, pv_cp=np.int16)
    for f in FIELDS:
        np.save(path / f"{f}.npy", np.stack([np.asarray(r[f]) for r in recs]).astype(dtypes[f]))


def build_shards(src: str, out_dir: str, *, max_positions: int, shard_size: int = 100_000,
                 pov: str = "stm", n_pv: int = 4, max_line: int = 8, workers: int = 1,
                 max_bytes: int | None = None, skip: int = 0, min_depth: int = 0) -> int:
    """Stream `src`, write shards. Returns the number of positions written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    existing = len(list(out.glob("shard_*")))
    lines = open_lines(src, max_bytes)
    for _ in range(skip):
        next(lines, None)
    jobs = ((c, pov, n_pv, max_line) for c in _chunks(lines, 2000))

    pool = mp.Pool(workers) if workers > 1 else None
    results = pool.imap(_parse_chunk, jobs) if pool else map(_parse_chunk, jobs)
    buf: list[dict] = []
    total, shard_id = 0, existing
    try:
        for recs in results:
            buf.extend(r for r in recs if r["depth"] >= min_depth)
            while len(buf) >= shard_size or (buf and total + len(buf) >= max_positions):
                take = min(len(buf), shard_size, max_positions - total)
                if take <= 0:
                    buf = []
                    break
                _write_shard(out / f"shard_{shard_id:05d}", buf[:take])
                buf, total, shard_id = buf[take:], total + take, shard_id + 1
                print(f"[data] wrote shard {shard_id - 1} ({total} positions)", flush=True)
            if total >= max_positions:
                break
    finally:
        if pool:
            pool.terminate()
    return total


# ---------------------------------------------------------------------------- dataset
class ShardDataset(Dataset):
    """Random-access view over all shards under `data_dir` (memory-mapped, low RAM)."""

    def __init__(self, data_dir: str, part: str = "all", n_val: int = 1, aug_p: float = 0.0):
        """part: "train" (all but the last n_val shards), "val" (the last n_val shards) or "all".

        aug_p: probability that a position is replaced by one further along its own best Stockfish line
        (see `advance_along_line`): free extra policy/value data from exactly the positions a look-ahead ends on."""
        self.aug_p = aug_p
        self.shards = sorted(Path(data_dir).glob("shard_*"))
        if part == "train" and len(self.shards) > n_val:
            self.shards = self.shards[:-n_val]
        elif part == "val" and len(self.shards) > n_val:
            self.shards = self.shards[-n_val:]
        if not self.shards:
            raise FileNotFoundError(f"no shards in {data_dir}")
        self.arrays = [{f: np.load(s / f"{f}.npy", mmap_mode="r") for f in FIELDS} for s in self.shards]
        self.sizes = np.array([len(a["cp"]) for a in self.arrays])
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, i: int) -> dict:
        s = int(np.searchsorted(self.offsets, i, side="right") - 1)
        j = i - int(self.offsets[s])
        a = self.arrays[s]
        item = dict(
            squares=torch.from_numpy(a["squares"][j].astype(np.int64)),
            castle=torch.tensor(int(a["castle"][j])),
            ep=torch.tensor(int(a["ep"][j])),
            halfmove=torch.tensor(int(a["halfmove"][j])),
            cp=torch.tensor(float(a["cp"][j])),
            pv_moves=torch.from_numpy(a["pv_moves"][j].astype(np.int64)),
            pv_cp=torch.from_numpy(a["pv_cp"][j].astype(np.float32)),
        )
        if self.aug_p > 0 and random.random() < self.aug_p:
            item = advance_along_line(item)
        return item


def advance_along_line(item: dict, plies: int | None = None) -> dict:
    """Replay the first `plies` (random if None) moves of the best stored line and return the position reached,
    re-expressed in the root frame (mover = white). Targets: the line's next move (policy), the line's score from the
    new mover's point of view (value) and the rest of the line. Returns `item` unchanged when no line is available."""
    pv, pv_cp = item["pv_moves"][0], item["pv_cp"][0]
    n = int((pv >= 0).sum())
    if n < 2:
        return item
    j = plies or random.randint(1, n - 1)
    try:
        board = board_from_tokens(item["squares"].tolist(), int(item["castle"]), int(item["ep"]), int(item["halfmove"]))
        for m in pv[:j].tolist():
            board.push(index_to_move(m))
        root, flipped = to_root_frame(board)
        conv = lambda m: move_to_index(mirror_move(index_to_move(m)) if flipped else index_to_move(m))  # noqa: E731
        nxt, rest = conv(int(pv[j])), [conv(int(m)) for m in pv[j + 1:n]]
        enc = encode_root_board(root)
    except Exception:
        return item
    pv_moves = torch.full_like(item["pv_moves"], -1)
    pv_moves[0, 0] = nxt
    if rest:
        pv_moves[0, 1:1 + len(rest)] = torch.tensor(rest)
    cp = torch.zeros_like(item["pv_cp"])
    cp[0] = pv_cp * (-1.0 if j % 2 else 1.0)
    return dict(squares=torch.from_numpy(np.asarray(enc.squares).astype(np.int64)), castle=torch.tensor(int(enc.castle)),
                ep=torch.tensor(int(enc.ep)), halfmove=torch.tensor(int(enc.halfmove)), cp=cp[0].clone(),
                pv_moves=pv_moves, pv_cp=cp)


# -------------------------------------------------------------------------- targets
def cp_to_wdl(cp: torch.Tensor, scale: float = 150.0, draw_margin: float = 60.0) -> torch.Tensor:
    """Heuristic centipawn -> (win, draw, loss) probabilities for the mover.

    P(win) = sigmoid((cp - m) / s), P(loss) = sigmoid((-cp - m) / s), draw = the rest.
    """
    win = torch.sigmoid((cp - draw_margin) / scale)
    loss = torch.sigmoid((-cp - draw_margin) / scale)
    draw = (1.0 - win - loss).clamp_min(0.0)
    p = torch.stack([win, draw, loss], dim=-1)
    return p / p.sum(-1, keepdim=True)


def cp_to_score(cp: torch.Tensor) -> torch.Tensor:
    """Expected game score in [0,1] (win=1, draw=.5) used for RL rewards."""
    p = cp_to_wdl(cp)
    return p[..., 0] + 0.5 * p[..., 1]
