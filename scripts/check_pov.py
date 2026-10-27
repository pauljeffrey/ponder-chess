"""Decide whether `cp` in a Lichess eval file is from white's or the side-to-move's point of view.

Correlates cp with the material balance separately for white-to-move and black-to-move positions:
  * stm convention  -> cp correlates positively with (mover material - opponent material) for BOTH sides
  * white convention-> positive for white-to-move, NEGATIVE for black-to-move.
Usage: python scripts/check_pov.py path/to/sample.jsonl[.zst]   (a few thousand lines are plenty)
"""
import json
import sys
from itertools import islice
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chessagent.data import open_lines  # noqa: E402

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def main(path: str, n: int = 20000):
    cps = {chess.WHITE: [], chess.BLACK: []}
    mats = {chess.WHITE: [], chess.BLACK: []}
    for raw in islice(open_lines(path), n):
        try:
            r = json.loads(raw)
            b = chess.Board(r["fen"])
            ev = max(r["evals"], key=lambda e: e.get("depth", 0))
            pv = ev["pvs"][0]
        except (ValueError, KeyError, IndexError):
            continue
        if "cp" not in pv:
            continue
        mat = sum(VAL[p.piece_type] * (1 if p.color == b.turn else -1) for p in b.piece_map().values())
        cps[b.turn].append(pv["cp"])
        mats[b.turn].append(mat)
    corr = {c: np.corrcoef(cps[c], mats[c])[0, 1] for c in cps if len(cps[c]) > 10}
    w, bl = corr.get(chess.WHITE), corr.get(chess.BLACK)
    print(f"corr(cp, mover material) white-to-move={w:.3f}  black-to-move={bl:.3f}")
    print("=> use --pov", "stm" if bl > 0 else "white")


if __name__ == "__main__":
    main(sys.argv[1])
