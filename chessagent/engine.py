"""Inference: policy -> top-k root moves -> decoder variations -> leaf evaluation -> decision.

The python-chess board is the source of truth for rules: illegal moves are masked at every step,
so the network never has to be trusted with legality. All network work is batched, which is what
keeps a decision at ~0.2-1 s on a laptop CPU for the base model.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import chess
import numpy as np
import torch

from .board import encode_root_board, to_root_frame
from .model import ChessNet, wdl_to_score
from .moves import VOCAB_SIZE, index_to_move, mirror_move, move_to_index
from .train import load_checkpoint


@dataclass
class Level:
    k: int            # candidate root moves
    depth: int        # plies per variation (including the root move)
    temperature: float  # 0 = always the best-scored candidate
    loops: int | None = None  # encoder/decoder passes; None = all the model has


TAU_REPLY = 0.1     # soft-min temperature over a candidate's replies (same value in training: rl.py)

LEVELS = {
    1: Level(k=3, depth=1, temperature=1.2, loops=1),
    2: Level(k=3, depth=2, temperature=0.6, loops=1),
    3: Level(k=4, depth=4, temperature=0.25),
    4: Level(k=4, depth=6, temperature=0.0),
    5: Level(k=4, depth=8, temperature=0.0),
}


def _tensors(boards: list[chess.Board], device):
    """Encode root-frame (white-to-move) boards into a model batch."""
    encs = [encode_root_board(b) for b in boards]
    return (
        torch.tensor(np.stack([e.squares for e in encs]), dtype=torch.long, device=device),
        torch.tensor([e.castle for e in encs], dtype=torch.long, device=device),
        torch.tensor([e.ep for e in encs], dtype=torch.long, device=device),
        torch.tensor([e.halfmove for e in encs], dtype=torch.long, device=device),
    )


def _legal_mask(boards: list[chess.Board], device) -> torch.Tensor:
    mask = torch.zeros(len(boards), VOCAB_SIZE, dtype=torch.bool, device=device)
    for i, b in enumerate(boards):
        for m in b.legal_moves:
            mask[i, move_to_index(m)] = True
    return mask


class Engine:
    def __init__(self, checkpoint: str | None = None, device: str = "cpu", threads: int | None = None):
        if threads:
            torch.set_num_threads(threads)
        self.device = torch.device(device)
        self.trained = checkpoint is not None
        self.step = None
        if checkpoint:
            self.model = load_checkpoint(checkpoint, self.device)
            self.step = getattr(self.model, "ckpt_step", None)
        else:  # untrained fallback so the app still boots (plays randomly-ish)
            from .config import preset
            self.model = ChessNet(preset("tiny")).to(self.device).eval()

    @torch.no_grad()
    def analyse(self, board: chess.Board, level: int = 4, k: int | None = None, depth: int | None = None,
                temperature: float | None = None, line_weight: float = 0.25, seed: int | None = None,
                loops: int | None = None, branch: int = 1, lookahead: bool = True,
                rollout: str = "policy", progress=None) -> dict:
        """progress: optional callback(dict) fed with the variations as they grow, one call per ply of look-ahead."""
        t0 = time.perf_counter()
        lv = LEVELS[max(1, min(5, level))]
        k = k or lv.k
        depth = depth or lv.depth
        temperature = lv.temperature if temperature is None else temperature
        loops = loops or lv.loops
        branch = max(1, branch)
        if rollout == "decoder" and not self.model.cfg.has_decoder:
            raise ValueError("this model has no line decoder: use rollout='policy'")
        gen = torch.Generator().manual_seed(seed) if seed is not None else None

        root, flipped = to_root_frame(board)
        legal = list(root.legal_moves)
        if not legal:
            raise ValueError("no legal moves")

        # ---- 1. policy + value at the root --------------------------------------------------
        mem = self.model.encode(*_tensors([root], self.device), loops=loops)
        logits = self.model.policy_logits(mem)[0].masked_fill(~_legal_mask([root], self.device)[0], float("-inf"))
        root_score = wdl_to_score(self.model.value_logits(mem))[0].item()
        logp = torch.log_softmax(logits, -1)
        k = min(k, len(legal))
        top = torch.topk(logp, k)
        cand_idx = top.indices.tolist()
        cand_logp = top.values.tolist()

        def real(idx: int) -> chess.Move:
            m = index_to_move(idx)
            return mirror_move(m) if flipped else m

        if not lookahead:      # instinct only: the policy head decides, no decoder, no board re-encoding
            real0 = real
            z = torch.tensor(cand_logp)
            pick = int(torch.multinomial(torch.softmax(z / temperature, -1), 1, generator=gen)) if temperature > 0 else 0
            ev = root_score if board.turn == chess.WHITE else 1.0 - root_score
            cands = []
            for idx, lp in zip(cand_idx, cand_logp):
                mv = real0(idx); san = board.san(mv)
                cands.append(dict(uci=mv.uci(), san=san, prior=math.exp(lp), score_white=ev, line=[san], line_uci=[mv.uci()],
                                  branches=[dict(line=[san], line_uci=[mv.uci()], score_white=ev, critical=True)]))
            return dict(move=cands[pick]["uci"], san=cands[pick]["san"], eval_white=ev, root_eval_white=ev, candidates=cands,
                        chosen=pick, trained=self.trained, lookahead=False, ms=round((time.perf_counter() - t0) * 1000))

        # ---- 2. variations from the line decoder (legality-masked): per candidate the decoder's top-`branch` replies,
        #         each continued greedily to `depth` plies ----------------------------------------------------------
        from .rl import softmin, tree_boards     # lazy: rl imports this module
        first = torch.tensor(cand_idx, dtype=torch.long, device=self.device)
        nb_eff = branch if (branch > 1 and depth >= 2) else 1
        root_ev = root_score if board.turn == chess.WHITE else 1.0 - root_score
        last_emit = [0.0]

        def line_san(idxs):
            b, out = board.copy(stack=False), []
            for idx in idxs:
                mv = real(idx)
                out.append(b.san(mv))
                b.push(mv)
            return out

        def on_ply(bs, ls):      # stream the variations as they grow (leaf scores follow once the lines are complete)
            now = time.perf_counter()
            if progress is None or (now - last_emit[0] < 0.05 and len(ls[0]) < depth and len(ls[0]) > 1):
                return
            last_emit[0] = now
            cp = []
            for i in range(k):
                brs = []
                for r in range(nb_eff):
                    ln = line_san(ls[i * nb_eff + r])
                    if not any(b["line"] == ln for b in brs):
                        brs.append(dict(line=ln, score_white=None, critical=False))
                cp.append(dict(san=brs[0]["line"][0], prior=math.exp(cand_logp[i]), score_white=None, line=brs[0]["line"], branches=brs))
            progress(dict(partial=True, candidates=cp, chosen=-1, root_eval_white=root_ev, lookahead=True, ms=None))

        boards, lines, bvalid = tree_boards(self.model, mem.expand(k, -1, -1), [root] * k, first, depth, branch, loops,
                                          policy=rollout == "policy", on_ply=on_ply if progress else None)
        nb = bvalid.shape[1]

        # ---- 3. evaluate leaves (root mover's expected score): the encoder re-reads each end position -----------------
        scores: list[float | None] = [None] * len(boards)
        need = []
        for i, b in enumerate(boards):
            if b.is_checkmate():
                scores[i] = 1.0 if b.turn == chess.BLACK else 0.0  # side to move is mated
            elif b.is_game_over(claim_draw=False):
                scores[i] = 0.5
            else:
                need.append(i)
        if need:
            leaf = [to_root_frame(boards[i])[0] for i in need]
            s = wdl_to_score(self.model.value_logits(self.model.encode(*_tensors(leaf, self.device), loops=loops))).tolist()
            for i, v in zip(need, s):
                scores[i] = v if boards[i].turn == chess.WHITE else 1.0 - v
        # a candidate is worth the soft-min over its replies (the opponent picks the reply that is worst for us)
        sc_t = torch.tensor(scores, dtype=torch.float32).view(k, nb)
        agg = softmin(sc_t, bvalid.cpu(), TAU_REPLY).tolist()

        # ---- 4. decision ------------------------------------------------------------------------
        def logit(p: float) -> float:
            p = min(max(p, 1e-4), 1 - 1e-4)
            return math.log(p / (1 - p))

        z = torch.tensor([logit(s) + line_weight * lp for s, lp in zip(agg, cand_logp)])
        if temperature > 0:
            probs = torch.softmax(z / temperature, -1)
            pick = int(torch.multinomial(probs, 1, generator=gen))
        else:
            pick = int(z.argmax())

        # ---- 5. package (convert back to the real board's frame) ------------------------------
        cands = []
        for i in range(k):
            branches = []
            for r in range(nb):
                e = i * nb + r
                if not bvalid[i, r]:
                    continue
                b = board.copy(stack=False)
                line_san = []
                for idx in lines[e]:
                    mv = real(idx)
                    line_san.append(b.san(mv))
                    b.push(mv)
                branches.append(dict(line=line_san, line_uci=[real(x).uci() for x in lines[e]],
                                     score_white=scores[e] if board.turn == chess.WHITE else 1.0 - scores[e]))
            crit = min(range(len(branches)), key=lambda j: branches[j]["score_white"] if board.turn == chess.WHITE
                       else -branches[j]["score_white"])             # the reply that is worst for the side to move
            for j, br in enumerate(branches):
                br["critical"] = j == crit
            white_score = agg[i] if board.turn == chess.WHITE else 1.0 - agg[i]
            cands.append(dict(uci=real(lines[i * nb][0]).uci(), san=branches[0]["line"][0], prior=math.exp(cand_logp[i]),
                              score_white=white_score, line=branches[crit]["line"], line_uci=branches[crit]["line_uci"],
                              branches=branches))
        chosen = cands[pick]
        return dict(
            move=chosen["uci"], san=chosen["san"], eval_white=chosen["score_white"],
            root_eval_white=root_score if board.turn == chess.WHITE else 1.0 - root_score,
            candidates=cands, chosen=pick, trained=self.trained, lookahead=True,
            ms=round((time.perf_counter() - t0) * 1000),
        )
