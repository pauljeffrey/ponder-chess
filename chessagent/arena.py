"""Play the network against Stockfish (UCI) in many games at once.

Stockfish runs on the CPU, one engine process per game; Ponder's look-ahead (policy lines replayed on real boards,
end positions read by the value head) is batched across all games that are waiting for Ponder's move, so a GPU serves
dozens of games at the cost of one search. Used by the game-outcome RL (rl_games.py) and the gauntlet script.
"""
from __future__ import annotations

import concurrent.futures as cf
import math
import random
import time
from dataclasses import dataclass, field

import chess
import chess.engine
import torch

from .board import to_root_frame
from .engine import _tensors
from .moves import index_to_move, mirror_move, move_to_index
from .rl import _mask, eval_boards, greedy_boards

OPENINGS = [
    "e2e4 e7e5 g1f3 b8c6", "e2e4 c7c5 g1f3 d7d6", "e2e4 e7e6 d2d4 d7d5", "e2e4 c7c6 d2d4 d7d5",
    "d2d4 d7d5 c2c4 e7e6", "d2d4 g8f6 c2c4 g7g6", "d2d4 g8f6 c2c4 e7e6", "c2c4 e7e5 b1c3 g8f6",
    "g1f3 d7d5 g2g3 g8f6", "g1f3 g8f6 c2c4 b7b6", "e2e4 e7e5 g1f3 g8f6", "d2d4 d7d5 g1f3 g8f6",
    "e2e4 d7d6 d2d4 g8f6", "d2d4 f7f5 g2g3 g8f6", "e2e4 g7g6 d2d4 f8g7",
]


def _logit(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-4, 1 - 1e-4)
    return torch.log(p / (1 - p))


@torch.no_grad()
def search_moves(model, boards: list[chess.Board], k: int = 4, depth: int = 4, temperature: float = 0.0,
                 line_weight: float = 0.25):
    """The look-ahead decision for many real boards at once (same procedure as Engine.analyse with rollout='policy').

    Returns (moves as UCI strings, records) where records hold, per board, what the policy saw at the root (encoded
    root-frame board, legal mask) and the root move it finally chose (in that frame): what the RL step reinforces."""
    dev = next(model.parameters()).device
    frames = [to_root_frame(b) for b in boards]
    roots, flips = [f[0] for f in frames], [f[1] for f in frames]
    enc, legal = _tensors(roots, dev), _mask(roots, dev)
    logp = torch.log_softmax(model.policy_logits(model.encode(*enc)).masked_fill(~legal, float("-inf")), -1)
    top = logp.topk(min(k, logp.shape[-1]), -1)
    cand, clp = top.indices.cpu(), top.values.cpu()
    valid = torch.isfinite(clp)
    meta = [(b, c) for b in range(len(boards)) for c in range(cand.shape[1]) if valid[b, c]]
    lb = greedy_boards(model, None, [roots[b] for b, _ in meta], torch.tensor([int(cand[b, c]) for b, c in meta]),
                       depth, None, policy=True)[0]
    score = eval_boards(model, lb, dev)[0].cpu()
    z = torch.full(cand.shape, float("-inf"))
    for n, (b, c) in enumerate(meta):
        z[b, c] = _logit(score[n]) + line_weight * clp[b, c]
    pick = torch.multinomial(torch.softmax(z / temperature, -1), 1)[:, 0] if temperature > 0 else z.argmax(-1)
    moves, act = [], []
    for b in range(len(boards)):
        idx = int(cand[b, int(pick[b])])
        mv = index_to_move(idx)
        moves.append((mirror_move(mv) if flips[b] else mv).uci())
        act.append(idx)
    rec = dict(sq=enc[0].cpu(), castle=enc[1].cpu(), ep=enc[2].cpu(), hm=enc[3].cpu(), legal=legal.cpu(),
               act=torch.tensor(act))
    return moves, rec


class StockfishPool:
    """`n` Stockfish processes, each set to a (nominal) Elo before every game."""

    def __init__(self, path: str = "/usr/games/stockfish", n: int = 16, hash_mb: int = 16):
        self.engines = [chess.engine.SimpleEngine.popen_uci(path) for _ in range(n)]
        for e in self.engines:
            e.configure({"Threads": 1, "Hash": hash_mb})
        self.pool = cf.ThreadPoolExecutor(max_workers=n)

    def set_elo(self, i: int, elo: int | None):
        if elo is None or elo >= 3190:
            self.engines[i].configure({"UCI_LimitStrength": False})
        else:
            self.engines[i].configure({"UCI_LimitStrength": True, "UCI_Elo": max(1320, int(elo))})

    def play(self, jobs: list[tuple[int, chess.Board]], seconds: float) -> list[chess.Move]:
        futs = [self.pool.submit(lambda i=i, b=b: self.engines[i].play(b, chess.engine.Limit(time=seconds)).move) for i, b in jobs]
        return [f.result() for f in futs]

    def evals(self, boards: list[chess.Board], depth: int = 10) -> list[float]:
        """Full-strength centipawn score (side to move's view) per board; mates count as +-1000."""
        def score(e, b):
            if b.is_checkmate():
                return -1000
            if b.is_game_over():
                return 0
            return e.analyse(b, chess.engine.Limit(depth=depth))["score"].relative.score(mate_score=1000)

        def worker(i):                                   # one engine handles its share one position at a time
            return [(n, score(self.engines[i], boards[n])) for n in range(i, len(boards), len(self.engines))]

        out = [0.0] * len(boards)
        for f in [self.pool.submit(worker, i) for i in range(len(self.engines))]:
            for n, v in f.result():
                out[n] = v
        return out

    def close(self):
        self.pool.shutdown()
        for e in self.engines:
            try:
                e.quit()
            except Exception:
                pass


@dataclass
class Game:
    elo: int | None
    ponder_white: bool
    board: chess.Board = field(default_factory=chess.Board)
    recs: list = field(default_factory=list)      # (fen before Ponder's move, record index into the batch of records)
    fens: list = field(default_factory=list)      # positions where Ponder was to move, and after its move
    result: float | None = None                   # Ponder's score: 1 win, 0.5 draw, 0 loss
    how: str = ""
    ponder_ms: float = 0.0
    ponder_moves: int = 0


def play_games(model, sf: StockfishPool, elos: list, k: int = 4, depth: int = 4, temperature: float = 0.0,
               sf_seconds: float = 0.1, max_plies: int = 200, collect: bool = False, rng: random.Random | None = None,
               judge: StockfishPool | None = None):
    """Play len(elos) games at once (colours alternate). Games still running at max_plies are adjudicated by `judge`
    (a full-strength pool): a lead of 300 cp or more wins, otherwise a draw. Returns (games, records) with records stacked over every
    Ponder move when collect=True: dict of tensors plus 'game' (index into games) and 'fen' (position before the move)."""
    rng = rng or random.Random()
    games = [Game(elo=e, ponder_white=(i % 2 == 0)) for i, e in enumerate(elos)]
    for i, g in enumerate(games):
        sf.set_elo(i, g.elo)
        for u in rng.choice(OPENINGS).split():
            g.board.push_uci(u)
    chunks, owner, fens, mvs = [], [], [], []
    while True:
        live = [i for i, g in enumerate(games) if g.result is None]
        for i in live:
            g = games[i]
            out = g.board.outcome(claim_draw=True)
            if out is not None:
                g.result = 0.5 if out.winner is None else float(out.winner == g.ponder_white)
                g.how = out.termination.name
            elif g.board.ply() >= max_plies:
                g.result, g.how = 0.5, "ply cap"
        live = [i for i in live if games[i].result is None]
        if not live:
            break
        mine = [i for i in live if (games[i].board.turn == chess.WHITE) == games[i].ponder_white]
        theirs = [i for i in live if i not in set(mine)]
        if mine:
            t0 = time.perf_counter()
            moves, rec = search_moves(model, [games[i].board for i in mine], k, depth, temperature)
            dt = (time.perf_counter() - t0) / len(mine)
            for n, i in enumerate(mine):
                g = games[i]
                if collect:
                    fens.append(g.board.fen())
                    mvs.append(moves[n])
                    owner.append(i)
                g.board.push_uci(moves[n])
                g.ponder_ms += 1000 * dt
                g.ponder_moves += 1
            if collect:
                chunks.append(rec)
        if theirs:
            for i, mv in zip(theirs, sf.play([(i, games[i].board) for i in theirs], sf_seconds)):
                games[i].board.push(mv)
    capped = [g for g in games if g.how == "ply cap"]
    if judge is not None and capped:
        for g, cp in zip(capped, judge.evals([g.board for g in capped], 8)):
            white_cp = cp if g.board.turn == chess.WHITE else -cp
            g.result = 0.5 if abs(white_cp) < 300 else float((white_cp > 0) == g.ponder_white)
            g.how = "adjudicated"
    records = None
    if collect and chunks:
        records = {key: torch.cat([c[key] for c in chunks]) for key in chunks[0]}
        records["game"] = torch.tensor(owner)
        records["fen"], records["move"] = fens, mvs
    return games, records


def elo_from_score(opp_elo: float, score: float) -> float:
    s = min(max(score, 0.02), 0.98)
    return opp_elo - 400 * math.log10(1 / s - 1)
