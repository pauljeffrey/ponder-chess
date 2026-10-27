"""GRPO fine-tuning against the Stockfish analysis stored in the Lichess evals (no live engine).

For every dataset position we sample G (move, variation) rollouts from the model and score them
with the multi-PV analysis that ships with the position:

  move reward   = score(sampled move) - score(best PV move)           (<= 0, "regret" in expected score)
                  a legal move absent from the multi-PV list is assumed worse than the worst listed one
  line reward   = 0.5 * fraction of the Stockfish PV for that move that the sampled variation
                  reproduces *in order* from ply 2 on (earlier failure => less reward)
  illegal       = -0.5 when the sampled variation contains an impossible move (it is cut there)

Advantages are normalised inside each group of G samples (GRPO); no value network is needed.
A frozen reference model adds a KL penalty, and a small supervised loss keeps the value head honest.

Decision term (search in the loop). Inference does not play the policy's move; it takes the top-k moves, calculates a
variation for each, scores the leaf of each variation with the value head and plays softmax(logit(leaf) + w*log prior).
The decision term trains exactly that choice: the k candidates are enumerated (no sampling noise), p = that softmax, and
the loss is -E_p[move reward]. The gradient reaches the value head / encoder (through the leaf evaluations) and the
policy (through the prior), so "which variation ends in the best position" is rewarded by Stockfish's verdict on the
first move. Lines are generated greedily and legally here, as at inference; their own training is the sampled-line term.

RL v2 adds three signals that grade the look-ahead itself against the stored Stockfish analysis:
  rank    - cross-entropy between the engine's decision distribution p (over the k candidates) and Stockfish's own
            preference q = softmax(regret / tau_q) over the same candidates: the final choice is scored against
            Stockfish's final choice, not only through an expected regret.
  ground  - leaf values get a target. (a) When a candidate's generated variation reproduces the stored Stockfish line
            of that first move for its whole length, the leaf is trained towards that line's stored score. (b) The stored
            lines are replayed for d plies (random d <= depth), and the position reached is trained towards the line's
            score, so the value head is graded on exactly the kind of position a leaf is. Both are from the root
            mover's point of view.
  line    - the sampled-line reward is the in-order matched prefix of the stored line for the sampled first move
            (both colours, so black replies count), now at weight `line_reward` (v1: 0.5).
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import chess
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .board import board_from_tokens, to_root_frame
from .config import TrainConfig
from .data import ShardDataset, cp_to_score
from .losses import compute_losses
from .engine import _tensors
from .model import ChessNet, wdl_to_score
from .moves import VOCAB_SIZE, index_to_move, move_to_index
from .train import load_checkpoint, pick_device, save_checkpoint, to_device

UNLISTED_MARGIN_CP = 200.0


def _root_boards(batch) -> list[chess.Board]:
    sq, castle, ep, hm = (batch[k].cpu() for k in ("squares", "castle", "ep", "halfmove"))
    return [board_from_tokens(sq[i].tolist(), int(castle[i]), int(ep[i]), int(hm[i])) for i in range(len(sq))]


def _mask(boards, device):
    m = torch.zeros(len(boards), VOCAB_SIZE, dtype=torch.bool, device=device)
    for i, b in enumerate(boards):
        for mv in b.legal_moves:
            m[i, move_to_index(mv)] = True
    return m


@torch.no_grad()
def sample_lines(model: ChessNet, memory: torch.Tensor, boards: list[chess.Board], first: torch.Tensor, max_len: int):
    """Sample variations *without* legality masking so the decoder learns the rules.

    first [N] long -> (tokens [N,max_len] with -1 padding (the illegal token, if any, is kept as the
    last entry), illegal [N] bool, final boards (before the illegal move, if any))
    """
    N = len(boards)
    tokens = torch.full((N, max_len), -1, dtype=torch.long, device=memory.device)
    tokens[:, 0] = first
    illegal = torch.zeros(N, dtype=torch.bool)
    boards = [b.copy(stack=False) for b in boards]
    alive = list(range(N))
    for i in alive:
        boards[i].push(index_to_move(int(first[i])))
    alive = [i for i in alive if not boards[i].is_game_over()]
    for t in range(1, max_len):
        if not alive:
            break
        seq = tokens[alive, :t].clamp_min(0)
        probs = torch.softmax(model.decode(memory[alive], seq)[:, -1], -1)
        pick = torch.multinomial(probs, 1).squeeze(1)
        still = []
        for i, idx in zip(alive, pick.tolist()):
            tokens[i, t] = idx
            mv = index_to_move(idx)
            if mv in boards[i].legal_moves:
                boards[i].push(mv)
                if not boards[i].is_game_over():
                    still.append(i)
            else:
                illegal[i] = True
        alive = still
    return tokens, illegal.to(memory.device), boards


def move_reward(first, pv_moves, pv_cp):
    """Regret in expected score of root moves `first` [B,G] against the stored multi-PV (<= 0)."""
    B, G = first.shape
    valid = pv_moves[:, :, 0] >= 0
    cp_best = pv_cp.masked_fill(~valid, -1e9).max(1).values
    cp_worst = pv_cp.masked_fill(~valid, 1e9).min(1).values - UNLISTED_MARGIN_CP
    same = (pv_moves[:, None, :, 0] == first[:, :, None]) & valid[:, None, :]      # [B,G,K]
    listed = same.any(-1)
    cp_move = torch.where(listed, (same * pv_cp[:, None, :]).sum(-1), cp_worst[:, None].expand(B, G))
    return cp_to_score(cp_move) - cp_to_score(cp_best)[:, None], same, listed


def rewards(first, tokens, illegal, pv_moves, pv_cp, line_reward: float = 0.5, leaf=None, leaf_weight: float = 0.0):
    """first [B,G]; tokens [B,G,L]; pv_moves [B,K,L]; pv_cp [B,K] -> reward [B,G] and diagnostics."""
    B, G, L = tokens.shape
    r_move, same, listed = move_reward(first, pv_moves, pv_cp)

    ref = (same[..., None] * pv_moves[:, None, :, :]).sum(2)                          # PV line of the sampled move [B,G,L]
    ref_len = (same[..., None] * (pv_moves[:, None, :, :] >= 0)).sum(2).sum(-1) - 1   # plies after the root move
    match = (tokens[..., 1:] == ref[..., 1:]) & (ref[..., 1:] >= 0)
    prefix = match.long().cumprod(-1).sum(-1)                                         # in-order matched plies
    r_line = line_reward * prefix / ref_len.clamp_min(1) * listed * (ref_len > 0)
    r_leaf, gap = torch.zeros_like(r_move), 0.0
    if leaf is not None and leaf_weight > 0:
        # the encoder re-scores the position the line ends in; a faithful line ends where Stockfish says the move leads
        valid = pv_moves[:, :, 0] >= 0
        sf = r_move + cp_to_score(pv_cp.masked_fill(~valid, -1e9).max(1).values)[:, None]     # stored score of that first move
        use = (listed & ~illegal).float()
        err = (leaf - sf).abs() * use
        r_leaf = -leaf_weight * err
        gap = (err.sum() / use.sum().clamp_min(1)).item()
    r = r_move + r_line + r_leaf - 0.5 * illegal.float()
    return r, dict(regret=r_move.mean().item(), line=r_line.mean().item(), leaf_gap=gap,
                   illegal=illegal.float().mean().item(), listed=listed.float().mean().item(),
                   pass_at_g=((r_move > -0.02).any(1)).float().mean().item())


@torch.no_grad()
def _extend(model: ChessNet, memory: torch.Tensor, boards, lines, plies: int, loops=None):
    """Extend legal greedy variations in place by `plies` moves (boards already carry `lines`)."""
    for _ in range(plies):
        alive = [i for i, b in enumerate(boards) if not b.is_game_over()]
        if not alive:
            break
        seq = torch.tensor([lines[i] for i in alive], dtype=torch.long, device=memory.device)
        dec = model.decode(memory[alive], seq, loops=loops)[:, -1].masked_fill(~_mask([boards[i] for i in alive], memory.device),
                                                                  float("-inf"))
        for i, idx in zip(alive, dec.argmax(-1).tolist()):
            boards[i].push(index_to_move(idx))
            lines[i].append(idx)


@torch.no_grad()
def _policy_moves(model: ChessNet, boards, loops=None, topn: int = 1):
    """The encoder's policy head played on the real boards (each re-encoded from scratch): top-`topn` legal moves per
    board, as indices in the same absolute frame the decoder lines use."""
    from .board import to_root_frame
    from .engine import _tensors
    from .moves import mirror_move
    dev = next(model.parameters()).device
    frames = [to_root_frame(b) for b in boards]
    roots = [f[0] for f in frames]
    mem = model.encode(*_tensors(roots, dev), loops=loops)
    lg = model.policy_logits(mem).masked_fill(~_mask(roots, dev), float("-inf"))
    top = lg.topk(min(topn, lg.shape[-1]), -1)
    out = []
    for n, (_, flipped) in enumerate(frames):
        mv = []
        for ix, v in zip(top.indices[n].tolist(), top.values[n].tolist()):
            if v > float("-inf"):
                m = index_to_move(ix)
                mv.append(move_to_index(mirror_move(m) if flipped else m))
        out.append(mv)
    return out


def _extend_policy(model: ChessNet, boards, lines, plies: int, loops=None):
    for _ in range(plies):
        alive = [i for i, b in enumerate(boards) if not b.is_game_over()]
        if not alive:
            break
        for i, mv in zip(alive, _policy_moves(model, [boards[i] for i in alive], loops)):
            boards[i].push(index_to_move(mv[0]))
            lines[i].append(mv[0])


@torch.no_grad()
def greedy_boards(model: ChessNet, memory: torch.Tensor, boards: list[chess.Board], first: torch.Tensor, depth: int,
                  loops=None, policy: bool = False):
    """Legal greedy variations (as at inference) -> the board at the end of each line."""
    boards = [b.copy(stack=False) for b in boards]
    lines = [[int(f)] for f in first]
    for b, f in zip(boards, lines):
        b.push(index_to_move(f[0]))
    if policy:
        _extend_policy(model, boards, lines, depth - 1, loops)
    else:
        _extend(model, memory, boards, lines, depth - 1, loops)
    return boards, lines


@torch.no_grad()
def tree_boards(model: ChessNet, memory: torch.Tensor, boards, first: torch.Tensor, depth: int, branch: int, loops=None,
                policy: bool = False):
    """Per root move: the decoder's top-`branch` replies, each continued greedily to `depth` plies.

    Returns (leaf boards [N*branch], lines, valid [N, branch]); with branch = 1 this is greedy_boards."""
    N = len(boards)
    if branch <= 1 or depth < 2:
        bs, ls = greedy_boards(model, memory, boards, first, depth, loops, policy)
        return bs, ls, torch.ones(N, 1, dtype=torch.bool, device=memory.device)
    bd = [b.copy(stack=False) for b in boards]
    for b, f in zip(bd, first.tolist()):
        b.push(index_to_move(f))
    alive = [i for i, b in enumerate(bd) if not b.is_game_over()]
    replies: dict[int, list[int]] = {}
    if alive and policy:
        for i, mv in zip(alive, _policy_moves(model, [bd[i] for i in alive], loops, branch)):
            replies[i] = mv
    elif alive:
        dec = model.decode(memory[alive], first[alive][:, None], loops=loops)[:, -1].masked_fill(
            ~_mask([bd[i] for i in alive], memory.device), float("-inf"))
        top = dec.topk(min(branch, dec.shape[-1]), -1)
        for n, i in enumerate(alive):
            replies[i] = [ix for ix, v in zip(top.indices[n].tolist(), top.values[n].tolist()) if v > float("-inf")]
    out_b, out_l = [None] * (N * branch), [None] * (N * branch)
    valid = torch.zeros(N, branch, dtype=torch.bool)
    for i in range(N):
        rs = replies.get(i, [])
        for r in range(branch):
            e = i * branch + r
            if r < len(rs):
                x = bd[i].copy(stack=False)
                x.push(index_to_move(rs[r]))
                out_b[e], out_l[e], valid[i, r] = x, [int(first[i]), rs[r]], True
            elif r == 0:                                     # game over after the first move
                out_b[e], out_l[e], valid[i, 0] = bd[i].copy(stack=False), [int(first[i])], True
            else:                                            # fewer legal replies than `branch`: masked placeholder
                out_b[e], out_l[e] = out_b[i * branch].copy(stack=False), list(out_l[i * branch])
    if policy:
        _extend_policy(model, out_b, out_l, depth - 2, loops)
    else:
        _extend(model, memory.repeat_interleave(branch, 0), out_b, out_l, depth - 2, loops)
    return out_b, out_l, valid.to(memory.device)


def softmin(s: torch.Tensor, valid: torch.Tensor, tau: float) -> torch.Tensor:
    """Smooth minimum over the valid replies [N, b]: the opponent picks the reply that is worst for the root mover."""
    n = valid.sum(-1).clamp_min(1).float()
    return -tau * (torch.logsumexp((-s / tau).masked_fill(~valid, float("-inf")), -1) - n.log())   # = mean when equal


def stored_line_boards(boards, pv_moves, pv_cp, d: int):
    """Replay each stored Stockfish line for d plies: (boards reached, root-mover target scores, (b, j) owners)."""
    out, tgt, own = [], [], []
    pm, pc = pv_moves.cpu(), pv_cp.cpu()
    for b in range(len(boards)):
        for j in range(pm.shape[1]):
            if pm[b, j, 0] < 0:
                continue
            ln = pm[b, j, :d].tolist()
            if len(ln) < d or min(ln) < 0:
                continue
            bd = boards[b].copy(stack=False)
            for idx in ln:
                bd.push(index_to_move(idx))
            if bd.is_game_over(claim_draw=False):
                continue
            out.append(bd)
            own.append((b, j))
    if own:
        tgt = cp_to_score(torch.stack([pc[b, j] for b, j in own]))
    return out, tgt, own


def leaf_scores(model, leaves, root_white_flags=None):
    """Root-mover expected score of each non-terminal leaf (grad flows); `leaves` are root-frame boards."""
    mem = model.encode(*_tensors([to_root_frame(b)[0] for b in leaves], next(model.parameters()).device))
    s = wdl_to_score(model.value_logits(mem))
    white = torch.tensor([b.turn == chess.WHITE for b in leaves], device=s.device)
    return torch.where(white, s, 1.0 - s).float()


def eval_boards(model, boards, device):
    """Root-mover expected score of root-frame boards: exact for mate / stalemate, otherwise the encoder's value head.
    Returns (scores [N], indices of the boards the network scored)."""
    fixed, need = torch.zeros(len(boards)), []
    for i, b in enumerate(boards):
        if b.is_checkmate():
            fixed[i] = 1.0 if b.turn == chess.BLACK else 0.0                        # side to move is mated
        elif b.is_game_over(claim_draw=False):
            fixed[i] = 0.5
        else:
            need.append(i)
    score = fixed.to(device)
    if need:
        score = score.index_put((torch.tensor(need, device=device),), leaf_scores(model, [boards[i] for i in need]))
    return score, need


def decision_loss(model: ChessNet, out, boards, logp_all, batch, k: int, depth: int, line_weight: float, tau: float,
                  tau_q: float = 0.05, branch: int = 1, tau_m: float = 0.1):
    """-E_p[reward] where p = softmax over the k candidates of logit(score) + line_weight * log prior, and a candidate's
    score is the soft-min over its `branch` decoder replies of the leaf score (branch = 1: one greedy line)."""
    device = logp_all.device
    B = len(boards)
    top = logp_all.detach().topk(k, -1)
    valid = torch.isfinite(top.values)                                              # fewer than k legal moves
    cand = torch.where(valid, top.indices, top.indices[:, :1].expand(B, k))        # [B,k]
    cand_lp = logp_all.gather(1, cand)                                              # grad -> policy

    flat_boards = [b for b in boards for _ in range(k)]
    leaves, lines, bvalid = tree_boards(model, out["memory"].detach().repeat_interleave(k, 0), flat_boards,
                                        cand.reshape(-1), depth, branch)
    nb = bvalid.shape[1]
    score_e, need = eval_boards(model, leaves, device)                              # grad -> value head, encoder
    need_set = set(need)
    score = softmin(score_e.view(B * k, nb), bvalid, tau_m).view(B, k)

    z = torch.logit(score.clamp(1e-4, 1 - 1e-4)) + line_weight * cand_lp
    z = z.masked_fill(~valid, float("-inf"))
    p = torch.softmax(z / tau, -1)
    r, same, listed = move_reward(cand, batch["pv_moves"], batch["pv_cp"])
    with torch.no_grad():
        pick = z.argmax(-1)
        stock_best = r.masked_fill(~valid, -1e9).argmax(-1)                          # Stockfish's choice among the candidates
        diag = dict(dec_exp=(p * r).sum(1).mean().item(),
                    dec_greedy=r.gather(1, pick[:, None]).mean().item(),         # what the engine would play
                    policy_top1=r[:, 0].mean().item(),                           # what the bare policy would play
                    agree=(pick == stock_best).float().mean().item(),
                    recall=(r.max(1).values > -0.02).float().mean().item())      # a near-best stored move is in the top-k
    # rank: the decision distribution should match Stockfish's preference over the same candidates
    q = torch.softmax((r / tau_q).masked_fill(~valid, float("-inf")), -1).detach()
    rank = -(q * torch.log_softmax(z, -1).masked_fill(~valid, 0.0)).sum(1).mean()

    # ground (a): a generated variation == the stored Stockfish line for its first move -> its leaf has a target
    pm, pc = batch["pv_moves"].cpu(), batch["pv_cp"].cpu()
    own, tgt = [], []
    for i in range(len(leaves)):
        c_idx = i // nb
        b, c = divmod(c_idx, k)
        if i not in need_set or not bvalid[c_idx, i % nb] or not valid[b, c] or not listed[b, c]:
            continue
        j = int(same[b, c].float().argmax())
        if len(lines[i]) == depth and pm[b, j, :depth].tolist() == lines[i]:
            own.append(i)
            tgt.append(float(cp_to_score(pc[b, j])))
    if own:
        ground = F.binary_cross_entropy(score_e[own].clamp(1e-4, 1 - 1e-4), torch.tensor(tgt, device=device))
    else:
        ground = score_e.sum() * 0.0
    diag["matched"] = len(own) / max(1, int(bvalid.sum()))

    # recall (pass@k for the policy): probability mass on *any* near-best stored move should be high
    sm, sc = batch["pv_moves"][:, :, 0], cp_to_score(batch["pv_cp"].float())
    sv = sm >= 0
    near = sv & (sc >= sc.masked_fill(~sv, -1.0).max(1, keepdim=True).values - 0.02)
    mass = logp_all.gather(1, sm.clamp_min(0)).masked_fill(~near, float("-inf")).logsumexp(-1)
    recall_l = -mass.mean()

    # branch: the decoder's explored replies are matched against the stored Stockfish line of that first move, ply by ply
    # (both colours, whole variation). Expected match over the explored replies, weighted by how good Stockfish
    # thinks the first move is -> gradient into the decoder's reply distribution (and encoder), pass@b style.
    branch_l = score_e.sum() * 0.0
    hit = tot = 0
    pref_best = 0.0
    if True:
        pos, rows, rews = [], [], []
        for c_idx in range(B * k):
            b, c = divmod(c_idx, k)
            if not valid[b, c] or not listed[b, c]:
                continue
            j = int(same[b, c].float().argmax())
            ref = pm[b, j, :depth].tolist()
            den = max(1, min(depth, sum(1 for x in ref if x >= 0)) - 1)
            reps, rw = [], []
            for rr in range(nb):
                e = c_idx * nb + rr
                if not bvalid[c_idx, rr] or len(lines[e]) < 2:
                    continue
                n = 0
                for a, bb in zip(lines[e][1:], ref[1:]):
                    if a != bb or bb < 0:
                        break
                    n += 1
                reps.append(lines[e][1])
                rw.append(n / den)
            if reps:
                pos.append((b, c)); rows.append((c_idx, reps)); rews.append(rw)
                tot += 1
                hit += any(x > 0 for x in rw)
                pref_best += max(rw)
        if rows:
            first_t = cand.reshape(-1)
            idx = torch.tensor([ci for ci, _ in rows], device=device)
            mem_c = out["memory"].repeat_interleave(k, 0)[idx]
            logits = model.decode(mem_c, first_t[idx][:, None])[:, -1]
            after = []
            for ci, _ in rows:
                bd = flat_boards[ci].copy(stack=False)
                bd.push(index_to_move(int(first_t[ci])))
                after.append(bd)
            logits = logits.masked_fill(~_mask(after, device), float("-inf"))
            loss_rows = []
            for n, (ci, reps) in enumerate(rows):
                pr = torch.softmax(logits[n, reps], -1)
                loss_rows.append((pr * torch.tensor(rews[n], device=device)).sum())
            w = torch.stack([q[b, c] for b, c in pos]).detach()
            branch_l = -(w * torch.stack(loss_rows)).sum() / B
    diag["hit_b"] = hit / max(1, tot)
    diag["pref_b"] = pref_best / max(1, tot)
    return -(p * r).sum(1).mean(), rank, ground, dict(recall=recall_l, branch=branch_l), diag


def grpo(cfg: TrainConfig, init_ckpt: str, group: int = 6, beta: float = 0.02, w_sup: float = 0.1,
         w_dec: float = 1.0, k: int = 4, depth: int = 4, line_weight: float = 0.25, tau: float = 1.0,
         w_rank: float = 0.0, w_ground: float = 0.0, line_reward: float = 0.5, leaf_reward: float = 0.0,
         branch: int = 1, tau_m: float = 0.1, w_recall: float = 0.0, w_branch: float = 0.0) -> Path:
    torch.manual_seed(cfg.seed)
    device = pick_device(cfg.device)
    model = load_checkpoint(init_ckpt, "cpu").to(device).train()
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    ds = ShardDataset(cfg.data_dir)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    L = model.cfg.max_line_len
    out_dir, step, t0 = Path(cfg.out_dir), 0, time.time()
    print(f"[rl] {len(ds)} positions | group={group} beta={beta} device={device}")

    while step < cfg.steps:
        for batch in dl:
            batch = to_device(batch, device)
            boards = _root_boards(batch)
            B = len(boards)
            out = model(batch["squares"], batch["castle"], batch["ep"], batch["halfmove"])
            legal = _mask(boards, device)
            logp_all = F.log_softmax(out["policy"].masked_fill(~legal, float("-inf")), -1)

            with torch.no_grad():
                ref_out = ref(batch["squares"], batch["castle"], batch["ep"], batch["halfmove"])
                ref_logp_all = F.log_softmax(ref_out["policy"].masked_fill(~legal, float("-inf")), -1)
                first = torch.multinomial(logp_all.exp(), group, replacement=True)            # [B,G]
                flat_boards = [b for b in boards for _ in range(group)]
                mem_bg = out["memory"].detach().repeat_interleave(group, 0)
                tokens, illegal, fin = sample_lines(model, mem_bg, flat_boards, first.reshape(-1), L)
                tokens = tokens.view(B, group, L)
                illegal = illegal.view(B, group)
                leaf = None
                if leaf_reward > 0:                                   # encoder re-scores where each sampled line ends
                    leaf = eval_boards(model, fin, device)[0].view(B, group)
                r, diag = rewards(first, tokens, illegal, batch["pv_moves"], batch["pv_cp"], line_reward, leaf, leaf_reward)
                adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)

            # ---- policy-gradient terms (teacher-forced re-scoring of the sampled rollouts) ----
            lp_move = logp_all.gather(1, first)                                                 # [B,G]
            ref_lp = ref_logp_all.gather(1, first)
            kl = (ref_lp - lp_move).exp() - (ref_lp - lp_move) - 1                              # k3 estimator
            flat_tok = tokens.view(B * group, L)
            dec_logits = model.decode(out["memory"].repeat_interleave(group, 0), flat_tok[:, :-1].clamp_min(0))
            tgt = flat_tok[:, 1:]
            tok_lp = -F.cross_entropy(dec_logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1),
                                      ignore_index=-1, reduction="none").view(B * group, L - 1)
            n_tok = (tgt >= 0).sum(1).clamp_min(1)
            lp_line = (tok_lp.sum(1) / n_tok).view(B, group)
            pg = -(adv * (lp_move + lp_line)).mean()
            sup = compute_losses(model, batch, cfg.policy_temp_cp)
            loss = pg + beta * kl.mean() + w_sup * (sup["policy"] + sup["value"] + sup["line"])
            dd = {}
            if w_dec > 0:
                dec_l, rank_l, ground_l, extra, dd = decision_loss(model, out, boards, logp_all, batch, k, min(depth, L),
                                                            line_weight, tau, branch=branch, tau_m=tau_m)
                loss = loss + w_dec * dec_l + w_rank * rank_l + w_ground * ground_l \
                    + w_recall * extra["recall"] + w_branch * extra["branch"]
                dd.update(rank=rank_l.item(), ground=ground_l.item())
                if w_ground > 0:                                        # ground (b): stored lines replayed to a leaf
                    d = int(torch.randint(1, min(depth, L) + 1, ()))
                    sb, st, _ = stored_line_boards(boards, batch["pv_moves"], batch["pv_cp"], d)
                    if sb:
                        g2 = F.binary_cross_entropy(leaf_scores(model, sb).clamp(1e-4, 1 - 1e-4), st.to(device))
                        loss = loss + w_ground * g2
                        dd["ground_pv"] = g2.item()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % cfg.log_every == 0 or step == 1:
                print(f"[rl] step {step:>5} loss {loss.item():.3f} reward {r.mean().item():+.3f} "
                      f"regret {diag['regret']:+.3f} pass@{group} {diag['pass_at_g']:.2f} "
                      f"line {diag['line']:.3f} gap {diag['leaf_gap']:.3f} illegal {diag['illegal']:.2f} kl {kl.mean().item():.4f} "
                      + (f"| search {dd['dec_greedy']:+.3f} vs policy {dd['policy_top1']:+.3f} (exp {dd['dec_exp']:+.3f}) "
                         f"agree {dd['agree']:.2f} recall {dd['recall']:.2f} matched {dd['matched']:.2f} "
                         f"pass@b {dd['hit_b']:.2f} pref {dd['pref_b']:.2f} rank {dd['rank']:.3f} ground {dd['ground']:.3f}/{dd.get('ground_pv', float('nan')):.3f} "
                         if dd else "") + f"| {time.time() - t0:.0f}s", flush=True)
            if step % cfg.save_every == 0:
                save_checkpoint(out_dir / "last.pt", model, step)
            if step >= cfg.steps:
                break
    path = out_dir / "last.pt"
    save_checkpoint(path, model, step)
    print(f"[rl] saved {path}")
    return path
