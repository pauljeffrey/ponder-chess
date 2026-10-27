"""ChessNet: encoder + policy head + value head + line decoder.

                 squares/extras
                       |
              +--------v---------+
              |  Board encoder   |   66 tokens: [CLS] [STATE] + 64 squares
              |  (pre-LN xformer |   token = piece-embedding + square-embedding
              |   + 2D relative  |   attention carries a learned bias indexed by the
              |   position bias) |   (rank, file) offset between two squares
              +----+--------+----+
                   |        |
        CLS token  |        | 64 square tokens
                   v        v
             Value head   Policy head  -> logits over MOVE_VOCAB (pointer: from-square x to-square)
             (W/D/L)           |
                               |  top-k root moves (masked to legal moves outside the net)
                               v
                     Line decoder (causal, cross-attends to the encoder output)
                     input  : root move m1, m2, ... (teacher-forced or generated)
                     output : distribution over the next move of the variation

Everything is in the "root frame": the side to move is always white (see board.py).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .moves import FROM_IDX, PROMO_IDX, TO_IDX, VOCAB_SIZE

N_SQ = 64
N_EXTRA = 2          # CLS + STATE
N_TOK = N_SQ + N_EXTRA


class Attention(nn.Module):
    """Multi-head attention. Self-attention when ctx is None, cross-attention otherwise."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)

    def forward(self, x, ctx=None, bias=None, causal: bool = False):
        B, T, D = x.shape
        ctx = x if ctx is None else ctx
        q = self.q(x).view(B, T, self.h, self.dk).transpose(1, 2)
        k, v = self.kv(ctx).view(B, ctx.shape[1], 2, self.h, self.dk).permute(2, 0, 3, 1, 4)
        if causal:
            mask = torch.ones(T, ctx.shape[1], dtype=torch.bool, device=x.device).tril()
            bias = torch.zeros(T, ctx.shape[1], device=x.device, dtype=q.dtype).masked_fill(~mask, float("-inf"))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.o(out.transpose(1, 2).reshape(B, T, D))


class FeedForward(nn.Module):
    def __init__(self, d: int, mult: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d * mult), nn.GELU(), nn.Dropout(dropout), nn.Linear(d * mult, d))

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, cfg.n_heads)
        self.ff = FeedForward(d, cfg.ff_mult, cfg.dropout)
        # learned bias per head for every (rank offset, file offset) in [-7, 7]^2
        self.rel_bias = nn.Parameter(torch.zeros(cfg.n_heads, 15, 15))

    def forward(self, x, rel_index):
        H = self.rel_bias.shape[0]
        sq_bias = self.rel_bias[:, rel_index[0], rel_index[1]]            # [H, 64, 64]
        bias = x.new_zeros(H, N_TOK, N_TOK)
        bias[:, N_EXTRA:, N_EXTRA:] = sq_bias
        x = x + self.attn(self.n1(x), bias=bias.unsqueeze(0))
        return x + self.ff(self.n2(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_attn = Attention(d, cfg.n_heads)
        self.cross_attn = Attention(d, cfg.n_heads)
        self.ff = FeedForward(d, cfg.ff_mult, cfg.dropout)

    def forward(self, x, memory):
        x = x + self.self_attn(self.n1(x), causal=True)
        x = x + self.cross_attn(self.n2(x), ctx=memory)
        return x + self.ff(self.n3(x))


def _rel_index() -> torch.Tensor:
    sq = torch.arange(64)
    rank, file = sq // 8, sq % 8
    dr = rank[None, :] - rank[:, None] + 7   # [64, 64]
    df = file[None, :] - file[:, None] + 7
    return torch.stack([dr, df])


class ChessNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # --- input embeddings -------------------------------------------------
        self.piece_emb = nn.Embedding(13, d)
        self.square_emb = nn.Embedding(N_SQ, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.castle_proj = nn.Linear(4, d)
        self.ep_emb = nn.Embedding(9, d)
        self.halfmove_proj = nn.Linear(1, d)

        # --- encoder ----------------------------------------------------------
        self.layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.n_layers))
        self.final_norm = nn.LayerNorm(d)
        # optional (cfg.loop_embed): one learned vector per pass; the parameter-free default re-applies the stack as is
        if cfg.loop_embed:
            self.enc_loop_emb = nn.Embedding(cfg.enc_loops, d)
            self.dec_loop_emb = nn.Embedding(cfg.dec_loops, d)
        self.register_buffer("rel_index", _rel_index(), persistent=False)

        # --- value head (win / draw / loss for the side to move) --------------
        self.value_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3))

        # --- policy head (pointer over from/to squares) -----------------------
        self.pol_q = nn.Linear(d, d)
        self.pol_k = nn.Linear(d, d)
        self.move_bias = nn.Parameter(torch.zeros(VOCAB_SIZE))
        self.promo_bias = nn.Parameter(torch.zeros(5))
        self.register_buffer("from_idx", torch.tensor(FROM_IDX), persistent=False)
        self.register_buffer("to_idx", torch.tensor(TO_IDX), persistent=False)
        self.register_buffer("promo_idx", torch.tensor(PROMO_IDX), persistent=False)

        # --- line decoder -----------------------------------------------------
        self.move_emb = nn.Embedding(VOCAB_SIZE, d)    # also used (tied) as the output projection
        self.ply_emb = nn.Embedding(cfg.max_line_len, d)
        self.dec_layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.dec_layers))
        self.dec_norm = nn.LayerNorm(d)
        self.dec_bias = nn.Parameter(torch.zeros(VOCAB_SIZE))

        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ encoder
    def _n_loops(self, requested, max_loops):
        """Training: a random pass count per batch (so every depth works). Inference: requested or max."""
        if requested is not None:
            return max(1, min(requested, max_loops))
        if self.training and max_loops > 1:
            return int(torch.randint(1, max_loops + 1, ()).item())
        return max_loops

    def encode(self, squares, castle, ep, halfmove, loops=None):
        """squares [B,64] long, castle [B] long bitmask, ep [B] long (8 = none), halfmove [B] -> [B,66,d]

        loops: how many times to apply the (weight-shared) encoder stack; None = max (eval) / random (train).
        """
        B = squares.shape[0]
        x = self.piece_emb(squares) + self.square_emb.weight[None]
        bits = ((castle[:, None] >> torch.arange(4, device=castle.device)) & 1).float()
        state = (self.castle_proj(bits) + self.ep_emb(ep)
                 + self.halfmove_proj((halfmove.float() / 100.0)[:, None]))[:, None]
        x = torch.cat([self.cls.expand(B, -1, -1), state, x], dim=1)
        # depth recurrence: the embedding above runs once; everything above it is re-applied (no extra parameters)
        for i in range(self._n_loops(loops, self.cfg.enc_loops)):
            if self.cfg.loop_embed:
                x = x + self.enc_loop_emb.weight[i]
            for layer in self.layers:
                x = layer(x, self.rel_index)
        return self.final_norm(x)

    # -------------------------------------------------------------------- heads
    def value_logits(self, memory):
        return self.value_head(memory[:, 0])                       # [B,3]  (win, draw, loss)

    def policy_logits(self, memory):
        sq = memory[:, N_EXTRA:]
        scores = self.pol_q(sq) @ self.pol_k(sq).transpose(1, 2) / math.sqrt(sq.shape[-1])  # [B,64,64]
        logits = scores[:, self.from_idx, self.to_idx]              # [B,V]
        return logits + self.move_bias + self.promo_bias[self.promo_idx]

    def decode(self, memory, line_in, loops=None):
        """Teacher-forced / step-wise decoding.

        memory [B,66,d]; line_in [B,T] vocab ids (m1..mT, all >= 0) -> logits [B,T,V]
        where logits[:, t] is the distribution of move t+2 (the move after line_in[:, t]).
        """
        T = line_in.shape[1]
        pos = torch.arange(T, device=line_in.device).clamp_max(self.cfg.max_line_len - 1)   # past the trained length: reuse the last slot
        x = self.move_emb(line_in) + self.ply_emb(pos)[None]
        for i in range(self._n_loops(loops, self.cfg.dec_loops)):
            if self.cfg.loop_embed:
                x = x + self.dec_loop_emb.weight[i]
            for layer in self.dec_layers:
                x = layer(x, memory)
        x = self.dec_norm(x)
        return x @ self.move_emb.weight.t() + self.dec_bias

    def forward(self, squares, castle, ep, halfmove, loops=None):
        mem = self.encode(squares, castle, ep, halfmove, loops)
        return dict(memory=mem, policy=self.policy_logits(mem), value=self.value_logits(mem))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def wdl_to_score(value_logits: torch.Tensor) -> torch.Tensor:
    """Expected score in [0,1] for the side to move: P(win) + 0.5 * P(draw)."""
    p = value_logits.softmax(-1)
    return p[..., 0] + 0.5 * p[..., 1]
