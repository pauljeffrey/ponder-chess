# How the Chess Agent works

Code: [chessagent/model.py](chessagent/model.py) (network), [chessagent/engine.py](chessagent/engine.py) (inference),
[chessagent/losses.py](chessagent/losses.py) + [chessagent/train.py](chessagent/train.py) (pretraining),
[chessagent/rl.py](chessagent/rl.py) (GRPO).

## 0. The one convention that simplifies everything: the root frame

The side to move is always **white, moving up the board**. A black-to-move position is mirrored (a1<->a8, colors
swapped) before it enters the network, and moves are mirrored back on the way out. The network therefore never has to
learn "and now do everything again for black". Moves are `(from, to, promotion)` triples drawn from a fixed
vocabulary of ~1,900 geometrically possible moves ([moves.py](chessagent/moves.py)); castling is the king's two-square move.

## 1. Encoder: "read the board"

**Input tokens (66).**

| token | built from |
|---|---|
| CLS | one learned vector |
| STATE | linear(castling rights, 4 bits) + embedding(en-passant file, 9 values) + linear(halfmove clock / 100) |
| 64 squares | `piece_embedding(13 kinds)` + `square_embedding(64)` |

A square token therefore contains *what* is on the square and *where* it is. Nothing is pooled or reordered
afterwards, so token *i* is "square *i*" in every layer. (Summing two embeddings equals concatenating them and
applying a learned linear map, but keeps the width at `d_model`.)

**Layer** (pre-LayerNorm transformer): `x += Attn(LN(x))`, then `x += FFN(LN(x))`. Attention is ordinary multi-head
attention plus a **learned bias per head indexed by the (rank offset, file offset) between two squares** (a 15x15 table
per head). That is what lets a head specialise as "look along the file", "knight jump", "diagonal neighbour" without
the model having to rediscover board geometry from scratch. CLS and STATE attend to everything with no geometric bias.

**Output.** 66 vectors of width `d_model` (384 for `base`), after a final LayerNorm. Two heads read them:

* **Value head** (CLS token -> MLP -> 3 logits: win / draw / loss for the side to move).
* **Policy head** (pointer): `q = W_q * square_tokens`, `k = W_k * square_tokens`, `scores[f,t] = q_f . k_t / sqrt(d)` gives a
  64x64 grid. Each vocabulary move reads its own `(from, to)` cell, plus a learned per-move bias and a promotion-type
  bias. So "which piece, which square" is answered by comparing the two squares' representations, and
  disambiguation (Ngf3 vs Nef3) is free because the from-square is explicit.

## 2. Decoder: "calculate a variation"

### 2.1 What flows from the encoder into the decoder

Exactly one thing: the **memory**, the encoder's final output `[B, 66, d]` (CLS, STATE and the 64 square vectors after the
last LayerNorm). It is computed **once per root position** and shared by every candidate line and every decoding step.
Nothing else crosses over: the decoder gets no board tensor, no policy logits and no value. The policy head, the value
head and the decoder are three readers of the same memory.

```
board ──► embed once ──► ENCODER stack (x n passes) ──► memory [66,d] ──┬─► value head (CLS)  → W/D/L
                                                                         ├─► policy head       → top-k root moves
                                                                         └─► keys / values for cross-attention
root move m1 (from policy top-k) ─► DECODER input embedding ─► [self-attn → cross-attn → FFN] x layers (x n passes)
                                                                         └─► logits over 1,900 moves = next ply
```

### 2.2 Decoder input

At step *t* the input sequence is the moves of the line so far, `m1 … mt`, each embedded as
`move_embedding(m) + ply_embedding(position in line)`. `m1` is the root move itself (taken from the policy's top-k at
inference, from the stored Stockfish PV in pretraining). All moves are in the root frame, so a black reply is simply a
move of the opposite colour's pieces.

### 2.3 One decoder layer, and what each attention looks at

1. **Causal self-attention** (queries, keys, values all from the line's own move tokens). Move *t* can see moves
   `1..t`, never later ones. This is where the decoder keeps track of what it has already played: "I moved the knight
   to f5, the opponent took it, ..." It never sees the real board after those moves.
2. **Cross-attention** (queries from the line's hidden states; keys and values are linear maps of the 66 memory
   tokens). This is how the line reads the board: a head can learn to fetch "what stands on the from-square of the
   last move", "which of my pieces attack this square", "where is the enemy king". It has no geometric bias; the
   geometry was already baked into the memory tokens by the encoder.
3. **FFN**, each sub-layer wrapped in pre-LayerNorm and a residual.

### 2.4 What it predicts

After the last layer a LayerNorm and a projection tied to the move embedding give **logits over the move vocabulary
at every position**: row *t* is the distribution of move *t+1*, i.e. the move that follows `m_t`. Teacher forcing
trains all rows at once (input `m1..m7`, target `m2..m8`, up to 4 PVs per position, padding ignored). At inference
only the last row is used, to extend the line by one move.

Honest limit of this design: the decoder reads the **root** board only. After move 1 the position exists solely as
the decoder's hidden state, which is a learned approximation of the real position. That is why every generated move is
masked to legal moves on the real python-chess board, and why very long lines drift. It re-runs the whole prefix each
step (no KV cache); with 8 plies that costs nothing.

### 2.5 Producing several variations at inference ([engine.py](chessagent/engine.py))

1. Encode the root once. Mask the policy to legal moves and take the top-*k* (default 4).
2. Build *k* boards, push each candidate. Run the decoder **on all *k* lines as one batch**, step by step: each step
   masks logits to the legal moves of that line's board, takes the argmax, pushes it. Lines that end (mate, draw)
   drop out of the batch.
3. Evaluate the *k* leaf positions with one batched encoder pass (value head), converted to the root mover's point of
   view. Checkmate / stalemate leaves get exact scores.
4. Decide: `z_i = logit(leaf_score_i) + 0.25 * log prior_i`, then argmax (or sample from `softmax(z / T)` at low
   strength levels). Cost: 1 root pass + (depth-1) tiny decoder steps (batch *k*) + 1 leaf pass (batch *k*).

## 3. Pretraining

**Data.** Lichess evaluation database: a position plus Stockfish analysis, often with several principal variations.
For each position we keep the analysis with the most lines (then the deepest), up to 4 PVs x 8 plies, everything converted
to the root frame and the mover's point of view (the database stores scores from White's point of view; checked on the
real file). Stored as ~144 bytes/position.

**Targets and losses** (all computed from one forward pass; any missing piece is masked, not dropped):

| head | target | loss |
|---|---|---|
| policy | softmax over the PVs' first moves of `score_cp / 100` (so a close second-best keeps probability mass) | soft cross-entropy |
| value | heuristic centipawn -> (win, draw, loss) curve of the best line | cross-entropy |
| decoder | the PV continuations, teacher-forced | token cross-entropy, padding ignored |

Total = policy + value + line (equal weights). AdamW, cosine schedule with warmup, bf16 on GPU, grad-clip 1.0, held-out
shard for validation (policy / value / line loss, top-1 and top-4 agreement with Stockfish's best move).
Illegal moves are not penalised in pretraining; the network learns legality softly, and inference masks the rest.

## 4. RL: GRPO against the stored analysis

There is no live Stockfish. Every position ships with the engine's verdict on its best few first moves
(`pv_cp`, `pv_moves`), and the reward is computed from that. Two terms are trained together.

### 4.1 Decision term: the softmax over variations (search in the loop)

This is the choice the engine really makes at inference, so RL trains it directly:

```
policy → top-k legal root moves m_1..m_k                      (log prior_i, with gradient)
for each m_i: greedy legal variation of `depth` plies          (decoder, no gradient, as at inference)
leaf_i = value head on the final position of that variation    (encoder + value head, WITH gradient)
z_i    = logit(leaf_score_i) + 0.25 * log prior_i
p_i    = softmax_i(z_i)                                         ← "which variation leads to the best position"
reward_i = score(m_i) − score(Stockfish best)                   ≤ 0, from the stored multi-PV
loss   = − Σ_i p_i · reward_i
```

The k candidates are enumerated rather than sampled, so the expectation `Σ p_i · reward_i` is exact and has no
sampling noise (a baseline would not change its gradient). Gradients reach:

* the **value head and encoder**, through the leaf scores: positions at the end of variations that lead to
  good Stockfish-rated first moves are pushed up, the ones behind bad first moves are pushed down;
* the **policy**, through the prior inside `z_i` (and through the top-k set of the next step).

The log line prints `search` (the regret of what the full engine would play), `policy` (the regret of the bare
policy's top-1) and `exp` (the softmax expectation). **`search` better than `policy` is the evidence that
variations help; if it is not, the search is not earning its compute.** This is checked, not assumed.

### 4.2 Sampled-line term (trains the decoder)

Per position sample **G** rollouts: a root move from the legal-masked policy, then a variation from the decoder
(**not** legality-masked, so mistakes are visible and punishable).

* `move reward` = same regret as above (≤ 0). A legal move absent from the multi-PV list is assumed worse than the
  worst listed one.
* `line reward` = 0.5 × fraction of the stored PV for that move reproduced *in order*; an early divergence stops the
  count, so depth of failure matters.
* `illegal` = −0.5 and the variation is cut there.
* Advantage = (reward − group mean) / group std inside each position's group; no value network.
* Loss = −advantage × (log-prob of the move + mean log-prob of the line tokens) + β·KL(policy ‖ frozen reference)
  (k3 estimator). This is the term that teaches the decoder on its own, imperfect rollouts instead of
  teacher-forced prefixes, which is where compounding errors live.

### 4.3 Total

`loss = sampled-line term + decision term + 0.1 × supervised loss`. The supervised loss keeps the value head
calibrated and the policy from drifting while the reward moves things around.

Why bother when pretraining already sees the same labels? Imitation learns the *top* move. RL optimises the
*decision the engine makes*, with a graded penalty for second-best and bad moves, through the same variation-and-leaf
pipeline that runs at play time.

### 4.4 RL v3: grading the look-ahead itself

The leaf is never read from the decoder. The decoder writes moves, python-chess plays them on a board, and the
**encoder re-reads the end position** (CLS -> value head). The decoder therefore influences the decision only through
*which* position it hands the encoder, and the decision loss cannot back-propagate through discrete moves. v3 closes that
loop with rewards:

* **Leaf-consistency reward (decoder).** For each sampled line the encoder scores where it ends; the line is rewarded for
  `-|leaf - Stockfish's stored score of its first move|` (a faithful line ends where Stockfish says the move leads), on top of
  the in-order matched-prefix reward for both colours.
* **Rank.** Cross-entropy between the engine's decision distribution over the k candidates and Stockfish's preference over the
  same candidates, `softmax(regret / 0.05)`: the final choice is graded against Stockfish's final choice.
* **Ground.** The leaf gets a target: when a generated line reproduces the stored Stockfish line, the leaf is trained to
  that line's score; stored lines are also replayed to a random depth and the position reached is trained to the line's score.
* **Branching.** Per candidate the decoder's top-`b` replies are explored (each continued greedily); the candidate's score is
  a soft-min over its replies (`-tau * (logsumexp(-s/tau) - log n)`, tau = 0.1), i.e. the opponent is assumed to pick the
  worst reply for us. `b = 1` is the earlier single greedy line. The same function runs in training and at inference.

* **Policy pass@k (recall).** `-log` of the probability mass on *any* near-best stored move (within 0.02 expected score of
  Stockfish's best): the policy is rewarded for putting a best move among its candidates, not for matching one exact label.
* **Branch match (decoder, pass@b).** For every candidate whose first move Stockfish listed, each explored reply is scored by
  how many consecutive plies of the stored Stockfish line it reproduces (both colours, whole variation, not only the leaf).
  The loss is the expected match over the explored replies under the decoder's own reply distribution, weighted by how good
  Stockfish thinks the first move is. A candidate's variation is credited if *any* branch gets it right. The log shows
  `pass@b` (a branch reproduces the stored reply) and `pref` (best matched fraction of the stored line).

Run: `scripts/rl.py --w-rank 0.5 --w-ground 1.0 --line-reward 1.0 --leaf-reward 0.5 --branch 2 --w-recall 0.5 --w-branch 1.0`.

## 5. Recurrence ("thinking deeper with the same weights") and test-time compute

**Looped depth.** The input embedding is computed **once**. Everything above it, the full transformer stack up to the
last layer, is then applied `n` times in a row, feeding its own output back in. There are **no extra parameters**:
`base` with 1 pass and `base` with 2 passes are the same 32.0M weights; the second just costs about twice the
compute. (The decoder does the same with its own embedding and layer stack.) There is no "pass number" input; the
stack only sees the residual stream it is refining.

During training a random pass count `1..enc_loops` is drawn per batch, so every depth gives a usable answer and
the model can be asked for more passes at inference (`loops`, wired into [engine.py](chessagent/engine.py); low
strength levels use 1). A model trained with `enc_loops=1`, like the current pretraining run, has never seen a second
pass and should not be run with more. To get a model that benefits, train with `--model base_loop` (same weights,
costs up to 2x compute per step). Whether extra passes beat extra width/depth at equal compute on chess is empirical.

**Variation search knobs** (inference only, no retraining): `k` (candidate moves), `depth` (plies per line), `loops`,
temperature. All k lines run as one batch, so raising k from 4 to 8-16 costs far less than 2-4x on a GPU and scales
roughly linearly on CPU. A natural next step is an adaptive budget: spend more k / depth when the top candidates'
scores are close or the position is tactical, and little when one move dominates.

## 6. Honest limits, including what a leaf value is worth

**What a "leaf value" is.** It is the value head's guess, from the *same network* that produced the candidates,
about a position that was never played: the end of a greedy line the decoder imagined. Nothing verifies it.

* **The reward does not grade the leaf.** Stockfish's verdict (stored cp) exists only for the *root* position and its
  first few moves. RL therefore teaches leaf values only indirectly: "a leaf that ends up behind a good first move
  should look good". It never learns "this leaf is +0.8" from ground truth.
* **Off-distribution inputs.** The value head was trained on positions from real games. Leaves come from imagined
  continuations, possibly with decoder mistakes, mid-capture or mid-check ("horizon effect"), which look unlike
  anything in the data. Errors there are largest exactly where tactics matter.
* **Correlated errors.** The value head and the policy share the encoder and the data. If the net misreads a
  position, the prior and the leaf value usually misread it the same way, so the softmax cannot correct it.
* **One greedy line, not minimax.** Each candidate gets a single opponent reply chain, the decoder's guess, not the
  opponent's best. A move that loses to a refutation the decoder did not choose still gets a good leaf score.
* **Reward noise.** Moves missing from the stored multi-PV are given a fixed assumed penalty; the stored depth varies
  by position; the cp→win/draw/loss curve is a heuristic, not fitted to game results.
* **What it does buy:** a cheap second opinion per candidate, exact scores for mates and stalemates, and a trainable
  tie-break between close candidates. How much strength that adds is measured by `evaluate.py` (policy-only vs search
  at several budgets) and by the `search` vs `policy` numbers in the RL log, not claimed here.
* **No playing-strength number exists yet.** Held-out agreement with Stockfish is a proxy; real strength needs games
  against a reference engine.
* **Upgrade path.** Branch at the opponent's replies (a tiny batched MCTS), and ground leaf values by supervising
  them against the stored cp whenever a generated line reproduces a stored PV.
