# Ponder

A small transformer chess agent: pretrain on Stockfish analysis (Lichess evals), fine-tune with reinforcement learning
against the same analysis, and play with a look-ahead (lines replayed on a real board, end positions re-read by the
encoder). Served through FastAPI with a web UI that can also stage Ponder vs Stockfish games.
Deployment (Dokploy backend + Vercel frontend): see [DEPLOY.md](DEPLOY.md).

```
chessagent/model.py    encoder + policy head + value head + line decoder   <- the architecture
chessagent/moves.py    move vocabulary (pointer head targets), mirroring
chessagent/board.py    board <-> tokens, root-frame (mover = white) encoding
chessagent/data.py     Lichess eval JSONL -> numpy shards -> Dataset, WDL targets
chessagent/losses.py   policy / value / line-decoder losses
chessagent/train.py    supervised pretraining loop
chessagent/rl.py       GRPO against the multi-PV analysis stored in the dataset
chessagent/engine.py   inference: policy -> variations -> leaf values -> decision
server/app.py          FastAPI (/api/move, /api/move/stream, /api/health) + static web UI
web/                   the game interface (chess.js for rules)
```

## Mac smoke test (no downloads)

```bash
source .venv/bin/activate
python scripts/make_synthetic.py --n 1500 --pov white            # fake eval file (~45 s)
python scripts/prepare_data.py --src data/synthetic/evals.jsonl --pov white --max-positions 1500
python scripts/train.py --model tiny --steps 150 --batch-size 32
python scripts/rl.py --ckpt runs/pretrain/last.pt --steps 6 --batch-size 4 --group 4
CHESS_CKPT=runs/pretrain/last.pt uvicorn server.app:app --port 8000   # open http://localhost:8000
pytest -q tests
```

## Real data

```bash
# 1. tiny real sample (~100 MB download) and sign check
python scripts/prepare_data.py --src https://database.lichess.org/lichess_db_eval.jsonl.zst \
    --max-bytes 100000000 --max-positions 50000 --pov stm --out data/sample
python scripts/check_pov.py <(curl -s -r 0-30000000 https://database.lichess.org/lichess_db_eval.jsonl.zst | zstd -dc 2>/dev/null)
#    -> rerun prepare_data with the --pov it prints

# 2. GPU box: stream, never decompress to disk (shards ~144 B/position)
python scripts/prepare_data.py --src https://database.lichess.org/lichess_db_eval.jsonl.zst \
    --max-positions 50000000 --shard-size 500000 --workers 16 --pov <stm|white>
python scripts/train.py --model base --steps 200000 --batch-size 512 --workers 8
python scripts/rl.py --ckpt runs/pretrain/last.pt --steps 5000 --batch-size 64 --group 8
```

## Deploy

`CHESS_CKPT=/path/last.pt CHESS_THREADS=4 uvicorn server.app:app --host 0.0.0.0 --port 8000`
(`CHESS_CORS=https://yourfront.end` only if the UI is hosted elsewhere; the UI also takes `?api=https://host`).
