# Deploying Ponder

Two parts: the **backend** (FastAPI + PyTorch on CPU, Docker) and the **frontend** (the static `web/` folder).
Model weights are too large for git (up to ~128 MB each), so they live outside the repo.

## 1. Weights

Export slim fp16 files and put them where the server can fetch them:

```bash
python scripts/export_weights.py runs/box/<checkpoint>.pt weights/ponder-64m.pt --strip-decoder   # repeat per model
```

`models.json` lists the files the server expects (`weights/<name>.pt`). Upload them to a Hugging Face model repo
(public, or private with a token), then give the backend the base URL:

```
PONDER_WEIGHTS_BASE=https://huggingface.co/BeardedMonster/ponder/resolve/main
HF_TOKEN=...            # only if the repo is private
```

On start the container downloads any missing file into `/app/weights`; mount a volume there so it only happens once.

## 2. Backend on Dokploy (Contabo)

1. Dokploy -> project -> **Create Service -> Application**, source **GitHub** (this repo), branch `main`.
2. Build type **Dockerfile** (path `./Dockerfile`), no build args.
3. **Environment**:
   - `PONDER_WEIGHTS_BASE`, `HF_TOKEN` (see above)
   - `CHESS_CORS=https://<your-vercel-domain>` (comma-separate several)
   - `CHESS_THREADS=2` (torch threads per search; searches run one at a time)
   - `PONDER_MAX_COST=4000` (cap on candidates x replies x depth, so one visitor cannot hog the CPU)
4. **Advanced -> Volumes**: mount a volume at `/app/weights`.
5. **Domains**: add e.g. `ponder-api.<yourdomain>`, container port `8000`, HTTPS (Let's Encrypt). HTTPS is required because
   the Vercel page is https.
6. Deploy. Check `https://ponder-api.<yourdomain>/api/health` and `/api/models`.

`--strip-decoder` drops the line decoder (a third of the parameters), which the deployed look-ahead does not use: the
look-ahead replays the policy head on a real board, and the page shows the variations growing live
(`POST /api/move/stream`, newline-delimited JSON). A stripped 32M model is ~44 MB on disk and ~90 MB in RAM, a
stripped 64M model ~85 MB on disk and ~170 MB in RAM, so all models fit comfortably on a shared machine.

## 3. Frontend on Vercel

1. Set the backend address in `web/config.js`: `window.PONDER_API = "https://ponder-api.<yourdomain>";` and push.
2. Vercel -> **Add New Project**, import this repo, **Root Directory: `web`**, Framework Preset **Other**, no build command.
3. Deploy. Stockfish for watch mode is served from `web/vendor/` (runs in the browser, nothing to configure).

## 4. Portfolio

Add the Vercel URL to `src/data/portfolio.ts` in the portfolio repo (an `iframe` item) and push; Vercel redeploys.
