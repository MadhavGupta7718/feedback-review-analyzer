# Deployment

**Status: NOT DEPLOYED.** The configuration below is written and was rehearsed locally with production settings (see
"Production rehearsal"). No Render or Vercel service exists yet. Creating them needs the owner's Render and Vercel accounts and
a push of the local commits to GitHub; the owner asked to keep commits local for now.

## Shape

| Part | Host | What runs | Needs |
|---|---|---|---|
| API | Render (free web service, Python) | `uvicorn backend.app.main:app`, serving the committed `artifacts/analytics.db` read-only | `requirements-api.txt` only: FastAPI, uvicorn, pydantic |
| Dashboard | Vercel (static) | `frontend/` built by Vite | `VITE_API_BASE_URL` at build time |
| Pipeline and Qwen | The GPU laptop (offline) | `python -m ml.pipeline`, `scripts/generate_brief.py` | CUDA GPU, models in `D:\huggingface` |

Nothing heavy is deployed:

- no model, no torch, no Hugging Face token and no dataset;
- the deployed database holds only redacted synthetic reviews plus aggregate reports;
- the API runs with `BRIEF_MODE=precomputed`, so the product brief is the stored, validated Qwen brief (`qwen_precomputed`), with the template as fallback.

**Docker is not required.** There is one stateless Python process and one static site. Render and Vercel build both natively
from the repo, so a container would add a build step without adding isolation this project needs.

## 1. Push the code (owner)

```powershell
cd D:\Microsoft\feedback-review-analyzer
git push origin main
```

## 2. API on Render

1. Render dashboard → **New** → **Blueprint** → pick the `feedback-review-analyzer` repo. Render reads `render.yaml`:
   - build command: `pip install -r requirements-api.txt`
   - start command: `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
   - health check: `/health`
   - `PYTHON_VERSION=3.11.9`, `ANALYTICS_DB=artifacts/analytics.db`, `BRIEF_MODE=precomputed`, offline flags
   - `autoDeploy: false`, so you deploy manually
2. Render asks for the two `sync: false` variables:
   - `CORS_ORIGINS`: the Vercel production URL, e.g. `https://feedback-review-analyzer.vercel.app` (no trailing slash).
     If Vercel is not created yet, enter a placeholder, then update it after step 3 and redeploy.
   - `CORS_ORIGIN_REGEX` (optional), for this project's preview URLs only, e.g.
     `^https://feedback-review-analyzer-[a-z0-9-]+-<your-vercel-team>\.vercel\.app$`. Leave it empty if you don't need it.
3. Deploy, then check:
   ```powershell
   curl https://<render-service>.onrender.com/health
   # expect {"status":"ok","database":"ok","reviews":10104,...}
   ```
   The free plan sleeps after inactivity, so the first request can take about 30–60 s.

## 3. Dashboard on Vercel

1. Vercel → **Add New Project** → import the repo, then set:
   - **Root Directory** `frontend`
   - **Framework** Vite
   - **Build Command** `npm run build`
   - **Output Directory** `dist`
2. Environment variable: `VITE_API_BASE_URL=https://<render-service>.onrender.com` (no trailing slash). Vite bakes it
   into the bundle at build time, so redeploy after changing it.
3. `frontend/vercel.json` provides the SPA rewrite, so deep links like `/radar?issue=theme_005` work. It also sets the `nosniff`,
   `Referrer-Policy` and `X-Frame-Options: DENY` headers.
4. Put the resulting URL into Render's `CORS_ORIGINS` and redeploy the API.

## 4. Verify after deploying

Only call it deployed once every check below passes:

- [ ] `GET /health` returns `ok` with 10,104 reviews
- [ ] Dashboard sidebar shows "API ok · 10,104 reviews"; Overview, Radar (**View why**), Evidence, Sentiment, Data Health render
- [ ] Product Brief → Generate → path `qwen_precomputed`, fallback reason mentions `BRIEF_MODE=precomputed`
- [ ] `curl -H "Origin: https://example.com" -I https://<render-service>.onrender.com/health` has **no**
      `access-control-allow-origin` header
- [ ] `GET /reviews?limit=500` → 422 (bounded pagination)

## Updating the data

Run the pipeline on the laptop, generate the brief, run the tests, commit the new `artifacts/analytics.db` and redeploy:

```powershell
.\.venv\Scripts\python.exe -m ml.pipeline                 # or --source path\to\reviews.csv
.\.venv\Scripts\python.exe scripts\generate_brief.py      # GPU; stores the validated Qwen brief in the DB
.\.venv\Scripts\python.exe -m pytest
```

Before committing a database built from **real** reviews, confirm you are allowed to publish the redacted text. For
private data, keep the repo private or host the database outside Git.

## Production rehearsal (done locally, 2026-09-27)

This reproduced the Render and Vercel setup on the laptop:

- **API:** a fresh virtual environment with only `requirements-api.txt` (no torch, no transformers). It ran `uvicorn` on port 8001 with
  `BRIEF_MODE=precomputed`, `CORS_ORIGINS=http://127.0.0.1:4173`, and the Hugging Face cache pointed at an empty folder.
- **Dashboard:** built with `VITE_API_BASE_URL=http://127.0.0.1:8001` and served by `vite preview` on port 4173.

| Check | Result |
|---|---|
| Backend tests in the minimal environment | 37 passed |
| `GET /health` | `ok`, 10,104 reviews |
| `/model-info` brief engine | `qwen_installed: false`, `live_generation_possible: false` |
| `POST /product-brief {"engine":"auto"}` | `generation_path: qwen_precomputed`, `access-control-allow-origin: http://127.0.0.1:4173` |
| Request from a foreign origin | no CORS header |
| Production build in the browser | Complaint Radar with **View why** for Battery (111 → 243, +118.92%, `208 x 2.19 = 455.35`) rendered from the port 8001 API |

Not rehearsed: Render's and Vercel's own build machines, HTTPS, cold starts and real hostnames. These need the actual deployment.
