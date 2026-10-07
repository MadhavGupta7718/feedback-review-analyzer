# Feedback & Review Analyzer

Microsoft Innovate — Problem 17 (Student Edition): **“10,000 Reviews, No Time to Read Them.”**

Repository: [github.com/MadhavGupta7718/feedback-review-analyzer](https://github.com/MadhavGupta7718/feedback-review-analyzer)

---

## Problem

Product and support teams receive thousands of reviews. Reading them by hand does not scale. Star ratings alone miss what people actually wrote (broken packaging, scalp issues, delayed delivery, praise). Teams need a way to turn a large CSV of reviews into a short, explainable picture of sentiment, themes, and rising complaints—without exposing personal data.

---

## Solution

Upload a review CSV. An offline pipeline cleans text, redacts PII, scores **3-class sentiment** (negative / neutral / positive), discovers **themes**, and—when timestamps exist—runs **Complaint Radar** and **data drift**. Each upload becomes its own SQLite analytics database. FastAPI serves JSON; a React dashboard shows one active batch at a time.

**Accuracy is not claimed per upload.** Global Model Validation reports TEST metrics on a held-out Amazon clothing split after a teacher→student training path (see [Measured results](#measured-results)).

### Product guarantees

| Guarantee | Meaning |
|-----------|---------|
| Traceable | Themes and radar evidence resolve to real review IDs and redacted verbatims in the batch DB |
| Validated (global) | Accuracy / macro recall come only from the Amazon holdout TEST set (`amazon_sentiment_eval.json`) |
| Private | PII is redacted before storage; the API and UI never serve raw review text with PII |
| Monitored | When dates exist, sentiment / theme / volume / length drift is compared across windows |
| Explainable radar | Status rules and per-issue calculations are visible in the UI (View why) |

---

## What the system does

```text
CSV upload
  → clean / validate / dedupe
  → PII redaction
  → 3-class sentiment (Amazon fine-tuned RoBERTa)
  → embeddings (MiniLM) + theme discovery (HDBSCAN)
  → Complaint Radar + drift (if dates present)
  → SQLite analytics DB
  → FastAPI → React dashboard
```

Optional: **Product Brief** (Qwen2.5-3B or deterministic template) turns analytics into a one-page summary. Numbers are filled from a validated fact sheet; the LLM does not invent metrics.

---

## Architecture (short)

| Layer | Role |
|-------|------|
| `ml/pipeline.py` | Offline GPU-capable pipeline; writes per-batch SQLite |
| `backend/app/` | FastAPI over the active DB; batch upload + activate |
| `frontend/` | Vite + React dashboard |
| `artifacts/models/amazon_roberta_sentiment/` | Product sentiment weights (local; not committed) |
| `artifacts/reports/amazon_sentiment_eval.json` | Global Model Validation metrics |
| `D:\huggingface` (or `HF_HOME`) | Cached base models (CardiffNLP, MiniLM, Qwen) |

Heavy models stay on the GPU machine. A CPU-only API deploy can serve a precomputed DB with `requirements-api.txt` and never load torch.

More detail: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Models

| Model | Role |
|-------|------|
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | Teacher for text-only labels; fallback if fine-tune missing |
| Fine-tune in `artifacts/models/amazon_roberta_sentiment/` | **Product** 3-class sentiment on uploads |
| `sentence-transformers/all-MiniLM-L6-v2` | Embeddings for themes |
| `Qwen/Qwen2.5-3B-Instruct` (optional) | Product Brief prose only |

### Training path (current)

1. Prepare Amazon clothing reviews (`scripts/prepare_amazon_clothing.py`).
2. Label with the CardiffNLP teacher on **text only**; stars ignored for training (`scripts/teacher_label_amazon.py`). Teacher vs star-map agreement was about **77.9%** (stars are noisy, especially neutrals).
3. Fine-tune a student RoBERTa on teacher labels (`scripts/finetune_amazon_sentiment.py`).
4. Hyperparameter sweep selected **Trial A** (`lr=1e-5`, batch size 16, max length 128, 3 epochs).

Split: deterministic **80 / 10 / 10** by `sha256(text) % 100` (no random leakage).

---

## Measured results

Source of truth: `artifacts/reports/amazon_sentiment_eval.json` (Model Validation page).

| Metric | Trial A (teacher labels, selected) |
|--------|-------------------------------------|
| TEST accuracy | **0.9616** |
| TEST macro recall | **0.9207** |
| TEST neg / neu / pos recall | **0.9686 / 0.8121 / 0.9813** |
| Best validation macro recall | **0.9051** |
| Holdout size | 4,812 TEST reviews |

Interpretation: TEST scores measure **agreement with the teacher**, not human gold labels.

Historical star-label student (before teacher path): TEST accuracy **0.8315**, macro recall **0.7363**.

Full tables and older studies: [`docs/RESULTS.md`](docs/RESULTS.md).

---

## Dashboard pages

| Route | Page | Purpose |
|-------|------|---------|
| `/batches` | Upload & batches | Upload CSV, poll job, activate a batch |
| `/sentiment` | Model Validation | Global Amazon TEST metrics |
| `/` | Executive Overview | KPIs, sentiment mix, emerging complaints |
| `/themes` | Themes | Theme list and detail |
| `/radar` | Complaint Radar | Statuses, growth, View why, classification rules |
| `/evidence` | Evidence | Search redacted reviews |
| `/health` | Data Health | Ingestion, PII, traceability, drift, model status |
| `/brief` | Product Brief | Auto / Qwen / Template one-page summary |

---

## Repository layout

```text
feedback-review-analyzer/
  ml/                 Offline pipeline (clean, PII, sentiment, themes, radar, drift, brief)
  backend/            FastAPI API + batch upload
  frontend/           React (Vite) dashboard
  scripts/            Train, download models, sweep, brief, demo helpers
  artifacts/reports/  Eval JSON, hyperparam sweep, verification logs
  docs/               Handbook, API, architecture, results, deployment
  report_build/       Regenerate Word guides
  tests/              Pytest unit / integration / optional GPU model tests
  experiment.md       List of removed offline experiment files
```

---

## Requirements

- **OS:** Windows or Linux (examples below use PowerShell)
- **Python:** 3.11
- **Node.js:** 20+ (22 recommended) for the frontend
- **GPU:** Optional but recommended for training and full pipeline uploads; CUDA if using the torch wheel below
- **Disk:** Several GB for Hugging Face caches (Qwen alone is ~6 GB)

---

## Quick start

### 1. Clone and Python env

```powershell
cd D:\Microsoft\feedback-review-analyzer
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` if needed (`HF_HOME`, `ANALYTICS_DB`, `CORS_ORIGINS`, `BRIEF_MODE`, `VITE_API_BASE_URL`).

### 2. Download base models (once)

```powershell
.\.venv\Scripts\python.exe scripts\download_models.py
.\.venv\Scripts\python.exe scripts\verify_models.py
```

Caches under `HF_HOME` (default `D:\huggingface`).

### 3. Train the product sentiment model (once)

Preferred teacher path (matches current Model Validation):

```powershell
.\.venv\Scripts\python.exe scripts\prepare_amazon_clothing.py
.\.venv\Scripts\python.exe scripts\teacher_label_amazon.py
.\.venv\Scripts\python.exe scripts\finetune_amazon_sentiment.py `
  --data data\interim\amazon_clothing\reviews_teacher_roberta.csv `
  --epochs 3 --batch-size 16 --lr 1e-5 --max-length 128
```

Writes:

- `artifacts/models/amazon_roberta_sentiment/`
- `artifacts/reports/amazon_sentiment_eval.json`

Optional sweep (after a baseline exists):

```powershell
.\.venv\Scripts\python.exe scripts\run_hyperparam_sweep.py
```

### 4. Frontend deps

```powershell
cd frontend
npm ci
cd ..
```

### 5. Run API and dashboard

Terminal 1:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --port 8000
```

Terminal 2:

```powershell
cd frontend
npm run dev
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173).

1. Confirm **Model Validation** shows Trial A metrics.
2. Go to **Upload & batches**, drop a CSV, wait for the job, click **View**.
3. Explore Overview, Themes, Radar, Evidence, Data Health, Product Brief.

Each upload is stored as `artifacts/cache/batches/{id}.db`. Activating a batch points the API at that file.

### API-only (no training, existing DB)

```powershell
pip install -r requirements-api.txt
uvicorn backend.app.main:app --port 8000
```

Set `ANALYTICS_DB` to a prepared database. Upload that loads models still needs the full `requirements.txt` environment.

---

## Upload CSV schema

| Field | Required | Accepted column names |
|-------|----------|------------------------|
| text | Yes | `text`, `review`, `review_text`, `content`, `body`, `comment`, `feedback` |
| date | No | `created_at`, `date`, `timestamp`, `review_date`, … |
| rating | No | `rating`, `score`, `stars`, … |

- Without usable dates, Complaint Radar and drift show a clear empty state; sentiment and themes still run.
- Dates are **never invented** for user uploads.
- Typical limit: about 80 MB per upload (API enforced).

---

## Complaint Radar (when dates exist)

Compares the last **14 days** of reviews to the previous **14 days**.

Status order (first match wins):

`NO_DATA` → `NOT_A_COMPLAINT` (neg ratio &lt; 0.5) → `INSUFFICIENT_EVIDENCE` → `NEW` → `EMERGING` (≥ +50%) → `DECLINING` (≤ −25%, or soft ≤ −15% with falling trend) → `STABLE`

Priority is a sort key only (negative volume × growth factor); it does not choose the status.

---

## Product Brief

| Engine | Behavior |
|--------|----------|
| Auto | Live Qwen if installed + CUDA; else stored precomputed brief; else template |
| Qwen | Prefer Qwen, with the same fallbacks |
| Template | Deterministic brief only; never loads Qwen |

`BRIEF_MODE` in `.env`: `auto` | `precomputed` | `template`.

Offline generation on GPU:

```powershell
.\.venv\Scripts\python.exe scripts\generate_brief.py
```

---

## Tests

```powershell
# Unit / API (no GPU required for most)
.\.venv\Scripts\python.exe -m pytest tests backend/tests -q

# Frontend
cd frontend
npm test
cd ..

# Optional live model tests (GPU + local HF cache)
.\.venv\Scripts\python.exe -m pytest -m models -q
```

Details: [`docs/TESTING.md`](docs/TESTING.md).

---

## Documentation

| Document | Contents |
|----------|----------|
| [`docs/PROJECT_HANDBOOK.md`](docs/PROJECT_HANDBOOK.md) | Full product and engineering story |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Pipeline, privacy, radar, brief paths |
| [`docs/API.md`](docs/API.md) | HTTP endpoints |
| [`docs/DATASET.md`](docs/DATASET.md) | Upload schema and train corpus |
| [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) | Models and known weaknesses |
| [`docs/RESULTS.md`](docs/RESULTS.md) | Measured numbers |
| [`docs/DEMO.md`](docs/DEMO.md) | Judge / mentor demo flow |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | Render + Vercel notes |
| [`docs/DEVELOPMENT_LOG.md`](docs/DEVELOPMENT_LOG.md) | Phase audit trail |
| [`docs/TESTING.md`](docs/TESTING.md) | Test suites |
| [`experiment.md`](experiment.md) | Removed offline experiment files |
| `docs/*.docx` | Generated Word guides (`report_build/`) |

Regenerate the final master guide:

```powershell
.\.venv\Scripts\python.exe report_build\build_final_master_guide.py
```

---

## Environment variables (common)

| Variable | Purpose |
|----------|---------|
| `HF_HOME` / `HF_HUB_CACHE` | Local Hugging Face cache |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` | Prefer local-only loads at runtime |
| `ANALYTICS_DB` | Active SQLite path for the API |
| `CORS_ORIGINS` | Allowed browser origins |
| `BRIEF_MODE` | `auto` / `precomputed` / `template` |
| `VITE_API_BASE_URL` | Frontend → API base URL |

See `.env.example`.

---

## Deployment

Configuration exists for a light API (Render) + static UI (Vercel) serving a precomputed DB without torch. Pipeline and live Qwen stay on the GPU laptop. See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

---

## Known limitations

- Theme and radar quality is harder on messy real reviews than on clean synthetic templates.
- Teacher-label TEST metrics measure agreement with CardiffNLP, not human annotation.
- Free-text person names without a cue may not be redacted.
- Radar and drift need real timestamps and enough date span (about two 14-day windows) to flag growth.
- One-day datasets still get sentiment and themes; emerging / declining flags usually will not.
- Large model weights and raw CSVs are gitignored; clone alone is not enough to score uploads until models and a fine-tune are present locally.

---

## License / data

Public Hugging Face checkpoints only; no HF token required for the listed models. Do not commit raw review CSVs or PII. Training and upload data stay under `data/` (gitignored) or local batch DBs under `artifacts/cache/`.
