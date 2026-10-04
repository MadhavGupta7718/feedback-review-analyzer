# Feedback & Review Analyzer — "10,000 Reviews, No Time to Read Them"

Microsoft hackathon project (problem 17, Student Edition: Feedback & Review Analyzer).

Upload a review CSV and get themes, complaint radar, evidence, and data health — powered by a **3-class RoBERTa
sentiment model fine-tuned on Amazon clothing reviews**. Accuracy is reported once, globally, on that model's held-out
TEST split (Model Validation). Uploaded batches get analytics only — no per-upload accuracy page.

Enterprise guarantees:

- **Traceable:** every theme and complaint links to real review IDs and redacted verbatims.
- **Validated (global):** accuracy / macro recall come from the Amazon holdout TEST set only.
- **Private:** PII is redacted before anything reaches the database, API, or dashboard.
- **Monitored:** when review dates exist, sentiment / theme / volume drift is tracked between windows.
- **Complaint Radar:** flags growing complaints when timestamps are present; otherwise shows a clear “dates required” empty state.

## Measured results (Amazon fine-tune)

| | |
|---|---|
| Training data | Amazon clothing CSV (`Review` + `Cons_rating` → neg/neu/pos) |
| Split | Deterministic 80/10/10 by `sha256(text)` |
| Model | Fine-tuned `cardiffnlp/twitter-roberta-base-sentiment-latest` → `artifacts/models/amazon_roberta_sentiment/` |
| Headline metrics | TEST accuracy + macro recall in `artifacts/reports/amazon_sentiment_eval.json` |

## Quick start

Requirements: Windows or Linux, Python 3.11, Node 22. An NVIDIA GPU is optional (used automatically for train / pipeline).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
.\.venv\Scripts\python.exe scripts\download_models.py      # one-time base models
cd frontend; npm ci; cd ..
```

### 1) Train the product sentiment model (once)

```powershell
.\.venv\Scripts\python.exe scripts\prepare_amazon_clothing.py
.\.venv\Scripts\python.exe scripts\finetune_amazon_sentiment.py --epochs 3 --batch-size 16
```

This writes:

- `artifacts/models/amazon_roberta_sentiment/`
- `artifacts/reports/amazon_sentiment_eval.json` (served by Model Validation)

### 2) Run API + dashboard

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --port 8000
# other terminal:
cd frontend; npm run dev
```

Open http://127.0.0.1:5173 → **Upload & batches** to drop a CSV. Each upload becomes
`artifacts/cache/batches/{id}.db`. Use **View** to activate it and open Overview / Themes / Radar / Evidence / Data Health.

API-only (serves an existing DB; no torch needed unless you upload):

```powershell
pip install -r requirements-api.txt
uvicorn backend.app.main:app --port 8000
```

### CSV columns

| Field | Required | Accepted names |
|---|---|---|
| text | yes | text, review, review_text, content, body, comment, feedback |
| date | no | created_at, date, timestamp, review_date, … — without dates, Radar + Drift are unavailable |
| rating | no | rating, score, stars, … |

Dates are never invented for uploads. Synthetic dates exist only on the Amazon **training** interim CSV.

## Documentation

| Document | Contents |
|---|---|
| `docs/PROJECT_HANDBOOK.md` | **Full project story:** backend, frontend, model selection, recall hit-and-trial, results |
| `docs/ARCHITECTURE.md` | Pipeline, privacy boundary, radar rules |
| `docs/API.md` | Endpoints including `/batches` and `/model/evaluation` |
| `docs/DATASET.md` | Upload schema and Amazon train corpus |
| `docs/MODEL_CARD.md` | Models and known weaknesses |
| `docs/TESTING.md` | How to run test suites |
| `docs/RESULTS.md` | Measured results |
| `docs/DEMO.md` | Demo storyline |
| `docs/DEPLOYMENT.md` | Render + Vercel notes |
| `docs/DEVELOPMENT_LOG.md` | Audit trail |

## Known limitations

- Theme / radar quality is harder on free-form real reviews than on synthetic templates.
- Star-rating weak labels (3★ = neutral) are noisy; Model Validation reflects that.
- Names in free text without a cue are not redacted.
- Radar / drift need real timestamps on the uploaded file.
- **Not deployed yet.** Configuration is ready for local rehearsal.
