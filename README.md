# Feedback & Review Analyzer — "10,000 Reviews, No Time to Read Them"

Microsoft hackathon project (problem 17, Student Edition: Feedback & Review Analyzer).

A tool that reads a batch of reviews and reports the main themes, common complaints and overall sentiment in a
dashboard, with enterprise guarantees:

- **Traceable:** every theme and complaint links to real review IDs and redacted verbatims; an audit checks every link on
  every run.
- **Validated:** sentiment accuracy is measured against 5,000 labelled Sentiment140 tweets (0.7646 binary accuracy,
  reported alongside two stricter scorings).
- **Private:** PII is redacted before anything reaches the database, API, dashboard, logs or LLM (100% recall on 628
  planted PII rows, 0 false-positive rows).
- **Monitored:** sentiment, theme mix, volume and review-length drift are tracked between time windows.
- **Complaint Radar:** flags complaints that are *growing*, not just the biggest ones, and shows the exact rule and
  calculation behind every flag (**View why**).
- **Product brief:** a local Qwen2.5-3B (4-bit on the laptop GPU) words the summary. It never sees a number: a
  validator rejects unsafe output and code fills in every value, with a deterministic fallback.

## Measured results

| | |
|---|---|
| End-to-end pipeline, 10,104 reviews | 46.7 s on the RTX 4050 vs 455.4 s on the CPU (9.8×) |
| Sentiment (Sentiment140, n = 5,000) | accuracy 0.7646, macro F1 0.7634 (binary); 0.6014 strict 3-class |
| Themes (synthetic, planted ground truth) | 27 themes, 12/13 planted themes recovered, ARI 0.64 |
| Complaint Radar | detects planted NEW (Failing Payment 0 → 140), EMERGING (Battery +119%), STABLE, DECLINING |
| Tests | 148 Python + 3 GPU model tests + 18 frontend tests, all passing |

Everything measured is in `docs/RESULTS.md`, and the phase-by-phase audit trail, including failed attempts, is in
`docs/DEVELOPMENT_LOG.md`.

## Quick start

Requirements: Windows or Linux, Python 3.11, Node 22. An NVIDIA GPU is optional (used automatically).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
.\.venv\Scripts\python.exe scripts\download_models.py      # one-time, public models, no token
cd frontend; npm ci; cd ..

.\.venv\Scripts\python.exe scripts\run_demo.py --serve     # API :8000 + dashboard http://127.0.0.1:5173
```

You can browse the committed, precomputed database without downloading any model:

```powershell
pip install -r requirements-api.txt
uvicorn backend.app.main:app --port 8000
```

To analyse your own reviews, run `python -m ml.pipeline --source reviews.csv`. The file needs a text column and a date
column; see `docs/DATASET.md`.

## Documentation

| Document | Contents |
|---|---|
| `docs/ARCHITECTURE.md` | Pipeline, privacy boundary, traceability, radar rules, brief paths |
| `docs/API.md` | Endpoints, parameters, response shapes, errors, env vars |
| `docs/DATASET.md` | Sentiment140 validation, synthetic dataset design, data-handling rules, CSV upload |
| `docs/MODEL_CARD.md` | The three models, measured performance, guardrails, known weaknesses |
| `docs/TESTING.md` | How to run each suite and what it covers |
| `docs/RESULTS.md` | Measured results only |
| `docs/DEMO.md` | Offline demo and a 5-minute storyline |
| `docs/DEPLOYMENT.md` | Render + Vercel steps (not deployed yet) and the local production rehearsal |
| `docs/DEVELOPMENT_LOG.md` | Chronological audit trail with every failed attempt |

## Datasets

| Dataset | Use |
|---|---|
| Sentiment140 (1.6M tweets, binary labels) | Dataset validation, sentiment accuracy, real-text PII audit |
| Synthetic "Nimbus" app reviews (deterministic, seed 42) | 10K demo batch with planted ground truth for themes, Complaint Radar, drift and PII |

Raw data is never committed or served. See `docs/DATASET.md`.

## Known limitations

- Theme and radar quality is measured on synthetic reviews generated from templates, so results on real reviews will be lower.
- Sentiment140 has no neutral class, and the model's neutral recall on the synthetic reviews is only 0.33.
- Names in free text without a cue ("this is X", "Mr X") are not redacted.
- Live Qwen needs a CUDA GPU and takes 1–3 minutes; CPU hosts serve the stored, validated brief.
- Radar statuses show association, not cause.
- **Not deployed yet.** The configuration is ready and rehearsed locally.
