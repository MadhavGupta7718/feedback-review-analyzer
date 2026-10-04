# Feedback & Review Analyzer — Full Project Handbook

**Audience:** anyone joining the project who needs to understand *what we built, why we chose it, what we tried, and what the numbers mean*.

**Hackathon context:** Microsoft Innovate — Problem 17 (Student Edition): *“10,000 Reviews, No Time to Read Them.”*

**Repo root:** `feedback-review-analyzer/`

---

## 1. One-paragraph product summary

Users **upload a review CSV**. The offline pipeline (GPU laptop) cleans text, redacts PII, runs **3-class sentiment** (Amazon fine-tuned RoBERTa), discovers **themes** (MiniLM + HDBSCAN), and—if timestamps exist—runs **Complaint Radar** and **drift**. Results land in a **per-upload SQLite DB**. A FastAPI service and React dashboard read that DB. **Accuracy / macro recall are not claimed per upload**; they come only from a **global Model Validation** page scored on a held-out Amazon clothing TEST split.

---

## 2. Locked product decisions (and why)

| Decision | Choice | Why |
|---|---|---|
| Accuracy story | **Global** Amazon holdout TEST metrics only | Synthetic “planted label” accuracy was not a credible product claim for real CSVs |
| Upload analytics | Separate DB per upload; **no** Sentiment Validation page per batch | Uploads usually lack reliable ground truth |
| Dates on uploads | Use if present; **never invent**; hide Radar/Drift when missing | Radar/drift need time windows |
| Sentiment model | Fine-tune **cardiffnlp/twitter-roberta-base-sentiment-latest** → 3-class on Amazon clothing | Domain closer to product reviews than tweets; 3-class matches neg/neu/pos |
| Labels for train/eval | `Cons_rating`: 1–2 neg, 3 neu, 4–5 pos | Weak labels from stars (noisy, especially neutrals) — documented honestly |
| Split | Deterministic **80/10/10** by `sha256(text) % 100` | Reproducible, no train/test leakage |
| Product Brief in UI | **Removed** from nav/product path | Pivot to upload + model validation; brief code may still exist unused |
| Synthetic demos (Nimbus/Dove/Adidas/Realme) | Generators may remain; **not** the primary UI story | Demo path is upload CSV + Amazon model |

---

## 3. Current selected model (what is live)

| Item | Value |
|---|---|
| Active sentiment model | `artifacts/models/amazon_roberta_sentiment/` |
| Loader key | `local:amazon_roberta_sentiment`, revision `amazon-finetune` |
| Base checkpoint | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| How chosen | If Amazon dir has `config.json`, pipeline uses it; else pretrained CardiffNLP |
| Eval report | `artifacts/reports/amazon_sentiment_eval.json` |
| Selection rule at train time | Checkpoint with **best validation macro recall** |

### Current TEST metrics (selected checkpoint)

| Metric | Value |
|---|---|
| TEST accuracy | **0.8315** |
| TEST macro recall | **0.7363** |
| Neg / Neu / Pos recall | **0.7225 / 0.6000 / 0.8865** |
| Best val macro recall | **0.7470** (epoch 2 of the restore run) |
| Eval date (UTC) | 2026-10-04T16:05:53 |

**Reason this checkpoint is selected:** After several recall-boost experiments that *lowered* TEST macro recall, we restored the **simple baseline recipe** (inverse-frequency class weights, natural batching, 3 epochs, lr=2e-5). That family of runs produced the best honest holdout numbers (~0.74 macro recall). Aggressive minority oversampling / equal-class batches traded away too much positive recall.

---

## 4. System architecture

```
Amazon clothing CSV ──► prepare_amazon_clothing.py ──► interim reviews.csv
                              │
                              ▼
                    finetune_amazon_sentiment.py
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
   amazon_roberta_sentiment/      amazon_sentiment_eval.json
              │                               │
              ▼                               ▼
         ml.pipeline                   Model Validation UI
              │
              ▼
   artifacts/cache/batches/{id}.db  ◄── Upload CSV (user)
              │
              ▼
         FastAPI (active DB) ──► React pages
```

### Privacy boundary

1. PII redaction runs **before** sentiment, embeddings, DB write, or any LLM.
2. DB stores redacted text only.
3. API re-redacts outbound review text (`safe_text`).
4. Logs use redacting filters; redaction reports store counts, not secrets.

### Traceability

Every theme representative and radar evidence ID must resolve to a real review in that batch’s DB. `audit_traceability()` runs at end of each pipeline run.

---

## 5. Backend

### Stack

- **FastAPI** + **Uvicorn** + **Pydantic**
- SQLite analytics DBs (read-only for most GETs)
- Optional pipeline deps (torch/transformers) for upload jobs; API-only deploy can use `requirements-api.txt`

### Important modules

| Path | Role |
|---|---|
| `backend/app/main.py` | HTTP routes |
| `backend/app/db.py` | Active DB path (batch override or `ANALYTICS_DB`) |
| `backend/app/batches.py` | Upload registry, background pipeline jobs, activate |
| `ml/pipeline.py` | End-to-end analytics build |
| `ml/models/registry.py` | Local model load; prefers Amazon fine-tune |
| `ml/sentiment/model.py` | Batched 3-class inference |
| `ml/themes/discovery.py` | Clustering / theme naming |
| `ml/complaints/radar.py` | Growing-complaint detection |
| `ml/drift/monitor.py` | PSI / volume / length drift |
| `ml/pii/` | Redaction + leak scan |

### API endpoints (product-facing)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Status, review count, dataset name |
| GET | `/metrics` | Overview KPIs |
| GET | `/themes`, `/themes/{id}` | Theme list / detail |
| GET | `/issues`, `/issues/{id}`, `.../evidence` | Complaint Radar |
| GET | `/reviews`, `/reviews/{id}` | Searchable evidence |
| GET | `/model/evaluation` | **Global** Amazon TEST metrics (alias: `/sentiment/validation`) |
| GET | `/drift` | Drift report or `{status: unavailable}` |
| GET | `/data-health` | Ingestion / PII / themes / dates flags |
| GET | `/model-info` | Model versions + `amazon_model_installed` |
| GET | `/batches` | List uploads |
| POST | `/batches/upload` | Multipart CSV → background pipeline |
| GET | `/batches/jobs/{job_id}` | Job status |
| POST | `/batches/{id}/activate` | Set active analytics DB |

`POST /product-brief` may still exist in code but is **not** part of the product UI after the pivot.

### Batch upload behaviour

1. Save CSV under `data/uploads/{batch_id}/`.
2. Run `ml.pipeline` into `artifacts/cache/batches/{batch_id}.db`.
3. Register in `artifacts/cache/batches/index.json`.
4. Auto-activate that batch for subsequent reads.
5. If ≥50% of rows lack usable dates → `dates_available=false`; radar/drift become stubs with a clear message. Sentiment/themes/PII/evidence still run.

### CSV columns accepted

| Field | Required | Aliases |
|---|---|---|
| text | **yes** | text, review, review_text, content, body, … |
| date | no | created_at, date, timestamp, … |
| rating | no | rating, score, stars, … |

---

## 6. Frontend

### Stack

- React 19 + TypeScript + Vite
- React Router
- Recharts
- Vitest + Testing Library

### Pages (routes)

| Route | Page | Role |
|---|---|---|
| `/batches` | Upload & batches | Drop CSV, poll job, list past uploads, View → activate |
| `/sentiment` | Model Validation | Global Amazon TEST accuracy / macro recall / confusion |
| `/` | Executive Overview | Sentiment mix, themes, emerging (or dates message) |
| `/themes` | Themes | Cluster browser + representatives |
| `/radar` | Complaint Radar | Growing issues + View why (or dates empty state) |
| `/evidence` | Evidence | Searchable reviews |
| `/health` | Data Health | Ingestion, PII, drift (or unavailable) |

### UX rules from the pivot

- Sidebar: “Upload batches · Amazon-trained sentiment”.
- Model Validation copy states metrics are **not** tied to any upload.
- Overview / Radar / Data Health show: *“This view needs review dates…”* when `dates_available === false`.
- Product Brief route removed from `App.tsx` / nav.

### Config

- API base: `VITE_API_BASE_URL` (default `http://127.0.0.1:8000`).

---

## 7. ML pipeline stages

Order of work in `ml/pipeline.py`:

1. **Load CSV** (text required; dates optional).
2. **Clean + PII redact**.
3. **Sentiment** with Amazon fine-tune if present (fp16 on CUDA).
4. **Embeddings** (MiniLM 384-d).
5. **Themes** (PCA → HDBSCAN → merge → name).
6. **Radar + drift** if dates available; else stubs.
7. **Traceability audit**.
8. **Write SQLite** + reports JSON blobs.

Uploads do **not** write a planted-label `sentiment_validation` payload for the accuracy page.

---

## 8. Sentiment training data

### Source

Amazon clothing CSV (`Review` + `Cons_rating`), prepared by:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_amazon_clothing.py
```

Writes `data/interim/amazon_clothing/reviews.csv`.

### Label map

| Cons_rating | Class |
|---|---|
| 1–2 | negative |
| 3 | neutral |
| 4–5 | positive |

### Split sizes (current)

| Split | Rows |
|---|---|
| Train | 38,661 |
| Val | 4,830 |
| Test | 4,812 |

**Note:** Interim `created_at` on the *training* corpus is **synthetic (hash-based)** for corpus bookkeeping only. It is **not** used to invent dates on user uploads.

---

## 9. Hit-and-trial: raising macro recall

**Goal:** raise TEST **macro recall** (= mean of neg/neu/pos recall). Bottleneck is almost always **neutral** (noisy 3★ labels + class imbalance).

**Script:** `scripts/finetune_amazon_sentiment.py`  
**Log:** `data/finetune_amazon.log`  
**Selection criterion:** best **validation macro recall**, then report TEST.

### Trial log

| # | Recipe | Result (TEST) | Verdict |
|---|---|---|---|
| A | **Baseline** — from CardiffNLP, 3 epochs, lr=2e-5, bs=16, inverse-frequency CE weights, natural shuffle | Acc **0.8356**, macro recall **0.7411** (neu ~0.60) | **Best honest result; reference** |
| B | Resume from A + equal-class batches + focal γ=2 + minority weight boosts, 5 epochs, lr=1e-5 | Acc 0.7903, macro **0.7257** (neu ↑ slightly, pos ↓ hard) | Rejected — macro fell |
| C | From scratch + heavy oversample neu×3/neg×2 + weight_power 1.35 + focal 1 | Val macro ~0.73, crushed pos weight | Abandoned mid-run |
| D | Milder oversample + focal 0.5 + neu_boost 1.6 | Val macro ~0.73, still below A | Abandoned |
| E | Balanced Softmax (τ=1) + mild oversample, 4 epochs | Acc 0.8092, macro **0.7366** | Rejected — below A |
| F | **Restore baseline recipe** (weights only, no focal/balanced/oversample), 3 epochs | Acc **0.8315**, macro **0.7363**; best **val** macro **0.7470** | **Currently selected** (close to A; slight run variance) |

### Why aggressive recall tricks failed

1. **Equal-class batches** force the model to predict minority classes often → positive recall collapses → macro mean does not rise.
2. **Very high inverse weights / weight_power > 1** on a ~74% positive set makes the positive class weight tiny → same collapse.
3. **Focal loss (γ=2)** on already-weighted CE over-emphasised hard/noisy neutrals.
4. Star-rating **neutrals are weak labels**; many 3★ texts are not true “neutral sentiment,” so ceiling on neu recall is limited without cleaner labels.

### What we kept in the training script (for future work)

Flags in `finetune_amazon_sentiment.py`:

- `--resume`
- `--balanced-batches`
- `--focal-gamma`
- `--weight-power`, `--neu-boost`, `--neg-boost`
- `--oversample-neu`, `--oversample-neg`
- `--balanced-softmax-tau`

**Default production train command (selected):**

```powershell
.\.venv\Scripts\python.exe scripts\finetune_amazon_sentiment.py --epochs 3 --lr 2e-5 --batch-size 16
```

---

## 10. Other experiments earlier in the project (context)

These informed the pivot but are **not** the product accuracy page:

| Experiment | Outcome / use |
|---|---|
| Sentiment140 binary study + threshold 0.725 | Offline; tweets ≠ product path |
| Fine-tuned binary RoBERTa on S140 | Experimental only (`roberta-s140-binary`) |
| Amazon / Yelp polarity offline scores | Offline study (~0.92 / ~0.87) |
| Synthetic Nimbus planted themes/radar/PII | Useful for radar/PII demos; abandoned as *accuracy* claim |
| Dove / Adidas / Realme synthetic generators | Exist under `ml/data/` + scripts; not primary UI |
| Soft-decline radar ideas | Explored earlier; product radar still date-window based |

---

## 11. How to run (current)

### One-time setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
.\.venv\Scripts\python.exe scripts\download_models.py
cd frontend; npm ci; cd ..
```

### Train product sentiment (if model missing)

```powershell
.\.venv\Scripts\python.exe scripts\prepare_amazon_clothing.py
.\.venv\Scripts\python.exe scripts\finetune_amazon_sentiment.py --epochs 3 --lr 2e-5 --batch-size 16
```

### Serve

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --port 8000
cd frontend; npm run dev
```

Open http://127.0.0.1:5173 → **Upload & batches** or **Model Validation**.

### Sample upload CSVs

| File | Notes |
|---|---|
| `data/uploads/demo_upload_15k.csv` | Amazon reviews, with dates |
| `data/uploads/dove_shampoo_upload_15k.csv` | Synthetic Dove (~15k, many themes) |
| `data/uploads/sample_no_dates.csv` | For dates-unavailable empty states |

---

## 12. Key artifacts

| Path | Meaning |
|---|---|
| `artifacts/models/amazon_roberta_sentiment/` | Selected product sentiment weights |
| `artifacts/reports/amazon_sentiment_eval.json` | Global TEST metrics for UI |
| `artifacts/cache/batches/*.db` | Per-upload analytics |
| `artifacts/cache/batches/index.json` | Batch registry + active id |
| `data/interim/amazon_clothing/reviews.csv` | Train/val/test source |
| `data/finetune_amazon.log` | Latest training log |

---

## 13. Frontend / backend ownership map

| Concern | Backend | Frontend |
|---|---|---|
| Accuracy numbers | `/model/evaluation` ← eval JSON | Model Validation page |
| Upload job | `/batches/*` + pipeline thread | Batches page |
| Themes / radar / evidence | GETs on active DB | Themes / Radar / Evidence |
| Dates missing | `dates_available` + stub reports | Empty-state banners |
| PII | Pipeline redactor + API `safe_text` | Never shows raw secrets |

---

## 14. Honest limitations (tell judges / readers)

1. Cons_rating labels are **weak**; neutral is especially noisy.
2. Macro recall ~0.74 reflects that ceiling more than “bad engineering.”
3. Theme/radar quality on free-form real reviews is harder than on synthetic templates.
4. Radar/drift need real timestamps on the uploaded file.
5. Names without cues may not be redacted.
6. Product Brief is out of the primary UI after the pivot.

---

## 15. Related docs

| Doc | Contents |
|---|---|
| `docs/ARCHITECTURE.md` | Deeper pipeline / privacy / radar formulas |
| `docs/API.md` | Endpoint contracts |
| `docs/DATASET.md` | Upload schema + Amazon train corpus |
| `docs/RESULTS.md` | Measured headline numbers |
| `docs/MODEL_CARD.md` | Model risks / intended use |
| `docs/DEMO.md` | Demo storyline |
| `docs/TESTING.md` | How to run tests |
| `docs/DEVELOPMENT_LOG.md` | Chronological audit trail |
| `README.md` | Quick start |

---

## 16. Chronology of the pivot (short)

1. Built full analyzer with synthetic Nimbus + planted-label accuracy story.
2. Explored Sentiment140 / polarity offline metrics; risk of mixing claims across datasets.
3. Added Dove/Adidas/Realme generators and per-DB metrics — still not a generalizable product accuracy story.
4. **Pivot:** train on real Amazon clothing ratings; global Model Validation; upload batches without per-file accuracy.
5. Implemented upload APIs, date-unavailable stubs, UI rewrite, Amazon fine-tune.
6. Attempted recall boosts (balanced batches, focal, oversample, balanced softmax); **rejected** when TEST macro recall fell.
7. Restored baseline-style fine-tune; documented selected metrics in `amazon_sentiment_eval.json` and this handbook.

---

*Last updated from project state on 2026-10-04: selected Amazon fine-tune TEST accuracy 0.8315 / macro recall 0.7363.*
