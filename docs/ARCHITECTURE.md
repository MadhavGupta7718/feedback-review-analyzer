# Architecture

## Principle

The **deterministic pipeline is the source of truth**. Every number, theme, status and evidence link is computed by
ordinary code and stored in one SQLite file. The API and dashboard only read that file. The LLM (Qwen) is a presentation
layer for two sections of the product brief. It never produces numbers, IDs or theme names, and its output is rejected
unless a validator accepts it.

```
                 offline, on the GPU laptop                               online (CPU is enough)
 ┌──────────────────────────────────────────────────────────────┐     ┌──────────────────────────────┐
 │ CSV batch ─► clean + PII redaction ─► RoBERTa sentiment        │     │ FastAPI (read-only SQLite)   │
 │                     │                 MiniLM embeddings       │     │   /themes /issues /reviews   │
 │                     │                   └► PCA ► HDBSCAN ►     │     │   /drift /data-health ...    │
 │                     │                      merge ► themes      │ DB  │   POST /product-brief        │
 │                     │                        └► Complaint Radar├────►│     qwen_live │ precomputed  │
 │                     │                        └► drift monitor  │     │     │ template               │
 │                     └► traceability audit ► analytics.db       │     └──────────────┬───────────────┘
 │ scripts/generate_brief.py: Qwen 4-bit ► validator ► DB         │                    │ JSON
 └──────────────────────────────────────────────────────────────┘     ┌──────────────▼───────────────┐
                                                                      │ React dashboard (Vite)       │
                                                                      └──────────────────────────────┘
```

## Components

| Path | Responsibility |
|---|---|
| `ml/data/sentiment140.py` | Streaming validation of the 1.6M-row CSV (one pass, never fully in memory); deterministic samples |
| `ml/data/synthetic.py` | Deterministic synthetic review generator (seed 42) with planted themes, trends, PII and dirty rows |
| `ml/preprocessing/clean.py` | Mojibake repair, HTML entities, null/empty rejection, ingestion-duplicate removal, then redaction |
| `ml/pii/redactor.py` | Ordered regex rules → typed placeholders (`[EMAIL]`, `[PHONE]` …); returns counts only, never the matched values |
| `ml/pii/leak_scan.py` | Independent, stricter scanner used by tests, the API tests, the pipeline audit and the brief validator |
| `ml/sentiment/model.py` | RoBERTa 3-class sentiment, length-sorted batching, fp16 on CUDA |
| `ml/embeddings/encoder.py` | MiniLM 384-d L2-normalised embeddings with an on-disk cache keyed by SHA-256 of model name + texts |
| `ml/themes/discovery.py` | PCA → HDBSCAN (batch-size-adaptive) → micro-cluster merge → noise reassignment → coverage-lift naming → representatives |
| `ml/complaints/radar.py` | 14-day current vs previous window, safe growth, statuses, transparent priority formula, evidence, segment associations |
| `ml/drift/monitor.py` | PSI (sentiment, theme mix), volume change, KS test on review length, weekly series |
| `ml/pipeline.py` | Orchestrates the stages, times them, runs `audit_traceability()`, writes `artifacts/analytics.db` |
| `ml/models/registry.py`, `ml/hardware.py` | Local-only model loading (`local_files_only=True`), runtime GPU/VRAM detection and Qwen plan |
| `ml/brief/` | `facts.py` fact sheet + slots, `template.py` deterministic brief, `qwen_writer.py`, `validator.py`, `service.py` path selection |
| `backend/app/` | FastAPI service, read-only DB access, input validation, CORS, generic errors |
| `frontend/` | React 19 + TypeScript dashboard (7 pages), lazy-loaded, API base from `VITE_API_BASE_URL` |

## Data flow and privacy boundary

1. Raw text exists only in the input CSV and in memory during cleaning. Redaction happens **before** sentiment, embeddings,
   storage, logging or any LLM prompt.
2. `analytics.db` stores `text_redacted` only; there is no raw-text column (checked by a test).
3. The API re-passes every outgoing review text through the redactor (`safe_text`) as defence in depth.
4. Logs go through `RedactingFilter`; redaction reports contain counts, not values.
5. Qwen receives a fact sheet with **slot names instead of numbers**, theme names, keywords and at most one redacted,
   digit-masked quote per emerging theme. It never receives the full batch. The prompt is leak-scanned before generation.

## Traceability

- Each theme stores `representative_ids` (the most central members); each radar issue stores current-window negative
  evidence reviews. These are real rows in the `reviews` table.
- `audit_traceability()` runs at the end of every pipeline run and checks every evidence link: the review exists, it belongs
  to the same theme, radar evidence is negative, and the counts are recomputable from the reviews table. It also checks for residual PII.
- `tests/integration/test_analytics_db.py` re-checks the stored database independently with SQL.

## Complaint Radar

For each complaint theme (at least 50% negative), mentions are counted in the current 14-day window and in the previous
14-day window.

- **NEW:** previous = 0 and current ≥ 30.
- **EMERGING:** growth ≥ +50% and current ≥ 30.
- **DECLINING:** growth ≤ −25%.
- **STABLE:** otherwise.
- **INSUFFICIENT_EVIDENCE:** fewer than 30 current mentions, fewer than 3 negative evidence reviews, or the batch
  does not cover a full previous window.

Priority is calculated as:

```
priority = current_negative_mentions × (1 + clip(growth_pct / 100, 0, 3))   [NEW: × 4]
```

This favours complaints that are growing over complaints that are merely large. Every issue carries its reasons, the
calculation strings, the thresholds and weekly counts, and the dashboard's **View why** panel shows them. Segment
associations (app version, platform) are reported only when lift ≥ 1.3 and support ≥ 20, and always with non-causal
wording.

## Product brief paths

| Path | When | What produces the summary |
|---|---|---|
| `qwen_live` | `BRIEF_MODE=auto`, Qwen installed, CUDA GPU present | Qwen now, validated |
| `qwen_precomputed` | live not possible, and a validated brief is stored in the DB | Qwen earlier (by `scripts/generate_brief.py`), validated |
| `template` | otherwise, or `BRIEF_MODE=template`, or validation failed | deterministic template |

- The response always contains `generation_path` and, when it fell back, `fallback_reasons`.
- Top complaints, emerging issues, evidence quotes, associations and caveats are deterministic in every path.

The validator rejects Qwen output when:

- it contains raw digits or unknown slots;
- it mentions review IDs that are not in the fact sheet;
- it mixes one theme's numbers into another theme's sentence;
- it reverses a from/to direction;
- it uses wording that contradicts the status (for example "decrease" for a NEW or EMERGING theme);
- it uses causal or certainty language;
- it invents theme names in quotes;
- it misses an emerging theme;
- the leak scanner flags PII.

Accepted output gets its slot values filled in by code.

## Deployment shape

The pipeline and Qwen need the GPU laptop. The deployed API needs only FastAPI, uvicorn and pydantic
(`requirements-api.txt`), and serves the committed, redacted `analytics.db` with `BRIEF_MODE=precomputed`.
The frontend is a static build. **Docker is not required**: there is one stateless Python process and one static site, and
both Render and Vercel build them natively. See `docs/DEPLOYMENT.md`.
