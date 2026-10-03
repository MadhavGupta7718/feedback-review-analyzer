# API

FastAPI service in `backend/app/main.py`. It serves the precomputed, PII-redacted `analytics.db` (path from
`ANALYTICS_DB`, default `artifacts/analytics.db`) and opens it read-only for each request. Interactive docs are at `/docs` (Swagger)
and `/redoc`.

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --port 8000          # full environment
# or, API only (no torch / transformers needed):
pip install -r requirements-api.txt; uvicorn backend.app.main:app --port 8000
```

## Guarantees

- **No raw data:** no endpoint returns raw review text or the raw dataset. Every review text passes through the redactor again before
  it is sent (`safe_text`).
- **Validated input:** path parameters are validated by regex (`theme_\d{3}`, review IDs `[A-Za-z0-9_-]{1,40}`), text search is
  capped at 60 characters and escaped for `LIKE`, and pagination is bounded (`limit` ≤ 200).
- **Generic errors:** errors return only generic messages, never stack traces or internal paths.
  - `422 {"error":"invalid_request","detail":"Invalid parameter(s): limit"}`
  - `404 {"detail":"Theme not found"}`
  - `503 {"error":"service_unavailable","detail":"Analytics data is not available."}` when the database is missing
  - `500 {"error":"internal_error","detail":"An internal error occurred."}`
- **CORS:** only origins listed in `CORS_ORIGINS` (comma-separated) or matching `CORS_ORIGIN_REGEX` are allowed. Methods are
  `GET` and `POST`; there are no credentials.

## Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | `status`, `database`, `reviews` count, `source`, `dataset`, `generated_at_utc`; 503 `degraded` if the DB is missing |
| GET | `/metrics` | Overview KPIs (sentiment split, top complaints, emerging issues, executive summary), dataset info, pipeline seconds, device |
| GET | `/themes` | All themes (name, keywords, size, sentiment counts, radar status, growth, priority, coherence) |
| GET | `/themes/{theme_id}` | One theme with representative reviews, its radar record, and app-version / platform breakdown |
| GET | `/issues` | Complaint Radar: `window`, `formula`, `params`, `rules` (text), `issues` sorted by status then priority |
| GET | `/issues/{theme_id}` | One radar issue with reasons, calculation strings, weekly counts, segment associations, evidence IDs |
| GET | `/issues/{theme_id}/evidence` | The evidence reviews (radar evidence and theme representatives) as full redacted review records |
| GET | `/reviews` | Paginated redacted reviews: `total`, `limit`, `offset`, `reviews` |
| GET | `/reviews/{review_id}` | One review with class probabilities, theme similarity and where it is used as evidence |
| GET | `/sentiment/validation` | This-DB sentiment eval: 3-class accuracy/macro recall, confusion, optional binary on non-neutral truth |
| GET | `/drift` | Current-vs-previous drift (sentiment PSI, theme PSI, volume, length KS) and weekly drift vs baseline |
| GET | `/data-health` | Ingestion counts, rejects, duplicates, PII redactions by type, this-batch recall summary, traceability audit |
| GET | `/model-info` | Model revisions, offline verification, hardware of the pipeline run, performance, brief engine status |
| POST | `/product-brief` | Product brief (see below) |

### Query parameters

- **`/themes`:**
  - `complaints_only` (bool)
  - `sort` = `size` | `negative_pct` | `priority` | `growth` | `name`
  - `q` (matches name or keywords)
- **`/issues`:** `status`, comma-separated, for example `NEW,EMERGING`. Valid statuses are NEW, EMERGING, STABLE, DECLINING, INSUFFICIENT_EVIDENCE,
  NOT_A_COMPLAINT and NO_DATA.
- **`/reviews`:**
  - `theme_id`
  - `sentiment` = `negative` | `neutral` | `positive`
  - `q` (substring of redacted text)
  - `limit` (1–200, default 25)
  - `offset`

### Review record

This example is taken from the real `/reviews` response:

```json
{"review_id": "R10098", "created_at": "2026-08-23T23:56:55", "rating": 2, "platform": "android", "app_version": "5.3",
 "text": "Support keeps sending the same copy paste answer. Very disappointing service. Very frustrating!!",
 "sentiment": "negative", "confidence": 0.9561, "theme_id": "theme_009", "theme_name": "Support Service",
 "theme_assignment": "cluster", "pii_redactions": 0, "source": "synthetic"}
```

- `text` is always the redacted text. PII appears only as placeholders such as `[EMAIL]`, `[PHONE]`, `[ORDER_ID]` or `[PERSON]`.
- `/reviews/{id}` adds `p_negative`, `p_neutral`, `p_positive`, `theme_similarity` and `used_as_evidence`.

### Radar issue (`/issues/theme_005`, abridged)

```json
{"theme_id": "theme_005", "name": "Failing Payment", "status": "NEW", "previous_mentions": 0, "current_mentions": 140,
 "growth_pct": null, "growth_label": "NEW", "negative_mentions_current": 135, "negative_ratio": 0.9643, "trend": "rising",
 "priority": 540.0, "reasons": "0 mentions in previous window, 140 in current window (>= 30)",
 "calculation": {"growth": "previous = 0 -> growth undefined, classified NEW", "negative_ratio": "135 / 140 = 96.4%",
                 "priority": "135 x 4.00 = 540.0",
                 "window": {"current": ["2026-08-09T23:56:55", "2026-08-23T23:56:55"], "previous": ["2026-07-26T23:56:55", "2026-08-09T23:56:55"]},
                 "thresholds": {"min_current_mentions": 30, "min_growth_pct": 50.0, "min_negative_ratio": 0.5, "min_evidence_reviews": 3}},
 "weekly_counts": [1, 1, 1, 1, 4, 1, 2, 1, 0, 0, 46, 94],
 "evidence_review_ids": ["R09905", "R08598", "R09580", "R09760", "R10077"], "associations": [], "formula": "...", "theme": {"...": "..."}}
```

## POST /product-brief

Request: `{"engine": "auto" | "qwen" | "template"}` (default `auto`).

- **`auto` and `qwen`:** try live Qwen if `BRIEF_MODE=auto`, Qwen is installed and a CUDA GPU is present. Otherwise they use the stored,
  already-validated Qwen brief. Otherwise they use the template.
- **`template`:** always returns the deterministic template.

The response always includes:

| Field | Meaning |
|---|---|
| `generation_path` | `qwen_live`, `qwen_precomputed` or `template` |
| `fallback_reasons` | why a Qwen path was not used, e.g. `live Qwen unavailable (BRIEF_MODE=precomputed, model not installed, no CUDA GPU)` |
| `validation` | for Qwen paths: `passed`, `errors`, the 11 `checks`, `attempts` |
| `executive_summary`, `suggested_investigation_areas` | written by Qwen (Qwen paths) or the template |
| `top_complaints`, `emerging_complaints`, `evidence`, `possible_associations`, `caveats` | always deterministic |
| `model`, `model_revision`, `written_by_model` | on Qwen paths |

The first live Qwen call on the laptop GPU takes 1–3 minutes (model load plus up to two attempts). On CPU-only hosts it returns
immediately with `qwen_precomputed` or `template`.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `ANALYTICS_DB` | `artifacts/analytics.db` | Database to serve |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Allowed browser origins |
| `CORS_ORIGIN_REGEX` | unset | Optional pattern, e.g. for Vercel preview URLs |
| `BRIEF_MODE` | `auto` | `auto` / `precomputed` / `template` (see above) |
| `HF_HOME`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` | see `.env.example` | Local model cache; offline mode |
