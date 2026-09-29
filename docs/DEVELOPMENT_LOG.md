# Development Log

Permanent chronological audit trail. Failed attempts are never deleted. All numbers below are copied
from actual command output (full JSON outputs live in `artifacts/reports/`).

---

## PHASE 0 — ENVIRONMENT (2026-09-27)

**Objective:** inspect the machine and set up a reproducible environment before writing application code.

**Initial state:** `D:\Microsoft` contained only `innovate_2_prompt.txt` and the Sentiment140 CSV. No repo,
no ML packages installed globally, no `D:\huggingface` cache.

**Decisions taken with the project owner**
- Datasets: Sentiment140 for validation/sentiment evaluation + a deterministic synthetic product-review
  dataset (planted ground truth) for the 10K demo batch and test cases.
- Repo: private GitHub repo `feedback-review-analyzer`, code in `D:\Microsoft\feedback-review-analyzer`.
- Deployment target: FastAPI serving precomputed artifacts on Render + React frontend on Vercel.

**Commands executed (working dir `D:\Microsoft\feedback-review-analyzer`)**

| Command | Result |
|---|---|
| `nvidia-smi` | NVIDIA GeForce RTX 4050 Laptop GPU, driver 592.82, CUDA 13.1, **6141 MiB VRAM** |
| `python -m venv .venv` | Python 3.11.9 venv on D: |
| `pip install torch --index-url .../cu128` | torch 2.11.0+cu128 installed (≈2.6 GB wheel, pip cache on `D:\pip-cache`) |
| CUDA smoke test (2048×2048 matmul on `cuda`) | `2.11.0+cu128 12.8 True NVIDIA GeForce RTX 4050 Laptop GPU` / `matmul ok True` |
| `pip install -r requirements.txt` | transformers 5.17.0, sentence-transformers 6.1.0, bitsandbytes 0.50.2, hdbscan 0.8.44, fastapi 0.141.1 … (pinned in `requirements.txt`) |
| `python scripts/env_check.py` | exit 0, `PHASE 0: PASS` |

**Measured environment (`artifacts/reports/env_check.json`)**
- GPU: RTX 4050 Laptop, total VRAM **6.0 GB**, free at check time 4.95 GB, compute capability 8.9 (bf16 capable)
- System RAM: 15.65 GB total, **6.22 GB available** at check time (Cursor + browser running)
- Disk: C: 74.1 GB free, D: 35.7 GB free
- HF cache env vars resolve to `D:\huggingface\...` inside Python
- Node v22.21.1, git 2.53.0

**Engineering decision:** 16 GB is system RAM; the GPU has 6 GB. Qwen2.5-3B fp16 weights (~6.2 GB) cannot
fit, so the Qwen plan must be decided at runtime from measured free VRAM (expected: 4-bit NF4 via bitsandbytes).

**Status:** PASS — **Gate:** PROCEED

---

## PHASE 1 — DATASET VALIDATION (2026-09-27)

**Objective:** validate the real Sentiment140 file without loading it into memory; create deterministic samples.

**Implementation:** `ml/data/sentiment140.py` (stdlib `csv` streaming, one pass), `scripts/validate_dataset.py`.

**Attempt 1** — `python scripts/validate_dataset.py` → exit 0, status PASS, scan 77.2 s.
Finding: **1,685 duplicated tweet IDs, and all 1,685 carry conflicting labels** (same tweet labelled 0 and 4).
Attempt 1 kept the first occurrence in the sampling pool — that would inject contradictory ground truth
into the evaluation sample.

**Fix:** every copy of a conflicting ID is now excluded from the sampling pool (`excluded_conflicting_ids`), and a
new check `conflicting_label_ids_excluded_from_samples` was added.

**Attempt 2 (retest)** — exit 0, status PASS. Measured (`artifacts/reports/dataset_validation.json`):

| Check | Measured |
|---|---|
| Path | `D:\Microsoft\sentiment analysis\archive\training.1600000.processed.noemoticon.csv` (prompt path `D:\sentiment analysis\...` does not exist on this machine) |
| Size / encoding | 238,803,811 bytes / latin-1 |
| Rows / columns | 1,600,000 / 6 on every row (0 malformed) |
| Labels | 0: 800,000 · 4: 800,000 (no neutral class) |
| Missing values | 0 in every column; 0 empty texts |
| Duplicate IDs | 1,685 (all label-conflicting, excluded from samples) |
| Duplicate texts | 23,300 (case-insensitive exact) |
| Dates | 2009-04-06 → 2009-06-25 (Apr 100,025 · May 576,367 · Jun 923,608), 0 unparseable |
| Text length | min 3 · median 68 · mean 73.31 · p95 136 · max 373 |
| HTML entities / URLs / @mentions | 94,458 / 76,463 / 746,432 rows |
| Flag column | `NO_QUERY` for all rows |
| Eligible rows per label after exclusions | 798,315 / 798,315 |

**Samples (seed 42, gitignored except the fixture):** eval 5,000 (2,500/label), batch 10,000 (5,000/label, disjoint
from eval), fixture 200 (committed, usernames pseudonymised with SHA-256 prefix).

**Synthetic dataset:** `ml/data/synthetic.py` + `scripts/generate_synthetic.py`. First run produced 9,892 rows; weekly
means for `general_praise`/`neutral_mixed` were raised to reach a ~10K batch → **10,192 rows** (seed 42), 84 distinct
days (2026-06-01 → 2026-08-23), planted PII in ~6% of rows, 8 empty/null texts, ~0.8% ingestion duplicates,
6 mojibake rows. Two consecutive runs produced an identical SHA-256 file hash.

**Tests:** `pytest tests/unit/test_datasets.py` → `6 passed in 9.61s`.

**Known limitation:** Sentiment140 has only ~11 weeks of 2009 tweets and no product themes, so theme/radar quality
is demonstrated on the synthetic dataset, whose planted patterns make results checkable but are easier than real data.

**Status:** PASS WITH KNOWN LIMITATION — **Gate:** PROCEED

**Git note:** the first push carried an automatic `Co-authored-by: Cursor` trailer injected by the IDE shell wrapper
around `git commit`. The owner asked for no tool attribution; a repo `commit-msg` hook did not help (trailer is added
afterwards), so commits are now created through a small Python helper outside the repo that calls git directly.
The initial commit was rewritten and force-pushed (owner approved) — verified via `gh api .../commits/main`.

---

## PHASE 2 — MODEL DOWNLOAD AND VERIFICATION (2026-09-27)

**Implementation:** `ml/models/registry.py` (local-only loaders, `local_files_only=True`, states NOT_INSTALLED /
LOAD_FAILED / LOADED), `ml/hardware.py` (runtime GPU/VRAM/RAM detection + Qwen plan), `scripts/download_models.py`,
`scripts/verify_models.py`.

**Download** — `python scripts/download_models.py --skip-verify` → exit 0, `MODEL SETUP: PASS`. Anonymous (`token=False`).

| Model | Revision | Disk (measured) | Download time |
|---|---|---|---|
| cardiffnlp/twitter-roberta-base-sentiment-latest | 3216a57f2a0d | 0.47 GB (pytorch_model.bin; repo has no root safetensors) | 46.9 s |
| sentence-transformers/all-MiniLM-L6-v2 | 1110a243fdf4 | 0.09 GB | 10.5 s |
| Qwen/Qwen2.5-3B-Instruct | aa8e72537993 | 5.76 GB | 475.5 s |

D: free 28.1 GB → 21.78 GB (6.31 GB used). C: free 72.86 → 72.85 GB (no model data on C:). HF env vars verified inside Python.
Optional `dslim/bert-base-NER` NOT downloaded (see Phase 3 decision).

**Offline verification** — run with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HTTP(S)_PROXY=http://127.0.0.1:9`
(an actual request to huggingface.co failed → `network_blocked=True`). Result `MODEL VERIFICATION: PASS`.
- RoBERTa: labels `[negative, neutral, positive]`, load 14.69 s, probabilities sum to 1; "I love this app…" → positive 0.9886,
  "worst update ever…" → negative 0.9548, "The app was updated on Tuesday." → neutral 0.8171.
- MiniLM: dim 384; cos("battery drains quickly","battery dies fast") = **0.6757** vs cos(…,"payment failed") = **−0.0087**;
  batch-1 vs batch-3 max abs diff 8.9e-8.
- Qwen: GPU **NVIDIA GeForce RTX 4050 Laptop GPU**, total VRAM **6.0 GB**, free 4.9 GB, compute 8.9, system RAM available 4.64 GB.
  Plan chosen at runtime: `cuda`, `bfloat16` compute, **bnb 4-bit NF4** (4.9 GB free < 7.2 GB needed for fp16).
  Load 20.71 s, GPU memory 1.93 GB (peak 1.99 GB after generation), 55 tokens in 13.77 s (greedy, max 120 new tokens).
- Missing model → `ModelNotInstalledError` with actionable message (no download attempted).

**Finding (manual inspection, not caught by the automated check):** Qwen's verification output was
"Battery Drain theme growth from previous period (142 to 50) with 184% increase…". All numbers were preserved (check passed)
but the **direction was reversed** (previous was 50, current 142). Number-set preservation is therefore insufficient;
Phase 12 must use a design where Qwen cannot state or reorder numbers (placeholder slots filled deterministically).

**Status:** PASS (Qwen generation quality issue carried to Phase 12 as a design requirement) — **Gate:** PROCEED

---

## PHASE 3 — PII REDACTION (2026-09-27)

**Implementation:** `ml/pii/redactor.py` (ordered regex rules: email, URL, order/account/customer IDs, card-like 13–19 digits,
phones, bare 10–12 digit phones, handles, person names via self-identification cues, honorifics and a small first-name
gazetteer), `ml/pii/leak_scan.py` (independent stricter scanner), `RedactingFilter` for logs, `ml/preprocessing/clean.py`.
Redaction returns only counts; raw matched values are never returned or logged.

**Attempt 1:** `pytest tests/unit/test_pii.py` → 1 failed: `+44 7700 900123` not redacted (phone groups limited to 5 digits).
Fix: allow 2–8 digit groups + `+` followed by 7–14 digits. Retest → 32 passed.

**Sentiment140 audit (real text, `scripts/pii_audit.py`)** found residual patterns in 18/10,000 tweets. Manual review:
- false alarms in the scanner: "awww." read as `www.`, "<3333333" hearts, "05312009" date, "1000000" → scanner tightened
- real misses fixed: unformatted phone `09166279004`, chained handle `@a@DevineNews`, honorific name `Mr Brooks`,
  URL tail after a broken link `[URL] /event.php?eid=90207194129`
- regression tests added for each case → `44 passed`.

**Measured (after fixes):**
- Synthetic batch (planted ground truth): recall **628/628 = 100%** across all 9 types; **0** false-positive rows out of 9,476
  rows without planted PII; residual leak scan: none.
- Sentiment140 10K: USER 4,835 · URL 541 · PERSON 19 · EMAIL 5 · PHONE 2 redactions.

**Engineering decision — NER:** `dslim/bert-base-NER` not used. On product reviews, customer names appear mainly in
self-identification ("this is X writing"), which the rules catch. NER on tweets would mostly redact public figures
(e.g. "Ashley Tisdale") and damage meaning. Known limitation: free-text third-party names without a cue
(e.g. "Judith Marie Keenan" in one tweet) are not redacted by rules.

**Status:** PASS WITH KNOWN LIMITATION — **Gate:** PROCEED

---

## PHASE 4 — SENTIMENT (2026-09-27)

**Implementation:** `ml/sentiment/model.py` (length-sorted batching, fp16 on CUDA), `ml/evaluation/metrics.py`,
`scripts/evaluate_sentiment.py`. Evaluation sample: 5,000 tweets (2,500/label, seed 42), redacted before inference.

**Methodology (Sentiment140 has no neutral ground truth):** `binary_forced` (positive iff P(pos) > P(neg), every example
scored — headline), `strict_3class` (neutral prediction = wrong), `abstain` (neutral predictions excluded, coverage reported).

| Metric (n = 5,000) | binary_forced | strict_3class | abstain |
|---|---|---|---|
| Accuracy | **0.7646** | 0.6014 | 0.8112 (coverage 0.7414) |
| Macro F1 | 0.7634 | 0.6906 | 0.811 |
| Negative P / R / F1 | 0.8087 / 0.6932 / 0.7465 | 0.8561 / 0.576 / 0.6887 | — |
| Positive P / R / F1 | 0.7315 / 0.836 / 0.7803 | 0.7738 / 0.6268 / 0.6926 | — |

Confusion (true → predicted neg/neu/pos): negative 1440 / 602 / 458 · positive 242 / 691 / 1567. Neutral prediction rate 25.86%.

**Attempt 1 bug:** the CPU comparison used the first 500 rows, which were all negative (file is ordered by source line) →
meaningless CPU metrics. Fixed with an evenly spaced subset. Retest: CPU vs GPU label agreement **0.998**, accuracy
0.764 on both.

**Performance (GPU fp16):** bs16 349 rev/s · bs32 716 · bs64 955 · **bs128 968 rev/s (peak 0.457 GB)** · bs256 937.
CPU fp32 bs32: **21.1 rev/s** (8 threads). Two GPU runs: identical labels, max prob diff 0.0.

**Synthetic 3-class (weak template labels, not human annotation):** accuracy 0.8882, macro F1 0.7811; neutral recall only 0.3344
(model pushes mild reviews to positive/negative).

**Status:** PASS — **Gate:** PROCEED

---

## PHASE 5 — EMBEDDINGS (2026-09-27)

`ml/embeddings/encoder.py` (L2-normalised, cached as `.npy` artifacts), `scripts/benchmark_embeddings.py`.
dim 384, norms 0.99999994–1.0000001, GPU reproducibility max diff 1.8e-7, CPU vs GPU max diff 2.1e-7.
GPU: 10,000 reviews in 4.99 s (**2,005 rev/s**, bs128, peak 0.191 GB). CPU: 1,000 in 6.54 s (153 rev/s).

**Status:** PASS — **Gate:** PROCEED

---

## PHASE 6 — THEME DISCOVERY (2026-09-27)

**Attempt 1** (PCA 50 → HDBSCAN mcs 40): 74 clusters, 22.7% noise, 45.6 s. Homogeneity 0.97 but completeness 0.54,
ARI 0.228 — each template phrasing became its own cluster (e.g. battery split into several). Rejected.

**Fix:** stage-2 merge of HDBSCAN micro-clusters by centroid cosine similarity (average linkage). Sweep
(`scripts/theme_sweep.py`, 63 configs, `artifacts/reports/theme_sweep.json`): best ARI 0.658 at PCA 20 / mcs 80 / merge 0.65.
mcs 80 cannot represent themes < 80 reviews (bad for emerging issues) → chose **mcs = max(15, 0.4% of batch) = 40**,
merge 0.65, PCA 20 (ARI ~0.64, trade-off documented).

**Attempt 2 naming:** c-TF-IDF bigram names picked detail phrases ("Reinstalled Twice" for login). Replaced with
coverage-lift naming over stemmed words (score = cov_c · ln(cov_c / cov_all)) → "Crashes Phone", "Login Password",
"Battery", "Notifications", "Failing Payment", "Delivery Tracking".

**Attempt 3:** on Sentiment140 the `[USER]` placeholder created an artificial "replies" cluster → placeholders stripped from
embedding input (display text unchanged).

**Final (synthetic 10K):** 27 themes, 12/13 planted themes recovered as a cluster majority (dark_mode_request, 24 reviews,
is below the minimum theme size — by design), ARI 0.64, NMI ≈0.82, homogeneity 0.914, completeness 0.735, 2.6% unassigned.
**Sentiment140 10K:** 95% HDBSCAN noise, 3 small themes ("Hurts Headache" etc.) — tweets have no dense product topics;
HDBSCAN correctly declines to invent themes.

**Known limitations:** praise/neutral reviews split into several sub-themes; some names are weak ("Load Fast" for slow
performance); parameters were tuned on synthetic data with templated phrasing.

**Status:** PASS WITH KNOWN LIMITATION — **Gate:** PROCEED

---

## PHASE 7 — EVIDENCE TRACEABILITY (2026-09-27)

Representatives = most central members (cosine to centroid), identical texts skipped. Radar evidence = current-window
negative reviews of the theme ranked by centroid similarity. `audit_traceability()` runs inside the pipeline;
`tests/integration/test_analytics_db.py` re-checks the DB (evidence → real review of the same theme, radar evidence negative,
theme counts = review table counts, radar counts recomputable with SQL, no PII in the full DB dump, no raw text column).
Pipeline audit: 213 links checked, 0 problems, 0 residual PII patterns.

**Status:** PASS — **Gate:** PROCEED

---

## PHASES 8–9 — COMPLAINT RADAR + DRIFT (2026-09-27)

`ml/complaints/radar.py` (14-day current vs previous window, safe growth, NEW when previous = 0, minimum 30 mentions + 3 negative
evidence reviews, ≥ 50% negative, ≥ +50% growth; transparent priority formula; segment associations with non-causal wording).
`ml/drift/monitor.py` (PSI for sentiment/theme, reviews/day change, KS on length).

Tests: `tests/unit/test_radar.py` **19 passed** (100→200 = +100%, 100→50 = −50%, 100→100 = 0%, 0→100 = NEW with no inf/NaN,
high-volume/low-growth, low-volume/high-growth → INSUFFICIENT_EVIDENCE, identical/missing/malformed periods, no reviews).
`tests/unit/test_drift.py` **7 passed** (identical → none, small change → none, large change → significant).

**Pipeline result on synthetic 10K (`python -m ml.pipeline`, 35.9 s on GPU):**
Failing Payment **NEW** (0 → 140, 96% negative) · Battery **EMERGING** (111 → 243, +119%, 86% negative) · Crashes STABLE (−1%)
· Login DECLINING (−46%) · small themes INSUFFICIENT_EVIDENCE — matches every planted pattern.
Drift current vs previous window: sentiment PSI 0.0028 (none), theme PSI 0.5863 (significant), volume +16.4% (none), length KS 0.054 (none).

**Status:** PASS — **Gate:** PROCEED

---

## PHASE 10 — FASTAPI BACKEND + SQLITE (2026-09-27)

`backend/app/main.py` serves the precomputed, already-redacted `artifacts/analytics.db` (opened read-only per request).
Endpoints: `/health`, `/metrics`, `/themes`, `/themes/{id}`, `/issues`, `/issues/{id}`, `/issues/{id}/evidence`, `/reviews`,
`/reviews/{id}`, `/sentiment/validation`, `/drift`, `/data-health`, `/model-info`, `POST /product-brief`.
Hardening: path-parameter regex validation, escaped LIKE search, bounded pagination, generic 422/500 messages (no stack traces),
503 when the DB is missing, CORS from `CORS_ORIGINS`, every outgoing text re-passed through the redactor (`safe_text`).
No endpoint exposes the raw dataset. `ml/brief/` added: `facts.py` (deterministic fact sheet + slots), `template.py`
(deterministic fallback brief), `service.py` (qwen_live → qwen_precomputed → template, with recorded fallback reasons).

**Failed attempt 1:** API PII test scanned the serialised JSON as one string → ISO timestamps were flagged as phone-like digit runs.
Fix: scan each string value; skip values that are pure ISO timestamps or hex hashes.
**Failed attempt 2:** 3 remaining false positives — `training.1600000.processed…` (round number), the executive summary's
`2026-06-01 to 2026-08-23` (ISO dates), and `Windows-10-10.0.26200-SP0` (OS version in hardware metadata).
Fix: leak scanner treats exact ISO dates and round numbers (`[1-9]\d{0,2}0{4,}`) as benign; the API test skips the
`hardware_at_pipeline_run` machine-metadata subtree. A first version of the ISO-date rule allowed trailing digits, which would
have hidden a phone number directly after a date; tightened to exact dates and a regression test added
(`"2026-06-01 555 123 4567"` must still be flagged).

Tests: `backend/tests/test_api.py` **35 passed**; full suite **112 passed**.

**Status:** PASS — **Gate:** PROCEED

Note (user instruction, 2026-09-27 20:31 IST): from this point commits are kept local only (no push).

---

## PHASE 11 — REACT DASHBOARD (2026-09-27)

`frontend/` — React 19 + Vite 8 + TypeScript 5.9 + Recharts 3 + React Router 7. Pages: Executive Overview, Themes (search/sort,
detail panel with weekly trend, per-version breakdown, representative reviews), Complaint Radar (status filter, **View why**
panel: reasons, calculation strings, thresholds, formula, weekly bars with the current window highlighted, segment associations,
current-window evidence reviews), Evidence (theme / sentiment / text filters, pagination, review drawer with probabilities),
Sentiment Validation (three scorings, confusion matrices, throughput, synthetic 3-class with neutral-recall caveat, methodology),
Data Health (ingestion counts, PII by type + recall, traceability audit, drift table with theme names, weekly PSI chart with
thresholds, model verification), Product Brief (engine selector, generate button, generation-path badge, fallback reasons,
evidence quotes linked to review IDs). Loading / error (with retry) / empty states everywhere; API base from `VITE_API_BASE_URL`.
Pages are lazy-loaded (largest chunk 359 kB instead of one 715 kB bundle). `vercel.json` adds SPA rewrites + security headers.

Tests (`frontend/src/test/app.test.tsx`, Vitest + Testing Library, fixtures generated from the real API by
`scripts/dump_api_samples.py --fixtures`): **18 passed** — headline numbers, API unreachable, 503 message, VIEW WHY shows the
calculation/reasons/evidence, evidence belongs to the theme, theme detail, search filter, evidence query parameters, review drawer,
three sentiment scorings, data health, brief shows generation path + request body, fallback reasons, and a PII-pattern scan of the
rendered text of five pages. `tsc -b` clean; `vite build` OK.

**Failed attempt 1:** Vitest forks pool timed out starting its worker on Windows ("Timeout waiting for worker to respond").
Fix: `pool: "threads"`.
**Failed attempt 2:** first test failed after lazy-loading pages (cold dynamic import > 1 s default wait). Fix: Testing Library
`asyncUtilTimeout` 5 s.
**Failed attempt 3 (browser):** a top-level `Suspense` blanked the whole app (sidebar included) while a page chunk loaded.
Fix: `Suspense` moved inside the layout around `<Outlet/>`.
**Bug found while typing responses:** `/issues` returns `rules` as a text block, not a dict; the type and the rendering were corrected.

Browser verification (local backend on :8000 + Vite on :5173, screenshots taken): Overview (KPIs, donut, emerging list, top
complaints chart), Radar → View why for Failing Payment (0 → 140, `135 x 4.00 = 540.0`, W11–W12 highlighted, evidence reviews),
Themes (Battery detail: rising weekly curve, v5.3 peak, representative reviews), Sentiment Validation, Data Health all rendered
with live data. The in-browser click on "Generate product brief" was blocked by the agent's tool policy; the same request was
verified over HTTP (`POST /product-brief` → 200, `access-control-allow-origin: http://127.0.0.1:5173`, `generation_path: template`)
and by the component tests.

Observation: Failing Payment has 140 of 152 mentions on app v5.3, but the radar reports no segment association. This is correct:
in the synthetic data the version is a function of time, so every current-window review is v5.3 and there is no within-window skew.
The per-version chart on the Themes page still shows the concentration.

**Status:** PASS WITH KNOWN LIMITATION (brief button click verified by tests + HTTP, not by in-browser click) — **Gate:** PROCEED

---

## PHASE 12 — QWEN PRODUCT BRIEF (2026-09-27)

Design (motivated by the Phase 2 finding that Qwen kept the numbers but reversed "50 → 142" into "142 to 50"):
- `ml/brief/qwen_writer.py` — Qwen2.5-3B-Instruct (bnb 4-bit NF4 on the RTX 4050, 1.92 GB VRAM, loaded with
  `local_files_only`) writes **only** the executive summary and the investigation areas. It never sees numeric values:
  the prompt lists slot names with their meaning (`{E1.current} = mentions in the current window`), theme names,
  keywords and one redacted quote per emerging theme with digits masked. The prompt is leak-scanned before generation.
  Top complaints, emerging list, evidence quotes, associations and caveats remain the deterministic template output.
- `ml/brief/validator.py` — rejects: missing sections, raw digits outside slots (even correct ones), unknown slots,
  review IDs not in the fact sheet, a sentence mixing one theme's name with another theme's numbers, reversed
  `from {Ek.current} to {Ek.previous}`, decrease wording for NEW/EMERGING or increase wording for DECLINING themes,
  causal words, certainty words, invented quoted theme names, missing emerging themes, leak-scanner hits. Then
  substitutes slot values deterministically. One retry with targeted feedback, then fallback.
- `ml/brief/service.py` — `BRIEF_MODE` now `auto | precomputed | template` (previously `template` still served a
  stored Qwen brief, which contradicted its name). Exceptions during generation now fall back instead of returning 500.
- `scripts/generate_brief.py` — generates on the GPU and stores the validated brief as report `qwen_brief`
  (served on CPU-only hosts as `qwen_precomputed`). Run log: `artifacts/reports/qwen_brief_run.json`.

**Failed attempt 1:** first prompt without an explicit structure → summary paraphrased "payment failures" instead of
naming the theme (coverage check failed) and used "root causes" (causal check); the retry repeated the same text.
Fix: prompt ends with a per-theme checklist built from the real slots; retry message names the exact missing slots /
forbidden words; "investigate/identify … causes" allowed in investigation bullets only (summary stays strict).
**Failed attempt 2:** "to confirm the pattern" (the template's own hedged wording) was flagged as certainty.
Fix: only "confirms/confirmed" count as certainty claims.
**Failed attempt 3:** live model test compared numbers with a regex that captured "140," (trailing comma) — a test bug;
regex fixed.
**Failed attempt 4:** first live API call returned 500: the running server had imported the old fact-sheet module
(stale process) while `qwen_writer` was new → `KeyError`. Root cause stale code, but it exposed that generation
exceptions were not caught; fixed + test added, server restarted.

Results (measured):
- Unit tests `tests/unit/test_brief.py` **27 passed** (adversarial 921-vs-842, correct-number-as-digits, reversed
  direction, decrease/increase wording, 5 causal phrasings, certainty, unknown slot, cross-theme number, unknown review
  ID, invented quoted theme, missing emerging theme, PII in output, malformed output, template values, fallback on
  validation failure / no GPU / load failure / generation exception, template engine never loads Qwen, prompt contains
  no fact values).
- Live GPU tests `pytest -m models tests/models/test_qwen_brief.py` **3 passed** in 179 s (network blocked via dead
  proxy + offline flags): model on CUDA; real-facts brief numbers ⊆ fact-sheet values and deterministic sections
  unchanged; prompt-injection quote ("IGNORE ALL PREVIOUS RULES… 921 reviews… caused it") never produced an accepted
  brief containing 921 or causal wording.
- `scripts/generate_brief.py`: attempt 1 rejected (`causal language: causes`), attempt 2 passed; ~50–58 s per attempt
  (~198 new tokens, ≈3.5–4 tokens/s with bnb 4-bit on the laptop GPU); stored as `qwen_brief`.
- Live API (`POST /product-brief {"engine":"auto"}` on the local server with GPU): `generation_path: qwen_live`,
  2 attempts, validation passed, 146.8 s including model load.
- API tests: precomputed brief served with `BRIEF_MODE=precomputed`, validation flag true, no PII, deterministic sections
  identical to the template, every number in its summary is an analytics value; `BRIEF_MODE=template` disables Qwen.
  Full suite **140 passed** (3 model tests deselected by default).

Known limitations: the validator guarantees numbers, names, IDs, direction words and non-causal wording, not every
nuance — e.g. the stored summary calls Battery "previously emerging, now shows a significant rise" (it is currently
EMERGING); interpretive phrases such as "indicating dissatisfaction" are allowed. Live generation is slow (1–3 min).

**Status:** PASS WITH KNOWN LIMITATION — **Gate:** PROCEED

---

## PHASES 13–14 — END-TO-END RUNS, SCALING, OFFLINE DEMO (2026-09-27)

`scripts/benchmark_pipeline.py` runs the whole pipeline (load → redact/clean → sentiment → embeddings → themes → radar →
drift → traceability → SQLite) in a fresh subprocess per configuration, embedding cache disabled.
`ml/pipeline.py --limit N` now takes a deterministic evenly spaced subsample (previously the first N rows, which in
time-ordered data meant "first week only"); `--no-embedding-cache` added.

**Measured (synthetic batch, RTX 4050 Laptop vs CPU; `artifacts/reports/pipeline_benchmark.json`):**

| reviews | GPU total | CPU total | GPU sentiment rev/s | CPU sentiment rev/s | GPU embed s | CPU embed s | traceability |
|---|---|---|---|---|---|---|---|
| 100 | 19.2 s | 21.4 s | 88 | 19 | 0.33 | 0.67 | PASS |
| 999 | 16.5 s | 55.6 s | 588 | 27 | 0.66 | 5.71 | PASS |
| 10,104 | 46.7 s | 455.4 s | 805 | 28 | 5.69 | 64.27 | PASS |

GPU is 9.8× faster end to end at 10K (model load ≈ 13 s dominates small batches).
Sentiment140 10K batch end to end on GPU (`pipeline_benchmark_sentiment140.json`): 9,999 tweets, 43.1 s, 824 rev/s,
traceability PASS, 3 themes (open-ended tweets rarely form product themes — documented limitation from Phase 6).

**Failed attempt (found by the benchmark):** small batches produced no useful themes — 0 themes at 100 reviews, 2 at 999
(ARI −0.006) — because `min_cluster_size = max(15, 0.4% n)` and a fixed `min_samples = 10` were tuned for 10K.
Investigation: `scripts/theme_scaling.py` (grid over min_cluster_size × min_samples at 100 / 250 / 500 / 1K / 2.5K / 10K,
scored against planted ground truth; `artifacts/reports/theme_scaling.json`).
Fix: `min_cluster_size = clip(1% n, 3, 40)`, `min_samples = clip(n / 1000, 3, 10)` — identical (40 / 10) at 10K, so the
main results are unchanged. With the rule: 100 → 6 themes (ARI 0.604), 250 → 19 (0.571), 500 → 21 (0.640),
999 → 19 (0.808), 2.5K → 18 (0.662), 10K → 27 (0.640). Re-benchmark on GPU: 100 → 6 themes, 999 → 19 themes
(`pipeline_benchmark_small_adaptive.json`).
Related fix: a cached-embedding run recorded MiniLM `revision: null` in `model_versions`; the cache path now records the
local revision.

The main database and the stored Qwen brief were regenerated after these changes (pipeline 36.3 s, 27 themes,
traceability PASS; brief attempt 1 rejected for causal wording, attempt 2 passed).

`scripts/run_demo.py` — offline demo (dead proxy + HF offline flags for the process and its children): console walkthrough
of overview, radar with VIEW WHY calculation and evidence, sentiment validation, data health and product brief, all through
the real API code; `--fresh N` re-runs the pipeline first; `--serve` starts API + dashboard. Verified: default walkthrough
(brief path `qwen_precomputed`), and `--fresh 1000` (16.9 s on GPU, 19 themes, traceability PASS, brief falls back to the
template because the fresh database has no stored Qwen brief).
Observation: in a 1,000-review subsample no issue reaches NEW/EMERGING because the radar's evidence threshold is absolute
(≥ 30 current-window mentions); Failing Payment has ≈ 14 there. This is intended: small batches do not raise alarms on
thin evidence.

Tests: `tests/unit/test_pipeline_utils.py` (subsample determinism). Full suite **143 passed** (+3 model tests deselected);
frontend **18 passed**.

**Status:** PASS — **Gate:** PROCEED

---

## PHASES 15–16 — DEPLOYMENT PREPARATION + DOCUMENTATION (2026-09-27)

**Deployment configuration**
- `requirements-api.txt` (fastapi, uvicorn, pydantic, pinned): the API serves the precomputed DB and needs no torch or transformers.
- `render.yaml` Blueprint:
  - Python 3.11.9 (`.python-version`), `BRIEF_MODE=precomputed`, offline flags, `/health` check, `autoDeploy: false`.
  - `CORS_ORIGINS` and `CORS_ORIGIN_REGEX` are `sync: false`, so they are entered in the Render dashboard rather than committed.
- `frontend/vercel.json` (Phase 11): SPA rewrite plus security headers.
- Docker is not used. There is one stateless Python process and one static site, and both platforms build them natively.

**Production rehearsal (local):**
- **Setup:** a fresh virtual environment with only `requirements-api.txt`, running `uvicorn` on port 8001 with `BRIEF_MODE=precomputed`, `CORS_ORIGINS=http://127.0.0.1:4173` and the Hugging Face cache pointed at a nonexistent folder. The frontend was built with `VITE_API_BASE_URL=http://127.0.0.1:8001` and served by `vite preview` on port 4173.
- **Results:**
  - backend tests in that environment: **37 passed**;
  - `/health` ok with 10,104 reviews;
  - `/model-info` reports `qwen_installed: false` and `live_generation_possible: false`;
  - `POST /product-brief` (auto) returns `qwen_precomputed` with the CORS header for port 4173;
  - a foreign origin gets no CORS header;
  - in the browser, the production build rendered Complaint Radar → **View why** for Battery (111 → 243, `208 x 2.19 = 455.35`, evidence reviews) from the port 8001 API.
- **Note:** the first screenshot after navigating still showed the previous page of the tab (the Product Brief page from the dev server on port 5173). A second screenshot after the load showed the correct page. It is noted here so the stale image is not mistaken for evidence.

**Found while documenting:** the `pii_audit` report copied into `analytics.db` still contained the 23 redacted Sentiment140 tweets kept for
manual review. The API never served them (`/data-health` exposes only the synthetic recall figures), but the database is the file that
gets deployed, and the dataset must not be copied unnecessarily.
- **Fix:** `ml/pipeline.py` drops `redacted_examples_for_manual_review` from the database copy. The existing DB was patched the same way (with a VACUUM), so the stored Qwen brief stays valid. The standalone report file keeps the examples as the manual-review audit record.
- **Test added:** `test_no_sentiment140_text_in_reports`.

**Gap closed (problem statement: "reads a batch of reviews"):** until now the pipeline accepted only the two built-in datasets.
`--source path\to\reviews.csv` now loads any review CSV:
- column names are matched case-insensitively (text, date, and optional rating, platform, version, ID);
- ISO and `d/m/Y` or `m/d/Y` dates are accepted, and timezones are converted to UTC;
- unsafe or duplicate IDs are replaced with `U000001`-style IDs;
- a file without a text or date column is rejected with an explanation. Dates are never invented, because the radar and drift compare time windows;
- the DB stores the source kind `csv` and the file name, not the local path (the first draft stored the full path, which `/health` would have exposed).

Failed attempt: the date parser included a `%Z` format meant for "PDT" timestamps. Python's `strptime` does not parse such timezone names,
and the unit test caught it, so the format was removed (Sentiment140 has its own parser).
Verified end to end: the synthetic batch was re-saved with columns `Review, Date, Stars, Version` (no IDs, no platform) and run with
`python -m ml.pipeline --source artifacts\cache\upload_test.csv --db artifacts\cache\upload_test.db --no-embedding-cache`.
It took 44.0 s on the GPU and produced the same 27 themes and radar statuses (Failing Payment NEW 0 → 140, Battery EMERGING 111 → 243). Traceability passed,
`/health` reports source `csv`, and review IDs are `U0xxxxx`.

**Documentation written:**
- `README.md` (updated)
- `docs/RESULTS.md` (measured results only, each with its report file)
- `docs/ARCHITECTURE.md`
- `docs/MODEL_CARD.md`
- `docs/DATASET.md`
- `docs/TESTING.md`
- `docs/API.md` (the example payloads were first written from memory with wrong field names, then corrected from real API responses)
- `docs/DEMO.md`
- `docs/DEPLOYMENT.md`

**Tests:** full Python suite **148 passed** (3 model tests deselected); frontend **18 passed**.

**Status:**
- Deployment preparation: PASS.
- Actual deployment: **BLOCKED**. It needs the owner's Render and Vercel accounts and a push to GitHub; the owner asked to keep commits local. Nothing is claimed as deployed.

**Gate:** PROJECT COMPLETE LOCALLY. Deployment is pending the owner's action (steps in `docs/DEPLOYMENT.md`).

## PHASE 17 — SENTIMENT ACCURACY STUDY (2026-09-28)

Goal: raise the real, measured sentiment accuracy of `cardiffnlp/twitter-roberta-base-sentiment-latest` on
Sentiment140 without tuning on test data. Owner decisions:
- a new 80/10/10 split of all 1.6M rows, so the old 5K number is no longer directly comparable;
- fine-tune a separate binary model as an experiment, and keep the 3-class model in the pipeline;
- about 2 hours of fine-tuning;
- 1–2 comparison models for evaluation only.

All numbers are in `docs/RESULTS.md` ("Sentiment accuracy study").

**Inspection.** The pipeline already follows the model card: mentions → `@user`, links → `http`, 128 tokens, no lower-casing,
emojis kept. Label mapping is by `id2label` name, not position. The old 5K sample reproduced 0.7646 exactly before any change.

**Steps**
1. `scripts/build_s140_split.py`: text-group hash split (67 s).
   - Split sizes: train 1,262,463, validation 157,077, test 156,705.
   - Excluded: 3,370 conflicting-ID rows and 20,385 conflicting-text rows.
   - Cross-split overlap: 0.
2. `sentiment_experiments.py val-study` (34 min, about 394 tweets/s including the cache):
   - 9 preprocessing variants, with neutral strategies A–D for each;
   - the negation subset;
   - McNemar tests against the current pipeline;
   - confidence analysis;
   - threshold tuning on validation. Selected: current preprocessing, threshold 0.725.
3. `sentiment_experiments.py test-eval`: one test run. Baseline 0.7767, tuned threshold 0.7806.
4. `finetune_sentiment.py matrix`: 5 runs, 4.4 h in total, selected ft01. Then `finetune_sentiment.py final`: full validation 0.8732, then one test run, 0.8730.
5. `sentiment_experiments.py compare`: siebert 0.7520 and DistilBERT 0.7073 on test.
6. `sentiment_experiments.py errors`: category error rates for the pretrained and fine-tuned models.
7. `scripts/evaluate_sentiment.py` moved from the 5K sample to the test split. It adds neutral strategies, the tuned-threshold scoring and an `accuracy_study` summary, which the Sentiment Validation page shows in a new card.

**Decisions and why**
- Preprocessing unchanged: no PII-safe variant gained at least 0.1 pp with p < 0.01. Removing mentions was significantly *worse* (−0.1 pp).
- Strategy C (neutral → nearer class by probability) stays the headline. A and B add a fixed bias, and D drops 27% of the data.
- The threshold is reported as an additional scoring, not a replacement. The product emits 3-class labels, so the threshold only affects binary evaluation. The unchanged `binary_forced` number stays the headline and feeds the brief caveat.
- The fine-tuned model is not integrated. It cannot output neutral, and its gain is measured on Sentiment140's emoticon-derived labels, not on product reviews.
- The comparison models were worse, so RoBERTa stays.
- No class weighting: all splits are 49.9–50.0% positive.

**Failed attempts and problems**
- The first fine-tuning launch crashed on step 1 with `shape '[-1, 3]' is invalid`. Replacing the classifier head updated `config.num_labels`, but not the model's own `num_labels` attribute that the loss uses. Fixed by setting both.
- The fine-tuning took longer than budgeted. A run at batch 32 took 37 min alone on the GPU, and 45–66 min while inference jobs shared it. The batch-16 run took 79 min. An attempt to stop the process after three runs was blocked, so the full planned matrix (5 runs, 4.4 h) ran to the end.
- Updating the `sentiment_validation` report stored in `artifacts/analytics.db`, and regenerating the Qwen brief (whose caveat quotes the accuracy), were blocked while the owner was away. The owner approved the step the next morning:
  - the DB report was replaced;
  - the brief caveat now reads "held-out labelled Sentiment140 test split: 77.7%";
  - the Qwen brief was regenerated and passed validation;
  - the frontend fixtures were re-dumped from the API.

  After that: pytest 156 passed, frontend 18 passed.
- The first `evaluate_sentiment.py` run compared a batch-128 run with a batch-64 run and reported "identical labels: false" (0.06% of labels differ from fp16 padding). The dashboard would have shown "reproducible: no". The check now repeats identical settings (identical, max difference 0.0) and reports the batch-size agreement separately (0.9994).
- Error-analysis examples (redacted tweets) were first written into the committed report. They were moved to the gitignored cache, so no dataset text is committed.
- One frontend run timed out on the first test (59 s run while the GPU jobs were loading the machine). The re-run passed in 12.7 s.

**Tests:**
- New: `tests/unit/test_sentiment_eval.py` (8 tests) and `tests/models/test_sentiment_model.py` (13 tests, CPU and GPU).
- Full Python suite: **156 passed**.
- Sentiment model tests: **13 passed**.
- Frontend: **18 passed**, `tsc` clean, build OK.
