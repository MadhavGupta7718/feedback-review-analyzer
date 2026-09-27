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
