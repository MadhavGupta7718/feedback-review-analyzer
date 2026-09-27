# Results

Measured results only. Every number is copied from command output; the full JSON lives in
`artifacts/reports/` (file named in each section). Hardware: laptop with NVIDIA RTX 4050 Laptop GPU
(6 GB VRAM), 16 GB system RAM, Windows 11, Python 3.11.9, torch 2.11.0+cu128. How each number was produced is
recorded in `docs/DEVELOPMENT_LOG.md`.

## Headline

| What | Result |
|---|---|
| Sentiment accuracy on 5,000 labelled Sentiment140 tweets (binary, every example scored) | **0.7646** (macro F1 0.7634) |
| PII recall on the synthetic batch (628 planted PII rows, 9 types) | **100%** (628/628), 0 false-positive rows out of 9,476 |
| Evidence traceability audit (theme/complaint → review IDs) | PASS: 213 links checked, 0 problems |
| Planted synthetic themes recovered | 12 of 13 (the missing one, 24 reviews, is below the minimum theme size) |
| Complaint Radar on the synthetic batch | NEW, EMERGING, STABLE and DECLINING planted patterns all detected |
| End-to-end pipeline, 10,104 reviews | **46.7 s on the GPU vs 455.4 s on the CPU (9.8× faster)** |

## Sentiment validation (`sentiment_validation.json`)

Model `cardiffnlp/twitter-roberta-base-sentiment-latest` (revision `3216a57f2a0d`), fp16 on CUDA.
Sample: 5,000 Sentiment140 tweets, stratified 2,500 per label, seed 42; tweets whose ID appears with conflicting labels
are excluded. PII is redacted before inference. Sentiment140 has **no neutral ground truth**, so three
scorings are reported separately:

| Scoring | n | Accuracy | Macro F1 | Neg P / R | Pos P / R |
|---|---|---|---|---|---|
| binary_forced (positive iff P(pos) > P(neg)), headline | 5,000 | 0.7646 | 0.7634 | 0.8087 / 0.6932 | 0.7315 / 0.8360 |
| strict_3class (a neutral prediction counts as wrong) | 5,000 | 0.6014 | 0.6906 | 0.8561 / 0.5760 | 0.7738 / 0.6268 |
| abstain (neutral predictions excluded) | 3,707 (coverage 0.7414) | 0.8112 | 0.8110 | 0.8561 / 0.7587 | 0.7738 / 0.8662 |

- Confusion matrix (rows are the true label; columns are predicted negative / neutral / positive): negative 1440 / 602 / 458; positive 242 / 691 / 1567. The model predicted neutral for 25.86% of tweets.
- Reproducibility: two GPU runs gave identical labels (max probability difference 0.0). CPU and GPU labels agree on 0.998 of a 500-tweet subset, with accuracy 0.764 on both devices.
- Synthetic 3-class check (weak template labels, not human annotation; 10,104 reviews): accuracy 0.8882, macro F1 0.7811, neutral recall **0.3344**. The model pushes mild reviews towards positive or negative.

Sentiment throughput on the GPU (fp16, 5,000 tweets): batch size 16 gives 366 rev/s, 32 gives 617, 64 gives 939, **128 gives 956 (peak 0.457 GB)** and 256 gives 934.
On the CPU (fp32, batch size 32, 8 threads): 21.1 rev/s.

## PII redaction (`pii_audit.json`)

| Batch | Result |
|---|---|
| Synthetic 10,192 rows (628 rows with planted PII) | recall 76/76 account IDs, 79/79 cards, 85/85 customer IDs, 71/71 emails, 72/72 handles, 71/71 order IDs, 64/64 person names, 67/67 phones, 43/43 URLs; **0 false-positive rows**; residual leak scan clean |
| Sentiment140 10,000 tweets (no ground truth) | redactions: 4,835 user handles, 541 URLs, 19 person names, 5 emails, 2 phones; residual leak scan flags **1** long digit run for manual review |

Known limitation: a person's name that appears in free text without a cue (for example "this is X writing" or "Mr X")
is not redacted. One such tweet was found in the manual review.

## Theme discovery (`theme_experiment_synthetic.json`, `theme_sweep.json`, `theme_scaling.json`)

Pipeline: MiniLM embeddings, then PCA 20, then HDBSCAN, then a merge of micro-clusters (cosine 0.65), then reassignment of noise points (cosine 0.60).
Scores are measured against the synthetic dataset's planted ground truth:

| Batch | Themes | ARI | Other |
|---|---|---|---|
| Synthetic 10,104 | 27 | 0.640 | NMI ≈ 0.82, homogeneity 0.914, completeness 0.735, 2.6% unassigned |
| Sentiment140 10K | 3 | n/a | 95% HDBSCAN noise: open-ended tweets have no dense product topics |

Small batches, after adapting `min_cluster_size` and `min_samples` to the batch size:

| Reviews | 100 | 250 | 500 | 999 | 2,500 | 10,104 |
|---|---|---|---|---|---|---|
| Themes | 6 | 19 | 21 | 19 | 18 | 27 |
| ARI | 0.604 | 0.571 | 0.640 | 0.808 | 0.662 | 0.640 |

## Complaint Radar and drift (synthetic batch, current 14 days vs previous 14 days)

| Complaint | Previous | Current | Growth | Negative | Status | Priority |
|---|---|---|---|---|---|---|
| Failing Payment | 0 | 140 | NEW | 96% | NEW | 540.0 |
| Battery | 111 | 243 | +118.92% | 86% | EMERGING | 455.4 |
| Crashes Phone | 262 | 259 | −1% | 92% | STABLE | 239.0 |
| Login Password | 70 | 38 | −46% | 89% | DECLINING | 34.0 |

These match the planted patterns: payment failures start after an app release, battery drain ramps up, crashes stay flat and login
problems fade. Drift between the two windows: sentiment PSI 0.0028 (none), theme PSI 0.5863 (significant), volume
+16.4% (none), review-length KS 0.054 (none).

## Performance (`pipeline_benchmark*.json`, `embedding_benchmark.json`)

The whole pipeline, from load through redaction, sentiment, embeddings, themes, radar, drift and the traceability audit to SQLite, runs in a fresh process for each configuration with the embedding cache off:

| Reviews | GPU total | CPU total | GPU sentiment rev/s | CPU sentiment rev/s | GPU embeddings | CPU embeddings |
|---|---|---|---|---|---|---|
| 100 | 19.2 s | 21.4 s | 88 | 19 | 0.33 s | 0.67 s |
| 999 | 16.5 s | 55.6 s | 588 | 27 | 0.66 s | 5.71 s |
| 10,104 | 46.7 s | 455.4 s | 805 | 28 | 5.69 s | 64.27 s |

- Sentiment140 10K on the GPU: 9,999 tweets in 43.1 s end to end (824 rev/s sentiment); the traceability audit passed.
- Embeddings alone: 10,000 texts in 4.99 s on the GPU (2,005 rev/s, peak 0.191 GB), versus 153 rev/s on the CPU.
- Model load takes about 13 s, which dominates small batches.

## Qwen product brief (`qwen_brief_run.json`)

- **Setup:** `Qwen/Qwen2.5-3B-Instruct` (revision `aa8e72537993`). The plan is chosen at runtime: bnb 4-bit NF4 with bf16 compute, because only 4.95 GB of VRAM was free and fp16 needs 7.2 GB. It loads in 23.2 s and uses 1.92 GB of GPU memory (peak 1.96 GB).
- **Stored run:** attempt 1 was rejected by the validator (`causal language: causes`). Attempt 2 passed all 11 checks. Each attempt took about 53–56 s for about 198 new tokens, roughly 3.5–4 tokens/s.
- **Live API call on the laptop** (`POST /product-brief`, engine auto): 146.8 s including model load; the generation path was `qwen_live`.
- **Live GPU tests:** 3 passed in 179 s. Numbers in the brief are a subset of the fact-sheet values. A prompt-injection review ("IGNORE ALL PREVIOUS RULES … 921 reviews … caused it") never produced an accepted brief containing 921 or causal wording.

## Tests (see `docs/TESTING.md`)

| Suite | Result |
|---|---|
| Python unit + integration + API (`pytest`) | 148 passed, 3 GPU model tests deselected by default |
| GPU model tests (`pytest -m models`) | 3 passed |
| API tests in a minimal environment with no torch or transformers (`requirements-api.txt`) | 37 passed |
| Frontend (Vitest + Testing Library) | 18 passed; `tsc -b` clean; production build OK (largest chunk 359 kB) |

## Deployment

**Not deployed.** The Render and Vercel configuration is written and was rehearsed locally with production settings (see
`docs/DEPLOYMENT.md`). Deploying needs the owner's Render and Vercel accounts and a push to GitHub, which the owner
deferred.
