# Model Card

All three models are public, downloaded anonymously (no Hugging Face token), stored in `D:\huggingface`, and loaded with
`local_files_only=True`. Production code never downloads a model. A missing model raises `ModelNotInstalledError`
with instructions to run `scripts/download_models.py`. Measured values come from `artifacts/reports/model_verification.json`,
`sentiment_validation.json`, `embedding_benchmark.json` and `qwen_brief_run.json`.

| Model | Revision | Disk | Role |
|---|---|---|---|
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | `3216a57f2a0d` | 0.47 GB | Sentiment (negative / neutral / positive) |
| `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4` | 0.09 GB | 384-d embeddings for theme discovery and evidence ranking |
| `Qwen/Qwen2.5-3B-Instruct` | `aa8e72537993` | 5.76 GB | Optional wording of two product-brief sections |

## 1. Sentiment: twitter-roberta-base-sentiment-latest

- **Why:** trained on tweets, so short and informal review text is in-domain. It has three classes, so mixed or neutral reviews are not forced into positive or negative.
- **Input:** PII-redacted, cleaned text, truncated to 128 tokens.
- **Output:** a label, class probabilities and a confidence score. Batches are sorted by length; fp16 on CUDA, fp32 on CPU.
- **Validation:** 5,000 labelled Sentiment140 tweets.
  - Binary-forced accuracy 0.7646, macro F1 0.7634.
  - Strict 3-class accuracy 0.6014.
  - Abstain accuracy 0.8112 at 74.14% coverage.

  See `docs/RESULTS.md` for per-class precision and recall and the confusion matrix.
- **Known weaknesses:**
  - Sentiment140 labels were produced from emoticons (distant supervision), so they are noisy, and they have no neutral class. That is why three scorings are reported instead of one flattering number.
  - On the synthetic reviews, neutral recall is only 0.33: the model labels mild or mixed reviews as positive or negative.
  - Negative recall on Sentiment140 is 0.69 (binary). The model often calls negative tweets neutral (602 of 2,500).
  - English only.
- **Speed:** 956 rev/s at batch size 128 on the RTX 4050 (peak 0.46 GB); 21 rev/s on the CPU.

## 2. Embeddings: all-MiniLM-L6-v2

- **Why:** small (0.09 GB), fast, and good at short-sentence semantic similarity. Verified: cos("battery drains quickly", "battery dies fast") = 0.676 vs cos(…, "payment failed") = −0.009.
- **Output:** L2-normalised 384-d vectors, cached as `.npy` artifacts (not in the database).
- **Reproducibility:** max difference between GPU runs 1.8e-7; between CPU and GPU 2.1e-7.
- **Speed:** 2,005 texts/s on the GPU, 153 texts/s on the CPU.
- **Known weaknesses:** themes inherit the embedding's view of similarity. Praise and neutral reviews split into several sub-themes, and theme names come from word statistics, so some are weak (for example "Load Fast" for a slow-performance theme).

## 3. Qwen2.5-3B-Instruct (presentation layer only)

- **Why:** the largest instruct model that runs locally on a 6 GB laptop GPU with headroom. At runtime the loader measures free VRAM: with 4.95 GB free (fp16 needs 7.2 GB), it chose bitsandbytes 4-bit NF4 with bf16 compute. That uses 1.92 GB of VRAM and generates about 3.5–4 tokens/s.
- **What it is allowed to do:** write the executive summary and the investigation areas of the product brief, using placeholders such as `{E1.current}`.
- **What it never does:**
  - It never sees or states a number: values are substituted by code after validation.
  - It never receives the full batch or raw PII: it gets one redacted, digit-masked quote per emerging theme.
  - It never produces the top complaints, evidence, statuses or caveats: those are deterministic.
- **Guardrails:**
  - The validator has 11 checks, covering format, known slots, known review IDs, no raw digits, slot/theme consistency, direction words, no causal language, hedged wording, known theme names, coverage of emerging themes, and no PII.
  - One retry is made with targeted feedback. If that fails, the brief falls back to the precomputed brief, then to the template, and the fallback reasons are reported.
- **Measured:**
  - The stored brief passed on attempt 2; attempt 1 was rejected for the word "causes".
  - The live GPU tests passed, including a prompt-injection review that asked for an invented number and a causal claim.
- **Known weaknesses:**
  - The validator guarantees facts, names, direction and non-causal wording, but not every nuance. The stored summary calls Battery "previously emerging, now shows a significant rise", even though it is currently EMERGING.
  - Interpretive phrases such as "indicating dissatisfaction" are allowed.
  - Live generation takes 1–3 minutes on the laptop and is unavailable on CPU-only hosts (there, `qwen_precomputed` is served).

## Not used

`dslim/bert-base-NER` was evaluated as an option and **not** downloaded. Rules catch customer self-identification in
product reviews. NER on tweets would mostly redact public figures and damage meaning, so it was left out (see the
Phase 3 entry in `docs/DEVELOPMENT_LOG.md`).

## Intended use and limits

- **Intended use:** summarising batches of English app or product reviews for product teams.
- **Every output needs human judgement.** Themes and radar statuses describe patterns in the text. They show association, not
  cause. Theme parameters were tuned on the synthetic dataset, and real review data will be messier.
