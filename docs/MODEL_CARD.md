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
- **Validation:** held-out Sentiment140 test split, 156,705 tweets (80/10/10 text-group split of all 1.6M rows, no overlap with train or validation).
  - Binary-forced accuracy 0.7766, macro F1 0.7757.
  - With the validation-tuned binary threshold (`SENTIMENT_BINARY_THRESHOLD = 0.725` on P(pos)/(P(pos)+P(neg))): 0.7806, macro F1 0.7806.
  - Strict 3-class accuracy 0.6024.
  - Abstain accuracy 0.8207 at 73.4% coverage.
  - The earlier 5,000-tweet sample gave 0.7646 and still reproduces exactly.

  See `docs/RESULTS.md` for per-class precision and recall, the confusion matrix and the full accuracy study.
- **Labelled reviews** (out of domain for a tweet model; no tuning; `scripts/evaluate_reviews.py`):
  - Amazon polarity: 0.9156 accuracy on 20,000 reviews.
  - Yelp polarity: 0.8719, where 3-star reviews count as positive and 48% of reviews exceed the 128-token limit.
  - The Sentiment140 fine-tuned variant scores lower on Amazon (0.8805), so its tweet gain does not carry over to reviews.
- **Accuracy study (summary):**
  - Preprocessing variants, negation handling and PII redaction change validation accuracy by at most 0.1 pp. None was adopted.
  - Neutral → nearest class by probability (strategy C) is the scoring used. Mapping neutral to a fixed label, or excluding it, is biased or not comparable.
  - Larger or other off-the-shelf models scored lower on test: siebert RoBERTa-large 0.7520, DistilBERT-SST-2 0.7073.
- **Experimental fine-tuned binary variant** (`artifacts/models/roberta-s140-binary`, not used by the pipeline):
  - Same checkpoint, 2-class head, fine-tuned on 100,000 train-split tweets.
  - Test accuracy 0.8730, macro F1 0.8730 (+9.63 pp over the pretrained baseline).
  - It cannot predict neutral, which the product needs for mixed reviews. It is also tuned to Sentiment140's emoticon-derived labels, so the gain may not carry over to product reviews.
- **Known weaknesses:**
  - Sentiment140 labels were produced from emoticons (distant supervision), so they are noisy, and they have no neutral class. That is why three scorings are reported instead of one flattering number.
  - On the synthetic reviews, neutral recall is only 0.33: the model labels mild or mixed reviews as positive or negative.
  - Negative recall on Sentiment140 is 0.71 (binary, 0.775 with the tuned threshold). The model often calls negative tweets neutral (19,099 of 78,357 on the test split).
  - Contrast and mixed-sentiment tweets ("but", "though") have the highest error rate (0.32 on validation).
  - English only.
- **Speed:** 918 rev/s at batch size 64 on the RTX 4050 (peak 0.38 GB); 22 rev/s on the CPU.

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
