# Results

Measured results only. Every number is copied from command output; the full JSON lives in
`artifacts/reports/` (file named in each section). Hardware: laptop with NVIDIA RTX 4050 Laptop GPU
(6 GB VRAM), 16 GB system RAM, Windows 11, Python 3.11.9, torch 2.11.0+cu128. How each number was produced is
recorded in `docs/DEVELOPMENT_LOG.md`.

## Headline

| What | Result |
|---|---|
| **Product model (Amazon clothing fine-tune)** TEST accuracy / macro recall (`amazon_sentiment_eval.json`, 4,812 holdout reviews) | **0.8315** / **0.7363** |
| Per-class TEST recall | neg **0.7225** · neu **0.6000** · pos **0.8865** |
| Split | 38,661 train / 4,830 val / 4,812 test (`sha256(text)%100` → 80/10/10) |
| Labels | Cons_rating → 1–2 neg, 3 neu, 4–5 pos (weak labels) |
| Training | 3 epochs, bs=16, lr=2e-5; best val macro recall **0.7470** (epoch 2) |
| Upload path | Per-batch analytics DB; **no** per-upload accuracy page |
| Sample upload pipeline (407 reviews, Amazon fine-tune loaded) | **23.6 s** GPU; traceability PASS |

### Historical offline studies (not the product accuracy page)

| What | Result |
|---|---|
| Sentiment140 holdout binary (pretrained cardiffnlp) | 0.7766 accuracy |
| Amazon / Yelp polarity offline | ~0.92 / ~0.87 |
| Synthetic Nimbus planted PII / themes / radar | See archive sections below |

## Sentiment validation (`sentiment_validation.json`)

Model `cardiffnlp/twitter-roberta-base-sentiment-latest` (revision `3216a57f2a0d`), fp16 on CUDA.
Evaluation set: the held-out **test split** of all 1.6M Sentiment140 tweets (156,705 tweets: 78,357 negative, 78,348
positive; see "Sentiment accuracy study" below for how the split was built). PII is redacted before inference.
Sentiment140 has **no neutral ground truth**, so the scorings are reported separately:

| Scoring | n | Accuracy | Macro F1 | Neg P / R | Pos P / R |
|---|---|---|---|---|---|
| binary_forced (positive iff P(pos) > P(neg)), headline | 156,705 | 0.7766 | 0.7757 | 0.8177 / 0.7119 | 0.7449 / 0.8413 |
| binary_threshold (positive iff P(pos)/(P(pos)+P(neg)) > 0.725, threshold tuned on validation only) | 156,705 | 0.7806 | 0.7806 | 0.7838 / 0.7750 | 0.7775 / 0.7862 |
| strict_3class (a neutral prediction counts as wrong) | 156,705 | 0.6024 | 0.6948 | 0.8673 / 0.5820 | 0.7814 / 0.6227 |
| abstain (neutral predictions excluded) | 115,023 (coverage 0.734) | 0.8207 | 0.8206 | 0.8673 / 0.7696 | 0.7814 / 0.8749 |

- Confusion matrix (rows are the true label; columns are predicted negative / neutral / positive): negative 45,605 / 19,099 / 13,653; positive 6,975 / 22,583 / 48,790. The model predicted neutral for 26.6% of tweets.
- Reproducibility: 5,000 test tweets scored twice with identical settings gave identical labels (max probability difference 0.0). Batch size 128 vs 64 changes 0.06% of labels (fp16 padding differences; max probability difference 0.0029). CPU and GPU labels agree on 100% of a 500-tweet subset (accuracy 0.788 on both).
- The earlier headline, 0.7646 on a stratified 5,000-tweet sample, is still reproduced exactly by the same model and code; it is a different, smaller sample and is not directly comparable with the test-split numbers.
- Synthetic 3-class check (weak template labels, not human annotation; 10,104 reviews): accuracy 0.8882, macro F1 0.7811, neutral recall **0.3344**. The model pushes mild reviews towards positive or negative.

Sentiment throughput on the GPU (fp16, 5,000 test tweets): batch size 16 gives 448 rev/s, 32 gives 753, **64 gives 918 (peak 0.381 GB)**, 128 gives 891 and 256 gives 845.
On the CPU (fp32, batch size 32, 8 threads): 21.9 rev/s.

## Sentiment accuracy study (`sentiment_study_*.json`, `sentiment_finetune.json`, `sentiment_error_analysis.json`)

Goal: raise the real, measured accuracy on Sentiment140 without tuning on the test data. Code:
`scripts/build_s140_split.py`, `scripts/sentiment_experiments.py`, `scripts/finetune_sentiment.py`.

**Split** (`s140_split.json`). All 1,600,000 rows were grouped by a normalised text key (lower case, HTML entities
decoded, mentions → `@user`, links → `http`, whitespace collapsed) and each group went to train / validation / test by a
seeded hash (80/10/10), so identical or trivially re-posted text never crosses splits (overlap checked: 0 in every pair).
Excluded, and counted: 3,370 rows whose tweet ID carries both labels, and 20,385 rows whose text group carries both
labels. Result: train 1,262,463, validation 157,077, test 156,705, each 49.9–50.0% positive, so no class weighting was
needed. Every test evaluation is appended to `sentiment_test_log.jsonl` (6 entries: the pretrained model once, each
comparison model once, the fine-tuned model once, and two re-runs of `evaluate_sentiment.py` with no selection).

**Neutral strategies** (pretrained model, validation / test accuracy):

| Strategy | Validation | Test | Comment |
|---|---|---|---|
| A: neutral → negative | 0.7199 | 0.7243 | fixed bias towards negative (positive recall 0.62) |
| B: neutral → positive | 0.7466 | 0.7465 | fixed bias towards positive (negative recall 0.58) |
| C: neutral → nearer class by the model's own probabilities | **0.7748** | **0.7767** | chosen: uses the model's evidence, no fixed bias, every tweet scored |
| D: exclude neutral | 0.8185 (coverage 0.7325) | 0.8207 (coverage 0.734) | not comparable: drops the hardest 27% |

C is the defensible choice because A and B inject a label bias by construction and D scores only an easier subset.
Within neutral predictions, 45% of tweets are truly negative, so the neutral class is not "mostly one label".

**Preprocessing** (validation, strategy C, McNemar test against the current pipeline):

| Variant | Accuracy | Changed inputs | McNemar p |
|---|---|---|---|
| current (repair → redact → tidy → `@user`/`http`) | 0.7748 | – | – |
| keep punctuation runs | 0.7748 | 7,439 | 0.29 |
| no ftfy repair | 0.7749 | 1,414 | 0.37 |
| remove links | 0.7748 | 7,796 | 0.94 |
| remove mentions | 0.7738 | 72,232 | 0.0005 (worse) |
| `#word` → `word` | 0.7747 | 3,658 | 0.02 |
| placeholders left as `[USER]`/`[URL]` | 0.7750 | 77,371 | 0.75 |
| raw tweet, unredacted (reference only) | 0.7753 | all | 0.20 |

No PII-safe variant met the selection rule (at least +0.1 pp with p < 0.01), so preprocessing is unchanged. The fully
raw, unredacted tweet is only 0.05 pp better and not significant, so PII redaction costs no measurable accuracy.
Negation: 40,135 validation tweets (25.6%) contain a negation word. Their accuracy is 0.7663 against 0.7748 overall.
No preprocessing step removes words, and the negation-word count changes in only 23 of 157,077 rows. Negation handling
therefore needs no fix.

**Confidence.** Accuracy rises with confidence: 0.573 when the top probability is below 0.5, 0.874 when it is 0.9 or
above. Mean confidence is 0.80 on correct and 0.72 on wrong predictions. The 26.8% of tweets predicted neutral are
0.655 accurate after strategy C, against 0.819 for the rest.

**Threshold** (validation only): the score P(pos)/(P(pos)+P(neg)) with a cut at **0.725** gave validation accuracy 0.7795,
against 0.7748 at 0.5. On test, the unchanged threshold gives **0.7806 vs 0.7767** (+0.39 pp; McNemar p = 1.4e-10;
4,940 tweets fixed, 4,321 broken). It also balances the classes: negative recall rises from 0.712 to 0.775 and positive
recall falls from 0.841 to 0.786.

**Fine-tuning a separate binary model** (`artifacts/models/roberta-s140-binary`, gitignored; the product model is untouched).
Initialised from the same checkpoint with a 2-class head copied from its negative and positive rows. Trained on 100,000
train-split tweets (50,000 per label) with production preprocessing: AdamW (weight decay 0.01), 6% warm-up then linear
decay, gradient clipping 1.0, bf16 autocast, max 64 tokens (0.02% of tweets are longer), seed 42, early stopping with
patience 1. Selection used a fixed stratified 20,000-tweet validation subset.

| Run | lr | Batch | Max epochs | Epochs run | Best epoch | Val-subset accuracy | Val-subset macro F1 | Training time |
|---|---|---|---|---|---|---|---|---|
| **ft01** (selected) | 1e-5 | 32 | 3 | 3 | 3 | **0.8756** | 0.8756 | 2,200 s |
| ft02 | 2e-5 | 32 | 3 | 3 | 3 | 0.8748 | 0.8747 | 2,689 s |
| ft03 | 3e-5 | 32 | 3 | 3 | 2 | 0.8750 | 0.8750 | 3,978 s |
| ft04 | 1e-5 | 16 | 3 | 3 | 2 | 0.8752 | 0.8752 | 4,728 s |
| ft05 | 1e-5 | 32 | 4 | 3 (early stop) | 2 | 0.8746 | 0.8745 | 2,330 s |

The five runs are within 0.1 pp of each other, which is inside the noise of a 20,000-tweet subset (standard error about
0.23 pp). ft02 to ft04 took longer because inference jobs shared the GPU. Selected ft01: full validation accuracy
**0.8732** (macro F1 0.8731). The tuned threshold (0.485) gained only 0.02 pp on validation, so the default 0.5 is kept.
Single test run: **0.8730** accuracy, macro F1 0.8730 (negative P/R 0.8708/0.8759, positive P/R 0.8751/0.8701). That
is **+9.63 pp over the pretrained baseline** (McNemar: 22,489 tweets fixed, 7,396 broken, p < 1e-300). Test throughput:
945 tweets/s.

**Comparison models** (evaluation only; production preprocessing, strategy C):

| Model | Validation | Test |
|---|---|---|
| cardiffnlp twitter-roberta-base-sentiment-latest (product) | 0.7748 | 0.7767 |
| `siebert/sentiment-roberta-large-english` (355M, binary) | 0.7528 | 0.7520 |
| `distilbert/distilbert-base-uncased-finetuned-sst-2-english` (binary) | 0.7075 | 0.7073 |

Neither comparison model beats the product model, so RoBERTa stays. siebert's training data includes a small
hand-labelled Sentiment140 test set, which could only have helped it.

**Error analysis** (validation; categories are automatic keyword/regex proxies, not manual annotation):

| Category | Share of tweets | Error rate, pretrained | Error rate, fine-tuned |
|---|---|---|---|
| all | 100% | 0.2252 | 0.1268 |
| contrast / mixed sentiment ("but", "though") | 10.4% | 0.322 | 0.163 |
| emoji / emoticon remnants | 1.8% | 0.271 | 0.145 |
| slang / abbreviations | 15.1% | 0.264 | 0.152 |
| question | 10.4% | 0.251 | 0.152 |
| elongated words (spelling proxy) | 8.2% | 0.245 | 0.135 |
| reply (starts with a mention) | 43.3% | 0.244 | 0.144 |
| negation | 25.6% | 0.234 | 0.133 |
| short text (4 words or fewer) | 10.5% | 0.183 | 0.104 |
| possible sarcasm (keyword proxy) | 0.4% | 0.160 | 0.077 |
| low binary margin (score within 0.1 of 0.5, each model's own score) | 4.7% (pretrained) | 0.481 | 0.468 |

Domain-specific language could not be measured automatically (no domain lexicon). Contrast and mixed-sentiment tweets are
the hardest category for the pretrained model. The fine-tuned model roughly halves the error in every category except the low-margin tweets, which stay near
chance: many Sentiment140 labels come from emoticons that were stripped from the text, so some tweets carry no
recoverable sentiment. Labels were never changed. Example tweets are kept only in the gitignored cache, so no dataset
text is committed.

**What changed in the product:** nothing in the 3-class labels. The validation-tuned binary threshold is now
`config.SENTIMENT_BINARY_THRESHOLD` and reported by `scripts/evaluate_sentiment.py` and the Sentiment Validation page, next
to the unchanged `binary_forced` headline. The fine-tuned binary model stays an experiment: the product needs a neutral
class, which a binary model cannot give.

## Sentiment on labelled reviews (`review_eval.json`)

Sentiment140 is tweets, while this project analyses product reviews, so the models were also scored on two public
review test sets. They were downloaded to `data/raw/reviews` (gitignored), and no review text is written to any report.
Each is a stratified random sample of 10,000 reviews per label (seed 42), preprocessed like production, including PII
redaction, with a 128-token limit. **Nothing was tuned on these datasets.** Labels come from star ratings:
- Amazon: 1–2 stars negative, 4–5 positive; 3-star reviews are not in the dataset.
- Yelp: 1–2 stars negative, 3–4 positive.

Accuracy (neutral → nearer class; macro F1 within 0.001 of accuracy unless shown):

| Model | Amazon polarity | Yelp polarity |
|---|---|---|
| **Product: cardiffnlp twitter-roberta-base-sentiment-latest** | **0.9156** | **0.8719** (F1 0.8711) |
| Product, Sentiment140 threshold 0.725 (not re-tuned) | 0.9149 | 0.8793 |
| Fine-tuned binary RoBERTa (trained on Sentiment140 only) | 0.8805 | 0.8818 |
| `distilbert-base-uncased-finetuned-sst-2-english` | 0.8889 | 0.8727 |
| `siebert/sentiment-roberta-large-english`, **contaminated** (see below) | 0.9609 | 0.9447 |

- **Product model on Amazon:** accuracy 0.9156 when every review is scored. It predicts neutral for 7.4% of reviews; excluding those gives 0.935 at 92.6% coverage. Strict 3-class accuracy is 0.8658. Negative recall is 0.895 and positive recall 0.937.
- **Yelp is harder:** 3-star reviews count as positive, so lukewarm reviews carry a positive label. Negative recall is 0.795 against 0.949 for positive. Also, 48% of Yelp reviews exceed 128 tokens and are truncated (29% on Amazon; mean length 170 and 101 tokens).
- **The Sentiment140 fine-tune does not transfer:** it is 3.5 pp *worse* than the product model on Amazon, and only 1.0 pp better on Yelp. Its +9.6 pp on tweets is mostly adaptation to Sentiment140's emoticon-derived labels. This supports keeping the pretrained model in the product.
- **siebert is contaminated:** it scores highest, but its published training data includes Amazon and Yelp reviews, so these are not unseen-domain results. It is shown for reference only and is not a fair comparison.

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
| Python unit + integration + API (`pytest`) | 163 passed, 16 model tests deselected by default |
| Model tests (`pytest -m models`) | Sentiment regression: 13 passed (CPU and GPU). Qwen brief: 3 passed |
| API tests in a minimal environment with no torch or transformers (`requirements-api.txt`) | 37 passed |
| Frontend (Vitest + Testing Library) | 18 passed; `tsc -b` clean; production build OK (largest chunk 359 kB) |

## Deployment

**Not deployed.** The Render and Vercel configuration is written and was rehearsed locally with production settings (see
`docs/DEPLOYMENT.md`). Deploying needs the owner's Render and Vercel accounts and a push to GitHub, which the owner
deferred.
