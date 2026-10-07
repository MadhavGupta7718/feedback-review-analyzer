# Offline / non-product experiment inventory

Files listed here are **not required** for the current product path:

**upload CSV → clean/PII → Amazon-trained sentiment → themes → radar/drift → FastAPI + React UI**  
(+ optional Qwen product brief, teacher-label training, hyperparam sweep evidence).

This document was written before cleanup. Items marked **DELETED** were removed from the repo after this list was created.

---

## Kept on purpose (not deleted)

| Path | Why kept |
|------|----------|
| `scripts/teacher_label_amazon.py`, `finetune_amazon_sentiment.py`, `prepare_amazon_clothing.py`, `run_hyperparam_sweep.py` | Current training / selection path |
| `scripts/download_models.py`, `verify_models.py`, `env_check.py`, `generate_brief.py`, `inspect_db.py`, `run_demo.py`, `dump_api_samples.py` | Ops / demo / fixtures |
| `ml/data/synthetic.py`, `ml/data/sentiment140.py` | Still imported by pipeline helpers and unit tests |
| `ml/sentiment/experiment.py`, `ml/evaluation/s140_split.py` | Still used by model/unit tests |
| `artifacts/reports/amazon_sentiment_eval.json` | Live Model Validation metrics (Trial A) |
| `artifacts/reports/hyperparam_sweep/` | Documented Trial A / B / C evidence |
| `artifacts/reports/qwen_brief_run.json`, `model_verification.json`, `env_check.json`, `dataset_validation.json`, `model_download.json`, `pip_freeze.txt` | Useful runtime / audit artifacts |
| Docs under `docs/`, `report_build/` | Mentions historical studies; left as narrative |

---

## 1. Local logs / error dumps — DELETED

Regenerable noise; not part of the product.

- `artifacts/reports/compare.err`
- `artifacts/reports/compare.log`
- `artifacts/reports/download_comparison.log`
- `artifacts/reports/download_models.log`
- `artifacts/reports/errors.err`
- `artifacts/reports/errors.log`
- `artifacts/reports/evaluate_sentiment.log`
- `artifacts/reports/finetune_matrix.log`
- `artifacts/reports/generate_synthetic.log`
- `artifacts/reports/test_eval.err`
- `artifacts/reports/test_eval.log`
- `artifacts/reports/theme_experiment_s140.log`
- `artifacts/reports/validate_dataset.log`
- `artifacts/reports/val_study.log`
- `artifacts/reports/verify_models.log`

---

## 2. Sentiment140 / star-study offline scripts — DELETED

Historical tweet/polarity accuracy work; not the Amazon teacher→student product path.

- `scripts/build_s140_split.py`
- `scripts/sentiment_experiments.py`
- `scripts/finetune_sentiment.py`
- `scripts/evaluate_sentiment.py`
- `scripts/evaluate_reviews.py`
- `scripts/validate_dataset.py`
- `scripts/pii_audit.py`
- `scripts/benchmark_embeddings.py`
- `scripts/benchmark_pipeline.py`

---

## 3. Theme experiment scripts — DELETED

Offline HDBSCAN sweeps vs planted truth; product uses fixed theme discovery in the pipeline.

- `scripts/theme_experiment.py`
- `scripts/theme_sweep.py`
- `scripts/theme_scaling.py`

---

## 4. Synthetic product CSV generators — DELETED

Not the primary UI story (upload real CSVs). Core `ml/data/synthetic.py` kept for tests/pipeline helpers.

- `scripts/generate_synthetic.py`
- `scripts/generate_dove_shampoo.py`
- `scripts/generate_adidas_shoes.py`
- `scripts/generate_realme_earbuds.py`
- `scripts/generate_borosil_bottle.py`
- `ml/data/dove_shampoo.py`
- `ml/data/adidas_shoes.py`
- `ml/data/realme_earbuds.py`
- `ml/data/borosil_bottle.py`
- `ml/data/product_batch.py`

---

## 5. Legacy Amazon Reviews 2023 converter — DELETED

Superseded by `prepare_amazon_clothing.py` for the clothing teacher/student path.

- `scripts/prepare_amazon_reviews.py`

---

## 6. Historical experiment report artifacts — DELETED

Offline study outputs referenced mainly from older handbook / RESULTS sections.

- `artifacts/reports/s140_split.json`
- `artifacts/reports/sentiment_study_compare.json`
- `artifacts/reports/sentiment_study_test.json`
- `artifacts/reports/sentiment_study_val.json`
- `artifacts/reports/sentiment_finetune.json`
- `artifacts/reports/sentiment_error_analysis.json`
- `artifacts/reports/sentiment_test_log.jsonl`
- `artifacts/reports/sentiment_validation.json` (Sentiment140 holdout; UI now uses `amazon_sentiment_eval.json`)
- `artifacts/reports/review_eval.json`
- `artifacts/reports/theme_experiment_synthetic.json`
- `artifacts/reports/theme_experiment_sentiment140.json`
- `artifacts/reports/theme_sweep.json`
- `artifacts/reports/theme_scaling.json`
- `artifacts/reports/pipeline_benchmark.json`
- `artifacts/reports/pipeline_benchmark_sentiment140.json`
- `artifacts/reports/pipeline_benchmark_small_adaptive.json`
- `artifacts/reports/embedding_benchmark.json`
- `artifacts/reports/pii_audit.json`
- `artifacts/reports/synthetic_summary.json`
- `artifacts/reports/model_download_comparison.json`

---

## Not deleted (local / gitignored)

These may still exist on disk but were never meant for git:

- `data/raw/`, `data/interim/` (CSVs)
- `artifacts/models/`, `artifacts/embeddings/`, `artifacts/cache/`
- `.venv/`
