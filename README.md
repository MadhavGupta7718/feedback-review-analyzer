# Feedback & Review Analyzer — "10,000 Reviews, No Time to Read Them"

Microsoft hackathon project (problem 17, Student Edition: Feedback & Review Analyzer).

A tool that reads a batch of reviews and reports the main themes, common complaints and overall
sentiment in a dashboard — with enterprise guarantees:

- **Traceable** — every theme and complaint links to real review IDs and redacted verbatims.
- **Validated** — sentiment accuracy is measured against a labelled Sentiment140 sample.
- **Private** — PII is redacted before anything reaches the database, API, dashboard, logs or LLM.
- **Monitored** — sentiment / theme / volume / length drift is tracked over time.
- **Complaint Radar** — flags complaints that are *growing*, not just the biggest ones, and shows the
  exact calculation behind every flag.

> Status: under active development. See `docs/DEVELOPMENT_LOG.md` for the verified phase-by-phase log
> and `docs/RESULTS.md` for measured results only.

## Datasets

| Dataset | Use |
|---|---|
| Sentiment140 (1.6M tweets, binary labels) | dataset validation, sentiment accuracy evaluation |
| Synthetic "Nimbus" app reviews (deterministic, seed 42) | 10K demo batch with planted ground truth for themes, Complaint Radar, drift and PII tests |

Raw data is never committed. See `docs/DATASET.md`.

## Quick start (development)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
.\.venv\Scripts\python.exe scripts\env_check.py
.\.venv\Scripts\python.exe scripts\validate_dataset.py
.\.venv\Scripts\python.exe scripts\generate_synthetic.py
```
