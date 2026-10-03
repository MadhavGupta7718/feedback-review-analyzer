"""Build Feedback_Review_Analyzer_Backend_Guide.docx — full backend / pipeline reference."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from docx_util import bullets, code, h1, h2, h3, new_doc, numbered, p, save, table  # noqa: E402

OUT_PRIMARY = Path(r"D:\Microsoft\Feedback_Review_Analyzer_Backend_Guide.docx")
OUT_COPY = ROOT / "docs" / "Feedback_Review_Analyzer_Backend_Guide.docx"


def build() -> list[Path]:
    doc = new_doc(
        "Backend & Pipeline Guide",
        "Complete reference for the offline ML pipeline, SQLite analytics store, FastAPI service, "
        "privacy boundary, evaluation, and how to analyse any new review CSV.",
    )

    h1(doc, "1. What the backend is")
    p(doc,
      "This project separates heavy analysis from serving. The backend has two layers:")
    numbered(doc, [
        "Offline pipeline (ml/pipeline.py) — runs once per batch on a GPU laptop if available. "
        "It cleans reviews, redacts PII, scores sentiment, discovers themes, runs Complaint Radar "
        "and drift, evaluates that batch only, and writes one SQLite file (analytics DB).",
        "Online API (backend/app/main.py) — a FastAPI process that opens that SQLite file "
        "read-only and returns JSON. It does not retrain models or re-cluster reviews on each request.",
    ])
    p(doc,
      "The product brief LLM (Qwen) is optional presentation on top of numbers that already exist "
      "in the DB. It never invents metrics; a validator rejects unsafe wording.")

    h1(doc, "2. Mental model (important)")
    bullets(doc, [
        "You do NOT train a separate model per dataset. One pretrained RoBERTa + MiniLM stack scores every batch.",
        "Each pipeline run produces a separate analytics DB. Amazon metrics live only in the Amazon DB; "
        "Nimbus only in Nimbus; Dove only in Dove.",
        "Opening the dashboard with --db path/to/batch.db shows that batch alone — never mixed numbers.",
        "Ground-truth columns (gt_sentiment, gt_theme, gt_pii) are used for scoring during the run, "
        "then stripped before the DB write so the API never serves labels.",
    ])

    h1(doc, "3. Repository layout (backend-relevant)")
    table(doc, ["Path", "Role"], [
        ["ml/pipeline.py", "End-to-end orchestrator; CLI entry python -m ml.pipeline"],
        ["ml/config.py", "Paths, model names, seed, ANALYTICS_DB default"],
        ["ml/preprocessing/clean.py", "Null/empty reject, duplicates, mojibake, then redaction"],
        ["ml/pii/redactor.py", "Regex PII → typed placeholders; counts only"],
        ["ml/pii/leak_scan.py", "Stricter residual PII scanner for audits/tests"],
        ["ml/sentiment/model.py", "cardiffnlp twitter-roberta-base-sentiment-latest (3-class)"],
        ["ml/embeddings/encoder.py", "sentence-transformers MiniLM, SHA-256 disk cache"],
        ["ml/themes/discovery.py", "PCA → HDBSCAN → merge → naming → representatives"],
        ["ml/complaints/radar.py", "14-day windows, NEW/EMERGING/DECLINING/…, priority"],
        ["ml/drift/monitor.py", "PSI + volume + length KS between windows"],
        ["ml/evaluation/batch_eval.py", "Per-DB accuracy/recall bundle written into reports"],
        ["ml/evaluation/metrics.py", "Binary + 3-class scoring helpers"],
        ["ml/data/synthetic.py", "Nimbus app mock (~21K) with planted truth"],
        ["ml/data/dove_shampoo.py", "Dove shampoo mock (~15K) with planted truth"],
        ["ml/brief/", "Fact sheet, template brief, Qwen writer, validator, service"],
        ["backend/app/main.py", "FastAPI routes"],
        ["backend/app/db.py", "Read-only SQLite helpers"],
        ["scripts/run_demo.py", "Walkthrough + optional API/UI launch"],
        ["scripts/generate_dove_shampoo.py", "Write Dove CSV"],
        ["artifacts/analytics.db", "Default demo DB (Nimbus after regen)"],
        ["artifacts/cache/*.db", "Other batches (Amazon, Dove, …)"],
        ["data/raw/reviews/*.csv", "Input CSVs (gitignored / local)"],
    ])

    h1(doc, "4. End-to-end pipeline stages")
    p(doc, "Command shape:")
    code(doc,
         "python -m ml.pipeline --source synthetic|path\\to\\reviews.csv --db artifacts\\cache\\my.db [--device cuda|cpu]")
    numbered(doc, [
        "Load — synthetic generator, Dove CSV, or any review CSV with text + date columns.",
        "Validate / clean / redact — drop empty text; remove ingestion duplicates; repair mojibake; "
        "redact PII before any model sees the text.",
        "Sentiment — RoBERTa 3-class probabilities; store label + confidence + p_negative/neutral/positive.",
        "Embeddings — MiniLM 384-d vectors (cached on disk by content hash).",
        "Themes — PCA + HDBSCAN clustering, merge micro-clusters, name themes, pick representatives.",
        "Complaint Radar — compare last 14 days vs previous 14 days per theme; assign status + priority.",
        "Drift — sentiment/theme PSI, volume change, review-length KS; weekly series.",
        "Traceability audit — every evidence ID must exist, match theme, and (for radar) be negative.",
        "Batch evaluation — sentiment / theme / radar / PII recall for THIS batch only → reports table.",
        "Write DB — reviews (redacted only), themes, issues, reports JSON blobs, model_versions, meta.",
    ])

    h2(doc, "4.1 CSV contract (any future dataset)")
    p(doc, "Column names are matched case-insensitively:")
    table(doc, ["Field", "Required?", "Accepted names"], [
        ["text", "yes", "text, review, review_text, content, body, comment, feedback"],
        ["created_at", "yes", "created_at, date, timestamp, review_date, time, at, datetime"],
        ["rating", "no", "rating, score, stars, star_rating → weak sentiment labels"],
        ["gt_sentiment", "no", "gt_sentiment, label, sentiment_label (neg/neu/pos aliases OK)"],
        ["gt_theme", "no", "gt_theme, theme_label, topic_label"],
        ["gt_pii", "no", "gt_pii, pii_types, planted_pii (; -separated types)"],
        ["platform", "no", "platform, os, device"],
        ["app_version", "no", "app_version, version, review_created_version"],
        ["review_id", "no", "review_id, id (else auto U000001…)"],
    ])
    p(doc,
      "If labels are absent, the pipeline still runs; sentiment validation / theme recall / radar recall / "
      "PII recall are simply omitted or limited to what can be computed (e.g. star weak labels only).")

    h2(doc, "4.2 Privacy boundary")
    bullets(doc, [
        "Raw text never enters analytics.db — only text_redacted.",
        "Redaction runs before sentiment, embeddings, storage, logs, and any LLM prompt.",
        "API calls safe_text() again on every outgoing review (defence in depth).",
        "PII reports store counts by type, never the matched secret values.",
        "Logs use a redacting filter.",
    ])

    h2(doc, "4.3 Complaint Radar rules (general, not dataset-specific)")
    bullets(doc, [
        "Windows: current = last window_days (default 14); previous = the 14 days before that.",
        "NEW — previous==0 and current ≥ effective min mentions.",
        "EMERGING — growth_pct ≥ +50%.",
        "DECLINING — growth_pct ≤ −25%, OR growth ≤ −15% with a falling multi-week trend (soft decline).",
        "STABLE — otherwise (after complaint / evidence gates).",
        "NOT_A_COMPLAINT — negative ratio below 50%.",
        "INSUFFICIENT_EVIDENCE — too few current mentions or negative evidence rows, or batch shorter than two windows.",
        "Min mentions scale down on small CSVs (<5000 rows) so tiny uploads are not stuck forever.",
        "Priority = current_negative_mentions × (1 + clip(growth/100, 0, 3)); NEW uses ×4.",
    ])

    h2(doc, "4.4 Per-DB evaluation (batch_eval)")
    p(doc, "Stored under reports keys sentiment_validation and batch_evaluation:")
    table(doc, ["Score", "When available", "Meaning"], [
        ["Sentiment 3-class accuracy + macro recall", "gt_sentiment or star ratings", "Argmax vs labels; macro = mean of neg/neu/pos recall"],
        ["Theme recall", "gt_theme planted", "Share of planted themes recovered by clustering"],
        ["Radar recall", "gt_theme + planted temporal keys", "Expected NEW/EMERGING/DECLINING/… patterns"],
        ["PII recall", "gt_pii planted", "Planted rows that received ≥1 redaction"],
        ["Mean available recalls", "any of the above", "Average of scores this batch can compute"],
    ])

    h1(doc, "5. Analytics SQLite schema (conceptual)")
    table(doc, ["Table", "Contents"], [
        ["meta", "generated_at_utc, source, dataset info, seed, theme/radar params, schema_version"],
        ["reviews", "review_id, created_at, rating, platform, app_version, text_redacted, sentiment, "
         "probs, theme_id, similarity, pii counts, source — NO raw text column"],
        ["themes", "theme_id + JSON (name, keywords, sizes, radar fields, representative_ids, …)"],
        ["issues", "Complaint Radar items (status, growth, reasons, calculation, evidence IDs, …)"],
        ["reports", "JSON blobs: overview, data_health, drift, traceability, performance, "
         "sentiment_validation, batch_evaluation, hardware, optional qwen_brief / pii_audit"],
        ["model_versions", "Which models/revisions/device were used for this run"],
    ])

    h1(doc, "6. FastAPI service")
    h2(doc, "6.1 How to run")
    code(doc,
         "# Full env (same venv as pipeline)\n"
         ".\\.venv\\Scripts\\python.exe -m uvicorn backend.app.main:app --port 8000\n\n"
         "# API-only deps (no torch) after DB already built\n"
         "pip install -r requirements-api.txt\n"
         "set ANALYTICS_DB=artifacts\\cache\\amazon_software.db\n"
         "uvicorn backend.app.main:app --port 8000\n\n"
         "# Or via demo helper (also starts Vite UI)\n"
         ".\\.venv\\Scripts\\python.exe scripts\\run_demo.py --serve --db artifacts\\cache\\dove_shampoo.db")
    p(doc, "Interactive OpenAPI: http://127.0.0.1:8000/docs — ReDoc: /redoc")

    h2(doc, "6.2 Guarantees")
    bullets(doc, [
        "Read-only DB connections (URI mode=ro).",
        "No raw dataset endpoints.",
        "Path params validated (theme_\\d{3}, review IDs [A-Za-z0-9_-]{1,40}).",
        "Search q capped at 60 chars; LIKE escaped; limit ≤ 200.",
        "Errors are generic (422 / 404 / 503 / 500) — no stack traces or internal paths.",
        "CORS: only CORS_ORIGINS (+ optional CORS_ORIGIN_REGEX); methods GET and POST; no credentials.",
    ])

    h2(doc, "6.3 Endpoints (detail)")
    table(doc, ["Method", "Path", "Purpose"], [
        ["GET", "/health", "Liveness + review count + dataset name; 503 degraded if DB missing"],
        ["GET", "/metrics", "Overview KPIs, dataset, pipeline seconds, device"],
        ["GET", "/themes", "Theme list; filters: complaints_only, sort, q"],
        ["GET", "/themes/{theme_id}", "Theme detail + representatives + radar slice + segments"],
        ["GET", "/issues", "Full radar list + window/formula/params/rules"],
        ["GET", "/issues/{theme_id}", "One issue with reasons, weekly_counts, associations"],
        ["GET", "/issues/{theme_id}/evidence", "Redacted evidence + representative reviews"],
        ["GET", "/reviews", "Paginated reviews; theme_id, sentiment, q, limit, offset"],
        ["GET", "/reviews/{review_id}", "One review + class probs + evidence usage"],
        ["GET", "/sentiment/validation", "This-DB sentiment metrics + optional overall recall block"],
        ["GET", "/drift", "Current-vs-previous and weekly drift reports"],
        ["GET", "/data-health", "Ingestion stats, PII audit summary, batch_evaluation, traceability"],
        ["GET", "/model-info", "Model revisions, hardware, performance, brief engine status"],
        ["POST", "/product-brief", "Body {engine: auto|qwen|template} → validated brief JSON"],
    ])

    h3(doc, "Product brief engines")
    bullets(doc, [
        "auto / qwen — try live Qwen on CUDA if BRIEF_MODE allows; else stored precomputed brief; else template.",
        "template — always deterministic, no GPU.",
        "generation_path is always returned: qwen_live | qwen_precomputed | template.",
        "Numbers/IDs in Qwen prose come from slot fill-in after validation — the model must write slot names, not digits.",
    ])

    h2(doc, "6.4 Environment variables")
    table(doc, ["Variable", "Default", "Purpose"], [
        ["ANALYTICS_DB", "artifacts/analytics.db", "Which SQLite file the API serves"],
        ["CORS_ORIGINS", "localhost:5173 variants", "Allowed browser origins"],
        ["CORS_ORIGIN_REGEX", "(unset)", "Optional regex for preview URLs"],
        ["BRIEF_MODE", "auto", "auto | precomputed | template"],
        ["HF_HOME / offline flags", "see .env.example", "Local Hugging Face cache / offline"],
    ])

    h1(doc, "7. Built-in datasets & how to add a new one")
    table(doc, ["Dataset", "How to build DB", "Typical size"], [
        ["Nimbus (synthetic app)", "python -m ml.pipeline --source synthetic --db artifacts/analytics.db", "~22K"],
        ["Amazon Software CSV", "pipeline --source data/raw/reviews/amazon_software_recent.csv "
         "--db artifacts/cache/amazon_software.db", "~12K"],
        ["Dove shampoo mock", "scripts/generate_dove_shampoo.py then pipeline on that CSV "
         "--db artifacts/cache/dove_shampoo.db", "~15.7K"],
        ["Your CSV", "pipeline --source path\\to\\file.csv --db artifacts/cache\\mine.db", "any"],
    ])
    p(doc, "Regenerate Dove CSV:")
    code(doc, ".\\.venv\\Scripts\\python.exe scripts\\generate_dove_shampoo.py")

    h1(doc, "8. Models (product path)")
    table(doc, ["Model", "Job", "Notes"], [
        ["cardiffnlp/twitter-roberta-base-sentiment-latest", "3-class sentiment", "Pretrained; same for every CSV"],
        ["sentence-transformers/all-MiniLM-L6-v2", "Embeddings / themes", "384-d, L2-normalised, disk cache"],
        ["Qwen2.5-3B (optional 4-bit)", "Brief wording only", "Never produces raw numbers; validated"],
    ])
    p(doc,
      "Sentiment140 tweet fine-tunes and tweet-corpus dashboards were removed from the product path. "
      "Archive scripts may still exist under scripts/ for offline study only.")

    h1(doc, "9. Testing (backend / ML)")
    bullets(doc, [
        "Unit: PII, radar rules, batch_eval, cleaning, brief validator, metrics helpers.",
        "Integration: tests/integration/test_analytics_db.py — evidence links, no raw PII, no raw-text column.",
        "Model tests (optional GPU): sentiment CPU/GPU agreement, Qwen smoke.",
        "Run: .\\.venv\\Scripts\\python.exe -m pytest -q",
    ])

    h1(doc, "10. Operational checklist")
    numbered(doc, [
        "Install Python 3.11 venv, torch (CUDA if available), requirements.txt, copy .env.example → .env.",
        "scripts/download_models.py once (public models, local cache).",
        "Generate or obtain a review CSV with text + dates.",
        "Run ml.pipeline into a dedicated --db path.",
        "Point ANALYTICS_DB or run_demo --db at that file.",
        "Start uvicorn (and optionally the Vite frontend).",
        "For a live Qwen brief on GPU: scripts/generate_brief.py or POST /product-brief with engine=auto.",
    ])

    h1(doc, "11. Known limitations")
    bullets(doc, [
        "Theme/radar quality on template synthetic data is easier than on messy real reviews.",
        "Star-rating weak labels (3★ = neutral) are noisy — Amazon-style macro recall will look lower.",
        "Names without honorific/cue patterns may not redact.",
        "Radar statuses are associations over time, not root-cause proof.",
        "Live Qwen needs CUDA and can take 1–3 minutes on first call.",
    ])

    paths = [save(doc, OUT_PRIMARY), save(doc, OUT_COPY)]
    return paths


if __name__ == "__main__":
    for path in build():
        print("Wrote", path)
