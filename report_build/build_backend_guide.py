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
        "Drift — sentiment/theme PSI, volume change, review-length KS; weekly series "
        "(see §4.4 Drift formulas). Stubbed when dates_available is false.",
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

    h2(doc, "4.3 Complaint Radar formulas (ml/complaints/radar.py)")
    p(doc, "Windows (default window_days = 14), ending at the latest review timestamp:")
    code(doc,
         "current  = (end - 14d, end]\n"
         "previous = (end - 28d, end - 14d]")
    p(doc, "Core quantities:")
    code(doc,
         "growth_pct     = (current - previous) / previous * 100     # None if previous == 0\n"
         "negative_ratio = (# negative reviews of theme in current) / (# theme reviews in current)\n"
         "acceleration_pp = ((c-b)/b - (b-a)/a) * 100               # last three weekly buckets a,b,c")
    p(doc, "Weekly trend: linear regression slope on weekly counts; relative = slope / mean(counts). "
           "rising if rel > 0.05, falling if rel < -0.05, else stable.")
    p(doc, "Min current mentions scales with batch size:")
    code(doc,
         "if n_reviews >= 5000: min_mentions = 30\n"
         "else: min_mentions = clamp(round(n_reviews * 0.015), floor=8, ceil=30)")
    p(doc, "Status rules (evaluated in order):")
    bullets(doc, [
        "NO_DATA — current == 0 and previous == 0",
        "NOT_A_COMPLAINT — negative_ratio < 0.50 (current, or whole batch if current empty)",
        "INSUFFICIENT_EVIDENCE — current < min_mentions, or negative evidence reviews < 3",
        "NEW — previous == 0 and current >= min_mentions",
        "EMERGING — growth_pct >= 50",
        "DECLINING — growth_pct <= -25, OR (growth_pct <= -15 AND weekly trend is falling AND previous >= min_mentions)",
        "STABLE — otherwise",
    ])
    p(doc, "Priority (sort key only; does not decide status), max_growth_bonus = 3:")
    code(doc,
         "growth_factor = 1 + clip(growth_pct / 100, 0, 3)     # NEW: growth_factor = 1 + 3 = 4\n"
         "priority      = current_negative_mentions * growth_factor")
    p(doc, "Segment association lift (app_version / platform), association not causation:")
    code(doc,
         "theme_share   = count(segment=v in theme current) / count(theme current with segment)\n"
         "overall_share = count(segment=v in all current) / count(all current with segment)\n"
         "lift          = theme_share / overall_share\n"
         "# keep if support >= 20 and lift >= 1.3")

    h2(doc, "4.4 Drift formulas (ml/drift/monitor.py)")
    p(doc,
      "Drift compares a reference window vs a current window (usually two consecutive 14-day periods "
      "from the batch’s own timestamps). Implementation: ml/drift/monitor.py. Served by GET /drift. "
      "When the upload has no usable dates, the pipeline writes a stub with status unavailable.")
    p(doc, "Four metrics are computed; overall_status is the worst of the four (none < moderate < significant).")

    h3(doc, "Sentiment and theme mix — Population Stability Index (PSI)")
    p(doc,
      "Convert each window to category shares (sentiment: negative / neutral / positive; themes: theme_id "
      "plus unassigned). Clip each share with epsilon ε = 1e-4, renormalize, then:")
    code(doc, "PSI = sum_i (p_cur_i - p_ref_i) * ln(p_cur_i / p_ref_i)")
    p(doc,
      "Jensen–Shannon distance (base 2) is also stored as a bounded [0, 1] companion (js_distance).")
    table(doc, ["PSI value", "Status"], [
        ["< 0.10", "none"],
        ["0.10 – 0.25", "moderate"],
        [">= 0.25", "significant"],
    ])

    h3(doc, "Volume — relative change in reviews per day")
    code(doc,
         "r = (#reviews) / (#days)\n"
         "change = (r_cur - r_ref) / r_ref")
    table(doc, ["|change|", "Status"], [
        ["< 20%", "none"],
        ["20% – 50%", "moderate"],
        [">= 50%", "significant"],
    ])

    h3(doc, "Review length — Kolmogorov–Smirnov (KS)")
    p(doc,
      "Two-sample KS test on character lengths of text_redacted. Uses statistic D and p-value "
      "(scipy.stats.ks_2samp). Status:")
    bullets(doc, [
        "none — if p >= 0.01, or D < 0.10",
        "moderate — if p < 0.01 and 0.10 <= D < 0.20",
        "significant — if p < 0.01 and D >= 0.20",
        "insufficient_data — fewer than 2 lengths in either window",
    ])
    p(doc, "Default thresholds (DriftThresholds): psi_moderate=0.10, psi_significant=0.25, "
           "volume_moderate=0.20, volume_significant=0.50, ks_moderate=0.10, ks_significant=0.20, ks_alpha=0.01.")

    h2(doc, "4.5 Sentiment scoring formulas (ml/sentiment/model.py, ml/evaluation/metrics.py)")
    p(doc, "Inference (product path): 3-class RoBERTa logits → softmax probabilities over {negative, neutral, positive}.")
    code(doc,
         "p_c = softmax(logits)_c\n"
         "label = argmax_c p_c\n"
         "confidence = max_c p_c")
    p(doc, "Held-out / labelled evaluation (Model Validation uses Amazon TEST):")
    code(doc,
         "precision_c = TP_c / (TP_c + FP_c)\n"
         "recall_c    = TP_c / (TP_c + FN_c)\n"
         "F1_c        = 2 * precision_c * recall_c / (precision_c + recall_c)\n"
         "accuracy    = (# correct labels) / N\n"
         "macro_recall = (recall_neg + recall_neu + recall_pos) / 3\n"
         "macro_f1     = (F1_neg + F1_neu + F1_pos) / 3")
    p(doc, "Amazon fine-tune class weights (inverse frequency, then optional boosts):")
    code(doc,
         "w_c = (N / (3 * count_c)) ** weight_power\n"
         "w_neu *= neu_boost;  w_neg *= neg_boost\n"
         "w = w / mean(w)     # keep mean ~1 for stable LR")
    p(doc, "Selected production train uses weight_power=1, neu_boost=1, neg_boost=1 (plain inverse-frequency CE).")

    h2(doc, "4.6 Theme formulas (ml/themes/discovery.py)")
    bullets(doc, [
        "Embeddings: MiniLM 384-d L2-normalised vectors.",
        "Clustering: PCA → HDBSCAN micro-clusters → average-linkage merge of centroids when cosine similarity ≥ merge_threshold.",
        "Assignment: cluster member, or nearest_centroid if cosine similarity high enough, else unassigned.",
    ])
    code(doc,
         "centroid_k = normalize(mean(embeddings of members_k))\n"
         "similarity(review, theme) = cosine(embedding, centroid) = embedding · centroid\n"
         "coherence(theme) = mean(cosine(member_i, centroid)) over members")
    p(doc, "Theme negative_pct on the dashboard = 100 * negative_count / size.")

    h2(doc, "4.7 Evaluation bundle formulas (batch_eval / global Model Validation)")
    p(doc, "Product accuracy page uses global Amazon holdout metrics (artifacts/reports/amazon_sentiment_eval.json). "
           "Optional per-batch planted/star scores may still be computed offline but are not the product accuracy UI.")
    code(doc,
         "theme_recall   = (# planted themes recovered) / (# planted themes)\n"
         "radar_recall   = (# expected radar patterns matched) / (# expected patterns)\n"
         "pii_recall     = (# planted-PII rows with ≥1 redaction) / (# planted-PII rows)\n"
         "mean_available = mean of whichever of {sentiment_macro_recall, theme_recall, radar_recall, pii_recall} exist")

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
