"""Build Feedback_Review_Analyzer_Final_Master_Guide.docx — single comprehensive project document."""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from docx.enum.text import WD_BREAK  # noqa: E402
from docx_util import bullets, code, h1, h2, h3, new_doc, numbered, p, save, table  # noqa: E402

OUT_PRIMARY = Path(r"D:\Microsoft\Feedback_Review_Analyzer_Final_Master_Guide.docx")
OUT_COPY = ROOT / "docs" / "Feedback_Review_Analyzer_Final_Master_Guide.docx"
OUT_FALLBACK = Path(r"D:\Microsoft\Feedback_Review_Analyzer_Final_Master_Guide_UPDATED.docx")
EVAL_JSON = ROOT / "artifacts" / "reports" / "amazon_sentiment_eval.json"
SWEEP_DIR = ROOT / "artifacts" / "reports" / "hyperparam_sweep"


def page_break(doc) -> None:
    para = doc.add_paragraph()
    para.add_run().add_break(WD_BREAK.PAGE)


def load_eval() -> dict:
    if not EVAL_JSON.exists():
        return {}
    return json.loads(EVAL_JSON.read_text(encoding="utf-8"))


def load_sweep_row(name: str) -> dict | None:
    path = SWEEP_DIR / name
    if not path.exists():
        return None
    r = json.loads(path.read_text(encoding="utf-8"))
    hp = r.get("hyperparams") or {}
    m = r.get("metrics") or {}
    test = m.get("test") or {}
    per = test.get("per_class") or m.get("per_class") or {}
    return {
        "lr": hp.get("lr"),
        "batch_size": hp.get("batch_size"),
        "max_length": hp.get("max_length"),
        "val_macro": r.get("best_val_macro_recall"),
        "acc": test.get("accuracy") or m.get("headline_accuracy"),
        "macro": test.get("macro_recall") or m.get("headline_macro_recall"),
        "neg": (per.get("negative") or {}).get("recall"),
        "neu": (per.get("neutral") or {}).get("recall"),
        "pos": (per.get("positive") or {}).get("recall"),
    }


def section_formulas(doc) -> None:
    h1(doc, "Part C — All formulas used in this project")
    p(doc,
      "Every metric shown in the dashboard or Model Validation traces back to one of the blocks below. "
      "Implementation paths are given so engineers can verify code.")

    h2(doc, "C.1 Text ingestion & cleaning (ml/preprocessing/clean.py)")
    numbered(doc, [
        "Validate — reject null_text, empty_text, no_content_after_cleaning (no word characters).",
        "Repair — ftfy mojibake fix, HTML entity decode, Unicode NFC, strip control characters.",
        "Redact — PII → typed placeholders ([EMAIL], [PHONE], …); raw text discarded after this step.",
        "Tidy — collapse punctuation runs (!!!… → max 3), normalize whitespace.",
    ])
    p(doc, "Ingestion duplicate removal (same text + created_at + platform + app_version):")
    code(doc, "dedupe_key = (text.strip().lower(), created_at, platform, app_version)")
    p(doc, "RoBERTa input normalisation (Twitter-trained checkpoint):")
    code(doc, "[USER] → @user ;  [URL] → http")

    h2(doc, "C.2 Sentiment inference (ml/sentiment/model.py)")
    code(doc,
         "p_c = softmax(logits)_c\n"
         "label = argmax_c p_c\n"
         "confidence = max_c p_c")
    p(doc, "Production loader prefers artifacts/models/amazon_roberta_sentiment/ when present.")

    h2(doc, "C.3 Sentiment evaluation (ml/evaluation/metrics.py, Model Validation page)")
    code(doc,
         "precision_c = TP_c / (TP_c + FP_c)\n"
         "recall_c    = TP_c / (TP_c + FN_c)\n"
         "F1_c        = 2 * precision_c * recall_c / (precision_c + recall_c)\n"
         "accuracy    = (# correct) / N\n"
         "macro_recall = (recall_neg + recall_neu + recall_pos) / 3\n"
         "macro_f1     = (F1_neg + F1_neu + F1_pos) / 3")
    p(doc, "Train/val/test split (Amazon corpus):")
    code(doc, "bucket = sha256(text.strip().lower()) % 100  →  <80 train, 80–89 val, ≥90 test")

    h2(doc, "C.4 Fine-tune class weights (scripts/finetune_amazon_sentiment.py)")
    code(doc,
         "w_c = (N / (3 * count_c)) ** weight_power\n"
         "w_neu *= neu_boost;  w_neg *= neg_boost\n"
         "w = w / mean(w)")

    h2(doc, "C.5 Teacher labeling (scripts/teacher_label_amazon.py)")
    p(doc,
      "Optional path: CardiffNLP RoBERTa labels review text only; stars kept as star_sentiment for comparison.")
    code(doc,
         "teacher_label = argmax_c softmax(cardiffnlp_logits)_c\n"
         "teacher_confidence = max_c p_c")

    h2(doc, "C.6 Themes (ml/themes/discovery.py)")
    bullets(doc, [
        "MiniLM 384-d embeddings, L2-normalised.",
        "PCA → HDBSCAN (min_cluster_size scales ~1% of batch, clamped 3–40).",
        "Merge micro-clusters when centroid cosine ≥ merge_threshold.",
        "Assign by cluster membership or nearest centroid if similarity ≥ noise_assign_threshold (0.60).",
    ])
    code(doc,
         "centroid_k = normalize(mean(embeddings of members_k))\n"
         "similarity(review, theme) = cosine(embedding, centroid)\n"
         "coherence(theme) = mean(cosine(member_i, centroid))\n"
         "negative_pct = 100 * negative_count / size")

    h2(doc, "C.7 Complaint Radar (ml/complaints/radar.py)")
    code(doc,
         "current  = (end - 14d, end]\n"
         "previous = (end - 28d, end - 14d]\n"
         "growth_pct     = (current - previous) / previous * 100   # None if previous==0\n"
         "negative_ratio = current_negatives / current_mentions\n"
         "acceleration_pp = ((c-b)/b - (b-a)/a) * 100   # last 3 weekly buckets")
    p(doc, "Weekly trend: linear regression slope / mean(weekly counts); rising if >0.05, falling if <-0.05.")
    code(doc,
         "if n_reviews >= 5000: min_mentions = 30\n"
         "else: min_mentions = clamp(round(n_reviews * 0.015), 8, 30)")
    p(doc, "Status order: NO_DATA → NOT_A_COMPLAINT (neg ratio < 0.5) → INSUFFICIENT_EVIDENCE → "
           "NEW → EMERGING (≥+50%) → DECLINING (≤−25% or soft ≤−15% + falling trend) → STABLE.")
    code(doc,
         "growth_factor = 1 + clip(growth_pct/100, 0, 3)   # NEW uses 1+3=4\n"
         "priority = current_negative_mentions * growth_factor")
    code(doc,
         "theme_share = count(segment=v in theme) / count(theme with segment)\n"
         "overall_share = count(segment=v in all current) / count(all with segment)\n"
         "lift = theme_share / overall_share   # report if support≥20 and lift≥1.3")

    h2(doc, "C.8 Drift (ml/drift/monitor.py) — Data Health UI")
    p(doc, "Compare reference vs current window (typically consecutive 14-day periods). "
           "overall_status = worst of four signals.")
    code(doc, "PSI = sum_i (p_cur_i - p_ref_i) * ln(p_cur_i / p_ref_i)   # shares ε-clipped, renormalized")
    table(doc, ["PSI", "Status"], [["< 0.10", "none"], ["0.10 – 0.25", "moderate"], [">= 0.25", "significant"]])
    code(doc,
         "r = reviews / days\n"
         "volume_change = (r_cur - r_ref) / r_ref")
    table(doc, ["|volume_change|", "Status"], [["< 20%", "none"], ["20% – 50%", "moderate"], [">= 50%", "significant"]])
    p(doc, "Review length: two-sample KS on len(text_redacted); D and p-value.")
    bullets(doc, [
        "none — p ≥ 0.01 or D < 0.10",
        "moderate — p < 0.01 and 0.10 ≤ D < 0.20",
        "significant — p < 0.01 and D ≥ 0.20",
    ])
    p(doc, "Weekly chart: sentiment PSI and theme PSI vs baseline week (same PSI family).")

    h2(doc, "C.9 Optional batch evaluation (offline / synthetic demos)")
    code(doc,
         "theme_recall = recovered_planted_themes / total_planted\n"
         "radar_recall = matched_expected_patterns / total_expected\n"
         "pii_recall   = planted_rows_redacted / total_planted_pii\n"
         "mean_available = mean of available recall scores")


def section_backend(doc) -> None:
    h1(doc, "Part D — Backend & pipeline (complete reference)")
    h2(doc, "D.1 Two layers")
    numbered(doc, [
        "Offline ml/pipeline.py — clean, PII, sentiment, embeddings, themes, radar, drift, SQLite write.",
        "Online backend/app/main.py — read-only FastAPI over the active analytics DB.",
    ])
    h2(doc, "D.2 Repository map")
    table(doc, ["Path", "Role"], [
        ["ml/pipeline.py", "Orchestrator: python -m ml.pipeline --source … --db …"],
        ["ml/config.py", "Paths, model IDs, AMAZON_SENTIMENT_DIR, HF cache"],
        ["ml/models/registry.py", "Local-only load; Amazon fine-tune first"],
        ["backend/app/batches.py", "Upload registry, background jobs, activate batch"],
        ["backend/app/db.py", "SQLite read helpers, active DB path"],
        ["scripts/teacher_label_amazon.py", "Text-only teacher labels for training CSV"],
        ["scripts/finetune_amazon_sentiment.py", "Fine-tune + amazon_sentiment_eval.json"],
        ["scripts/download_models.py", "One-time HF cache for CardiffNLP + MiniLM + Qwen"],
    ])
    h2(doc, "D.3 Pipeline command")
    code(doc,
         ".\\.venv\\Scripts\\python.exe -m ml.pipeline "
         "--source path\\to\\reviews.csv --db artifacts\\cache\\batches\\my.db --device cuda")
    h2(doc, "D.4 CSV contract")
    table(doc, ["Field", "Required?", "Aliases"], [
        ["text", "yes", "text, review, review_text, content, body, comment, feedback"],
        ["created_at", "no*", "created_at, date, timestamp, review_date, … (*required for radar/drift)"],
        ["rating", "no", "rating, score, stars, star_rating"],
        ["review_id", "no", "review_id, id (else U000001…)"],
        ["platform / app_version", "no", "For radar segment lift"],
    ])
    p(doc, "Ground-truth columns (gt_sentiment, gt_theme, gt_pii) used only during pipeline scoring; stripped before DB serve.")
    h2(doc, "D.5 SQLite schema (conceptual)")
    table(doc, ["Table", "Contents"], [
        ["meta", "source, dataset, dates_available, params, schema_version"],
        ["reviews", "review_id, text_redacted, sentiment, probs, theme_id, similarity, …"],
        ["themes", "theme_id + JSON (name, keywords, radar fields, representatives)"],
        ["issues", "Complaint Radar rows per theme"],
        ["reports", "overview, data_health, drift, traceability, performance JSON"],
        ["model_versions", "Sentiment + embedding revisions used for this run"],
    ])
    h2(doc, "D.6 FastAPI endpoints")
    table(doc, ["Method", "Path", "Purpose"], [
        ["GET", "/health", "Liveness, review count, dataset name"],
        ["GET", "/metrics", "Overview KPIs"],
        ["GET", "/themes, /themes/{id}", "Theme list and detail"],
        ["GET", "/issues, /issues/{id}, …/evidence", "Complaint Radar"],
        ["GET", "/reviews, /reviews/{id}", "Evidence search"],
        ["GET", "/sentiment/validation", "Alias: global Model Validation metrics"],
        ["GET", "/model/evaluation", "Same as above — Amazon TEST holdout"],
        ["GET", "/drift", "Drift report or unavailable stub"],
        ["GET", "/data-health", "Ingestion, PII, traceability, dates flag"],
        ["GET", "/model-info", "Installed models + hardware"],
        ["GET", "/batches", "List uploads + active_db path"],
        ["POST", "/batches/upload", "Multipart CSV → background pipeline"],
        ["GET", "/batches/jobs/{job_id}", "Upload job status"],
        ["POST", "/batches/{batch_id}/activate", "Switch active analytics DB"],
    ])
    h2(doc, "D.7 Batch upload flow")
    numbered(doc, [
        "Save CSV under data/uploads/{batch_id}/.",
        "Run pipeline → artifacts/cache/batches/{batch_id}.db.",
        "Register in artifacts/cache/batches/index.json; auto-activate.",
        "No per-upload accuracy in UI — only global Model Validation.",
        "Missing dates → dates_available=false; radar/drift empty states in UI.",
    ])
    h2(doc, "D.8 Privacy")
    bullets(doc, [
        "PII redaction before any model or DB write.",
        "DB stores text_redacted only; API safe_text() on output.",
        "Logs use redacting filters; reports store PII counts not secrets.",
    ])
    h2(doc, "D.9 Environment variables")
    table(doc, ["Variable", "Purpose"], [
        ["ANALYTICS_DB", "Default SQLite path for API"],
        ["VITE_API_BASE_URL", "Frontend → API origin"],
        ["CORS_ORIGINS", "Allowed browser origins"],
        ["HF_HOME / HF_HUB_CACHE", "Local model cache (no runtime download)"],
    ])


def section_frontend(doc) -> None:
    h1(doc, "Part E — Frontend & dashboard (complete reference)")
    h2(doc, "E.1 Stack & entry")
    bullets(doc, [
        "React 19 + TypeScript + Vite 8 + React Router 7 + Recharts.",
        "frontend/src/main.tsx → App.tsx routes inside Layout.tsx.",
        "API: frontend/src/api/client.ts, useApi.ts, types.ts.",
        "All review text from API is already redacted.",
    ])
    h2(doc, "E.2 Navigation (current product)")
    table(doc, ["Route", "Nav label", "Component"], [
        ["/batches", "Upload & batches", "pages/Batches.tsx"],
        ["/sentiment", "Model Validation", "pages/SentimentValidation.tsx"],
        ["/", "Overview", "pages/Overview.tsx"],
        ["/themes", "Themes", "pages/Themes.tsx"],
        ["/radar", "Complaint Radar", "pages/Radar.tsx"],
        ["/evidence", "Evidence", "pages/Evidence.tsx"],
        ["/health", "Data Health", "pages/DataHealth.tsx"],
    ])
    p(doc, "Product Brief (/brief) removed from nav and routes after pivot.")
    h2(doc, "E.3 How to run UI + API")
    code(doc,
         "# Terminal A\n"
         "cd D:\\Microsoft\\feedback-review-analyzer\n"
         ".\\.venv\\Scripts\\python.exe -m uvicorn backend.app.main:app --port 8000\n\n"
         "# Terminal B\n"
         "cd frontend && npm run dev\n\n"
         "# Browser: http://127.0.0.1:5173")
    p(doc, "No in-UI dataset picker — restart API with different ANALYTICS_DB or activate a batch from Upload page.")
    h2(doc, "E.4 Page-by-page")
    h3(doc, "Upload & batches (/batches)")
    bullets(doc, [
        "POST /batches/upload — drag/drop CSV; poll GET /batches/jobs/{id}.",
        "List batches; View activates POST /batches/{id}/activate.",
        "Sidebar note: Amazon-trained sentiment for all uploads.",
    ])
    h3(doc, "Model Validation (/sentiment)")
    bullets(doc, [
        "GET /model/evaluation — global Amazon TEST only.",
        "KPIs: accuracy, macro recall, per-class recall, confusion matrix.",
        "Banner: teacher or star label methodology from eval JSON ground_truth_note.",
        "Does not show accuracy for the currently open upload batch.",
    ])
    h3(doc, "Overview (/)")
    bullets(doc, [
        "GET /metrics + /themes — KPIs, sentiment pie, top complaints, emerging list, drift badge.",
        "Dates missing → banner linking to need for timestamps on radar/drift.",
    ])
    h3(doc, "Themes (/themes)")
    bullets(doc, [
        "Sort/filter; coherence, negative_pct, radar status; ?theme= deep link.",
        "Formulas: see Part C.6.",
    ])
    h3(doc, "Complaint Radar (/radar)")
    bullets(doc, [
        "Status chips; View why shows calculation strings, thresholds, weekly chart, lift associations.",
        "Formulas: Part C.7.",
    ])
    h3(doc, "Evidence (/evidence)")
    bullets(doc, [
        "Paginated reviews; drawer with softmax probs and theme similarity.",
    ])
    h3(doc, "Data Health (/health)")
    bullets(doc, [
        "Ingestion rejects/duplicates; PII chart; traceability PASS/FAIL.",
        "Drift table + weekly PSI chart when dates_available (Part C.8).",
        "Empty state when dates unavailable.",
    ])
    h2(doc, "E.5 UI source map")
    table(doc, ["Path", "Role"], [
        ["src/components/Layout.tsx", "Sidebar, health pill, nav"],
        ["src/components/ui.tsx", "PageHeader, Card, Kpi, badges"],
        ["src/lib/format.ts", "fmtPct, fmtInt, sentiment colours"],
        ["src/pages/DataHealth.tsx", "Drift section + ingestion"],
        ["src/pages/RadarCards.tsx", "Issue cards + View why"],
        ["src/test/fixtures/*.json", "Vitest mock API payloads"],
    ])
    h2(doc, "E.6 Frontend tests")
    code(doc, "cd frontend && npm test")


def build() -> list[Path]:
    ev = load_eval()
    test = (ev.get("metrics") or {}).get("test") or {}
    acc = test.get("accuracy") or (ev.get("metrics") or {}).get("headline_accuracy")
    macro = test.get("macro_recall") or (ev.get("metrics") or {}).get("headline_macro_recall")
    per = test.get("per_class") or {}
    neg_r = per.get("negative", {}).get("recall", "—")
    neu_r = per.get("neutral", {}).get("recall", "—")
    pos_r = per.get("positive", {}).get("recall", "—")
    label_note = ev.get("label_mapping", "See amazon_sentiment_eval.json")
    eval_date = (ev.get("evaluation_date_utc") or "")[:10]
    hp = ev.get("hyperparams") or {}

    doc = new_doc(
        "Final Master Guide — Feedback & Review Analyzer",
        f"Complete reference: product story, architecture, every formula, backend API, frontend pages, "
        f"training, results, and runbook. Generated {date.today().isoformat()}.",
    )
    p(doc, "Microsoft Innovate — Problem 17: “10,000 Reviews, No Time to Read Them.”")
    p(doc, "Repository: github.com/MadhavGupta7718/feedback-review-analyzer")

    h1(doc, "Part A — Executive summary")
    p(doc,
      "Users upload a review CSV. An offline GPU pipeline cleans text, redacts PII, runs 3-class sentiment "
      "(Amazon fine-tuned RoBERTa), discovers themes (MiniLM + HDBSCAN), and—when timestamps exist—runs "
      "Complaint Radar and drift monitoring. Each upload becomes its own SQLite analytics database. "
      "FastAPI serves JSON; React dashboard visualises one active batch at a time.")
    p(doc,
      "Accuracy is not claimed per upload. Model Validation shows global metrics on a held-out Amazon "
      "clothing TEST split only.")
    table(doc, ["Decision", "Choice"], [
        ["Sentiment model", "Trial A fine-tune in artifacts/models/amazon_roberta_sentiment/"],
        ["Training labels (current)", label_note[:90] + ("…" if len(label_note) > 90 else "")],
        ["Selected hyperparameters",
         f"lr={hp.get('lr', '1e-5')}, batch_size={hp.get('batch_size', 16)}, "
         f"epochs={hp.get('epochs', 3)}, max_length={hp.get('max_length', 128)}"],
        ["Upload analytics", "Separate DB per batch; batch APIs + activate"],
        ["Dates", "Use if present; never invent; hide radar/drift when missing"],
        ["UI accuracy", "Global Model Validation only"],
    ])
    if acc is not None:
        table(doc, ["Current TEST metrics — Trial A (selected)", "Value"], [
            ["Evaluation date", eval_date or "—"],
            ["Accuracy", f"{acc:.4f}"],
            ["Macro recall", f"{macro:.4f}" if macro else "—"],
            ["Neg / Neu / Pos recall", f"{neg_r} / {neu_r} / {pos_r}"],
            ["Best val macro recall", str(ev.get("best_val_macro_recall", "—"))],
        ])
    p(doc,
      "Interpretation: teacher-labeled TEST scores measure agreement with the CardiffNLP teacher on a "
      "text-hash holdout—not independent human gold labels.")

    h1(doc, "Part B — System architecture")
    code(doc,
         "User CSV → POST /batches/upload → ml.pipeline → *.db\n"
         "                    ↓\n"
         "Amazon train path: prepare_amazon_clothing.py → teacher_label_amazon.py (optional)\n"
         "                 → finetune_amazon_sentiment.py → model + amazon_sentiment_eval.json\n"
         "                    ↓\n"
         "FastAPI (active DB) ← activate batch ← React dashboard")
    numbered(doc, [
        "One sentiment + embedding stack scores every batch (no per-upload retrain).",
        "Ground truth columns never exposed via API after pipeline write.",
        "Traceability audit: every radar/theme evidence ID must resolve to a real review.",
    ])

    section_formulas(doc)
    page_break(doc)
    section_backend(doc)
    page_break(doc)
    section_frontend(doc)

    h1(doc, "Part F — ML training & experiments")
    h2(doc, "F.1 Amazon clothing corpus + teacher labels")
    code(doc,
         ".\\.venv\\Scripts\\python.exe scripts\\prepare_amazon_clothing.py\n"
         ".\\.venv\\Scripts\\python.exe scripts\\teacher_label_amazon.py\n"
         "# Selected production train (Trial A):\n"
         ".\\.venv\\Scripts\\python.exe scripts\\finetune_amazon_sentiment.py "
         "--data data\\interim\\amazon_clothing\\reviews_teacher_roberta.csv "
         "--epochs 3 --lr 1e-5 --batch-size 16 --max-length 128")
    bullets(doc, [
        "Teacher: cardiffnlp/twitter-roberta-base-sentiment-latest labels text only (stars ignored).",
        "Teacher mix (~48,303): positive 34,620 · negative 8,612 · neutral 5,071; ~78% agree with star map.",
        "Split: 38,661 train / 4,830 val / 4,812 test (sha256 text-hash 80/10/10).",
    ])

    h2(doc, "F.2 Selected production checkpoint (Trial A)")
    table(doc, ["Item", "Value"], [
        ["Hyperparameters", "lr=1e-5, batch_size=16, epochs=3, max_length=128"],
        ["TEST accuracy", f"{acc:.4f}" if acc is not None else "0.9616"],
        ["TEST macro recall", f"{macro:.4f}" if macro else "0.9207"],
        ["Neg / Neu / Pos recall", f"{neg_r} / {neu_r} / {pos_r}"],
        ["Best val macro recall", str(ev.get("best_val_macro_recall", "0.9051"))],
        ["Model path", "artifacts/models/amazon_roberta_sentiment/"],
        ["Eval report", "artifacts/reports/amazon_sentiment_eval.json"],
    ])

    h2(doc, "F.3 Hyperparameter sweep (safe params, teacher labels)")
    p(doc,
      "Safe sweep over learning rate, batch size, and max_length. Aggressive recipes (balanced batches, "
      "focal loss, heavy oversampling) were not re-run here—they hurt star-label macro recall earlier. "
      "Trials D (bs=32) and E (max_length=256) were stopped before completion; product model set to Trial A.")
    sweep_rows = [
        ("Baseline (before sweep)", "00_baseline_before_sweep.json"),
        ("A — lr=1e-5 (SELECTED)", "A_lr1e-5.json"),
        ("B — lr=3e-5", "B_lr3e-5.json"),
        ("C — batch_size=8", "C_bs8.json"),
    ]
    table_rows: list[list[str]] = []
    for label, fname in sweep_rows:
        row = load_sweep_row(fname)
        if not row:
            continue
        table_rows.append([
            label,
            f"lr={row['lr']}, bs={row['batch_size']}, max={row['max_length']}",
            f"{row['acc']:.4f}" if row["acc"] is not None else "—",
            f"{row['macro']:.4f}" if row["macro"] is not None else "—",
            f"{row['neg']:.4f}" if row["neg"] is not None else "—",
            f"{row['neu']:.4f}" if row["neu"] is not None else "—",
            f"{row['pos']:.4f}" if row["pos"] is not None else "—",
            f"{row['val_macro']:.4f}" if row["val_macro"] is not None else "—",
        ])
    table_rows.append(["D — batch_size=32", "lr=2e-5, bs=32, max=128", "—", "—", "—", "—", "—", "Stopped"])
    table_rows.append(["E — max_length=256", "lr=2e-5, bs=16, max=256", "—", "—", "—", "—", "—", "Not run"])
    table(doc,
          ["Trial", "Settings", "TEST acc", "TEST macro", "Neg", "Neu", "Pos", "Val macro"],
          table_rows)
    p(doc, "Sweep artifacts: artifacts/reports/hyperparam_sweep/ and artifacts/models/hyperparam_sweep/.")

    h2(doc, "F.4 Star-label era (historical, before teacher pivot)")
    table(doc, ["Cons_rating", "Class"], [["1–2", "negative"], ["3", "neutral"], ["4–5", "positive"]])
    table(doc, ["Recipe", "TEST accuracy", "TEST macro recall", "Outcome"], [
        ["Baseline inverse-freq CE", "~0.8356", "~0.7411", "Best star-label reference"],
        ["Balanced batches + focal", "~0.7903", "~0.7257", "Rejected"],
        ["Balanced softmax", "~0.8092", "~0.7366", "Rejected"],
        ["Restore baseline", "0.8315", "0.7363", "Last star checkpoint before teacher path"],
    ])

    h2(doc, "F.5 Models in product path")
    table(doc, ["Model", "Role"], [
        ["amazon_roberta_sentiment (Trial A)", "3-class sentiment on uploads"],
        ["all-MiniLM-L6-v2", "Embeddings / themes"],
        ["cardiffnlp (base)", "Teacher labeling + fallback if fine-tune missing"],
    ])

    h1(doc, "Part G — Runbook & troubleshooting")
    h2(doc, "G.1 First-time setup")
    code(doc,
         "python -m venv .venv\n"
         ".\\.venv\\Scripts\\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
         ".\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt\n"
         ".\\.venv\\Scripts\\python.exe scripts\\download_models.py\n"
         "cd frontend && npm ci")
    h2(doc, "G.2 Regenerate this Word document")
    code(doc, ".\\.venv\\Scripts\\python.exe report_build\\build_final_master_guide.py")
    h2(doc, "G.3 Tests")
    code(doc,
         ".\\.venv\\Scripts\\python.exe -m pytest -q\n"
         "cd frontend && npm test")
    h2(doc, "G.4 Common issues")
    table(doc, ["Symptom", "Fix"], [
        ["0 reviews / degraded health", "Point ANALYTICS_DB at a built .db or activate a batch"],
        ["Radar/drift empty", "Upload CSV needs real created_at dates"],
        ["Model Validation 404", "Run finetune; ensure amazon_sentiment_eval.json exists"],
        ["CORS errors", "Add Vite origin to CORS_ORIGINS"],
        ["MODEL NOT INSTALLED", "Run download_models.py with internet once"],
    ])

    h1(doc, "Part H — Limitations & honesty")
    numbered(doc, [
        "Teacher-labeled metrics are not human-gold accuracy.",
        "Star-labeled metrics were limited by noisy neutrals.",
        "Radar statuses describe temporal change, not root cause.",
        "Theme quality varies on messy real text vs synthetic templates.",
        "PII redaction is regex-based; some names may remain.",
        "Model weights and large CSVs are gitignored — clone + train/download locally.",
    ])

    h1(doc, "Part I — Related files in repo")
    table(doc, ["Document / path", "Contents"], [
        ["README.md", "Quick start"],
        ["docs/PROJECT_HANDBOOK.md", "Markdown handbook"],
        ["docs/RESULTS.md", "Measured numbers"],
        ["docs/API.md", "Endpoint contracts"],
        ["docs/ARCHITECTURE.md", "Pipeline depth"],
        ["docs/Feedback_Review_Analyzer_*_Guide.docx", "Split backend/UI/handbook Word guides"],
        ["artifacts/reports/amazon_sentiment_eval.json", "Live eval for Model Validation UI (Trial A)"],
        ["artifacts/reports/hyperparam_sweep/", "Per-trial eval JSON from safe hyperparameter sweep"],
        ["scripts/run_hyperparam_sweep.py", "Safe lr / batch / max_length sweep runner"],
        ["scripts/teacher_label_amazon.py", "CardiffNLP text-only teacher labeling"],
    ])

    p(doc, "— End of Final Master Guide —")

    paths: list[Path] = []
    for target in (OUT_COPY, OUT_FALLBACK, OUT_PRIMARY):
        try:
            paths.append(save(doc, target))
            print(f"Wrote {target}")
        except OSError as exc:
            print(f"Skipped ({exc.__class__.__name__}): {target}")
    if not paths:
        raise SystemExit("Could not write any output DOCX — close Word and retry")
    return paths


if __name__ == "__main__":
    build()
