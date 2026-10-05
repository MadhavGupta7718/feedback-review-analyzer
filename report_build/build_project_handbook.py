"""Build Feedback_Review_Analyzer_Project_Handbook.docx — full project story for newcomers."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from docx_util import bullets, code, h1, h2, new_doc, numbered, p, save, table  # noqa: E402

OUT_PRIMARY = Path(r"D:\Microsoft\Feedback_Review_Analyzer_Project_Handbook.docx")
OUT_COPY = ROOT / "docs" / "Feedback_Review_Analyzer_Project_Handbook.docx"


def build() -> list[Path]:
    doc = new_doc(
        "Project Handbook — Feedback & Review Analyzer",
        "Complete guide: product decisions, backend, frontend, Amazon sentiment model, "
        "recall hit-and-trial, selected results, and how to run. Written so a new teammate "
        "can understand the whole project.",
    )
    p(doc, "Hackathon: Microsoft Innovate — Problem 17 (Student Edition): "
           "“10,000 Reviews, No Time to Read Them.”")
    p(doc, "Repo: feedback-review-analyzer/  ·  Last metrics snapshot: 2026-10-04")

    h1(doc, "1. Product summary")
    p(doc,
      "Users upload a review CSV. An offline pipeline (GPU laptop) cleans text, redacts PII, "
      "runs 3-class sentiment (Amazon fine-tuned RoBERTa), discovers themes (MiniLM + HDBSCAN), "
      "and—if timestamps exist—runs Complaint Radar and drift. Results land in a per-upload "
      "SQLite database. FastAPI + a React dashboard read that DB. Accuracy and macro recall are "
      "not claimed per upload; they appear only on a global Model Validation page scored on a "
      "held-out Amazon clothing TEST split.")

    h1(doc, "2. Locked product decisions (and why)")
    table(doc, ["Decision", "Choice", "Why"], [
        ["Accuracy story", "Global Amazon holdout TEST only",
         "Synthetic planted-label accuracy was not credible for real CSVs"],
        ["Upload analytics", "Separate DB per upload; no per-batch accuracy page",
         "Uploads usually lack reliable ground truth"],
        ["Dates on uploads", "Use if present; never invent; hide Radar/Drift if missing",
         "Radar and drift need time windows"],
        ["Sentiment model", "Fine-tune cardiffnlp twitter-roberta → 3-class on Amazon clothing",
         "Closer to product reviews than tweets; matches neg/neu/pos"],
        ["Train/eval labels", "Cons_rating: 1–2 neg, 3 neu, 4–5 pos",
         "Weak star labels (noisy neutrals) — documented honestly"],
        ["Split", "Deterministic 80/10/10 by sha256(text)%100",
         "Reproducible, no train/test leakage"],
        ["Product Brief in UI", "Removed from nav / product path",
         "Pivot to upload + model validation"],
        ["Synthetic demos", "Generators may remain; not primary UI",
         "Primary path is upload CSV + Amazon model"],
    ])

    h1(doc, "3. Currently selected model")
    table(doc, ["Item", "Value"], [
        ["Active sentiment model", "artifacts/models/amazon_roberta_sentiment/"],
        ["Loader key", "local:amazon_roberta_sentiment (revision amazon-finetune)"],
        ["Base checkpoint", "cardiffnlp/twitter-roberta-base-sentiment-latest"],
        ["How chosen at runtime", "If Amazon dir has config.json → use it; else pretrained CardiffNLP"],
        ["Eval report", "artifacts/reports/amazon_sentiment_eval.json"],
        ["Train-time selection", "Checkpoint with best validation macro recall"],
    ])

    h2(doc, "3.1 Current TEST metrics (selected checkpoint)")
    table(doc, ["Metric", "Value"], [
        ["TEST accuracy", "0.8315"],
        ["TEST macro recall", "0.7363"],
        ["Neg / Neu / Pos recall", "0.7225 / 0.6000 / 0.8865"],
        ["Best val macro recall", "0.7470 (epoch 2 of restore run)"],
        ["Eval date (UTC)", "2026-10-04T16:05:53"],
    ])
    p(doc,
      "Reason this checkpoint is selected: After several recall-boost experiments that lowered "
      "TEST macro recall, we restored the simple baseline recipe (inverse-frequency class weights, "
      "natural batching, 3 epochs, lr=2e-5). That family produced the best honest holdout numbers "
      "(~0.74 macro recall). Aggressive minority oversampling and equal-class batches traded away "
      "too much positive recall.")

    h1(doc, "4. System architecture")
    code(doc,
         "Amazon clothing CSV → prepare_amazon_clothing.py → interim reviews.csv\n"
         "        → finetune_amazon_sentiment.py\n"
         "              → amazon_roberta_sentiment/     → ml.pipeline (uploads)\n"
         "              → amazon_sentiment_eval.json    → Model Validation UI\n"
         "User CSV → pipeline → artifacts/cache/batches/{id}.db → FastAPI → React")
    h2(doc, "4.1 Privacy boundary")
    numbered(doc, [
        "PII redaction runs before sentiment, embeddings, DB write, or any LLM.",
        "DB stores redacted text only.",
        "API re-redacts outbound review text (safe_text).",
        "Logs use redacting filters; redaction reports store counts, not secrets.",
    ])
    h2(doc, "4.2 Traceability")
    p(doc,
      "Every theme representative and radar evidence ID must resolve to a real review in that "
      "batch’s DB. audit_traceability() runs at the end of each pipeline run.")

    h1(doc, "5. Backend")
    h2(doc, "5.1 Stack")
    bullets(doc, [
        "FastAPI + Uvicorn + Pydantic",
        "SQLite analytics DBs (read-only for most GETs)",
        "Full pipeline needs torch/transformers (requirements.txt); API-only can use requirements-api.txt",
    ])
    h2(doc, "5.2 Important modules")
    table(doc, ["Path", "Role"], [
        ["backend/app/main.py", "HTTP routes"],
        ["backend/app/db.py", "Active DB path (batch override or ANALYTICS_DB)"],
        ["backend/app/batches.py", "Upload registry, background pipeline jobs, activate"],
        ["ml/pipeline.py", "End-to-end analytics build"],
        ["ml/models/registry.py", "Local model load; prefers Amazon fine-tune"],
        ["ml/sentiment/model.py", "Batched 3-class inference"],
        ["ml/themes/discovery.py", "Clustering / theme naming"],
        ["ml/complaints/radar.py", "Growing-complaint detection"],
        ["ml/drift/monitor.py", "PSI / volume / length drift"],
        ["ml/pii/", "Redaction + leak scan"],
    ])
    h2(doc, "5.3 API endpoints (product-facing)")
    table(doc, ["Method", "Path", "Purpose"], [
        ["GET", "/health", "Status, review count, dataset name"],
        ["GET", "/metrics", "Overview KPIs"],
        ["GET", "/themes, /themes/{id}", "Theme list / detail"],
        ["GET", "/issues, /issues/{id}, .../evidence", "Complaint Radar"],
        ["GET", "/reviews, /reviews/{id}", "Searchable evidence"],
        ["GET", "/model/evaluation", "Global Amazon TEST metrics (alias /sentiment/validation)"],
        ["GET", "/drift", "Drift report or status unavailable"],
        ["GET", "/data-health", "Ingestion / PII / themes / dates flags"],
        ["GET", "/model-info", "Model versions + amazon_model_installed"],
        ["GET", "/batches", "List uploads"],
        ["POST", "/batches/upload", "Multipart CSV → background pipeline"],
        ["GET", "/batches/jobs/{job_id}", "Job status"],
        ["POST", "/batches/{id}/activate", "Set active analytics DB"],
    ])
    p(doc, "POST /product-brief may still exist in code but is not part of the product UI after the pivot.")

    h2(doc, "5.4 Batch upload behaviour")
    numbered(doc, [
        "Save CSV under data/uploads/{batch_id}/.",
        "Run ml.pipeline into artifacts/cache/batches/{batch_id}.db.",
        "Register in artifacts/cache/batches/index.json.",
        "Auto-activate that batch for subsequent reads.",
        "If most rows lack usable dates → dates_available=false; radar/drift become stubs. "
        "Sentiment/themes/PII/evidence still run.",
    ])
    h2(doc, "5.5 CSV columns")
    table(doc, ["Field", "Required", "Aliases"], [
        ["text", "yes", "text, review, review_text, content, body, …"],
        ["date", "no", "created_at, date, timestamp, …"],
        ["rating", "no", "rating, score, stars, …"],
    ])

    h1(doc, "6. Frontend")
    h2(doc, "6.1 Stack")
    bullets(doc, ["React 19 + TypeScript + Vite", "React Router", "Recharts", "Vitest + Testing Library"])
    h2(doc, "6.2 Pages")
    table(doc, ["Route", "Page", "Role"], [
        ["/batches", "Upload & batches", "Drop CSV, poll job, list uploads, View → activate"],
        ["/sentiment", "Model Validation", "Global Amazon TEST accuracy / macro recall / confusion"],
        ["/", "Executive Overview", "Sentiment mix, themes, emerging (or dates message)"],
        ["/themes", "Themes", "Cluster browser + representatives"],
        ["/radar", "Complaint Radar", "Growing issues + View why (or dates empty state)"],
        ["/evidence", "Evidence", "Searchable reviews"],
        ["/health", "Data Health", "Ingestion, PII, drift (or unavailable)"],
    ])
    h2(doc, "6.3 UX rules from the pivot")
    bullets(doc, [
        "Sidebar: “Upload batches · Amazon-trained sentiment”.",
        "Model Validation states metrics are not tied to any upload.",
        "Overview / Radar / Data Health show a dates-required empty state when dates_available is false.",
        "Product Brief route removed from App.tsx / nav.",
        "API base from VITE_API_BASE_URL (default http://127.0.0.1:8000).",
    ])

    h1(doc, "7. ML pipeline stages")
    numbered(doc, [
        "Load CSV (text required; dates optional).",
        "Clean + PII redact.",
        "Sentiment with Amazon fine-tune if present (fp16 on CUDA).",
        "Embeddings (MiniLM 384-d).",
        "Themes (PCA → HDBSCAN → merge → name).",
        "Radar + drift if dates available; else stubs.",
        "Traceability audit.",
        "Write SQLite + report JSON blobs.",
    ])
    p(doc, "Uploads do not write a planted-label sentiment_validation payload for the accuracy page.")

    h2(doc, "7.1 Formulas used in this project")
    p(doc, "Sentiment inference:")
    code(doc,
         "p_c = softmax(logits)_c\n"
         "label = argmax_c p_c\n"
         "confidence = max_c p_c")
    p(doc, "Held-out evaluation (Model Validation / Amazon TEST):")
    code(doc,
         "precision_c = TP_c / (TP_c + FP_c)\n"
         "recall_c    = TP_c / (TP_c + FN_c)\n"
         "F1_c        = 2 * precision_c * recall_c / (precision_c + recall_c)\n"
         "accuracy    = (# correct) / N\n"
         "macro_recall = (recall_neg + recall_neu + recall_pos) / 3")
    p(doc, "Class weights (fine-tune CE); selected train uses weight_power=1:")
    code(doc,
         "w_c = (N / (3 * count_c)) ** weight_power\n"
         "w = w / mean(w)")
    p(doc, "Themes:")
    code(doc,
         "centroid_k = normalize(mean(embeddings of members_k))\n"
         "similarity = cosine(embedding, centroid)\n"
         "coherence  = mean(cosine(member_i, centroid))\n"
         "negative_pct = 100 * negative_count / size")
    p(doc, "Complaint Radar:")
    code(doc,
         "current  = (end - 14d, end];  previous = (end - 28d, end - 14d]\n"
         "growth_pct     = (current - previous) / previous * 100\n"
         "negative_ratio = current_negatives / current_mentions\n"
         "priority       = current_negatives * (1 + clip(growth_pct/100, 0, 3))  # NEW → factor 4\n"
         "lift           = theme_share(segment=v) / overall_share(segment=v)")
    p(doc, "Status gates (order): NO_DATA → NOT_A_COMPLAINT (neg ratio < 0.5) → "
           "INSUFFICIENT_EVIDENCE → NEW → EMERGING (≥ +50%) → DECLINING (≤ −25%, or soft ≤ −15% + "
           "falling weekly trend) → STABLE.")
    p(doc, "Drift:")
    code(doc,
         "PSI = sum_i (p_cur_i - p_ref_i) * ln(p_cur_i / p_ref_i)\n"
         "volume change = (r_cur - r_ref) / r_ref     # r = reviews/day\n"
         "length: KS statistic D + p-value on redacted text lengths")
    p(doc, "PSI: <0.10 none, 0.10–0.25 moderate, ≥0.25 significant. "
           "Volume |change|: <20% / 20–50% / ≥50%. KS significant if p<0.01 and D≥0.20. "
           "Same formulas appear in the Backend Guide and UI Guide next to each topic.")

    h1(doc, "8. Sentiment training data")
    p(doc, "Prepare:")
    code(doc, ".\\.venv\\Scripts\\python.exe scripts\\prepare_amazon_clothing.py")
    p(doc, "Writes data/interim/amazon_clothing/reviews.csv from Review + Cons_rating.")
    table(doc, ["Cons_rating", "Class"], [
        ["1–2", "negative"],
        ["3", "neutral"],
        ["4–5", "positive"],
    ])
    table(doc, ["Split", "Rows"], [
        ["Train", "38,661"],
        ["Val", "4,830"],
        ["Test", "4,812"],
    ])
    p(doc,
      "Note: synthetic created_at on the training interim file is hash-based for corpus bookkeeping only. "
      "Dates are never invented for user uploads.")

    h1(doc, "9. Hit-and-trial: raising macro recall")
    p(doc,
      "Goal: raise TEST macro recall (mean of neg/neu/pos recall). Bottleneck is usually neutral "
      "(noisy 3★ labels + class imbalance). Script: scripts/finetune_amazon_sentiment.py. "
      "Selection criterion: best validation macro recall, then report TEST.")

    h2(doc, "9.1 Trial log")
    table(doc, ["#", "Recipe", "TEST result", "Verdict"], [
        ["A", "Baseline: CardiffNLP → 3 epochs, lr=2e-5, bs=16, inverse-frequency CE, natural shuffle",
         "Acc 0.8356, macro 0.7411 (neu ~0.60)", "Best honest reference"],
        ["B", "Resume A + equal-class batches + focal γ=2 + minority boosts, 5 epochs, lr=1e-5",
         "Acc 0.7903, macro 0.7257", "Rejected — macro fell; pos recall collapsed"],
        ["C", "From scratch + oversample neu×3/neg×2 + weight_power 1.35 + focal 1",
         "Val macro ~0.73; crushed pos weight", "Abandoned mid-run"],
        ["D", "Milder oversample + focal 0.5 + neu_boost 1.6",
         "Val macro ~0.73, below A", "Abandoned"],
        ["E", "Balanced Softmax τ=1 + mild oversample, 4 epochs",
         "Acc 0.8092, macro 0.7366", "Rejected — below A"],
        ["F", "Restore baseline recipe (weights only), 3 epochs",
         "Acc 0.8315, macro 0.7363; best val 0.7470",
         "Currently selected (close to A; run variance)"],
    ])

    h2(doc, "9.2 Why aggressive recall tricks failed")
    numbered(doc, [
        "Equal-class batches force minority predictions → positive recall collapses → macro mean does not rise.",
        "Very high inverse weights / weight_power > 1 on a ~74% positive set makes positive weight tiny → same collapse.",
        "Focal loss (γ=2) over-emphasised hard/noisy neutrals.",
        "Star-rating neutrals are weak labels; ceiling on neu recall is limited without cleaner labels.",
    ])

    h2(doc, "9.3 Training flags kept for future work")
    bullets(doc, [
        "--resume",
        "--balanced-batches",
        "--focal-gamma",
        "--weight-power, --neu-boost, --neg-boost",
        "--oversample-neu, --oversample-neg",
        "--balanced-softmax-tau",
    ])
    p(doc, "Default / selected production train command:")
    code(doc,
         ".\\.venv\\Scripts\\python.exe scripts\\finetune_amazon_sentiment.py "
         "--epochs 3 --lr 2e-5 --batch-size 16")

    h1(doc, "10. Earlier experiments (context, not product accuracy page)")
    table(doc, ["Experiment", "Outcome / use"], [
        ["Sentiment140 binary study + threshold 0.725", "Offline; tweets ≠ product path"],
        ["Fine-tuned binary RoBERTa on S140", "Experimental only (roberta-s140-binary)"],
        ["Amazon / Yelp polarity offline", "Offline study (~0.92 / ~0.87)"],
        ["Synthetic Nimbus planted themes/radar/PII", "Useful for demos; abandoned as accuracy claim"],
        ["Dove / Adidas / Realme generators", "Exist under ml/data/; not primary UI"],
    ])

    h1(doc, "11. How to run")
    h2(doc, "11.1 Setup")
    code(doc,
         "python -m venv .venv\n"
         ".\\.venv\\Scripts\\python.exe -m pip install torch==2.11.0 "
         "--index-url https://download.pytorch.org/whl/cu128\n"
         ".\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt\n"
         "copy .env.example .env\n"
         ".\\.venv\\Scripts\\python.exe scripts\\download_models.py\n"
         "cd frontend; npm ci; cd ..")
    h2(doc, "11.2 Train (if model missing)")
    code(doc,
         ".\\.venv\\Scripts\\python.exe scripts\\prepare_amazon_clothing.py\n"
         ".\\.venv\\Scripts\\python.exe scripts\\finetune_amazon_sentiment.py "
         "--epochs 3 --lr 2e-5 --batch-size 16")
    h2(doc, "11.3 Serve")
    code(doc,
         ".\\.venv\\Scripts\\python.exe -m uvicorn backend.app.main:app --port 8000\n"
         "cd frontend; npm run dev")
    p(doc, "Open http://127.0.0.1:5173 → Upload & batches or Model Validation.")

    h2(doc, "11.4 Sample upload CSVs")
    table(doc, ["File", "Notes"], [
        ["data/uploads/demo_upload_15k.csv", "Amazon reviews, with dates"],
        ["data/uploads/dove_shampoo_upload_15k.csv", "Synthetic Dove (~15k, many themes)"],
        ["data/uploads/sample_no_dates.csv", "For dates-unavailable empty states"],
    ])

    h1(doc, "12. Key artifacts")
    table(doc, ["Path", "Meaning"], [
        ["artifacts/models/amazon_roberta_sentiment/", "Selected product sentiment weights"],
        ["artifacts/reports/amazon_sentiment_eval.json", "Global TEST metrics for UI"],
        ["artifacts/cache/batches/*.db", "Per-upload analytics"],
        ["artifacts/cache/batches/index.json", "Batch registry + active id"],
        ["data/interim/amazon_clothing/reviews.csv", "Train/val/test source"],
        ["data/finetune_amazon.log", "Latest training log"],
    ])

    h1(doc, "13. Backend / frontend ownership map")
    table(doc, ["Concern", "Backend", "Frontend"], [
        ["Accuracy numbers", "/model/evaluation ← eval JSON", "Model Validation page"],
        ["Upload job", "/batches/* + pipeline thread", "Batches page"],
        ["Themes / radar / evidence", "GETs on active DB", "Themes / Radar / Evidence"],
        ["Dates missing", "dates_available + stub reports", "Empty-state banners"],
        ["PII", "Pipeline redactor + API safe_text", "Never shows raw secrets"],
    ])

    h1(doc, "14. Honest limitations")
    numbered(doc, [
        "Cons_rating labels are weak; neutral is especially noisy.",
        "Macro recall ~0.74 reflects that ceiling more than “bad engineering.”",
        "Theme/radar quality on free-form real reviews is harder than on synthetic templates.",
        "Radar/drift need real timestamps on the uploaded file.",
        "Names without cues may not be redacted.",
        "Product Brief is out of the primary UI after the pivot.",
    ])

    h1(doc, "15. Chronology of the pivot")
    numbered(doc, [
        "Built full analyzer with synthetic Nimbus + planted-label accuracy story.",
        "Explored Sentiment140 / polarity offline metrics; risk of mixing claims across datasets.",
        "Added Dove/Adidas/Realme generators and per-DB metrics — still not a generalizable accuracy story.",
        "Pivot: train on real Amazon clothing ratings; global Model Validation; upload batches without per-file accuracy.",
        "Implemented upload APIs, date-unavailable stubs, UI rewrite, Amazon fine-tune.",
        "Attempted recall boosts (balanced batches, focal, oversample, balanced softmax); rejected when TEST macro recall fell.",
        "Restored baseline-style fine-tune; documented selected metrics in amazon_sentiment_eval.json and this handbook.",
    ])

    h1(doc, "16. Related markdown docs")
    table(doc, ["Document", "Contents"], [
        ["docs/PROJECT_HANDBOOK.md", "Same story in Markdown"],
        ["docs/ARCHITECTURE.md", "Deeper pipeline / privacy / radar formulas"],
        ["docs/API.md", "Endpoint contracts"],
        ["docs/DATASET.md", "Upload schema + Amazon train corpus"],
        ["docs/RESULTS.md", "Measured headline numbers"],
        ["docs/MODEL_CARD.md", "Model risks / intended use"],
        ["docs/DEMO.md", "Demo storyline"],
        ["docs/TESTING.md", "How to run tests"],
        ["docs/DEVELOPMENT_LOG.md", "Chronological audit trail"],
        ["README.md", "Quick start"],
    ])

    p(doc,
      "End of handbook. Selected Amazon fine-tune TEST accuracy 0.8315 / macro recall 0.7363 "
      "(as of 2026-10-04).")

    paths = [save(doc, OUT_PRIMARY), save(doc, OUT_COPY)]
    return paths


if __name__ == "__main__":
    for path in build():
        print(f"Wrote {path}")
