"""Build Feedback_Review_Analyzer_UI_Guide.docx — full frontend / dashboard reference."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from docx_util import bullets, code, h1, h2, h3, new_doc, numbered, p, save, table  # noqa: E402

OUT_PRIMARY = Path(r"D:\Microsoft\Feedback_Review_Analyzer_UI_Guide.docx")
OUT_COPY = ROOT / "docs" / "Feedback_Review_Analyzer_UI_Guide.docx"


def build() -> list[Path]:
    doc = new_doc(
        "Frontend & Dashboard Guide",
        "Complete reference for the React dashboard: every page, control, API call, "
        "how metrics stay scoped to the open analytics database, and how to run the UI.",
    )

    h1(doc, "1. What the frontend is")
    p(doc,
      "The UI is a React 19 + TypeScript single-page app (Vite) that talks only to the FastAPI backend. "
      "It never reads CSVs or SQLite directly. Whatever analytics DB the API was started with "
      "(ANALYTICS_DB or run_demo --db) is what every page displays — one batch at a time.")
    bullets(doc, [
        "Stack: React 19, React Router 7, Recharts, Vite 8, Vitest + Testing Library.",
        "Entry: frontend/src/main.tsx → BrowserRouter → App routes inside Layout.",
        "API base URL: import.meta.env.VITE_API_BASE_URL (default http://127.0.0.1:8000).",
        "All review text shown is already redacted by the API (placeholders like [EMAIL], [PHONE]).",
    ])

    h1(doc, "2. How to open the dashboard")
    h2(doc, "2.1 One-command demo (API + UI)")
    code(doc,
         "cd D:\\Microsoft\\feedback-review-analyzer\n"
         ".\\.venv\\Scripts\\python.exe scripts\\run_demo.py --serve\n"
         "# Nimbus default DB\n\n"
         ".\\.venv\\Scripts\\python.exe scripts\\run_demo.py --serve --db artifacts\\cache\\amazon_software.db\n"
         ".\\.venv\\Scripts\\python.exe scripts\\run_demo.py --serve --db artifacts\\cache\\dove_shampoo.db")
    p(doc, "Dashboard: http://127.0.0.1:5173 — API: http://127.0.0.1:8000 — Ctrl+C stops both.")

    h2(doc, "2.2 Manual (two terminals)")
    code(doc,
         "# Terminal A — API pointed at a DB\n"
         "set ANALYTICS_DB=artifacts\\cache\\dove_shampoo.db\n"
         ".\\.venv\\Scripts\\python.exe -m uvicorn backend.app.main:app --port 8000\n\n"
         "# Terminal B — Vite\n"
         "cd frontend\n"
         "npm ci\n"
         "npm run dev")

    h2(doc, "2.3 Switching datasets")
    p(doc,
      "There is no in-UI dataset picker. Stop the servers, restart with a different --db / ANALYTICS_DB, "
      "and refresh the browser. That is intentional: metrics never mix across batches.")

    h1(doc, "3. Application shell")
    h2(doc, "3.1 Layout & navigation")
    p(doc, "components/Layout.tsx renders a left sidebar + main outlet. Nav links:")
    table(doc, ["Route", "Label", "Page component"], [
        ["/", "Overview", "pages/Overview.tsx"],
        ["/themes", "Themes", "pages/Themes.tsx"],
        ["/radar", "Complaint Radar", "pages/Radar.tsx"],
        ["/evidence", "Evidence", "pages/Evidence.tsx"],
        ["/sentiment", "Sentiment Validation", "pages/SentimentValidation.tsx"],
        ["/health", "Data Health", "pages/DataHealth.tsx"],
        ["/brief", "Product Brief", "pages/ProductBrief.tsx"],
    ])
    bullets(doc, [
        "Sidebar footer shows API health (green/red) and review count from GET /health.",
        "Deep links: /themes?theme=theme_003 and /radar?issue=theme_005 open a detail drawer/panel.",
        "Shared UI primitives live in components/ui.tsx: PageHeader, Card, Kpi, badges, Loading, ErrorState.",
        "Formatting helpers in lib/format.ts: fmtInt, fmtPct, fmtRatio, fmtDate, sentiment colours.",
    ])

    h2(doc, "3.2 Data fetching pattern")
    bullets(doc, [
        "api/client.ts — typed fetch wrappers for every endpoint; throws on non-OK with detail message.",
        "api/useApi.ts — hook: { data, error, loading, reload } for one async loader.",
        "api/types.ts — TypeScript shapes aligned with backend JSON (including batch_evaluation).",
        "Pages call useApi(() => api.xyz()) and render Loading / ErrorState / content.",
    ])

    h1(doc, "4. Page-by-page guide")

    h2(doc, "4.1 Executive Overview (/)")
    p(doc, "API: GET /metrics (+ GET /themes for the complaint bar chart).")
    bullets(doc, [
        "Header: total reviews, date range, dataset name.",
        "Banner when dataset.kind is synthetic (demo data warning).",
        "KPI row: reviews analysed, negative %, positive %, theme count, emerging issue count, PII redactions.",
        "Emerging complaints list with status badges and View why → /radar?issue=…",
        "Largest complaint theme callout.",
        "Sentiment pie (neg/neu/pos) and top complaint bar chart (negative volume).",
        "Drift badge from overview.drift_status when present.",
    ])
    p(doc, "Purpose: one-screen executive summary of the batch currently loaded in the API.")

    h2(doc, "4.2 Themes (/themes)")
    p(doc, "API: GET /themes, GET /themes/{id} when a theme is selected.")
    bullets(doc, [
        "Sort: size | negative_pct | priority | growth | name.",
        "Toggle complaints_only; text filter q on name/keywords.",
        "Table/cards show size, sentiment mix, radar status, growth, priority, coherence.",
        "Selecting a theme opens detail: keywords, representatives (redacted quotes), "
        "version/platform breakdown, link into radar.",
        "Query string ?theme=theme_XXX auto-opens that theme.",
    ])
    p(doc, "Formulas displayed on this page (from theme discovery + radar fields):")
    code(doc,
         "negative_pct = 100 * negative_count / size\n"
         "coherence    = mean(cosine(member_embedding, theme_centroid))\n"
         "similarity   = cosine(review_embedding, theme_centroid)   # shown on Evidence drawer")
    p(doc, "Purpose: explore what customers talk about and drill into example reviews.")

    h2(doc, "4.3 Complaint Radar (/radar)")
    p(doc, "API: GET /issues, GET /issues/{id}, evidence via related calls.")
    bullets(doc, [
        "Lists issues sorted by status then priority (NEW / EMERGING first).",
        "Filter by status chips (NEW, EMERGING, STABLE, DECLINING, …).",
        "Each row: mentions previous→current, growth label, negative ratio, priority.",
        "View why panel shows: human reasons, calculation strings (growth, negative ratio, priority), "
        "thresholds, weekly counts chart, segment associations (non-causal wording), evidence IDs.",
        "Soft-decline rule (gradual drop + falling weekly trend) appears in reasons when used.",
        "?issue=theme_XXX deep-links into a specific issue.",
    ])
    p(doc, "Formulas in View why (same as backend §4.3):")
    code(doc,
         "windows: current = last 14d; previous = 14d before that\n"
         "growth_pct     = (current - previous) / previous * 100\n"
         "negative_ratio = current_negatives / current_mentions\n"
         "priority       = current_negatives * (1 + clip(growth_pct/100, 0, 3))  # NEW → factor 4\n"
         "lift           = theme_share(segment=v) / overall_share(segment=v)")
    p(doc, "Purpose: show which complaints are growing — not only which are largest — with full transparency.")

    h2(doc, "4.4 Evidence (/evidence)")
    p(doc, "API: GET /reviews (filters), GET /reviews/{id}.")
    bullets(doc, [
        "Paginated table of redacted reviews.",
        "Filters: theme, sentiment, free-text search q (server-side substring on redacted text).",
        "Row click opens a drawer: class probabilities, theme similarity, whether used as radar/theme evidence.",
        "No raw PII — only placeholders if something was redacted.",
    ])
    p(doc, "Drawer scores use the same inference formulas as the backend:")
    code(doc,
         "p_c = softmax(logits)_c\n"
         "label = argmax_c p_c ;  confidence = max_c p_c\n"
         "theme_similarity = cosine(review_embedding, theme_centroid)")
    p(doc, "Purpose: audit trail from any claim on Overview/Themes/Radar back to real review IDs.")

    h2(doc, "4.5 Model Validation / Sentiment (/sentiment)")
    p(doc, "API: GET /sentiment/validation — product path shows global Amazon clothing TEST metrics "
           "(not per-upload accuracy). Upload batches do not invent holdout labels.")
    bullets(doc, [
        "KPIs: 3-class accuracy, macro recall, per-class recall (neg/neu/pos), confusion matrix.",
        "Methodology text: Amazon Cons_rating → sentiment weak labels; 80/10/10 holdout; fine-tuned RoBERTa.",
        "Does NOT claim accuracy on an uploaded CSV.",
    ])
    p(doc, "Formulas for the numbers on this page:")
    code(doc,
         "precision_c = TP_c / (TP_c + FP_c)\n"
         "recall_c    = TP_c / (TP_c + FN_c)\n"
         "F1_c        = 2 * precision_c * recall_c / (precision_c + recall_c)\n"
         "accuracy    = (# correct) / N\n"
         "macro_recall = (recall_neg + recall_neu + recall_pos) / 3")
    p(doc, "Purpose: prove how well the production sentiment model scores on the held-out Amazon TEST split.")

    h2(doc, "4.6 Data Health (/health)")
    p(doc, "API: GET /data-health, GET /drift, GET /model-info, GET /themes (for names).")
    bullets(doc, [
        "Ingestion: input rows → processed, rejects breakdown, duplicates removed, mojibake repaired.",
        "PII redactions by type (bar chart) + overall PII recall when available.",
        "Upload-batch note / model accuracy pointer (global Model Validation for Amazon holdout).",
        "Traceability audit status (PASS/FAIL) and link counts.",
        "Drift section (below) when dates exist; otherwise empty state: dates required.",
        "Model/hardware footnotes from model-info.",
    ])
    p(doc, "Purpose: data quality + monitoring, still scoped to the open DB.")

    h2(doc, "4.6.1 Drift formulas shown on Data Health")
    p(doc,
      "The UI displays the drift report from GET /drift (computed in ml/drift/monitor.py). "
      "It compares a reference window vs a current window (usually two consecutive 14-day periods). "
      "If dates_available is false, show the unavailable message instead of charts.")
    p(doc, "Overall drift badge = worst of the four statuses below (none < moderate < significant).")

    p(doc, "Sentiment mix and theme mix — Population Stability Index (PSI). "
           "Category shares are clipped with ε = 1e-4, renormalized, then:")
    code(doc, "PSI = sum_i (p_cur_i - p_ref_i) * ln(p_cur_i / p_ref_i)")
    p(doc, "Companion metric: Jensen–Shannon distance (base 2), shown as js_distance in [0, 1].")
    table(doc, ["PSI", "Badge / status"], [
        ["< 0.10", "none"],
        ["0.10 – 0.25", "moderate"],
        [">= 0.25", "significant"],
    ])

    p(doc, "Volume — relative change in reviews per day:")
    code(doc,
         "r = (#reviews) / (#days)\n"
         "change = (r_cur - r_ref) / r_ref")
    table(doc, ["|change|", "Status"], [
        ["< 20%", "none"],
        ["20% – 50%", "moderate"],
        [">= 50%", "significant"],
    ])

    p(doc, "Review length — Kolmogorov–Smirnov (KS) on character lengths of redacted text "
           "(statistic D and p-value):")
    bullets(doc, [
        "none if p >= 0.01 or D < 0.10",
        "moderate if p < 0.01 and 0.10 <= D < 0.20",
        "significant if p < 0.01 and D >= 0.20",
    ])
    p(doc, "Weekly drift series (when present) reuses the same metric family week-by-week vs baseline.")

    h2(doc, "4.7 Product Brief (/brief)")
    p(doc, "API: POST /product-brief with { engine: auto | qwen | template }.")
    bullets(doc, [
        "Engine selector buttons; Generate triggers the POST.",
        "Shows generation_path and fallback_reasons when Qwen was not used live.",
        "Sections: executive summary, top complaints, emerging, evidence quotes, associations, "
        "suggested investigations, caveats.",
        "Caveats include this-batch sentiment accuracy / macro recall when present.",
        "Validation panel for Qwen paths (checks passed/failed).",
        "All quotes are redacted; numbers in template path are deterministic from the DB.",
    ])
    p(doc, "Purpose: stakeholder-ready narrative without letting the LLM invent metrics.")

    h1(doc, "5. Visual language & UX conventions")
    bullets(doc, [
        "StatusBadge colours for NEW / EMERGING / STABLE / DECLINING / …",
        "DriftBadge for ok / moderate / significant.",
        "Kpi tone: bad (negative), good (positive), warn (attention).",
        "Banner notes for synthetic/demo batches.",
        "ErrorState always offers Retry (reload).",
        "Charts via Recharts; keep tooltips; avoid cluttering the executive page with secondary widgets.",
    ])

    h1(doc, "6. Frontend source map")
    table(doc, ["Path", "Responsibility"], [
        ["src/App.tsx", "Route table"],
        ["src/main.tsx", "React root + BrowserRouter"],
        ["src/components/Layout.tsx", "Sidebar, nav, health pill"],
        ["src/components/ui.tsx", "Shared presentational components"],
        ["src/pages/*.tsx", "One file per screen"],
        ["src/api/client.ts", "HTTP client"],
        ["src/api/types.ts", "Shared TS types"],
        ["src/api/useApi.ts", "Async data hook"],
        ["src/lib/format.ts", "Number/date/percent formatting"],
        ["src/styles.css (or equivalent)", "Global layout / tables / badges"],
        ["src/test/app.test.tsx", "Route + page smoke tests with mock fixtures"],
        ["src/test/fixtures/*.json", "Canned API responses for Vitest"],
        ["src/test/mockApi.ts", "fetch stub router"],
    ])

    h1(doc, "7. Configuration")
    table(doc, ["Variable / file", "Meaning"], [
        ["VITE_API_BASE_URL", "API origin (default http://127.0.0.1:8000)"],
        ["frontend/.env / .env.local", "Optional Vite env overrides"],
        ["package.json scripts", "dev, build, preview, typecheck, test"],
    ])
    code(doc,
         "cd frontend\n"
         "npm ci\n"
         "npm run dev        # http://127.0.0.1:5173\n"
         "npm run build      # production bundle\n"
         "npm test           # vitest run")

    h1(doc, "8. Testing the UI")
    bullets(doc, [
        "Vitest + jsdom + Testing Library.",
        "mockFetch() stubs every API route from fixtures — tests do not need a live server.",
        "Coverage examples: Overview KPIs, Themes sort/filter, Radar View why, Evidence drawer, "
        "Sentiment this-DB copy, Data Health ingestion, Product Brief engine paths.",
        "After changing API shapes, refresh fixtures (scripts/dump_api_samples.py --fixtures) "
        "while the API serves a current DB.",
    ])

    h1(doc, "9. Privacy & safety in the UI")
    bullets(doc, [
        "UI never receives raw unredacted text if the API contract is honoured.",
        "Do not log full review payloads to the browser console in demos.",
        "CORS must allow the Vite origin; credentials are not used.",
        "If /health is 503, the shell shows degraded state — build/point a DB first.",
    ])

    h1(doc, "10. Typical demo walkthrough (5 minutes)")
    numbered(doc, [
        "Start run_demo --serve --db for the batch you want (Nimbus / Amazon / Dove).",
        "Overview — call out emerging issues and sentiment split.",
        "Radar — open View why on a NEW/EMERGING item; show formula + evidence IDs.",
        "Evidence — open one ID; show probabilities.",
        "Sentiment Validation — show this-DB accuracy and overall recall card.",
        "Data Health — traceability PASS + PII redaction counts.",
        "Product Brief — template or auto; read executive summary + caveats.",
    ])

    h1(doc, "11. Relationship to backend docs")
    p(doc,
      "Pipeline stages, SQLite tables, radar math, and evaluation semantics are documented in "
      "Feedback_Review_Analyzer_Backend_Guide.docx. The UI only renders what that backend exposes.")

    h1(doc, "12. Troubleshooting")
    table(doc, ["Symptom", "Likely cause", "Fix"], [
        ["API ok · 0 reviews / degraded", "Wrong or missing ANALYTICS_DB", "Rebuild DB; pass --db to run_demo"],
        ["CORS errors in browser", "Origin not allowed", "Add origin to CORS_ORIGINS"],
        ["Sentiment page empty / 404", "Old DB without sentiment_validation", "Re-run ml.pipeline on that source"],
        ["Brief stuck loading", "Live Qwen loading on GPU", "Wait 1–3 min or choose template engine"],
        ["Amazon vs Nimbus numbers mixed", "Restarted UI without switching API DB", "Restart API with the intended --db"],
    ])

    paths = [save(doc, OUT_PRIMARY), save(doc, OUT_COPY)]
    return paths


if __name__ == "__main__":
    for path in build():
        print("Wrote", path)
