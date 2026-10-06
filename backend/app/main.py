"""FastAPI service over the precomputed analytics database.

No endpoint exposes the raw dataset or raw review text; every review text passes through safe_text().
Errors return generic messages (no stack traces, no internal paths).
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, File, HTTPException, Path as PathParam, Query, Request, UploadFile  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from backend.app import batches as batch_reg  # noqa: E402
from backend.app import db  # noqa: E402
from ml import config  # noqa: E402
from ml.pii import get_safe_logger  # noqa: E402

log = get_safe_logger("api")
logging.basicConfig(level=logging.INFO)

THEME_ID = r"^theme_\d{3}$"
REVIEW_ID = r"^[A-Za-z0-9_-]{1,40}$"

app = FastAPI(title="Feedback & Review Analyzer API", version="1.0.0",
              description="Serves precomputed, PII-redacted review analytics. See docs/API.md.")
origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
origin_regex = os.environ.get("CORS_ORIGIN_REGEX") or None
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_origin_regex=origin_regex, allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type"], allow_credentials=False)


@app.exception_handler(db.DatabaseUnavailable)
async def _db_unavailable(_: Request, exc: db.DatabaseUnavailable):
    return JSONResponse(status_code=503, content={"error": "service_unavailable", "detail": "Analytics data is not available."})


@app.exception_handler(RequestValidationError)
async def _validation(_: Request, exc: RequestValidationError):
    fields = [".".join(str(p) for p in e.get("loc", [])[1:]) or "request" for e in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "invalid_request", "detail": f"Invalid parameter(s): {', '.join(fields)}"})


@app.exception_handler(Exception)
async def _unhandled(_: Request, exc: Exception):
    log.error("unhandled error: %s", type(exc).__name__)
    return JSONResponse(status_code=500, content={"error": "internal_error", "detail": "An internal error occurred."})


def _get(con, sql: str, args: tuple = ()):
    return con.execute(sql, args).fetchone()


@app.get("/health")
def health():
    try:
        with db.connect() as con:
            m = db.meta(con)
            n = _get(con, "SELECT COUNT(*) AS n FROM reviews")["n"]
    except db.DatabaseUnavailable:
        return JSONResponse(status_code=503, content={"status": "degraded", "database": "missing"})
    return {"status": "ok", "database": "ok", "reviews": n, "source": m.get("source"),
            "dataset": m.get("dataset", {}).get("name"), "generated_at_utc": m.get("generated_at_utc")}


@app.get("/metrics")
def metrics():
    with db.connect() as con:
        ov = db.report(con, "overview")
        perf = db.report(con, "performance") or {}
        m = db.meta(con)
    return {"overview": ov, "dataset": m.get("dataset"), "generated_at_utc": m.get("generated_at_utc"),
            "pipeline_seconds": perf.get("total_seconds"), "device": perf.get("device")}


SORT_KEYS = {"size": "size", "negative_pct": "negative_pct", "priority": "priority", "growth": "growth_pct", "name": "name"}


@app.get("/themes")
def list_themes(complaints_only: bool = False, sort: Literal["size", "negative_pct", "priority", "growth", "name"] = "size",
                q: str | None = Query(default=None, max_length=60)):
    with db.connect() as con:
        themes = [json.loads(r["data"]) for r in con.execute("SELECT data FROM themes")]
    if complaints_only:
        themes = [t for t in themes if t["is_complaint"]]
    if q:
        ql = q.lower()
        themes = [t for t in themes if ql in t["name"].lower() or any(ql in k for k in t["keywords"])]
    key = SORT_KEYS[sort]
    themes.sort(key=lambda t: (t.get(key) is None, t.get(key) if key == "name" else -(t.get(key) or 0)))
    return {"count": len(themes), "themes": [{k: v for k, v in t.items() if k != "representative_ids"} for t in themes]}


@app.get("/themes/{theme_id}")
def get_theme(theme_id: str = PathParam(pattern=THEME_ID)):
    with db.connect() as con:
        row = _get(con, "SELECT data FROM themes WHERE theme_id=?", (theme_id,))
        if row is None:
            raise HTTPException(404, detail="Theme not found")
        t = json.loads(row["data"])
        names = db.theme_names(con)
        reps = []
        for rid in t["representative_ids"]:
            r = _get(con, "SELECT * FROM reviews WHERE review_id=?", (rid,))
            if r:
                reps.append(db.review_dict(r, names))
        issue = _get(con, "SELECT data FROM issues WHERE theme_id=?", (theme_id,))
        by_version = [dict(r) for r in con.execute(
            "SELECT app_version AS value, COUNT(*) AS n FROM reviews WHERE theme_id=? AND app_version IS NOT NULL GROUP BY app_version ORDER BY app_version", (theme_id,))]
        by_platform = [dict(r) for r in con.execute(
            "SELECT platform AS value, COUNT(*) AS n FROM reviews WHERE theme_id=? AND platform IS NOT NULL GROUP BY platform", (theme_id,))]
    return {**{k: v for k, v in t.items() if k != "representative_ids"}, "representative_reviews": reps,
            "radar": json.loads(issue["data"]) if issue else None, "segments": {"app_version": by_version, "platform": by_platform}}


@app.get("/issues")
def list_issues(status: str | None = Query(default=None, pattern=r"^[A-Z_,]{1,120}$",
                                           description="comma separated, e.g. NEW,EMERGING")):
    wanted = set(status.split(",")) if status else None
    with db.connect() as con:
        meta = db.report(con, "radar_meta") or {}
        items = [json.loads(r["data"]) for r in con.execute("SELECT data FROM issues ORDER BY priority DESC")]
        m = db.meta(con)
    if wanted:
        items = [i for i in items if i["status"] in wanted]
    order = {"NEW": 0, "EMERGING": 1, "STABLE": 2, "DECLINING": 3, "INSUFFICIENT_EVIDENCE": 4, "NOT_A_COMPLAINT": 5, "NO_DATA": 6}
    items.sort(key=lambda i: (order.get(i["status"], 9), -(i.get("priority") or 0)))
    return {"count": len(items), "window": meta.get("window"), "formula": meta.get("formula"), "params": meta.get("params"),
            "rules": meta.get("rules"),
            "dates_available": meta.get("dates_available", m.get("dates_available", True)),
            "message": meta.get("message") or m.get("dates_message"),
            "status": meta.get("status") or ("ok" if items else "unavailable"),
            "issues": [{k: v for k, v in i.items() if k not in ("calculation",)} for i in items]}


@app.get("/issues/{theme_id}")
def get_issue(theme_id: str = PathParam(pattern=THEME_ID)):
    with db.connect() as con:
        row = _get(con, "SELECT data FROM issues WHERE theme_id=?", (theme_id,))
        if row is None:
            raise HTTPException(404, detail="Issue not found")
        meta = db.report(con, "radar_meta") or {}
        t = _get(con, "SELECT data FROM themes WHERE theme_id=?", (theme_id,))
    return {**json.loads(row["data"]), "formula": meta.get("formula"), "window": meta.get("window"),
            "theme": {k: v for k, v in json.loads(t["data"]).items() if k in ("theme_id", "name", "keywords", "size", "coherence")} if t else None}


@app.get("/issues/{theme_id}/evidence")
def issue_evidence(theme_id: str = PathParam(pattern=THEME_ID)):
    with db.connect() as con:
        if _get(con, "SELECT 1 FROM issues WHERE theme_id=?", (theme_id,)) is None:
            raise HTTPException(404, detail="Issue not found")
        names = db.theme_names(con)
        rows = con.execute(
            "SELECT r.*, e.kind, e.rank FROM evidence e JOIN reviews r ON r.review_id = e.review_id "
            "WHERE e.theme_id=? ORDER BY e.kind DESC, e.rank", (theme_id,)).fetchall()
    ev = [{**db.review_dict(r, names), "evidence_kind": r["kind"], "rank": r["rank"]} for r in rows]
    return {"theme_id": theme_id, "count": len(ev), "evidence": ev}


@app.get("/reviews")
def list_reviews(theme_id: str | None = Query(default=None, pattern=THEME_ID),
                 sentiment: Literal["negative", "neutral", "positive"] | None = None,
                 q: str | None = Query(default=None, max_length=60),
                 limit: int = Query(default=25, ge=1, le=200), offset: int = Query(default=0, ge=0, le=1_000_000)):
    where, args = [], []
    if theme_id:
        where.append("theme_id = ?")
        args.append(theme_id)
    if sentiment:
        where.append("sentiment = ?")
        args.append(sentiment)
    if q:
        where.append("text_redacted LIKE ? ESCAPE '\\'")
        args.append("%" + re.sub(r"([%_\\])", r"\\\1", q) + "%")
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    with db.connect() as con:
        names = db.theme_names(con)
        total = _get(con, f"SELECT COUNT(*) AS n FROM reviews {clause}", tuple(args))["n"]
        rows = con.execute(f"SELECT * FROM reviews {clause} ORDER BY created_at DESC, review_id LIMIT ? OFFSET ?",
                           (*args, limit, offset)).fetchall()
    return {"total": total, "limit": limit, "offset": offset, "reviews": [db.review_dict(r, names) for r in rows]}


@app.get("/reviews/{review_id}")
def get_review(review_id: str = PathParam(pattern=REVIEW_ID)):
    with db.connect() as con:
        r = _get(con, "SELECT * FROM reviews WHERE review_id=?", (review_id,))
        if r is None:
            raise HTTPException(404, detail="Review not found")
        names = db.theme_names(con)
        ev = [dict(x) for x in con.execute("SELECT theme_id, kind, rank FROM evidence WHERE review_id=?", (review_id,))]
    return {**db.review_dict(r, names), "p_negative": r["p_negative"], "p_neutral": r["p_neutral"], "p_positive": r["p_positive"],
            "theme_similarity": r["theme_similarity"], "used_as_evidence": ev}


@app.get("/sentiment/validation")
@app.get("/model/evaluation")
def model_evaluation():
    """Global Amazon holdout metrics for the fine-tuned product model (not per-upload)."""
    path = config.AMAZON_SENTIMENT_EVAL
    if not path.exists():
        raise HTTPException(404, detail="Model evaluation report not found. Run scripts/finetune_amazon_sentiment.py")
    rep = json.loads(path.read_text(encoding="utf-8"))
    m = rep.get("metrics") or {}
    test = m.get("test") or {}
    return {
        "evaluation_date_utc": rep.get("evaluation_date_utc"),
        "model": rep.get("model_dir") or rep.get("model_base"),
        "model_revision": "amazon-finetune",
        "dataset": rep.get("dataset"),
        "device": rep.get("device"),
        "ground_truth_note": rep.get("ground_truth_note"),
        "methodology": rep.get("methodology"),
        "scope": "global_model_evaluation",
        "sample": {
            "size": test.get("n") or (rep.get("split") or {}).get("test"),
            "label_counts": {k: (test.get("per_class") or {}).get(k, {}).get("support", 0)
                             for k in ("negative", "neutral", "positive")},
            "selection": "held-out Amazon clothing TEST split (sha256 text-hash 80/10/10)",
        },
        "metrics": {
            "headline_accuracy": m.get("headline_accuracy") or test.get("accuracy"),
            "headline_macro_recall": m.get("headline_macro_recall") or test.get("macro_recall"),
            "macro_recall": m.get("headline_macro_recall") or test.get("macro_recall"),
            "three_class": test,
            "per_class": m.get("per_class") or test.get("per_class"),
            "confusion_3x3": m.get("confusion_3x3") or test.get("confusion"),
            "neutral_prediction_rate": None,
        },
        "split": rep.get("split"),
        "hyperparams": rep.get("hyperparams"),
    }


@app.get("/drift")
def drift():
    with db.connect() as con:
        d = db.report(con, "drift")
    if d is None:
        raise HTTPException(404, detail="Drift report not available")
    return d


@app.get("/data-health")
def data_health():
    with db.connect() as con:
        dh = db.report(con, "data_health")
        pii = db.report(con, "pii_audit") or {}
        trace = db.report(con, "traceability")
        m = db.meta(con)
    s = pii.get("synthetic", {})
    return {**(dh or {}), "traceability": trace,
            "pii_audit": {
                "overall_recall": s.get("overall_recall"),
                "recall_by_type": s.get("recall_by_type"),
                "false_positive_row_rate": s.get("false_positive_row_rate"),
            },
            "dates_available": (dh or {}).get("dates_available", m.get("dates_available")),
            "dates_message": (dh or {}).get("dates_message") or m.get("dates_message"),
            "dataset_validation": {
                "note": "Upload batch analytics only. Model accuracy lives on Model Validation (Amazon holdout).",
                "rows": (dh or {}).get("processed_reviews"),
                "status": (trace or {}).get("status"),
            }}


@app.get("/model-info")
def model_info():
    from ml.brief.service import qwen_status

    try:
        with db.connect() as con:
            models = {r["model"]: json.loads(r["data"]) for r in con.execute("SELECT model, data FROM model_versions")}
            mv = db.report(con, "model_verification") or {}
            hw = db.report(con, "hardware")
            perf = db.report(con, "performance") or {}
            qb = db.report(con, "qwen_brief")
    except db.DatabaseUnavailable:
        models, mv, hw, perf, qb = {}, {}, None, {}, None
    ver = {name: {k: v for k, v in rec.items() if k in ("state", "revision", "disk_gb", "load_seconds", "checks", "plan",
                                                         "gpu_memory_allocated_gb", "generation_seconds", "embedding_dim", "similarity")}
           for name, rec in mv.get("models", {}).items()}
    return {"pipeline_models": models, "verification": ver, "verification_status": mv.get("status"),
            "verification_offline": mv.get("network_blocked"), "hardware_at_pipeline_run": hw,
            "performance": {k: v for k, v in perf.items() if k != "embedding"},
            "brief_engine": qwen_status(), "precomputed_qwen_brief": bool(qb),
            "amazon_model_installed": (config.AMAZON_SENTIMENT_DIR / "config.json").exists(),
            "amazon_eval_present": config.AMAZON_SENTIMENT_EVAL.exists()}


class BriefRequest(BaseModel):
    engine: Literal["auto", "qwen", "template"] = Field(default="auto")


@app.post("/product-brief")
def product_brief(req: BriefRequest):
    from ml.brief.service import generate_brief

    if not db.db_path().exists():
        raise db.DatabaseUnavailable()
    return generate_brief(db.db_path(), req.engine)


@app.get("/batches")
def batches_list():
    return {"batches": batch_reg.list_batches(), "active_db": str(db.db_path()) if db.db_path().exists() else None}


@app.post("/batches/upload")
async def batches_upload(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(422, detail="Upload a .csv file with a review text column")
    content = await file.read()
    if len(content) > 80_000_000:
        raise HTTPException(413, detail="File too large (max 80MB)")
    if not content.strip():
        raise HTTPException(422, detail="Empty file")
    job = batch_reg.start_upload(file.filename, content)
    return job


@app.get("/batches/jobs/{job_id}")
def batches_job(job_id: str):
    job = batch_reg.job_status(job_id)
    if job is None:
        raise HTTPException(404, detail="Job not found")
    return job


@app.post("/batches/{batch_id}/activate")
def batches_activate(batch_id: str):
    try:
        return batch_reg.activate(batch_id)
    except KeyError:
        raise HTTPException(404, detail="Batch not found")
