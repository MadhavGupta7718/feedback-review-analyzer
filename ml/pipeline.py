"""End-to-end offline pipeline -> artifacts/analytics.db (SQLite).

LOAD -> VALIDATE/DEDUPE -> PII REDACTION -> CLEAN -> SENTIMENT -> EMBEDDINGS -> THEMES -> COMPLAINTS
-> COMPLAINT RADAR -> DRIFT -> TRACEABILITY AUDIT -> SQLITE

Only redacted text is written. Raw text exists in memory for the duration of clean_batch() only.
Embeddings live in artifacts/embeddings/*.npy (an artifact store), not in the relational DB.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import config  # noqa: E402
from ml.complaints.radar import RadarParams, compute_radar  # noqa: E402
from ml.drift.monitor import drift_report  # noqa: E402
from ml.pii import get_safe_logger  # noqa: E402
from ml.pii.leak_scan import scan  # noqa: E402
from ml.preprocessing.clean import clean_batch  # noqa: E402

log = get_safe_logger("pipeline")
SYNTHETIC_CSV = config.DATA_DIR / "raw" / "synthetic_reviews.csv"
S140_BATCH = config.DATA_DIR / "interim" / "s140_batch_10k.csv"


class StageTimer:
    def __init__(self):
        import psutil

        self.proc = psutil.Process()
        self.stages: dict[str, dict] = {}
        self.peak_rss = 0

    def run(self, name: str, fn, *a, **kw):
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        rss = self.proc.memory_info().rss
        self.peak_rss = max(self.peak_rss, rss)
        self.stages[name] = {"seconds": round(time.perf_counter() - t0, 3), "rss_mb": round(rss / 1024**2, 1)}
        print(f"  [{name}] {self.stages[name]['seconds']}s", flush=True)
        return out


def load_source(source: str) -> tuple[list[dict], dict]:
    if source == "synthetic":
        if not SYNTHETIC_CSV.exists():
            from ml.data import synthetic

            synthetic.write_csv(synthetic.generate(seed=config.SEED, scale=2.0), SYNTHETIC_CSV)
        with open(SYNTHETIC_CSV, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            r["rating"] = int(r["rating"]) if r.get("rating") else None
            r["text"] = r["text"] if r["text"] != "" else None
        info = {"name": "Synthetic Nimbus app reviews", "kind": "synthetic", "path": "data/raw/synthetic_reviews.csv",
                "seed": config.SEED, "note": "Deterministic synthetic data with planted patterns. Not real customer reviews."}
        return rows, info
    if source == "sentiment140":
        raise ValueError(
            "Sentiment140 (tweets) was removed from the product path. This project is review-based. "
            "Use --source synthetic or --source path\\to\\reviews.csv"
        )
    if source.lower().endswith(".csv"):
        return load_csv(Path(source))
    raise ValueError(f"unknown source {source}")


CSV_ALIASES = {
    "text": ("text", "review", "review_text", "content", "body", "comment", "feedback"),
    "created_at": ("created_at", "date", "timestamp", "review_date", "time", "at", "datetime"),
    "rating": ("rating", "score", "stars", "star_rating"),
    "platform": ("platform", "os", "device"),
    "app_version": ("app_version", "version", "review_created_version"),
    "review_id": ("review_id", "id", "reviewid"),
    "gt_sentiment": ("gt_sentiment", "label", "sentiment_label", "sentiment_gt"),
    "gt_theme": ("gt_theme", "theme_label", "topic_label"),
    "gt_pii": ("gt_pii", "pii_types", "planted_pii"),
}
# ambiguous dates such as 06/07/2026 are read day-first
DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M")
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def parse_any_date(v: str | None) -> datetime | None:
    v = (v or "").strip()
    if not v:
        return None
    try:
        dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        dt = None
        for fmt in DATE_FORMATS:
            try:
                dt = datetime.strptime(v, fmt)
                break
            except ValueError:
                continue
    if dt is not None and dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def load_csv(path: Path) -> tuple[list[dict], dict]:
    """Any review CSV: a text column is required. Timestamps are optional — without them Radar/drift are
    marked unavailable (dates are never invented for uploads). Column names match CSV_ALIASES case-insensitively."""
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    raw_bytes = path.read_bytes()
    try:
        content = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        content = raw_bytes.decode("latin-1")
    reader = csv.DictReader(content.splitlines())
    header = {h.strip().lower(): h for h in (reader.fieldnames or [])}
    cols = {field: next((header[a] for a in aliases if a in header), None) for field, aliases in CSV_ALIASES.items()}
    if cols["text"] is None:
        raise ValueError(f"no review text column; expected one of {CSV_ALIASES['text']}")
    rows, seen_ids, bad_dates = [], set(), 0
    for n, r in enumerate(reader, 1):
        rid = (r.get(cols["review_id"]) or "").strip() if cols["review_id"] else ""
        if not SAFE_ID.match(rid) or rid in seen_ids:
            rid = f"U{n:06d}"
        seen_ids.add(rid)
        dt = parse_any_date(r.get(cols["created_at"])) if cols["created_at"] else None
        if cols["created_at"] and dt is None:
            bad_dates += 1
        elif not cols["created_at"]:
            bad_dates += 1
        rating = (r.get(cols["rating"]) or "").strip() if cols["rating"] else ""
        try:
            rating_v = int(round(float(rating))) if rating else None
        except ValueError:
            rating_v = None
        text = r.get(cols["text"])
        gt_sent = (r.get(cols["gt_sentiment"]) or "").strip().lower() if cols.get("gt_sentiment") else ""
        gt_sent = {"neg": "negative", "neu": "neutral", "pos": "positive",
                   "0": "negative", "2": "neutral", "4": "positive",
                   "1": "negative", "3": "neutral", "5": "positive"}.get(gt_sent, gt_sent)
        if gt_sent not in ("negative", "neutral", "positive"):
            gt_sent = None
        gt_theme = (r.get(cols["gt_theme"]) or "").strip() if cols.get("gt_theme") else None
        gt_pii = (r.get(cols["gt_pii"]) or "").strip() if cols.get("gt_pii") else None
        row = {"review_id": rid, "created_at": dt.isoformat() if dt else None, "text": text if text else None,
               "rating": rating_v, "platform": (r.get(cols["platform"]) or None) if cols["platform"] else None,
               "app_version": (r.get(cols["app_version"]) or None) if cols["app_version"] else None, "source": "csv"}
        if gt_sent:
            row["gt_sentiment"] = gt_sent
        if gt_theme:
            row["gt_theme"] = gt_theme
        if gt_pii:
            row["gt_pii"] = gt_pii
        rows.append(row)
    if not rows:
        raise ValueError("CSV has no data rows")
    has_dates = any(r.get("created_at") for r in rows)
    info = {"name": f"Uploaded CSV ({path.name})", "kind": "csv", "path": path.name,
            "columns": {k: v for k, v in cols.items() if v}, "rows_without_date": bad_dates,
            "dates_available": has_dates,
            "note": ("User-supplied reviews. Complaint Radar and drift require dates; "
                     + ("timestamps present." if has_dates else
                        "no usable dates — Radar/drift will be unavailable for this batch."))}
    return rows, info


def parse_dt(v) -> datetime | None:
    if isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(v) if v else None
    except (TypeError, ValueError):
        return None


def theme_rows(themes, labels, reviews, window_end: datetime, n_weeks: int) -> list[dict]:
    out = []
    for i, t in enumerate(themes):
        mem = [reviews[j] for j in t.member_indices]
        sc = Counter(r["sentiment"] for r in mem)
        weekly = [0] * n_weeks
        start = window_end - timedelta(days=7 * n_weeks)
        for r in mem:
            d = r["created_at"]
            if d and start < d <= window_end:
                weekly[min(n_weeks - 1, int((d - start).total_seconds() // (7 * 86400)))] += 1
        out.append({
            "theme_id": t.theme_id, "name": t.name, "keywords": t.keywords, "size": t.size, "core_size": t.core_size,
            "coherence": t.coherence, "negative_count": sc["negative"], "neutral_count": sc["neutral"],
            "positive_count": sc["positive"], "negative_pct": round(100 * sc["negative"] / max(1, t.size), 1),
            "representative_ids": [reviews[j]["review_id"] for j in t.representative_indices],
            "weekly_counts": weekly, "is_complaint": sc["negative"] / max(1, t.size) >= 0.5,
            "avg_rating": round(float(np.mean([r["rating"] for r in mem if r.get("rating")])), 2) if any(r.get("rating") for r in mem) else None,
        })
    return out


def weekly_drift(reviews: list[dict], end: datetime, n_weeks: int, theme_ids: list[str]) -> list[dict]:
    """Each week vs the first two weeks of the batch (baseline)."""
    start = end - timedelta(days=7 * n_weeks)
    weeks = [[] for _ in range(n_weeks)]
    for r in reviews:
        d = r["created_at"]
        if d and start < d <= end:
            weeks[min(n_weeks - 1, int((d - start).total_seconds() // (7 * 86400)))].append(r)
    base = weeks[0] + weeks[1] if n_weeks >= 2 else weeks[0]
    out = []
    for w, rows in enumerate(weeks):
        if not rows or w < 2:
            continue
        rep = drift_report(base, rows, 14, 7, theme_ids=theme_ids)
        out.append({"week": w + 1, "week_start": (start + timedelta(days=7 * w)).date().isoformat(), "n": len(rows),
                    **{k: {"value": v["value"], "status": v["status"]} for k, v in rep["metrics"].items()},
                    "overall_status": rep["overall_status"]})
    return out


def audit_traceability(reviews: list[dict], themes: list[dict], radar: dict) -> dict:
    by_id = {r["review_id"]: r for r in reviews}
    problems = []
    checked = 0
    for t in themes:
        for rid in t["representative_ids"]:
            checked += 1
            r = by_id.get(rid)
            if r is None:
                problems.append(f"{t['theme_id']}: representative {rid} does not exist")
            elif r["theme_id"] != t["theme_id"]:
                problems.append(f"{t['theme_id']}: representative {rid} belongs to {r['theme_id']}")
    for it in radar["items"]:
        for rid in it["evidence_review_ids"]:
            checked += 1
            r = by_id.get(rid)
            if r is None:
                problems.append(f"{it['theme_id']}: evidence {rid} does not exist")
            elif r["theme_id"] != it["theme_id"]:
                problems.append(f"{it['theme_id']}: evidence {rid} belongs to {r['theme_id']}")
            elif r["sentiment"] != "negative":
                problems.append(f"{it['theme_id']}: evidence {rid} is not negative")
    leaks = sum(1 for r in reviews if scan(r["text_redacted"]))
    return {"links_checked": checked, "problems": problems, "reviews_with_residual_pii_pattern": leaks,
            "status": "PASS" if not problems and leaks == 0 else "FAILED"}


SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE reviews (
  review_id TEXT PRIMARY KEY, created_at TEXT, rating INTEGER, platform TEXT, app_version TEXT,
  text_redacted TEXT NOT NULL, sentiment TEXT NOT NULL, confidence REAL NOT NULL,
  p_negative REAL, p_neutral REAL, p_positive REAL, theme_id TEXT, theme_assignment TEXT, theme_similarity REAL,
  pii_redaction_count INTEGER NOT NULL, source TEXT NOT NULL);
CREATE INDEX idx_reviews_theme ON reviews(theme_id);
CREATE INDEX idx_reviews_created ON reviews(created_at);
CREATE TABLE themes (theme_id TEXT PRIMARY KEY, name TEXT NOT NULL, data TEXT NOT NULL);
CREATE TABLE issues (theme_id TEXT PRIMARY KEY, status TEXT NOT NULL, priority REAL NOT NULL, data TEXT NOT NULL);
CREATE TABLE evidence (theme_id TEXT NOT NULL, review_id TEXT NOT NULL REFERENCES reviews(review_id), kind TEXT NOT NULL,
  rank INTEGER NOT NULL, PRIMARY KEY (theme_id, review_id, kind));
CREATE TABLE reports (key TEXT PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE model_versions (model TEXT PRIMARY KEY, data TEXT NOT NULL);
"""


def write_db(path: Path, meta: dict, reviews: list[dict], themes: list[dict], radar: dict, reports: dict, models: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.db")
    if tmp.exists():
        tmp.unlink()
    con = sqlite3.connect(tmp)
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO meta VALUES (?, ?)", [(k, json.dumps(v)) for k, v in meta.items()])
    con.executemany("INSERT INTO reviews VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
        (r["review_id"], r["created_at"].isoformat() if r["created_at"] else None, r.get("rating"), r.get("platform"),
         r.get("app_version"), r["text_redacted"], r["sentiment"], r["confidence"], r["p_negative"], r["p_neutral"],
         r["p_positive"], r["theme_id"], r["theme_assignment"], r["theme_similarity"], r["pii_redaction_count"], r["source"])
        for r in reviews])
    con.executemany("INSERT INTO themes VALUES (?,?,?)", [(t["theme_id"], t["name"], json.dumps(t)) for t in themes])
    con.executemany("INSERT INTO issues VALUES (?,?,?,?)", [(i["theme_id"], i["status"], i["priority"], json.dumps(i)) for i in radar["items"]])
    ev = []
    for t in themes:
        ev += [(t["theme_id"], rid, "representative", k) for k, rid in enumerate(t["representative_ids"])]
    for i in radar["items"]:
        ev += [(i["theme_id"], rid, "radar_evidence", k) for k, rid in enumerate(i["evidence_review_ids"])]
    con.executemany("INSERT INTO evidence VALUES (?,?,?,?)", ev)
    con.executemany("INSERT INTO reports VALUES (?,?)", [(k, json.dumps(v, default=str)) for k, v in reports.items()])
    con.executemany("INSERT INTO model_versions VALUES (?,?)", [(k, json.dumps(v)) for k, v in models.items()])
    con.commit()
    con.close()
    if path.exists():
        path.unlink()
    tmp.rename(path)


def subsample(rows: list[dict], limit: int | None) -> list[dict]:
    """Deterministic, evenly spaced subsample so a small batch still spans the whole time range."""
    if not limit or limit >= len(rows):
        return rows
    idx = np.linspace(0, len(rows) - 1, limit).round().astype(int)
    return [rows[i] for i in idx]


def run(source: str = "synthetic", limit: int | None = None, db_path: Path | None = None, device: str | None = None,
        use_embedding_cache: bool = True) -> dict:
    import torch

    from ml.embeddings.encoder import encode_cached
    from ml.evaluation import batch_eval
    from ml.hardware import detect_hardware
    from ml.sentiment.model import SentimentModel
    from ml.themes.discovery import ThemeParams, discover, embedding_texts, evaluate_against_truth

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    timer = StageTimer()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_all = time.perf_counter()
    print(f"Pipeline source={source} device={device}", flush=True)

    rows, dataset_info = timer.run("load", load_source, source)
    rows = subsample(rows, limit)
    cleaned, clean_rep = timer.run("validate_redact_clean", clean_batch, rows)
    for r in cleaned:
        r["created_at"] = parse_dt(r.get("created_at"))
        # keep gt_* through modelling so this batch can be scored into THIS database only
    ts_missing = sum(1 for r in cleaned if r["created_at"] is None)
    texts = [r["text_redacted"] for r in cleaned]

    sm = timer.run("load_sentiment_model", SentimentModel, device)
    out = timer.run("sentiment", sm.predict, texts, 64)
    for r, lab, conf, p in zip(cleaned, out.labels, out.confidences, out.probs):
        r.update(sentiment=lab, confidence=round(float(conf), 4), p_negative=round(float(p[0]), 4),
                 p_neutral=round(float(p[1]), 4), p_positive=round(float(p[2]), 4))
    sent_rev = sm.revision
    del sm
    if device == "cuda":
        torch.cuda.empty_cache()

    emb, emb_info = timer.run("embeddings", encode_cached, embedding_texts(texts), device, 128, use_embedding_cache)
    tres = timer.run("themes", discover, emb, texts, ThemeParams())
    for idx, r in enumerate(cleaned):
        lab = int(tres.labels[idx])
        r["theme_id"] = tres.themes[lab].theme_id if lab >= 0 else None
        r["theme_assignment"] = tres.assignment[idx]
        r["theme_similarity"] = round(float(tres.similarity[idx]), 4)
    theme_truth = [r.get("gt_theme") or "unknown" for r in cleaned]
    theme_eval = None
    if any(r.get("gt_theme") and r["gt_theme"] != "invalid" for r in cleaned):
        theme_eval = evaluate_against_truth(tres.labels, theme_truth, tres.themes)

    dated = [r for r in cleaned if r["created_at"]]
    dates_available = len(dated) > 0 and len(dated) >= max(1, int(0.5 * len(cleaned)))
    date_msg = ("This view needs review dates; none were found in this dataset (or most rows lack parseable dates)."
                if not dates_available else None)

    if dates_available:
        end = max(r["created_at"] for r in dated)
        first = min(r["created_at"] for r in dated)
        n_weeks = max(1, int(np.ceil((end - first).total_seconds() / (7 * 86400))))
        themes = theme_rows(tres.themes, tres.labels, cleaned, end, n_weeks)
        radar_reviews = [{"review_id": r["review_id"], "created_at": r["created_at"], "theme_id": r["theme_id"],
                          "sentiment": r["sentiment"], "confidence": r["confidence"], "similarity": r["theme_similarity"],
                          "app_version": r.get("app_version"), "platform": r.get("platform")} for r in cleaned]
        radar = timer.run("complaint_radar", compute_radar, radar_reviews, themes, RadarParams())
        radar_by_theme = {i["theme_id"]: i for i in radar["items"]}
        for t in themes:
            it = radar_by_theme[t["theme_id"]]
            t.update(radar_status=it["status"], growth_pct=it["growth_pct"], growth_label=it["growth_label"], priority=it["priority"])
        w = radar["window"]
        cur_start, prev_start = datetime.fromisoformat(w["current_start"]), datetime.fromisoformat(w["previous_start"])
        cur = [r for r in dated if cur_start < r["created_at"] <= end]
        prev = [r for r in dated if prev_start < r["created_at"] <= cur_start]
        theme_ids = [t["theme_id"] for t in themes]
        drift = timer.run("drift", lambda: {
            "status": "ok",
            "dates_available": True,
            "current_vs_previous": {**drift_report(prev, cur, 14, 14, theme_ids=theme_ids),
                                    "reference_window": [prev_start.isoformat(), cur_start.isoformat()],
                                    "current_window": [cur_start.isoformat(), end.isoformat()]},
            "weekly_vs_baseline": weekly_drift(dated, end, n_weeks, theme_ids),
            "note": "Drift is measured on the batch's own timestamps from the uploaded CSV."})
        date_range = [first.isoformat(), end.isoformat()]
        emerging = [i for i in radar["items"] if i["status"] in ("NEW", "EMERGING")]
        drift_status = drift["current_vs_previous"]["overall_status"]
        drift_metrics = {k: {"value": v["value"], "status": v["status"]} for k, v in drift["current_vs_previous"]["metrics"].items()}
    else:
        end = datetime.now(timezone.utc).replace(tzinfo=None)
        themes = theme_rows(tres.themes, tres.labels, cleaned, end, 1)
        for t in themes:
            t.update(radar_status="NO_DATA", growth_pct=None, growth_label="n/a", priority=0.0)
        radar = {
            "status": "unavailable",
            "dates_available": False,
            "message": date_msg,
            "params": {},
            "formula": None,
            "rules": None,
            "window": None,
            "items": [],
        }
        drift = {
            "status": "unavailable",
            "dates_available": False,
            "message": date_msg,
            "current_vs_previous": {"overall_status": "unavailable", "metrics": {}},
            "weekly_vs_baseline": [],
            "note": date_msg,
        }
        date_range = None
        emerging = []
        drift_status = "unavailable"
        drift_metrics = {}

    trace = timer.run("traceability_audit", audit_traceability, cleaned, themes, radar)

    # Per-upload batch evaluation (planted/star) is NOT used for the product accuracy page.
    # Global model metrics live in artifacts/reports/amazon_sentiment_eval.json.
    batch_metrics = {"scope": "upload_batch_analytics_only", "sentiment": None, "themes": None, "radar": None, "pii": None,
                     "overall_recall": None}

    sent_counts = Counter(r["sentiment"] for r in cleaned)
    n = len(cleaned)
    overview = {
        "total_reviews": n,
        "sentiment_counts": dict(sent_counts),
        "sentiment_pct": {k: round(100 * sent_counts[k] / n, 1) for k in ("negative", "neutral", "positive")},
        "n_themes": len(themes),
        "n_complaint_themes": sum(1 for t in themes if t["is_complaint"]),
        "top_complaint": max((t for t in themes if t["is_complaint"]), key=lambda t: t["negative_count"], default=None),
        "emerging": emerging,
        "pii_redactions_total": sum(clean_rep.pii_redactions.values()),
        "drift_status": drift_status,
        "date_range": date_range,
        "dates_available": dates_available,
        "dates_message": date_msg,
        "avg_rating": round(float(np.mean([r["rating"] for r in cleaned if r.get("rating")])), 2) if any(r.get("rating") for r in cleaned) else None,
    }
    overview["top_complaint"] = ({k: overview["top_complaint"][k] for k in ("theme_id", "name", "negative_count", "size", "negative_pct")}
                                 if overview["top_complaint"] else None)
    overview["emerging"] = [{k: i[k] for k in ("theme_id", "name", "status", "current_mentions", "previous_mentions", "growth_pct",
                                              "growth_label", "negative_ratio", "priority")} for i in overview["emerging"]]
    data_health = {
        "dataset": dataset_info,
        "input_rows": clean_rep.input_rows,
        "processed_reviews": n,
        "rejected": clean_rep.rejected,
        "ingestion_duplicates_removed": clean_rep.ingestion_duplicates_removed,
        "identical_text_rows_kept": clean_rep.identical_text_rows_kept,
        "missing_timestamps": ts_missing,
        "dates_available": dates_available,
        "dates_message": date_msg,
        "mojibake_repaired": clean_rep.mojibake_repaired,
        "html_entities_decoded": clean_rep.html_entities_decoded,
        "pii_redactions": clean_rep.pii_redactions,
        "rows_with_pii": clean_rep.rows_with_pii,
        "drift": drift_metrics,
        "theme_stats": tres.stats,
    }
    hw = detect_hardware().to_dict()
    perf = {"stages": timer.stages, "total_seconds": round(time.perf_counter() - t_all, 2),
            "peak_rss_mb": round(timer.peak_rss / 1024**2, 1), "device": device, "n_reviews": n,
            "sentiment_reviews_per_sec": round(n / out.seconds, 1),
            "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3) if device == "cuda" else None,
            "embedding": emb_info}
    reports = {"overview": overview, "data_health": data_health, "drift": drift, "traceability": trace, "performance": perf,
               "radar_meta": {k: v for k, v in radar.items() if k != "items"}, "hardware": hw,
               "batch_evaluation": batch_metrics}
    for key, fname in (("pii_audit", "pii_audit.json"), ("model_verification", "model_verification.json")):
        f = config.ARTIFACTS_DIR / "reports" / fname
        if f.exists():
            reports[key] = json.loads(f.read_text(encoding="utf-8"))
    if "pii_audit" in reports:
        reports["pii_audit"].pop("sentiment140_batch", None)
    if "model_verification" in reports:
        for m in reports["model_verification"].get("models", {}).values():
            m.pop("traceback", None)
    for r in cleaned:
        r.pop("gt_theme", None)
        r.pop("gt_pii", None)
        r.pop("gt_sentiment", None)
    sent_name = out.model
    models = {
        sent_name: {"revision": sent_rev, "device": device, "purpose": "sentiment", "batch_size": 64,
                    "dtype": "float16" if device == "cuda" else "float32"},
        config.EMBEDDING_MODEL: {"revision": emb_info.get("revision"), "device": device, "purpose": "embeddings / themes", "dim": 384},
    }
    meta = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "source": dataset_info["kind"], "dataset": dataset_info,
            "seed": config.SEED, "theme_params": tres.stats["params"], "radar_params": radar.get("params") or {},
            "limit": limit, "schema_version": 2, "dates_available": dates_available, "dates_message": date_msg}
    target = db_path or config.ANALYTICS_DB
    timer.run("write_db", write_db, target, meta, cleaned, themes, radar, reports, models)
    print(f"Wrote {target}  reviews={n} themes={len(themes)} traceability={trace['status']} total={perf['total_seconds']}s")
    return {"db": str(target), "overview": overview, "traceability": trace, "performance": perf, "data_health": data_health}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", help="synthetic | path to a review CSV")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--db", type=Path, default=None)
    ap.add_argument("--device", default=None, choices=[None, "cuda", "cpu"])
    ap.add_argument("--no-embedding-cache", action="store_true")
    args = ap.parse_args()
    res = run(args.source, args.limit, args.db, args.device, not args.no_embedding_cache)
    print(json.dumps({"overview": res["overview"], "traceability": res["traceability"],
                      "performance": {k: v for k, v in res["performance"].items() if k != "embedding"}}, indent=2, default=str))
    return 0 if res["traceability"]["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
