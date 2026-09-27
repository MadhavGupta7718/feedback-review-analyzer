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

            synthetic.write_csv(synthetic.generate(seed=config.SEED), SYNTHETIC_CSV)
        with open(SYNTHETIC_CSV, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            r["rating"] = int(r["rating"]) if r.get("rating") else None
            r["text"] = r["text"] if r["text"] != "" else None
        info = {"name": "Synthetic Nimbus app reviews", "kind": "synthetic", "path": "data/raw/synthetic_reviews.csv",
                "seed": config.SEED, "note": "Deterministic synthetic data with planted patterns. Not real customer reviews."}
        return rows, info
    if source == "sentiment140":
        from ml.data.sentiment140 import parse_s140_date

        with open(S140_BATCH, encoding="utf-8") as fh:
            raw = list(csv.DictReader(fh))
        rows = []
        for r in raw:
            dt = parse_s140_date(r["date"])
            rows.append({"review_id": f"S{r['id']}", "created_at": dt.isoformat() if dt else None, "text": r["text"],
                         "rating": None, "platform": None, "app_version": None, "source": "sentiment140",
                         "gt_sentiment": "negative" if r["target"] == "0" else "positive"})
        info = {"name": "Sentiment140 10K balanced batch", "kind": "sentiment140", "path": "data/interim/s140_batch_10k.csv",
                "seed": config.SEED + 1, "note": "Tweets from Apr-Jun 2009; binary labels; not product reviews."}
        return rows, info
    raise ValueError(f"unknown source {source}")


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
    from ml.hardware import detect_hardware
    from ml.sentiment.model import SentimentModel
    from ml.themes.discovery import ThemeParams, discover, embedding_texts

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
        r.pop("gt_theme", None), r.pop("gt_pii", None)
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

    dated = [r for r in cleaned if r["created_at"]]
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
        "current_vs_previous": {**drift_report(prev, cur, 14, 14, theme_ids=theme_ids),
                                "reference_window": [prev_start.isoformat(), cur_start.isoformat()],
                                "current_window": [cur_start.isoformat(), end.isoformat()]},
        "weekly_vs_baseline": weekly_drift(dated, end, n_weeks, theme_ids),
        "note": ("Drift is measured on the batch's own timestamps. " +
                 ("Synthetic timestamps: drift reflects planted patterns, not production data." if source == "synthetic"
                  else "Sentiment140 timestamps are 2009 tweet times."))})
    trace = timer.run("traceability_audit", audit_traceability, cleaned, themes, radar)

    sent_counts = Counter(r["sentiment"] for r in cleaned)
    n = len(cleaned)
    overview = {
        "total_reviews": n,
        "sentiment_counts": dict(sent_counts),
        "sentiment_pct": {k: round(100 * sent_counts[k] / n, 1) for k in ("negative", "neutral", "positive")},
        "n_themes": len(themes),
        "n_complaint_themes": sum(1 for t in themes if t["is_complaint"]),
        "top_complaint": max((t for t in themes if t["is_complaint"]), key=lambda t: t["negative_count"], default=None),
        "emerging": [i for i in radar["items"] if i["status"] in ("NEW", "EMERGING")],
        "pii_redactions_total": sum(clean_rep.pii_redactions.values()),
        "drift_status": drift["current_vs_previous"]["overall_status"],
        "date_range": [first.isoformat(), end.isoformat()],
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
        "mojibake_repaired": clean_rep.mojibake_repaired,
        "html_entities_decoded": clean_rep.html_entities_decoded,
        "pii_redactions": clean_rep.pii_redactions,
        "rows_with_pii": clean_rep.rows_with_pii,
        "drift": {k: {"value": v["value"], "status": v["status"]} for k, v in drift["current_vs_previous"]["metrics"].items()},
        "theme_stats": tres.stats,
    }
    hw = detect_hardware().to_dict()
    perf = {"stages": timer.stages, "total_seconds": round(time.perf_counter() - t_all, 2),
            "peak_rss_mb": round(timer.peak_rss / 1024**2, 1), "device": device, "n_reviews": n,
            "sentiment_reviews_per_sec": round(n / out.seconds, 1),
            "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3) if device == "cuda" else None,
            "embedding": emb_info}
    reports = {"overview": overview, "data_health": data_health, "drift": drift, "traceability": trace, "performance": perf,
               "radar_meta": {k: v for k, v in radar.items() if k != "items"}, "hardware": hw}
    for key, fname in (("sentiment_validation", "sentiment_validation.json"), ("pii_audit", "pii_audit.json"),
                       ("model_verification", "model_verification.json"), ("dataset_validation", "dataset_validation.json")):
        f = config.ARTIFACTS_DIR / "reports" / fname
        if f.exists():
            reports[key] = json.loads(f.read_text(encoding="utf-8"))
    if "model_verification" in reports:  # keep the report small and free of tracebacks
        for m in reports["model_verification"].get("models", {}).values():
            m.pop("traceback", None)
    models = {
        config.SENTIMENT_MODEL: {"revision": sent_rev, "device": device, "purpose": "sentiment", "batch_size": 64,
                                 "dtype": "float16" if device == "cuda" else "float32"},
        config.EMBEDDING_MODEL: {"revision": emb_info.get("revision"), "device": device, "purpose": "embeddings / themes", "dim": 384},
    }
    meta = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "source": source, "dataset": dataset_info,
            "seed": config.SEED, "theme_params": tres.stats["params"], "radar_params": radar["params"],
            "limit": limit, "schema_version": 1}
    target = db_path or config.ANALYTICS_DB
    timer.run("write_db", write_db, target, meta, cleaned, themes, radar, reports, models)
    print(f"Wrote {target}  reviews={n} themes={len(themes)} traceability={trace['status']} total={perf['total_seconds']}s")
    return {"db": str(target), "overview": overview, "traceability": trace, "performance": perf, "data_health": data_health}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "sentiment140"])
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
