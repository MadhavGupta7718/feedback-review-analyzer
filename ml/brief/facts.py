"""Verified fact sheet for the product brief, built ONLY from the analytics database.

Every number, name and review ID a brief may mention comes from here. Each fact has a slot name
(e.g. E1.current) so an LLM can reference a value without ever writing the digits itself.
Only redacted text is included, and only a handful of evidence quotes (never the whole batch).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

MAX_TOP = 5
MAX_EMERGING = 4
QUOTES_PER_ISSUE = 2


def _report(con: sqlite3.Connection, key: str) -> dict | None:
    row = con.execute("SELECT data FROM reports WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def build_facts(db_path: Path) -> dict:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        ov = _report(con, "overview")
        drift = _report(con, "drift")
        sv = _report(con, "sentiment_validation")
        meta = {k: json.loads(v) for k, v in con.execute("SELECT key, value FROM meta")}
        issues = [json.loads(d) for (d,) in con.execute("SELECT data FROM issues ORDER BY priority DESC")]
        themes = {tid: json.loads(d) for tid, d in con.execute("SELECT theme_id, data FROM themes")}

        def quotes(ids):
            out = []
            for rid in ids[:QUOTES_PER_ISSUE]:
                row = con.execute("SELECT review_id, text_redacted, sentiment FROM reviews WHERE review_id=?", (rid,)).fetchone()
                if row:
                    out.append({"review_id": row[0], "text": row[1], "sentiment": row[2]})
            return out

        slots: dict[str, object] = {
            "total_reviews": ov["total_reviews"],
            "negative_pct": ov["sentiment_pct"]["negative"],
            "neutral_pct": ov["sentiment_pct"]["neutral"],
            "positive_pct": ov["sentiment_pct"]["positive"],
            "n_themes": ov["n_themes"],
        }
        emerging = []
        for k, it in enumerate([i for i in issues if i["status"] in ("NEW", "EMERGING")][:MAX_EMERGING], start=1):
            key = f"E{k}"
            slots.update({f"{key}.name": it["name"], f"{key}.current": it["current_mentions"],
                          f"{key}.previous": it["previous_mentions"], f"{key}.growth": it["growth_label"],
                          f"{key}.negative_pct": round(it["negative_ratio"] * 100, 1), f"{key}.status": it["status"]})
            emerging.append({"key": key, "theme_id": it["theme_id"], "name": it["name"], "status": it["status"],
                             "current": it["current_mentions"], "previous": it["previous_mentions"],
                             "growth_label": it["growth_label"], "growth_pct": it["growth_pct"],
                             "negative_pct": round(it["negative_ratio"] * 100, 1), "trend": it["trend"],
                             "evidence_review_ids": it["evidence_review_ids"], "quotes": quotes(it["evidence_review_ids"]),
                             "associations": [a["statement"] for a in it["associations"]],
                             "calculation": it["calculation"]["growth"]})
        complaints = sorted((t for t in themes.values() if t["is_complaint"]), key=lambda t: -t["negative_count"])[:MAX_TOP]
        top = []
        for k, t in enumerate(complaints, start=1):
            key = f"C{k}"
            slots.update({f"{key}.name": t["name"], f"{key}.negative": t["negative_count"], f"{key}.size": t["size"],
                          f"{key}.negative_pct": t["negative_pct"], f"{key}.growth": t.get("growth_label", "n/a")})
            top.append({"key": key, "theme_id": t["theme_id"], "name": t["name"], "size": t["size"],
                        "negative_count": t["negative_count"], "negative_pct": t["negative_pct"],
                        "growth_label": t.get("growth_label", "n/a"), "status": t.get("radar_status"),
                        "representative_ids": t["representative_ids"][:3]})
        declining = [{"theme_id": i["theme_id"], "name": i["name"], "growth_label": i["growth_label"]}
                     for i in issues if i["status"] == "DECLINING"]
        dm = drift["current_vs_previous"]["metrics"] if drift else {}
        facts = {
            "dataset": meta.get("dataset", {}),
            "date_range": ov["date_range"],
            "slots": slots,
            "overview": {"total_reviews": ov["total_reviews"], "sentiment_pct": ov["sentiment_pct"], "n_themes": ov["n_themes"],
                         "pii_redactions_total": ov["pii_redactions_total"]},
            "top_complaints": top,
            "emerging": emerging,
            "declining": declining,
            "drift": {k: {"value": v["value"], "status": v["status"]} for k, v in dm.items()},
            "drift_overall": drift["current_vs_previous"]["overall_status"] if drift else None,
            "sentiment_accuracy": (sv or {}).get("metrics", {}).get("binary_forced", {}).get("accuracy"),
            "allowed_review_ids": sorted({r for e in emerging for r in e["evidence_review_ids"]} |
                                         {r for c in top for r in c["representative_ids"]}),
            "allowed_theme_names": sorted({t["name"] for t in themes.values()}),
        }
        return facts
    finally:
        con.close()


def facts_json_for_llm(facts: dict) -> str:
    """Compact, slot-keyed view for the LLM: values are shown so it understands them, slots are what it must write."""
    view = {
        "slots": facts["slots"],
        "emerging": [{"key": e["key"], "status": e["status"], "trend": e["trend"], "evidence_review_ids": e["evidence_review_ids"][:3],
                      "example_quotes": [q["text"] for q in e["quotes"]], "associations": e["associations"]} for e in facts["emerging"]],
        "top_complaints": [{"key": c["key"]} for c in facts["top_complaints"]],
        "declining": [d["name"] for d in facts["declining"]],
        "drift_overall": facts["drift_overall"],
    }
    return json.dumps(view, ensure_ascii=False)
