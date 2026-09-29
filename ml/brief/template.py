"""Deterministic template brief. Always available; used when Qwen is unavailable or fails validation."""
from __future__ import annotations

from datetime import datetime, timezone


def _pct(v) -> str:
    return f"{v:g}%"


def template_brief(facts: dict) -> dict:
    ov = facts["overview"]
    sp = ov["sentiment_pct"]
    em = facts["emerging"]
    top = facts["top_complaints"]
    lines = [
        f"Across {ov['total_reviews']:,} reviews ({facts['date_range'][0][:10]} to {facts['date_range'][1][:10]}), "
        f"{_pct(sp['negative'])} are negative, {_pct(sp['neutral'])} neutral and {_pct(sp['positive'])} positive, "
        f"grouped into {ov['n_themes']} themes."
    ]
    if em:
        parts = [f"{e['name']} ({'new: ' + str(e['current']) + ' mentions vs 0' if e['status'] == 'NEW' else e['growth_label'] + ', ' + str(e['previous']) + ' to ' + str(e['current']) + ' mentions'})"
                 for e in em]
        lines.append("Emerging complaints in the latest 14-day window: " + "; ".join(parts) + ".")
    if top:
        lines.append(f"The largest complaint theme by negative volume is {top[0]['name']} "
                     f"({top[0]['negative_count']:,} negative of {top[0]['size']:,} mentions).")
    if facts.get("declining"):
        lines.append("Declining: " + ", ".join(f"{d['name']} ({d['growth_label']})" for d in facts["declining"]) + ".")
    exec_summary = " ".join(lines)

    top_complaints = [f"{c['name']}: {c['negative_count']:,} negative of {c['size']:,} mentions ({_pct(c['negative_pct'])} negative), "
                      f"window change {c['growth_label']}." for c in top]
    emerging = []
    evidence = []
    associations = []
    investigate = []
    for e in em:
        if e["status"] == "NEW":
            emerging.append(f"{e['name']} [NEW]: {e['current']} mentions in the current window, 0 in the previous window; "
                            f"{_pct(e['negative_pct'])} negative; trend {e['trend']}.")
        else:
            emerging.append(f"{e['name']} [EMERGING]: {e['previous']} -> {e['current']} mentions ({e['growth_label']}); "
                            f"{_pct(e['negative_pct'])} negative; trend {e['trend']}. Calculation: {e['calculation']}.")
        for q in e["quotes"]:
            evidence.append({"theme": e["name"], "review_id": q["review_id"], "text": q["text"]})
        associations.extend(e["associations"])
        investigate.append(f"Review the {e['current']} current-window {e['name']} reports (evidence IDs "
                           f"{', '.join(e['evidence_review_ids'][:3])}) to confirm the pattern before prioritising a fix.")
    if not associations:
        associations.append("No segment (app version / platform) shows a notable association with the emerging complaints.")
    if facts.get("drift_overall") in ("moderate", "significant"):
        drifted = [k for k, v in facts["drift"].items() if v["status"] in ("moderate", "significant")]
        investigate.append(f"Feedback distribution changed between windows ({', '.join(drifted)} drift); re-check theme "
                           f"definitions if the change persists.")
    caveats = [
        "Statistics describe feedback volume and sentiment; they do not establish causes.",
        "All quotes are redacted verbatims from the analysed batch.",
    ]
    if facts.get("dataset", {}).get("kind") == "synthetic":
        caveats.append("This batch is synthetic demonstration data, not real customer feedback.")
    if facts.get("sentiment_accuracy") is not None:
        caveats.append(f"Sentiment model accuracy on the held-out labelled Sentiment140 test split: {facts['sentiment_accuracy'] * 100:.1f}% (binary evaluation).")
    return {
        "generation_path": "template",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "executive_summary": exec_summary,
        "top_complaints": top_complaints,
        "emerging_complaints": emerging,
        "evidence": evidence,
        "possible_associations": associations,
        "suggested_investigation_areas": investigate,
        "caveats": caveats,
    }
