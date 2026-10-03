"""Per-batch evaluation stored inside that batch's analytics DB.

Every metric here is computed only from the reviews in the current pipeline run. The dashboard that
serves that DB therefore shows accuracy / recall for THAT dataset only (Nimbus mock, Amazon CSV, …).
No cross-dataset or tweet corpus numbers are mixed in.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

import numpy as np

from ml.evaluation import metrics as M
from ml.pii.leak_scan import scan

LABEL_TO_I = {"negative": M.NEG, "neutral": M.NEU, "positive": M.POS}
I_TO_LABEL = {M.NEG: "negative", M.NEU: "neutral", M.POS: "positive"}

# Planted radar expectations for synthetic generators (theme key → expected status family).
# Only keys that appear as gt_theme in the current batch are scored.
PLANTED_RADAR = {
    # Nimbus app mock
    "payment_failure": {"NEW", "EMERGING"},
    "battery_drain": {"EMERGING", "STABLE"},
    "login_problems": {"DECLINING"},
    "app_crashes": {"STABLE", "INSUFFICIENT_EVIDENCE"},
    "dark_mode_request": {"INSUFFICIENT_EVIDENCE", "NOT_A_COMPLAINT", "NO_DATA"},
    # Dove shampoo mock
    "bottle_pump": {"NEW", "EMERGING"},
    "dry_hair": {"EMERGING", "STABLE"},
    "price_value": {"DECLINING"},
    "scalp_irritation": {"STABLE", "INSUFFICIENT_EVIDENCE"},
    "travel_size_request": {"INSUFFICIENT_EVIDENCE", "NOT_A_COMPLAINT", "NO_DATA"},
}


def stars_to_sentiment(rating: int | None) -> str | None:
    """Weak label from star ratings when a CSV has no explicit gt_sentiment."""
    if rating is None:
        return None
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    if rating >= 4:
        return "positive"
    return None


def _labels(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (y_true, y_pred) as class indices and a note describing the ground-truth source."""
    y_true, y_pred = [], []
    source = "none"
    for r in rows:
        gt = r.get("gt_sentiment")
        if not gt and r.get("rating") is not None:
            gt = stars_to_sentiment(r.get("rating"))
            source = "star_rating_weak_labels"
        elif gt:
            source = "planted_gt_sentiment" if source in ("none", "planted_gt_sentiment") else source
        if gt not in LABEL_TO_I or r.get("sentiment") not in LABEL_TO_I:
            continue
        y_true.append(LABEL_TO_I[gt])
        y_pred.append(LABEL_TO_I[r["sentiment"]])
    if source == "none" and y_true:
        source = "mixed"
    return np.array(y_true, dtype=int), np.array(y_pred, dtype=int), source


def _macro_recall(rep: dict) -> float:
    recalls = [rep["per_class"][c]["recall"] for c in ("negative", "neutral", "positive")]
    return round(float(np.mean(recalls)), 4)


def sentiment_report(rows: list[dict], model: str, revision: str | None, device: str) -> dict | None:
    y_true, y_pred, gt_source = _labels(rows)
    if len(y_true) == 0:
        return None
    three = M.evaluate_3class(y_true, y_pred)
    three["macro_recall"] = _macro_recall(three)
    # binary on non-neutral ground truth only (neg vs pos); neutral predictions map via p_pos vs p_neg when available
    mask = y_true != M.NEU
    binary = None
    if mask.any():
        kept = []
        for r in rows:
            gt = r.get("gt_sentiment") or stars_to_sentiment(r.get("rating"))
            if gt in LABEL_TO_I and r.get("sentiment") in LABEL_TO_I:
                kept.append(r)
        if kept and kept[0].get("p_negative") is not None:
            probs = np.array([[r["p_negative"], r["p_neutral"], r["p_positive"]] for r in kept], dtype=np.float32)
            forced = np.where(probs[:, M.POS] > probs[:, M.NEG], M.POS, M.NEG)
            binary = M.binary_report(y_true[mask], forced[mask])
        else:
            binary = M.binary_report(y_true[mask], y_pred[mask])
        binary["macro_recall"] = round((binary["negative"]["recall"] + binary["positive"]["recall"]) / 2, 4)
    counts = Counter(I_TO_LABEL[i] for i in y_true)
    return {
        "evaluation_date_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "model_revision": revision,
        "device": device,
        "dataset": "this analytics database only",
        "ground_truth_source": gt_source,
        "ground_truth_note": (
            "Labels come from planted synthetic ground truth."
            if gt_source == "planted_gt_sentiment"
            else "Labels are weak star-rating mappings (1–2 negative, 3 neutral, 4–5 positive). Not human sentiment annotation."
        ),
        "sample": {
            "size": int(len(y_true)),
            "label_counts": {k: int(counts.get(k, 0)) for k in ("negative", "neutral", "positive")},
            "selection": "all labelled reviews kept in this batch after cleaning",
        },
        "metrics": {
            "three_class": three,
            "binary_non_neutral_truth": binary,
            "headline_macro_recall": three["macro_recall"],
            "headline_accuracy": three["accuracy"],
        },
        "methodology": (
            "Per-batch evaluation. 3-class accuracy/precision/recall/F1 use the model's argmax label. "
            "Macro recall is the unweighted mean of per-class recall (the optimisation target for this project). "
            "Binary metrics (when present) score only rows whose ground truth is negative or positive."
        ),
    }


def theme_report(themes: list, theme_eval: dict | None) -> dict | None:
    if theme_eval is None:
        return None
    planted = [t for t in theme_eval["planted_themes"] if t != "invalid"]
    if not planted:
        return None
    recovered = [t for t in theme_eval["recovered_planted_themes"] if t != "invalid"]
    recall = len(set(recovered) & set(planted)) / max(1, len(set(planted)))
    return {
        "planted_themes": planted,
        "recovered_planted_themes": recovered,
        "missing_planted_themes": sorted(set(planted) - set(recovered)),
        "theme_recall": round(recall, 4),
        "ari_assigned": theme_eval.get("ari_assigned"),
        "nmi_assigned": theme_eval.get("nmi_assigned"),
        "weighted_purity": theme_eval.get("weighted_purity"),
        "n_themes_discovered": len(themes),
    }


def _majority_gt_theme(rows: list[dict], theme_id: str) -> str | None:
    counts = Counter(r.get("gt_theme") for r in rows if r.get("theme_id") == theme_id and r.get("gt_theme"))
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def radar_report(radar_items: list[dict], themes: list[dict], rows: list[dict] | None = None) -> dict | None:
    """Score planted temporal patterns when this batch carries gt_theme labels.

    Mapping preference (dataset-agnostic):
      1) majority gt_theme among reviews assigned to the discovered theme
      2) fallback: keyword match on the discovered theme name (legacy synthetic names)
    Only patterns listed in PLANTED_RADAR that appear in this batch are scored.
    """
    name_map = {
        "battery": "battery_drain", "payment": "payment_failure", "failing payment": "payment_failure",
        "login": "login_problems", "crash": "app_crashes", "dark": "dark_mode_request",
        "pump": "bottle_pump", "dry": "dry_hair", "price": "price_value",
        "scalp": "scalp_irritation", "itch": "scalp_irritation", "travel": "travel_size_request",
    }
    found: dict[str, str] = {}
    for it in radar_items:
        key = None
        if rows:
            key = _majority_gt_theme(rows, it["theme_id"])
        if not key:
            nm = (it.get("name") or "").lower()
            for needle, mapped in name_map.items():
                if needle in nm:
                    key = mapped
                    break
        if key and key in PLANTED_RADAR:
            # Prefer the first match; if duplicates, keep the higher-priority status already sorted by radar
            found.setdefault(key, it["status"])
    present = [k for k in PLANTED_RADAR if k in found or (rows and any(r.get("gt_theme") == k for r in rows))]
    if not present:
        return None
    hits = []
    for key in present:
        expected = PLANTED_RADAR[key]
        status = found.get(key)
        ok = status in expected if status else False
        hits.append({"planted": key, "observed_status": status, "expected_any_of": sorted(expected), "detected": ok})
    recall = sum(1 for h in hits if h["detected"]) / max(1, len(hits))
    return {"checks": hits, "radar_recall": round(recall, 4),
            "note": "Planted temporal patterns for this batch only (gt_theme present). Skipped on unlabelled CSVs."}


def pii_report(rows: list[dict]) -> dict | None:
    """Recall of planted PII types: a row with gt_pii must have pii_redaction_count > 0 and no leak-scan hits."""
    planted = [r for r in rows if r.get("gt_pii")]
    if not planted:
        return None
    caught, residual = 0, 0
    by_type: Counter = Counter()
    by_type_ok: Counter = Counter()
    for r in planted:
        types = [t for t in str(r["gt_pii"]).split(";") if t]
        for t in types:
            by_type[t] += 1
        ok = int(r.get("pii_redaction_count") or 0) > 0
        if ok:
            caught += 1
            for t in types:
                by_type_ok[t] += 1
        if scan(r.get("text_redacted") or ""):
            residual += 1
    n = len(planted)
    return {
        "rows_with_planted_pii": n,
        "rows_redacted": caught,
        "overall_recall": round(caught / n, 4),
        "recall_by_type": {t: f"{by_type_ok[t]}/{by_type[t]}" for t in sorted(by_type)},
        "residual_leak_scan_hits": residual,
        "note": "Planted PII rows in this batch only; not a cross-dataset audit.",
    }


def build(rows: list[dict], *, model: str, revision: str | None, device: str,
          theme_eval: dict | None, themes: list, radar_items: list[dict]) -> dict:
    """Full evaluation bundle for reports['sentiment_validation'] + reports['batch_evaluation']."""
    sent = sentiment_report(rows, model, revision, device)
    themes_r = theme_report(themes, theme_eval)
    radar_r = radar_report(radar_items, themes, rows=rows) if any(r.get("gt_theme") for r in rows) else None
    pii_r = pii_report(rows)
    recalls = {
        "sentiment_macro_recall": (sent or {}).get("metrics", {}).get("headline_macro_recall"),
        "theme_recall": (themes_r or {}).get("theme_recall"),
        "radar_recall": (radar_r or {}).get("radar_recall"),
        "pii_recall": (pii_r or {}).get("overall_recall"),
    }
    present = [v for v in recalls.values() if v is not None]
    return {
        "evaluation_date_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "metrics for this analytics database only",
        "sentiment": sent,
        "themes": themes_r,
        "radar": radar_r,
        "pii": pii_r,
        "overall_recall": {
            **recalls,
            "mean_available_recalls": round(float(np.mean(present)), 4) if present else None,
            "note": "Mean of whichever recall scores this batch can compute (sentiment / themes / radar / PII).",
        },
    }
