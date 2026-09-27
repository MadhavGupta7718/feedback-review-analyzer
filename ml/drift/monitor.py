"""Drift monitoring between a reference window and a current window.

Metric                 Method                                   Thresholds (none / moderate / significant)
sentiment distribution PSI over {negative, neutral, positive}   < 0.10 / 0.10-0.25 / >= 0.25
theme distribution     PSI over theme ids (+ unassigned)        < 0.10 / 0.10-0.25 / >= 0.25
review volume          relative change in reviews per day       < 20%  / 20-50%    / >= 50%
review length          KS statistic on character lengths        D < 0.10 or p >= 0.01 / D 0.10-0.20 / D >= 0.20 (p < 0.01)

PSI (population stability index) = sum((cur - ref) * ln(cur / ref)) with epsilon smoothing; these PSI
bands are the conventional credit-risk / model-monitoring thresholds. Jensen-Shannon distance is also
reported for the categorical metrics as a bounded [0, 1] companion.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

EPS = 1e-4


@dataclass
class DriftThresholds:
    psi_moderate: float = 0.10
    psi_significant: float = 0.25
    volume_moderate: float = 0.20
    volume_significant: float = 0.50
    ks_moderate: float = 0.10
    ks_significant: float = 0.20
    ks_alpha: float = 0.01


def _dist(values: list, categories: list) -> np.ndarray:
    c = Counter(values)
    total = sum(c[k] for k in categories)
    if total == 0:
        return np.full(len(categories), 1.0 / max(1, len(categories)))
    return np.array([c[k] / total for k in categories], dtype=float)


def psi(ref: np.ndarray, cur: np.ndarray) -> float:
    r = np.clip(ref, EPS, None)
    c = np.clip(cur, EPS, None)
    r, c = r / r.sum(), c / c.sum()
    return float(np.sum((c - r) * np.log(c / r)))


def classify_psi(value: float, t: DriftThresholds) -> str:
    if value >= t.psi_significant:
        return "significant"
    if value >= t.psi_moderate:
        return "moderate"
    return "none"


def categorical_drift(ref_values: list, cur_values: list, t: DriftThresholds, categories: list | None = None) -> dict:
    cats = categories or sorted(set(ref_values) | set(cur_values), key=str)
    r, c = _dist(ref_values, cats), _dist(cur_values, cats)
    value = psi(r, c)
    changes = sorted(({"category": str(k), "reference_share": round(float(a), 4), "current_share": round(float(b), 4),
                       "change_pp": round(float(b - a) * 100, 2)} for k, a, b in zip(cats, r, c)),
                     key=lambda x: -abs(x["change_pp"]))
    return {"metric": "PSI", "value": round(value, 4), "js_distance": round(float(jensenshannon(r, c, base=2)), 4),
            "status": classify_psi(value, t), "reference_n": len(ref_values), "current_n": len(cur_values),
            "top_changes": changes[:5]}


def volume_drift(ref_count: int, ref_days: float, cur_count: int, cur_days: float, t: DriftThresholds) -> dict:
    ref_rate = ref_count / ref_days if ref_days > 0 else 0.0
    cur_rate = cur_count / cur_days if cur_days > 0 else 0.0
    if ref_rate == 0:
        change = None
        status = "significant" if cur_rate > 0 else "none"
    else:
        change = (cur_rate - ref_rate) / ref_rate
        status = "significant" if abs(change) >= t.volume_significant else "moderate" if abs(change) >= t.volume_moderate else "none"
    return {"metric": "relative change in reviews/day", "reference_per_day": round(ref_rate, 2), "current_per_day": round(cur_rate, 2),
            "value": None if change is None else round(change, 4), "status": status}


def length_drift(ref_lengths: list[int], cur_lengths: list[int], t: DriftThresholds) -> dict:
    if len(ref_lengths) < 2 or len(cur_lengths) < 2:
        return {"metric": "KS", "value": None, "p_value": None, "status": "insufficient_data"}
    res = ks_2samp(ref_lengths, cur_lengths)
    d, pval = float(res.statistic), float(res.pvalue)
    if pval >= t.ks_alpha or d < t.ks_moderate:
        status = "none"
    elif d >= t.ks_significant:
        status = "significant"
    else:
        status = "moderate"
    return {"metric": "KS", "value": round(d, 4), "p_value": float(f"{pval:.3g}"), "status": status,
            "reference_mean": round(float(np.mean(ref_lengths)), 1), "current_mean": round(float(np.mean(cur_lengths)), 1)}


SEVERITY = {"none": 0, "insufficient_data": 0, "moderate": 1, "significant": 2}


def drift_report(ref: list[dict], cur: list[dict], ref_days: float, cur_days: float,
                 thresholds: DriftThresholds | None = None, theme_ids: list | None = None) -> dict:
    """ref / cur: review dicts with sentiment, theme_id (None = unassigned), text_redacted."""
    t = thresholds or DriftThresholds()
    themes = (theme_ids or sorted({r.get("theme_id") or "unassigned" for r in ref + cur})) + ["unassigned"]
    themes = list(dict.fromkeys(themes))
    metrics = {
        "sentiment": categorical_drift([r["sentiment"] for r in ref], [r["sentiment"] for r in cur], t,
                                       ["negative", "neutral", "positive"]),
        "theme": categorical_drift([r.get("theme_id") or "unassigned" for r in ref],
                                   [r.get("theme_id") or "unassigned" for r in cur], t, themes),
        "volume": volume_drift(len(ref), ref_days, len(cur), cur_days, t),
        "review_length": length_drift([len(r["text_redacted"]) for r in ref], [len(r["text_redacted"]) for r in cur], t),
    }
    worst = max(metrics.values(), key=lambda m: SEVERITY.get(m["status"], 0))["status"]
    return {"thresholds": asdict(t), "metrics": metrics,
            "overall_status": worst if worst != "insufficient_data" else "none",
            "method_note": __doc__.strip()}
