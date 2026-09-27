"""Complaint Radar: which complaints are GROWING, with the full calculation behind every flag.

Definitions (all transparent, all returned in the API so the dashboard can show "why")
  current window   the last `window_days` days of the batch, ending at the latest review timestamp
  previous window  the `window_days` days immediately before the current window
  growth_pct       (current - previous) / previous * 100          (None when previous == 0 -> status NEW)
  negative_ratio   negative reviews / all reviews of the theme in the current window
  acceleration_pp  growth of the last weekly period minus growth of the period before it (percentage points)

Status rules, evaluated in this order
  NO_DATA                current == 0 and previous == 0
  NOT_A_COMPLAINT        negative_ratio (current, or whole batch if current is empty) < min_negative_ratio
  INSUFFICIENT_EVIDENCE  current < min_current_mentions  or  negative evidence reviews < min_evidence_reviews
  NEW                    previous == 0 and current >= min_current_mentions
  EMERGING               growth_pct >= min_growth_pct
  DECLINING              growth_pct <= -decline_pct
  STABLE                 otherwise

Priority (only a sort key; it never decides the status)
  priority = negative mentions in current window x growth_factor
  growth_factor = 1 + clip(growth_pct / 100, 0, max_growth_bonus)   (NEW -> 1 + max_growth_bonus)
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

import numpy as np


@dataclass
class RadarParams:
    window_days: int = 14
    min_current_mentions: int = 30
    min_evidence_reviews: int = 3
    min_growth_pct: float = 50.0
    min_negative_ratio: float = 0.50
    decline_pct: float = 25.0
    max_growth_bonus: float = 3.0
    n_evidence: int = 5
    trend_bucket_days: int = 7
    min_segment_support: int = 20
    min_segment_lift: float = 1.3


PRIORITY_FORMULA = "priority = current_negative_mentions x (1 + clip(growth_pct / 100, 0, {cap}))  [NEW: x (1 + {cap})]"


def growth_pct(current: int, previous: int) -> float | None:
    """Safe growth: None when previous == 0 (caller decides NEW vs NO_DATA). Never inf / NaN."""
    if current < 0 or previous < 0:
        raise ValueError("counts must be non-negative")
    if previous == 0:
        return None
    return round((current - previous) / previous * 100.0, 2)


def growth_factor(g: float | None, previous: int, current: int, cap: float) -> float:
    if previous == 0 and current > 0:
        return 1.0 + cap
    if g is None:
        return 1.0
    return 1.0 + min(max(g / 100.0, 0.0), cap)


@dataclass
class RadarItem:
    theme_id: str
    name: str
    status: str
    current_mentions: int
    previous_mentions: int
    growth_pct: float | None
    growth_label: str
    negative_mentions_current: int
    negative_ratio: float
    negative_ratio_all: float
    total_mentions: int
    acceleration_pp: float | None
    trend: str
    weekly_counts: list = field(default_factory=list)
    evidence_review_ids: list = field(default_factory=list)
    priority: float = 0.0
    calculation: dict = field(default_factory=dict)
    associations: list = field(default_factory=list)
    reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _trend(counts: list[int]) -> str:
    if len(counts) < 3 or sum(counts) == 0:
        return "insufficient_periods"
    y = np.asarray(counts, dtype=float)
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    rel = slope / max(y.mean(), 1e-9)
    if rel > 0.05:
        return "rising"
    if rel < -0.05:
        return "falling"
    return "stable"


def _acceleration(counts: list[int]) -> float | None:
    if len(counts) < 3:
        return None
    a, b, c = counts[-3], counts[-2], counts[-1]
    if a == 0 or b == 0:
        return None
    return round(((c - b) / b - (b - a) / a) * 100.0, 2)


def weekly_buckets(dates: list[datetime], end: datetime, n_buckets: int, bucket_days: int) -> list[int]:
    counts = [0] * n_buckets
    start = end - timedelta(days=bucket_days * n_buckets)
    for d in dates:
        if start < d <= end:
            idx = min(n_buckets - 1, int((d - start).total_seconds() // (bucket_days * 86400)))
            counts[idx] += 1
    return counts


def compute_radar(reviews: list[dict], themes: list[dict], params: RadarParams | None = None,
                  as_of: datetime | None = None) -> dict:
    """reviews: dicts with review_id, created_at (datetime), theme_id (or None), sentiment, confidence,
    similarity, and optional segment fields (app_version, platform).
    themes: dicts with theme_id, name."""
    p = params or RadarParams()
    valid = [r for r in reviews if isinstance(r.get("created_at"), datetime)]
    malformed = len(reviews) - len(valid)
    if not valid:
        return {"params": asdict(p), "formula": PRIORITY_FORMULA.format(cap=p.max_growth_bonus), "items": [],
                "window": None, "malformed_timestamps": malformed, "note": "no reviews with valid timestamps"}
    end = as_of or max(r["created_at"] for r in valid)
    cur_start = end - timedelta(days=p.window_days)
    prev_start = cur_start - timedelta(days=p.window_days)
    first = min(r["created_at"] for r in valid)
    n_buckets = max(1, math.ceil((end - first).total_seconds() / (p.trend_bucket_days * 86400)))

    in_cur = [r for r in valid if cur_start < r["created_at"] <= end]
    in_prev = [r for r in valid if prev_start < r["created_at"] <= cur_start]
    has_previous_window = first <= cur_start
    seg_base = {}
    for key in ("app_version", "platform"):
        vals = [r.get(key) for r in in_cur if r.get(key)]
        if vals:
            seg_base[key] = {v: vals.count(v) / len(vals) for v in set(vals)}

    items: list[RadarItem] = []
    for t in themes:
        tid = t["theme_id"]
        mem = [r for r in valid if r.get("theme_id") == tid]
        cur = [r for r in in_cur if r.get("theme_id") == tid]
        prev = [r for r in in_prev if r.get("theme_id") == tid]
        c, pv = len(cur), len(prev)
        neg_cur = [r for r in cur if r.get("sentiment") == "negative"]
        neg_ratio = len(neg_cur) / c if c else 0.0
        neg_all = sum(1 for r in mem if r.get("sentiment") == "negative") / len(mem) if mem else 0.0
        g = growth_pct(c, pv)
        weekly = weekly_buckets([r["created_at"] for r in mem], end, n_buckets, p.trend_bucket_days)
        evidence = sorted(neg_cur, key=lambda r: (-(r.get("similarity") or 0), -(r.get("confidence") or 0), r["review_id"]))
        ev_ids = [r["review_id"] for r in evidence[: p.n_evidence]]

        reasons = []
        if c == 0 and pv == 0:
            status = "NO_DATA"
        elif (neg_ratio if c else neg_all) < p.min_negative_ratio:
            status = "NOT_A_COMPLAINT"
            reasons.append(f"negative ratio {round((neg_ratio if c else neg_all) * 100, 1)}% < {p.min_negative_ratio * 100:.0f}%")
        elif c < p.min_current_mentions or len(neg_cur) < p.min_evidence_reviews:
            status = "INSUFFICIENT_EVIDENCE"
            reasons.append(f"current mentions {c} < {p.min_current_mentions}" if c < p.min_current_mentions
                           else f"negative evidence reviews {len(neg_cur)} < {p.min_evidence_reviews}")
        elif not has_previous_window:
            status = "INSUFFICIENT_EVIDENCE"
            reasons.append("batch does not cover a full previous window")
        elif pv == 0:
            status = "NEW"
            reasons.append(f"0 mentions in previous window, {c} in current window (>= {p.min_current_mentions})")
        elif g is not None and g >= p.min_growth_pct:
            status = "EMERGING"
            reasons.append(f"growth {g}% >= {p.min_growth_pct}% with {round(neg_ratio * 100, 1)}% negative and {c} mentions")
        elif g is not None and g <= -p.decline_pct:
            status = "DECLINING"
            reasons.append(f"growth {g}% <= -{p.decline_pct}%")
        else:
            status = "STABLE"
            reasons.append(f"growth {g}% within (-{p.decline_pct}%, {p.min_growth_pct}%)")

        gf = growth_factor(g, pv, c, p.max_growth_bonus)
        priority = round(len(neg_cur) * gf, 2)
        if pv == 0 and c > 0:
            glabel = "NEW"
        elif g is None:
            glabel = "n/a"
        else:
            glabel = f"{'+' if g > 0 else ''}{g:.0f}%"

        associations = []
        for key, base in seg_base.items():
            vals = [r.get(key) for r in cur if r.get(key)]
            for v in sorted(set(vals)):
                share = vals.count(v) / len(vals)
                lift = share / base[v] if base.get(v) else 0
                if vals.count(v) >= p.min_segment_support and lift >= p.min_segment_lift:
                    associations.append({
                        "segment": key, "value": v, "theme_share": round(share, 4), "overall_share": round(base[v], 4),
                        "lift": round(lift, 2), "support": vals.count(v),
                        "statement": f"{round(share * 100)}% of current-window {t['name']} mentions are {key}={v}, "
                                     f"vs {round(base[v] * 100)}% of all current-window reviews (lift {lift:.2f}). "
                                     f"This is an association, not evidence of cause.",
                    })

        calc = {
            "window": {"current": [cur_start.isoformat(), end.isoformat()], "previous": [prev_start.isoformat(), cur_start.isoformat()]},
            "growth": (f"({c} - {pv}) / {pv} x 100 = {g}%" if g is not None else
                       ("previous = 0 -> growth undefined, classified NEW" if c > 0 else "no mentions in either window")),
            "negative_ratio": f"{len(neg_cur)} / {c} = {round(neg_ratio * 100, 1)}%" if c else "no current mentions",
            "priority": f"{len(neg_cur)} x {gf:.2f} = {priority}",
            "thresholds": {"min_current_mentions": p.min_current_mentions, "min_growth_pct": p.min_growth_pct,
                           "min_negative_ratio": p.min_negative_ratio, "min_evidence_reviews": p.min_evidence_reviews},
        }
        items.append(RadarItem(
            theme_id=tid, name=t["name"], status=status, current_mentions=c, previous_mentions=pv, growth_pct=g,
            growth_label=glabel, negative_mentions_current=len(neg_cur), negative_ratio=round(neg_ratio, 4),
            negative_ratio_all=round(neg_all, 4), total_mentions=len(mem), acceleration_pp=_acceleration(weekly),
            trend=_trend(weekly), weekly_counts=weekly, evidence_review_ids=ev_ids, priority=priority,
            calculation=calc, associations=sorted(associations, key=lambda a: -a["lift"])[:3], reasons=reasons,
        ))

    order = {"NEW": 0, "EMERGING": 1, "STABLE": 2, "DECLINING": 3, "INSUFFICIENT_EVIDENCE": 4, "NOT_A_COMPLAINT": 5, "NO_DATA": 6}
    items.sort(key=lambda i: (order[i.status], -i.priority, i.theme_id))
    return {
        "params": asdict(p),
        "formula": PRIORITY_FORMULA.format(cap=p.max_growth_bonus),
        "rules": __doc__.split("Status rules")[1].split("Priority")[0].strip(),
        "window": {"current_start": cur_start.isoformat(), "end": end.isoformat(), "previous_start": prev_start.isoformat(),
                   "current_reviews": len(in_cur), "previous_reviews": len(in_prev), "has_previous_window": has_previous_window},
        "malformed_timestamps": malformed,
        "n_buckets": n_buckets,
        "items": [i.to_dict() for i in items],
    }
