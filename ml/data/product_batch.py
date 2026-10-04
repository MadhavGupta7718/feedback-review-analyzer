"""Shared helpers for synthetic product-review batches (Dove / Adidas / Realme / …)."""
from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ml.data.synthetic import (
    CLOSERS,
    EMOJI,
    OPENERS,
    RATING,
    SyntheticReview,
    _pii_snippet,
)

FIELDS = ["review_id", "created_at", "rating", "platform", "app_version", "text", "source",
          "gt_theme", "gt_sentiment", "gt_pii"]

N_WEEKS = 12


def generate_product_reviews(
    *,
    weekly_means: dict[str, list[float]],
    sentiment_mix: dict[str, tuple[float, float, float]],
    phrases: dict[str, dict[str, list[str]]],
    line_by_week: list[str],
    start_date: datetime,
    source: str,
    id_prefix: str,
    platforms: list[str],
    platform_p: list[float],
    seed: int = 42,
    scale: float = 1.42,
    pii_rate: float = 0.05,
    duplicate_rate: float = 0.006,
) -> list[SyntheticReview]:
    if scale <= 0:
        raise ValueError("scale must be positive")
    if len(line_by_week) != N_WEEKS:
        raise ValueError(f"line_by_week must have {N_WEEKS} entries")
    rng = np.random.default_rng(seed)
    records: list[tuple[datetime, str, str]] = []
    for week in range(N_WEEKS):
        for theme, means in weekly_means.items():
            mean = means[week] * scale
            n = int(rng.poisson(mean)) if mean > 0 else 0
            for _ in range(n):
                ts = start_date + timedelta(weeks=week, seconds=int(rng.integers(0, 7 * 24 * 3600)))
                sentiment = str(rng.choice(["negative", "neutral", "positive"], p=sentiment_mix[theme]))
                records.append((ts, theme, sentiment))
    records.sort(key=lambda r: r[0])

    reviews: list[SyntheticReview] = []
    for i, (ts, theme, sentiment) in enumerate(records, start=1):
        ph = phrases[theme]
        core = str(rng.choice(ph[sentiment]))
        if rng.random() < 0.5:
            core = core[0].upper() + core[1:]
        parts = [str(rng.choice(OPENERS[sentiment])), core + ("." if rng.random() < 0.7 else ""),
                 str(rng.choice(ph["detail"]))]
        pii_types: list[str] = []
        if rng.random() < pii_rate:
            snippet, kind = _pii_snippet(rng)
            parts.append(snippet + ".")
            pii_types.append(kind)
        parts.append(str(rng.choice(CLOSERS[sentiment])))
        if rng.random() < 0.08:
            parts.append(str(rng.choice(EMOJI[sentiment])))
        text = " ".join(p for p in parts if p).strip()
        if rng.random() < 0.06:
            text = text.lower()
        if sentiment == "negative" and rng.random() < 0.04:
            text = text.rstrip(".!") + "!!!!!"
        week = min(N_WEEKS - 1, (ts - start_date).days // 7)
        platform = str(rng.choice(platforms, p=platform_p))
        reviews.append(SyntheticReview(
            review_id=f"{id_prefix}{i:05d}",
            created_at=ts.isoformat(),
            rating=int(rng.choice(RATING[sentiment])),
            platform=platform,
            app_version=line_by_week[week],
            text=text,
            source=source,
            gt_theme=theme,
            gt_sentiment=sentiment,
            gt_pii=";".join(pii_types),
        ))

    n = len(reviews)
    next_id = n + 1
    extra: list[SyntheticReview] = []
    for j in rng.choice(n, size=int(n * duplicate_rate), replace=False):
        src = reviews[int(j)]
        extra.append(SyntheticReview(**{**src.__dict__, "review_id": f"{id_prefix}{next_id:05d}"}))
        next_id += 1
    for bad in ["", "   ", None, None, "\n\t", ""]:
        src = reviews[int(rng.integers(0, n))]
        extra.append(SyntheticReview(**{**src.__dict__, "review_id": f"{id_prefix}{next_id:05d}", "text": bad,
                                        "gt_theme": "invalid", "gt_pii": ""}))
        next_id += 1
    reviews.extend(extra)
    reviews.sort(key=lambda r: r.created_at)
    return reviews


def write_csv(reviews: list[SyntheticReview], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in reviews:
            w.writerow({k: getattr(r, k) for k in FIELDS})
    return path
