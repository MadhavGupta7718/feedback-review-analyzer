"""Deterministic synthetic review generator for Dove shampoo (product reviews, not an app).

Same contract as the Nimbus generator: planted gt_theme / gt_sentiment / gt_pii and scripted
temporal patterns so a pipeline run can score that batch alone. Output is marked
source="synthetic_dove"; nothing here is a real customer review.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ml.data.synthetic import (
    CLOSERS,
    EMOJI,
    FIRST_NAMES,
    LAST_NAMES,
    OPENERS,
    RATING,
    SyntheticReview,
    _pii_snippet,
)

START_DATE = datetime(2025, 1, 6)
N_WEEKS = 12

# Poisson weekly means → ~11K at scale=1; use scale≈1.42 for ~15K
WEEKLY_MEANS: dict[str, list[float]] = {
    # surges after a formula refresh (emerging)
    "dry_hair":             [40, 40, 42, 45, 48, 50, 55, 70, 95, 130, 160, 180],
    # pump defects appear late with a new bottle lot (new)
    "bottle_pump":          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 70, 110],
    # price complaints fade after a promotion (declining)
    "price_value":          [110, 105, 100, 95, 85, 70, 55, 45, 35, 30, 25, 20],
    # steady high-volume complaint
    "scalp_irritation":     [95] * 12,
    "scent_too_strong":     [55] * 12,
    "residue_buildup":      [40] * 12,
    "lather_issues":        [35] * 12,
    "customer_support":     [30] * 12,
    "shipping_packaging":   [25] * 12,
    "moisturizing_praise":  [140] * 12,
    "scent_love":           [90] * 12,
    "general_praise":       [160] * 12,
    "neutral_mixed":        [70] * 12,
    # tiny wishlist theme (should often fail min-evidence as a complaint)
    "travel_size_request":  [2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6],
}

SENTIMENT_MIX: dict[str, tuple[float, float, float]] = {
    "dry_hair": (0.88, 0.08, 0.04),
    "bottle_pump": (0.92, 0.06, 0.02),
    "price_value": (0.80, 0.15, 0.05),
    "scalp_irritation": (0.90, 0.07, 0.03),
    "scent_too_strong": (0.70, 0.22, 0.08),
    "residue_buildup": (0.78, 0.16, 0.06),
    "lather_issues": (0.72, 0.20, 0.08),
    "customer_support": (0.85, 0.10, 0.05),
    "shipping_packaging": (0.75, 0.18, 0.07),
    "moisturizing_praise": (0.02, 0.08, 0.90),
    "scent_love": (0.03, 0.10, 0.87),
    "general_praise": (0.02, 0.08, 0.90),
    "neutral_mixed": (0.10, 0.80, 0.10),
    "travel_size_request": (0.10, 0.70, 0.20),
}

PHRASES: dict[str, dict[str, list[str]]] = {
    "dry_hair": {
        "negative": [
            "Dove shampoo left my hair dry and brittle after a week",
            "new Dove formula makes my ends feel straw-like",
            "hair feels dehydrated every time I use Dove",
            "Dove used to moisturize, now it strips my hair",
            "my curls are frizzy and dry since switching to Dove",
            "this Dove bottle dries my hair out badly",
            "after washing with Dove my hair snaps easily",
            "Dove shampoo ruined the softness I used to get",
        ],
        "neutral": [
            "Dove feels a bit drier on my hair than before",
            "noticed more dryness with this Dove formula, still usable",
            "mixed on moisture, Dove is okay but not rich",
        ],
        "positive": [
            "mostly like Dove, only a little dryness on the ends",
        ],
        "detail": ["", "I have color-treated hair.", "Started after the new label design.", "Using 3x a week.", "Tried the daily moisture line."],
    },
    "bottle_pump": {
        "negative": [
            "Dove pump broke after two uses and will not dispense",
            "shampoo bottle pump is stuck, Dove product wasted",
            "the Dove dispenser cracked and leaks everywhere",
            "pump mechanism on this Dove bottle is defective",
            "cannot get product out, Dove pump is jammed",
            "new Dove bottle pump sprays sideways uselessly",
        ],
        "neutral": [
            "Dove pump is stiff but still works if I press hard",
            "bottle pump on Dove feels flimsy",
        ],
        "positive": [
            "product is fine, only the Dove pump feels cheap",
        ],
        "detail": ["", "Bought a twin pack.", "Lot code on the bottom.", "Happened on the first refill."],
    },
    "price_value": {
        "negative": [
            "Dove shampoo is overpriced for what you get",
            "too expensive compared with store brands",
            "price jumped and Dove quality did not",
            "not worth the money anymore",
            "paying premium for Dove and still disappointed",
        ],
        "neutral": [
            "Dove is pricey but average in performance",
            "waiting for a sale on Dove, full price feels high",
        ],
        "positive": [
            "a bit costly but Dove still works for me",
        ],
        "detail": ["", "Bought at the supermarket.", "Family size bottle.", "Saw a cheaper multipack elsewhere."],
    },
    "scalp_irritation": {
        "negative": [
            "Dove shampoo made my scalp itchy and flaky",
            "burning scalp after washing with Dove",
            "irritation along my hairline from this Dove bottle",
            "my scalp is red and sore from Dove",
            "triggered dandruff-like flakes with Dove shampoo",
            "allergic-feeling itch every time I use Dove",
        ],
        "neutral": [
            "slight scalp tingle with Dove, not sure if irritation",
            "scalp feels sensitive after Dove, mild so far",
        ],
        "positive": [
            "hair looks fine, tiny itch that goes away",
        ],
        "detail": ["", "Sensitive scalp history.", "Rinsed thoroughly.", "Stopped after four washes."],
    },
    "scent_too_strong": {
        "negative": [
            "Dove scent is overpowering and gives me a headache",
            "fragrance is too strong, smells like perfume not shampoo",
            "cannot stand the artificial Dove smell",
            "scent lingers too long and feels chemical",
        ],
        "neutral": [
            "Dove fragrance is strong; okay if you like perfume scents",
            "scent is noticeable but fades after an hour",
        ],
        "positive": [
            "love Dove overall, scent is just a bit loud for me",
        ],
        "detail": ["", "Unscented would be better.", "Partner complained about the smell.", "Bathroom smells for hours."],
    },
    "residue_buildup": {
        "negative": [
            "Dove leaves a heavy residue and hair looks greasy by noon",
            "white buildup on my scalp from Dove shampoo",
            "hair feels coated and weighed down after Dove",
            "need a clarifying wash to undo Dove residue",
        ],
        "neutral": [
            "a little residue with Dove if I do not rinse long enough",
            "fine hair gets weighed down slightly by Dove",
        ],
        "positive": [
            "usually clean rinse, rare bit of residue",
        ],
        "detail": ["", "Hard water area.", "Used less product.", "Followed with conditioner."],
    },
    "lather_issues": {
        "negative": [
            "Dove barely lathers no matter how much I use",
            "foam is weak compared with older Dove bottles",
            "takes forever to work up a lather with this Dove",
        ],
        "neutral": [
            "lather is average, nothing special",
            "foams okay on second wash",
        ],
        "positive": [
            "lathers fine for me with Dove",
        ],
        "detail": ["", "Short hair.", "Thick curly hair.", "Used lukewarm water."],
    },
    "customer_support": {
        "negative": [
            "Dove support never replied about my damaged bottle",
            "refund request ignored by the seller for Dove shampoo",
            "customer service was unhelpful about a leaky Dove pack",
        ],
        "neutral": [
            "waiting on a reply from support about Dove",
            "chat bot only, no human yet",
        ],
        "positive": [
            "support eventually replaced my Dove bottle",
        ],
        "detail": ["", "Opened a ticket last week.", "Order was a multipack.", "Asked for a refund."],
    },
    "shipping_packaging": {
        "negative": [
            "Dove bottle arrived leaking all over the box",
            "packaging crushed and seal was broken",
            "shipment delayed a week and bottle was half empty",
        ],
        "neutral": [
            "box was dented but Dove bottle survived",
            "shipping took longer than estimated",
        ],
        "positive": [
            "arrived sealed and on time",
        ],
        "detail": ["", "Prime delivery.", "Warehouse packed poorly.", "Bubble wrap missing."],
    },
    "moisturizing_praise": {
        "negative": [
            "expected more moisture from Dove than I got",
        ],
        "neutral": [
            "Dove moisturizes okay, not dramatic",
        ],
        "positive": [
            "Dove leaves my hair soft and moisturized",
            "best moisture I have had from a drugstore shampoo",
            "hydrated my dry hair without weighing it down",
            "silky feel after every Dove wash",
            "my hair finally feels nourished with Dove",
            "moisture level is perfect for daily use",
        ],
        "detail": ["", "Dry hair type.", "Combined with Dove conditioner.", "Noticeable in a week."],
    },
    "scent_love": {
        "negative": [
            "wanted a lighter scent than this Dove",
        ],
        "neutral": [
            "Dove smell is fine, not memorable",
        ],
        "positive": [
            "love the fresh Dove scent, lasts all day",
            "smells clean and soft, not too soapy",
            "fragrance is gentle and pleasant",
            "get compliments on how my hair smells after Dove",
        ],
        "detail": ["", "Coconut variant.", "Classic Dove scent.", "Not overpowering for me."],
    },
    "general_praise": {
        "negative": [
            "expected more from the Dove brand",
        ],
        "neutral": [
            "Dove is fine, does the job",
        ],
        "positive": [
            "great everyday Dove shampoo, will repurchase",
            "reliable wash, hair looks healthy",
            "family favorite shampoo for years",
            "gentle enough for daily washing",
            "five stars, soft clean feel",
            "consistently good Dove quality",
        ],
        "detail": ["", "Been using Dove for months.", "Bought again.", "Works for the whole family."],
    },
    "neutral_mixed": {
        "negative": [
            "just meh, lots of small annoyances with Dove",
        ],
        "neutral": [
            "mixed feelings on Dove, some good some average",
            "neither impressed nor disappointed, just okay shampoo",
            "three-star Dove experience, middle of the road",
            "so-so results, nothing stood out",
            "neutral take: Dove is average for the price",
            "balanced review, equal pros and cons",
        ],
        "positive": [
            "pretty good overall with a few rough edges",
        ],
        "detail": ["", "Giving three stars.", "Might try another scent.", "Average drugstore option."],
    },
    "travel_size_request": {
        "negative": [
            "no travel size Dove option in stock is annoying",
        ],
        "neutral": [
            "feature request: please sell more Dove travel bottles",
            "would like a TSA-friendly Dove size someday",
            "curious if Dove will restock mini bottles",
        ],
        "positive": [
            "love Dove, a travel size would make it perfect",
        ],
        "detail": ["", "Not urgent.", "Wishlist only.", "Flying next month."],
    },
}

# Product "versions" = formula / packaging lines over time
LINE_BY_WEEK = ["classic", "classic", "classic", "classic",
                "daily_moisture", "daily_moisture", "daily_moisture", "daily_moisture",
                "daily_moisture_v2", "daily_moisture_v2",
                "daily_moisture_v2", "daily_moisture_v2"]


def generate(seed: int = 7, pii_rate: float = 0.05, duplicate_rate: float = 0.006,
             scale: float = 1.42) -> list[SyntheticReview]:
    """Generate Dove shampoo reviews (scale=1.42 ≈ 15K rows; deterministic for a given seed)."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    rng = np.random.default_rng(seed)
    records: list[tuple[datetime, str, str]] = []
    for week in range(N_WEEKS):
        for theme, means in WEEKLY_MEANS.items():
            mean = means[week] * scale
            n = int(rng.poisson(mean)) if mean > 0 else 0
            for _ in range(n):
                ts = START_DATE + timedelta(weeks=week, seconds=int(rng.integers(0, 7 * 24 * 3600)))
                sentiment = str(rng.choice(["negative", "neutral", "positive"], p=SENTIMENT_MIX[theme]))
                records.append((ts, theme, sentiment))
    records.sort(key=lambda r: r[0])

    reviews: list[SyntheticReview] = []
    for i, (ts, theme, sentiment) in enumerate(records, start=1):
        ph = PHRASES[theme]
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
        week = min(N_WEEKS - 1, (ts - START_DATE).days // 7)
        # retail channel as platform stand-in
        platform = str(rng.choice(["amazon", "walmart", "target", "grocery"], p=[0.55, 0.20, 0.15, 0.10]))
        reviews.append(SyntheticReview(
            review_id=f"D{i:05d}",
            created_at=ts.isoformat(),
            rating=int(rng.choice(RATING[sentiment])),
            platform=platform,
            app_version=LINE_BY_WEEK[week],
            text=text,
            source="synthetic_dove",
            gt_theme=theme,
            gt_sentiment=sentiment,
            gt_pii=";".join(pii_types),
        ))

    n = len(reviews)
    next_id = n + 1
    extra: list[SyntheticReview] = []
    for j in rng.choice(n, size=int(n * duplicate_rate), replace=False):
        src = reviews[int(j)]
        extra.append(SyntheticReview(**{**src.__dict__, "review_id": f"D{next_id:05d}"}))
        next_id += 1
    for bad in ["", "   ", None, None, "\n\t", ""]:
        src = reviews[int(rng.integers(0, n))]
        extra.append(SyntheticReview(**{**src.__dict__, "review_id": f"D{next_id:05d}", "text": bad,
                                        "gt_theme": "invalid", "gt_pii": ""}))
        next_id += 1
    reviews.extend(extra)
    reviews.sort(key=lambda r: r.created_at)
    return reviews


FIELDS = ["review_id", "created_at", "rating", "platform", "app_version", "text", "source",
          "gt_theme", "gt_sentiment", "gt_pii"]


def write_csv(reviews: list[SyntheticReview], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in reviews:
            w.writerow({k: getattr(r, k) for k in FIELDS})
    return path
