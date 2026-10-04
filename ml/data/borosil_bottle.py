"""Deterministic synthetic reviews for Borosil water bottles (~15K at scale=1.5)."""
from __future__ import annotations

from datetime import datetime

from ml.data.product_batch import generate_product_reviews, write_csv
from ml.data.synthetic import SyntheticReview

START_DATE = datetime(2025, 2, 10)

WEEKLY_MEANS: dict[str, list[float]] = {
    # insulation complaints surge after a new vacuum batch (emerging)
    "insulation_leak":      [28, 30, 32, 35, 40, 48, 62, 88, 120, 150, 175, 190],
    # lid/seal defects appear late (new)
    "lid_seal_leak":        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 65, 105],
    # price complaints fade after a sale (declining)
    "price_value":          [95, 90, 85, 80, 70, 55, 45, 35, 28, 22, 18, 14],
    "durability_dents":     [70] * 12,
    "taste_odor":           [50] * 12,
    "cleaning_mouth":       [42] * 12,
    "capacity_size":        [38] * 12,
    "customer_support":     [28] * 12,
    "shipping_packaging":   [24] * 12,
    "keeps_cold_praise":    [125] * 12,
    "general_praise":       [155] * 12,
    "neutral_mixed":        [65] * 12,
    "straw_lid_request":    [2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7],
}

SENTIMENT_MIX: dict[str, tuple[float, float, float]] = {
    "insulation_leak": (0.88, 0.08, 0.04),
    "lid_seal_leak": (0.92, 0.06, 0.02),
    "price_value": (0.80, 0.15, 0.05),
    "durability_dents": (0.78, 0.14, 0.08),
    "taste_odor": (0.82, 0.12, 0.06),
    "cleaning_mouth": (0.72, 0.20, 0.08),
    "capacity_size": (0.55, 0.30, 0.15),
    "customer_support": (0.85, 0.10, 0.05),
    "shipping_packaging": (0.75, 0.18, 0.07),
    "keeps_cold_praise": (0.03, 0.10, 0.87),
    "general_praise": (0.02, 0.08, 0.90),
    "neutral_mixed": (0.10, 0.80, 0.10),
    "straw_lid_request": (0.10, 0.70, 0.20),
}

PHRASES: dict[str, dict[str, list[str]]] = {
    "insulation_leak": {
        "negative": [
            "Borosil bottle does not keep water cold even for four hours",
            "ice melts too fast in this Borosil flask",
            "claimed 24-hour cold retention is false for my Borosil",
            "hot tea goes lukewarm in under two hours, Borosil insulation failed",
            "outer wall gets warm while drink is still hot — Borosil vacuum issue",
            "Borosil vacuum bottle lost insulation after a month",
        ],
        "neutral": [
            "Borosil insulation is okay but not as long as advertised",
            "average cold retention on this Borosil bottle",
        ],
        "positive": [
            "build is fine, only Borosil insulation could be better",
        ],
        "detail": ["", "Used with ice cubes.", "Office AC room.", "Compared with Hydro Flask."],
    },
    "lid_seal_leak": {
        "negative": [
            "Borosil lid leaks in my bag every day",
            "screw cap seal on Borosil is defective and drips",
            "water seeps from the Borosil mouth when tipped",
            "new Borosil bottle arrived with a warped lid seal",
            "Borosil sipper gasket came loose after two washes",
        ],
        "neutral": [
            "had to tighten the Borosil lid harder than expected",
            "slight drip if Borosil bottle is sideways overnight",
        ],
        "positive": [
            "bottle is good when upright, Borosil lid needs a better seal",
        ],
        "detail": ["", "Tried spare gasket.", "Backpack commute.", "Checked threads."],
    },
    "price_value": {
        "negative": [
            "Borosil bottle feels overpriced for the insulation I got",
            "not worth the money versus other steel bottles",
            "paid premium for Borosil and still disappointed",
            "cheaper flasks performed better than this Borosil",
        ],
        "neutral": [
            "okay Borosil value during a sale only",
            "mid-range price, mid-range Borosil experience",
        ],
        "positive": [
            "fair price for a Borosil bottle when it works",
        ],
        "detail": ["", "Amazon sale.", "Compared with Milton.", "Gift set."],
    },
    "durability_dents": {
        "negative": [
            "Borosil dented badly after one drop on tile",
            "powder coat chipped in a week of Borosil use",
            "bottom of Borosil bottle scratched and rust-looking spots",
            "steel body looks premium until the first dent",
        ],
        "neutral": [
            "Borosil shows light scuffs with daily bag use",
            "durable enough, not indestructible",
        ],
        "positive": [
            "Borosil survived a few knocks without leaking",
        ],
        "detail": ["", "Dropped once.", "School bag.", "Gym floor."],
    },
    "taste_odor": {
        "negative": [
            "metallic taste from the Borosil bottle ruins water",
            "plastic smell from Borosil lid never went away",
            "coffee flavor lingers forever in this Borosil flask",
            "Borosil leaves a weird aftertaste even after washing",
        ],
        "neutral": [
            "mild steel taste on day one with Borosil, faded later",
            "odor is okay if I wash Borosil daily",
        ],
        "positive": [
            "no aftertaste once Borosil was rinsed well",
        ],
        "detail": ["", "Used for lemon water.", "Tried baking soda wash.", "Only water now."],
    },
    "cleaning_mouth": {
        "negative": [
            "narrow Borosil mouth is impossible to clean properly",
            "mold risk in the Borosil lid threads",
            "brush does not reach the Borosil bottom easily",
            "dishwasher warped the Borosil plastic parts",
        ],
        "neutral": [
            "cleaning Borosil takes extra time but doable",
            "wide-mouth would help; current Borosil is manageable",
        ],
        "positive": [
            "cleaned my Borosil fine with a bottle brush",
        ],
        "detail": ["", "Hand wash only.", "Dishwasher attempt failed.", "Kids used it."],
    },
    "capacity_size": {
        "negative": [
            "750 ml Borosil is too small for office day",
            "Borosil feels bulky for a one-liter claim",
            "wish Borosil offered a true 1.5 L option",
        ],
        "neutral": [
            "size is okay for short trips, not full day",
            "Borosil capacity matches the label roughly",
        ],
        "positive": [
            "Borosil size fits my cup holder and bag",
        ],
        "detail": ["", "1L model.", "500 ml travel.", "Car holder."],
    },
    "customer_support": {
        "negative": [
            "Borosil support ignored my leak warranty claim",
            "replacement denied even with video of Borosil leaking",
            "chat bot only, no human for Borosil issues",
        ],
        "neutral": [
            "waiting on Borosil ticket response",
            "support asked for invoice photos",
        ],
        "positive": [
            "Borosil eventually replaced the defective lid",
        ],
        "detail": ["", "Opened ticket last week.", "Amazon seller vs brand.", "Warranty card."],
    },
    "shipping_packaging": {
        "negative": [
            "Borosil arrived with dented body and torn box",
            "no protective foam, Borosil scratched in transit",
            "shipment delayed and bottle seal was open",
        ],
        "neutral": [
            "box was dented but Borosil survived",
            "shipping took longer than estimated",
        ],
        "positive": [
            "arrived sealed and well packed",
        ],
        "detail": ["", "Prime delivery.", "Warehouse packed poorly.", "Gift wrap."],
    },
    "keeps_cold_praise": {
        "negative": [
            "expected longer cold hours from Borosil than I got",
        ],
        "neutral": [
            "Borosil keeps drinks cool enough for half a day",
        ],
        "positive": [
            "Borosil keeps ice water cold through my whole shift",
            "still icy after eight hours in this Borosil bottle",
            "best cold retention I have had in a steel flask",
            "hot coffee stayed hot till evening in Borosil",
            "insulation on this Borosil actually matches the claim",
        ],
        "detail": ["", "Filled with ice.", "Summer outdoors.", "Office thermos use."],
    },
    "general_praise": {
        "negative": ["expected more polish from Borosil"],
        "neutral": ["Borosil bottle is okay for the money"],
        "positive": [
            "solid everyday Borosil bottle, would recommend",
            "looks premium and works for school and office",
            "happy with this Borosil purchase overall",
            "reliable steel bottle from Borosil for the price",
            "family uses Borosil daily without complaints",
        ],
        "detail": ["", "Second Borosil product.", "Gifted to cousin.", "Used every day."],
    },
    "neutral_mixed": {
        "negative": ["mixed bag, some Borosil annoyances"],
        "neutral": [
            "mixed feelings on this Borosil bottle",
            "neither impressed nor disappointed, average Borosil",
            "three-star Borosil experience, middle of the road",
            "so-so results, nothing stood out",
            "balanced review: equal pros and cons on Borosil",
        ],
        "positive": ["pretty good Borosil flask with a few rough edges"],
        "detail": ["", "Giving three stars.", "Might try another model.", "Average flask."],
    },
    "straw_lid_request": {
        "negative": ["missing straw lid option on Borosil is annoying"],
        "neutral": [
            "wishlist: Borosil should sell a straw lid accessory",
            "curious if a Borosil spout lid exists for kids",
            "feature request only: easier sip lid someday",
        ],
        "positive": ["like Borosil overall, a straw lid would make it perfect"],
        "detail": ["", "Not urgent.", "Wishlist only.", "For kids."],
    },
}

LINE_BY_WEEK = [
    "hydra_pro", "hydra_pro", "hydra_pro", "hydra_pro",
    "hydra_pro_v2", "hydra_pro_v2", "hydra_pro_v2", "hydra_pro_v2",
    "hydra_pro_v2_seal", "hydra_pro_v2_seal",
    "hydra_pro_v2_seal", "hydra_pro_v2_seal",
]


def generate(seed: int = 19, pii_rate: float = 0.05, duplicate_rate: float = 0.006,
             scale: float = 1.5) -> list[SyntheticReview]:
    return generate_product_reviews(
        weekly_means=WEEKLY_MEANS,
        sentiment_mix=SENTIMENT_MIX,
        phrases=PHRASES,
        line_by_week=LINE_BY_WEEK,
        start_date=START_DATE,
        source="synthetic_borosil",
        id_prefix="B",
        platforms=["amazon", "flipkart", "borosil_store", "reliance_digital"],
        platform_p=[0.42, 0.33, 0.15, 0.10],
        seed=seed,
        scale=scale,
        pii_rate=pii_rate,
        duplicate_rate=duplicate_rate,
    )


__all__ = ["generate", "write_csv", "START_DATE", "WEEKLY_MEANS"]
