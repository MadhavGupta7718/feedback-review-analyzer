"""Deterministic synthetic reviews for Realme earbuds (~15K at scale=1.42)."""
from __future__ import annotations

from datetime import datetime

from ml.data.product_batch import generate_product_reviews, write_csv
from ml.data.synthetic import SyntheticReview

START_DATE = datetime(2025, 3, 3)

WEEKLY_MEANS: dict[str, list[float]] = {
    # battery complaints surge after a firmware period (emerging)
    "earbud_battery":       [32, 32, 35, 38, 42, 48, 60, 85, 110, 145, 175, 195],
    # pairing failures appear with a new lot (new)
    "pairing_failure":      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 70, 115],
    # price complaints fade after a price cut (declining)
    "price_value":          [100, 95, 90, 85, 75, 60, 48, 38, 30, 26, 20, 16],
    "sound_quality":        [95] * 12,
    "mic_call_quality":     [55] * 12,
    "ear_fit_comfort":      [48] * 12,
    "charging_case":        [40] * 12,
    "customer_support":     [30] * 12,
    "shipping_packaging":   [26] * 12,
    "bass_praise":          [130] * 12,
    "general_praise":       [160] * 12,
    "neutral_mixed":        [68] * 12,
    "anc_request":          [2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 6, 7],
}

SENTIMENT_MIX: dict[str, tuple[float, float, float]] = {
    "earbud_battery": (0.88, 0.08, 0.04),
    "pairing_failure": (0.92, 0.06, 0.02),
    "price_value": (0.80, 0.15, 0.05),
    "sound_quality": (0.78, 0.14, 0.08),
    "mic_call_quality": (0.82, 0.12, 0.06),
    "ear_fit_comfort": (0.70, 0.22, 0.08),
    "charging_case": (0.80, 0.14, 0.06),
    "customer_support": (0.85, 0.10, 0.05),
    "shipping_packaging": (0.75, 0.18, 0.07),
    "bass_praise": (0.03, 0.10, 0.87),
    "general_praise": (0.02, 0.08, 0.90),
    "neutral_mixed": (0.10, 0.80, 0.10),
    "anc_request": (0.10, 0.70, 0.20),
}

PHRASES: dict[str, dict[str, list[str]]] = {
    "earbud_battery": {
        "negative": [
            "Realme earbuds battery dies in under two hours now",
            "case charge drains overnight with these Realme buds",
            "left bud battery is much weaker than the right on Realme",
            "Realme battery life dropped hard after the update",
            "cannot finish a commute on one Realme charge",
            "Realme buds show 100% then die suddenly",
        ],
        "neutral": [
            "Realme battery seems a bit shorter than advertised",
            "average battery on these Realme earbuds",
        ],
        "positive": [
            "sound is fine, only Realme battery could be better",
        ],
        "detail": ["", "Used with ANC off.", "Android phone.", "Started after firmware update."],
    },
    "pairing_failure": {
        "negative": [
            "Realme earbuds will not pair with my phone at all",
            "keeps disconnecting every few minutes, Realme pairing is broken",
            "one Realme bud never connects after reset",
            "Bluetooth pairing loop on these Realme earbuds",
            "Realme buds forget the device after every charge",
        ],
        "neutral": [
            "had to re-pair Realme buds twice this week",
            "pairing is flaky on crowded Wi-Fi, Realme otherwise okay",
        ],
        "positive": [
            "audio is good when connected, Realme pairing needs work",
        ],
        "detail": ["", "Tried reset.", "iOS and Android both tested.", "New unit from sealed box."],
    },
    "price_value": {
        "negative": [
            "Realme earbuds feel overpriced after the battery issues",
            "not worth the money compared with other TWS options",
            "cheaper Realme models performed better for me",
            "price cut still does not fix the value gap",
        ],
        "neutral": [
            "okay value for Realme during a sale only",
            "mid-range price, mid-range Realme experience",
        ],
        "positive": [
            "fair price for what Realme offers when it works",
        ],
        "detail": ["", "Bought on Flipkart sale.", "Compared with boat.", "Exchange window open."],
    },
    "sound_quality": {
        "negative": [
            "Realme sound is muddy and lacks clarity",
            "treble is harsh on these Realme earbuds",
            "music sounds compressed compared with ads",
            "Realme EQ presets cannot fix the flat sound",
        ],
        "neutral": [
            "Realme sound is average for the price",
            "okay for podcasts, weak for detailed music",
        ],
        "positive": [
            "clear enough Realme sound for daily listening",
        ],
        "detail": ["", "Tried Bass Boost.", "AAC codec.", "Volume at 70%."],
    },
    "mic_call_quality": {
        "negative": [
            "callers cannot hear me on Realme earbuds outdoors",
            "mic picks up wind noise badly on Realme",
            "voice sounds distant during meetings with Realme buds",
            "Realme dual mic promise does not hold up",
        ],
        "neutral": [
            "indoor calls are fine, outdoor Realme mic is average",
            "mic quality is acceptable in a quiet room",
        ],
        "positive": [
            "colleagues said Realme call quality was clear indoors",
        ],
        "detail": ["", "Used for Zoom.", "Windy commute.", "Tried transparency mode."],
    },
    "ear_fit_comfort": {
        "negative": [
            "Realme buds fall out when I jog",
            "ear tips hurt after thirty minutes of Realme use",
            "no tip size fits my ears with these Realme earbuds",
        ],
        "neutral": [
            "fit is okay if I use the largest Realme tip",
            "slight pressure after long Realme sessions",
        ],
        "positive": [
            "Realme buds stay put and feel light",
        ],
        "detail": ["", "Tried all tip sizes.", "Gym use.", "Small ears."],
    },
    "charging_case": {
        "negative": [
            "Realme charging case lid hinge feels flimsy",
            "case does not charge the buds unless seated perfectly",
            "LED on the Realme case is confusing and inaccurate",
            "case battery percentage jumps randomly in the app",
        ],
        "neutral": [
            "Realme case is compact but finish feels cheap",
            "pocketable case, average build",
        ],
        "positive": [
            "case charges quickly enough for travel",
        ],
        "detail": ["", "USB-C cable included.", "Wireless charging not supported.", "Dropped once."],
    },
    "customer_support": {
        "negative": [
            "Realme support closed my ticket without fixing pairing",
            "warranty claim for dead Realme bud is stuck",
            "no useful help from Realme chat on battery drain",
        ],
        "neutral": [
            "waiting for Realme service centre appointment",
            "email reply was generic",
        ],
        "positive": [
            "Realme replaced my defective bud eventually",
        ],
        "detail": ["", "Opened app ticket.", "Invoice attached.", "Asked for refund."],
    },
    "shipping_packaging": {
        "negative": [
            "Realme box came open and seal was broken",
            "shipping delayed and package looked reused",
            "missing ear tips in the Realme accessory kit",
        ],
        "neutral": [
            "carton dented but Realme buds were okay",
            "delivery took longer than shown",
        ],
        "positive": [
            "sealed Realme pack arrived on time",
        ],
        "detail": ["", "Flipkart delivered.", "Seller was retailnet.", "Needed a reship."],
    },
    "bass_praise": {
        "negative": ["wanted cleaner mids from Realme"],
        "neutral": ["bass is fine, not thumping"],
        "positive": [
            "punchy Realme bass for the price",
            "love the low end on these Realme earbuds",
            "fun V-shaped sound, great for EDM on Realme",
            "bass boost actually works well on Realme",
        ],
        "detail": ["", "Hip-hop playlist.", "Musical app EQ.", "Volume warning ignored."],
    },
    "general_praise": {
        "negative": ["expected more polish from Realme"],
        "neutral": ["Realme buds are okay for the money"],
        "positive": [
            "solid daily Realme earbuds, would recommend",
            "good value TWS from Realme overall",
            "happy with this Realme purchase for commute and calls",
            "reliable enough Realme pair for the price",
        ],
        "detail": ["", "Second Realme audio product.", "Gifted to cousin.", "Used 4 hours a day."],
    },
    "neutral_mixed": {
        "negative": ["mixed bag, some Realme annoyances"],
        "neutral": [
            "mixed feelings on these Realme earbuds",
            "neither impressed nor disappointed, average Realme",
            "three-star Realme experience, middle of the road",
            "so-so results, nothing stood out",
            "balanced review: equal pros and cons on Realme",
        ],
        "positive": ["pretty good Realme buds with a few rough edges"],
        "detail": ["", "Giving three stars.", "Might try another model.", "Average TWS."],
    },
    "anc_request": {
        "negative": ["missing ANC on this Realme model is disappointing"],
        "neutral": [
            "wishlist: Realme should add stronger ANC on the next version",
            "curious if a Realme ANC firmware trick exists",
            "feature request only: better noise cancel someday",
        ],
        "positive": ["like Realme overall, ANC would make it perfect"],
        "detail": ["", "Not urgent.", "Wishlist only.", "Fly often."],
    },
}

LINE_BY_WEEK = [
    "buds_air5", "buds_air5", "buds_air5", "buds_air5",
    "buds_air6", "buds_air6", "buds_air6", "buds_air6",
    "buds_air6_fw2", "buds_air6_fw2",
    "buds_air6_fw2", "buds_air6_fw2",
]


def generate(seed: int = 13, pii_rate: float = 0.05, duplicate_rate: float = 0.006,
             scale: float = 1.55) -> list[SyntheticReview]:
    return generate_product_reviews(
        weekly_means=WEEKLY_MEANS,
        sentiment_mix=SENTIMENT_MIX,
        phrases=PHRASES,
        line_by_week=LINE_BY_WEEK,
        start_date=START_DATE,
        source="synthetic_realme",
        id_prefix="E",
        platforms=["amazon", "flipkart", "realme_store", "croma"],
        platform_p=[0.40, 0.35, 0.15, 0.10],
        seed=seed,
        scale=scale,
        pii_rate=pii_rate,
        duplicate_rate=duplicate_rate,
    )


__all__ = ["generate", "write_csv", "START_DATE", "WEEKLY_MEANS"]
