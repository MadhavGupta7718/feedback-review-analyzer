"""Deterministic synthetic reviews for Adidas shoes (~15K at scale=1.42)."""
from __future__ import annotations

from datetime import datetime

from ml.data.product_batch import generate_product_reviews, write_csv
from ml.data.synthetic import SyntheticReview

START_DATE = datetime(2025, 2, 3)

WEEKLY_MEANS: dict[str, list[float]] = {
    # sole wear surges after a new production run (emerging)
    "sole_wear":            [35, 35, 38, 40, 42, 48, 55, 75, 100, 140, 170, 190],
    # sizing QC issue appears late (new)
    "sizing_wrong":         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 65, 105],
    # price complaints fade after a sale (declining)
    "price_value":          [105, 100, 95, 90, 80, 65, 50, 40, 32, 28, 22, 18],
    "comfort_pain":         [90] * 12,
    "breathability":        [50] * 12,
    "color_fading":         [38] * 12,
    "shipping_packaging":   [28] * 12,
    "customer_support":     [32] * 12,
    "comfort_praise":       [145] * 12,
    "style_looks":          [100] * 12,
    "general_praise":       [155] * 12,
    "neutral_mixed":        [65] * 12,
    "wide_fit_request":     [2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6],
}

SENTIMENT_MIX: dict[str, tuple[float, float, float]] = {
    "sole_wear": (0.88, 0.08, 0.04),
    "sizing_wrong": (0.90, 0.08, 0.02),
    "price_value": (0.80, 0.15, 0.05),
    "comfort_pain": (0.88, 0.08, 0.04),
    "breathability": (0.72, 0.20, 0.08),
    "color_fading": (0.75, 0.18, 0.07),
    "shipping_packaging": (0.75, 0.18, 0.07),
    "customer_support": (0.85, 0.10, 0.05),
    "comfort_praise": (0.02, 0.08, 0.90),
    "style_looks": (0.03, 0.10, 0.87),
    "general_praise": (0.02, 0.08, 0.90),
    "neutral_mixed": (0.10, 0.80, 0.10),
    "wide_fit_request": (0.10, 0.70, 0.20),
}

PHRASES: dict[str, dict[str, list[str]]] = {
    "sole_wear": {
        "negative": [
            "Adidas sole wore out after only a few weeks of walking",
            "outsole on these Adidas shoes is peeling already",
            "tread is gone too fast for Adidas quality I expected",
            "shoes look new on top but the Adidas sole is shredded",
            "rubber sole cracked near the toe on my Adidas pair",
            "Adidas grip disappeared after light outdoor use",
        ],
        "neutral": [
            "Adidas sole wear seems a bit faster than my last pair",
            "noticing early sole wear on these Adidas shoes",
        ],
        "positive": [
            "mostly love the Adidas fit, sole wear is my only gripe",
        ],
        "detail": ["", "Size 9 UK.", "Mostly city walking.", "Started after the new batch."],
    },
    "sizing_wrong": {
        "negative": [
            "Adidas size runs totally wrong, ordered true to size and it does not fit",
            "these Adidas shoes are a full size smaller than labelled",
            "sizing chart failed me, Adidas pair is unwearably tight",
            "had to return Adidas shoes because size is inconsistent",
            "toe box on this Adidas model is wrongly sized",
        ],
        "neutral": [
            "Adidas sizing feels off compared with my usual size",
            "might need a half size up in this Adidas line",
        ],
        "positive": [
            "style is great, only the Adidas sizing confused me",
        ],
        "detail": ["", "Ordered online.", "Compared with Ultraboost.", "Exchange pending."],
    },
    "price_value": {
        "negative": [
            "Adidas shoes are overpriced for this build quality",
            "paying premium Adidas money for average durability",
            "not worth the MRP, better deals exist",
            "price jumped and Adidas quality did not",
        ],
        "neutral": [
            "Adidas is pricey but okay during a sale",
            "waiting for a discount on these Adidas shoes",
        ],
        "positive": [
            "a bit costly but Adidas still worth it for me",
        ],
        "detail": ["", "Bought during festival sale.", "Outlet price was better.", "Compared with Nike."],
    },
    "comfort_pain": {
        "negative": [
            "Adidas shoes hurt my heels after 20 minutes",
            "arch support is terrible, feet ache in these Adidas",
            "blisters from the Adidas collar rubbing my ankle",
            "too stiff out of the box and never broke in",
            "my knees feel the impact in these Adidas trainers",
        ],
        "neutral": [
            "Adidas comfort is average, need thicker socks",
            "slight hotspot on the right Adidas shoe",
        ],
        "positive": [
            "mostly comfortable Adidas pair after a short break-in",
        ],
        "detail": ["", "Wide feet.", "Used for gym.", "Stood all day at work."],
    },
    "breathability": {
        "negative": [
            "Adidas upper traps heat, feet get sweaty fast",
            "no airflow in these Adidas shoes on warm days",
            "mesh looks breathable but Adidas still feels hot",
        ],
        "neutral": [
            "Adidas breathability is okay indoors, warm outdoors",
            "average ventilation for an Adidas trainer",
        ],
        "positive": [
            "Adidas mesh keeps my feet cool enough",
        ],
        "detail": ["", "Summer use.", "Black colourway.", "Tried without socks once."],
    },
    "color_fading": {
        "negative": [
            "Adidas colour faded after two washes of the uppers",
            "white Adidas midsole yellowed very quickly",
            "dye transfer from these Adidas ruined my socks",
        ],
        "neutral": [
            "slight fading on the Adidas logo area",
            "colour holds okay if I clean Adidas gently",
        ],
        "positive": [
            "colour still looks fresh on my Adidas",
        ],
        "detail": ["", "Machine washed once by mistake.", "Kept indoors mostly.", "Light colourway."],
    },
    "shipping_packaging": {
        "negative": [
            "Adidas box arrived crushed and one shoe scuffed",
            "shipping delayed a week for my Adidas order",
            "wrong Adidas colour shipped in a damaged carton",
        ],
        "neutral": [
            "box was dented but Adidas shoes were fine",
            "delivery took longer than estimated",
        ],
        "positive": [
            "Adidas pair arrived sealed and on time",
        ],
        "detail": ["", "Amazon fulfillment.", "Seller pack was weak.", "Needed a replacement."],
    },
    "customer_support": {
        "negative": [
            "Adidas support ignored my sizing complaint",
            "refund for defective Adidas sole is still pending",
            "chat bot only, no human help on my Adidas return",
        ],
        "neutral": [
            "waiting on Adidas support about an exchange",
            "ticket opened, no update yet",
        ],
        "positive": [
            "Adidas support replaced my pair eventually",
        ],
        "detail": ["", "Opened a case last week.", "Warranty claim.", "Asked for store credit."],
    },
    "comfort_praise": {
        "negative": ["expected more cushioning from Adidas"],
        "neutral": ["Adidas comfort is fine, not cloud-like"],
        "positive": [
            "Adidas cushioning feels great for all-day wear",
            "super comfortable Adidas trainers for walking",
            "plush ride, my go-to Adidas pair now",
            "light and soft underfoot, love these Adidas",
        ],
        "detail": ["", "Daily commute.", "Recovered well after runs.", "Wide foot friendly."],
    },
    "style_looks": {
        "negative": ["looked different from the Adidas photos"],
        "neutral": ["Adidas design is fine, nothing flashy"],
        "positive": [
            "clean Adidas look, gets compliments",
            "stylish enough for casual and gym",
            "classic Adidas stripes never miss",
            "love the colourway on these Adidas shoes",
        ],
        "detail": ["", "Black/white.", "Matches my wardrobe.", "Photos were accurate."],
    },
    "general_praise": {
        "negative": ["expected more from the Adidas brand"],
        "neutral": ["Adidas is fine, does the job"],
        "positive": [
            "solid everyday Adidas shoes, will buy again",
            "reliable Adidas quality for the price I paid",
            "great all-rounder trainers from Adidas",
            "happy with this Adidas purchase overall",
        ],
        "detail": ["", "Second Adidas pair.", "Gift for my brother.", "Gym and errands."],
    },
    "neutral_mixed": {
        "negative": ["just meh, mixed experience with Adidas"],
        "neutral": [
            "mixed feelings on these Adidas shoes",
            "neither impressed nor disappointed, average Adidas",
            "three-star Adidas experience, middle of the road",
            "so-so results, nothing stood out",
            "balanced review: equal pros and cons on Adidas",
        ],
        "positive": ["pretty good Adidas pair with a few rough edges"],
        "detail": ["", "Giving three stars.", "Might try another model.", "Average trainer."],
    },
    "wide_fit_request": {
        "negative": ["no wide Adidas option in this model is annoying"],
        "neutral": [
            "wishlist: Adidas should offer a wide fit for this shoe",
            "curious if Adidas will add EE widths here",
            "feature request only: wider Adidas last someday",
        ],
        "positive": ["love Adidas, a wide size would make it perfect"],
        "detail": ["", "Not urgent.", "Wishlist only.", "Have wide feet."],
    },
}

LINE_BY_WEEK = [
    "ultraboost_22", "ultraboost_22", "ultraboost_22", "ultraboost_22",
    "ultraboost_23", "ultraboost_23", "ultraboost_23", "ultraboost_23",
    "ultraboost_23_refresh", "ultraboost_23_refresh",
    "ultraboost_23_refresh", "ultraboost_23_refresh",
]


def generate(seed: int = 11, pii_rate: float = 0.05, duplicate_rate: float = 0.006,
             scale: float = 1.45) -> list[SyntheticReview]:
    return generate_product_reviews(
        weekly_means=WEEKLY_MEANS,
        sentiment_mix=SENTIMENT_MIX,
        phrases=PHRASES,
        line_by_week=LINE_BY_WEEK,
        start_date=START_DATE,
        source="synthetic_adidas",
        id_prefix="A",
        platforms=["amazon", "myntra", "adidas_store", "flipkart"],
        platform_p=[0.45, 0.25, 0.15, 0.15],
        seed=seed,
        scale=scale,
        pii_rate=pii_rate,
        duplicate_rate=duplicate_rate,
    )


__all__ = ["generate", "write_csv", "START_DATE", "WEEKLY_MEANS"]
