"""Deterministic synthetic app-store review generator for "Nimbus" (a fictional shopping + wallet app).

Why synthetic data exists in this project
  Real CSVs often lack theme / Complaint-Radar ground truth. This generator plants KNOWN labels
  (theme, sentiment, PII, and scripted temporal patterns) so every downstream stage can be scored
  against the same batch that is stored in that batch's analytics DB. All output is marked
  source="synthetic"; nothing generated here is ever presented as a real customer review.

Planted temporal patterns (12 weekly periods)
  battery_drain       flat, then surges after the v5.2 release (emerging)
  payment_failure     absent, appears only in the last two weeks with v5.3 (new / emerging)
  login_problems      high, then declining after the v5.1 fix
  app_crashes         high volume, flat (high volume / low growth)
  dark_mode_request   tiny volume that grows fast (must NOT pass the minimum-evidence gate)
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

START_DATE = datetime(2026, 6, 1)
N_WEEKS = 12

# Expected reviews per week for each theme (Poisson means at scale=1 ≈ 10K rows). Index 0 = week 1.
# Default generate(scale=2) ≈ 20K rows for a richer demo and stabler recall estimates.
WEEKLY_MEANS: dict[str, list[float]] = {
    "battery_drain":        [30, 30, 30, 30, 30, 30, 30, 30, 45, 65, 110, 140],
    "payment_failure":      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 95],
    "login_problems":       [120, 120, 120, 120, 100, 80, 60, 50, 40, 35, 30, 25],
    "app_crashes":          [130] * 12,
    "slow_performance":     [50, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70],
    "notification_issues":  [35] * 12,
    "customer_support":     [45] * 12,
    "delivery_delays":      [55] * 12,
    "subscription_pricing": [40] * 12,
    "ui_praise":            [90] * 12,
    "general_praise":       [170] * 12,
    "dark_mode_request":    [2, 2, 2, 2, 2, 3, 3, 4, 4, 5, 6, 8],
    "neutral_mixed":        [90] * 12,
    # richer long-tail themes (scale>1 makes them large enough for clustering)
    "search_broken":        [18] * 12,
    "photo_upload":         [22] * 12,
    "privacy_ads":          [15] * 12,
}

# P(negative), P(neutral), P(positive) per theme
SENTIMENT_MIX: dict[str, tuple[float, float, float]] = {
    "battery_drain": (0.85, 0.10, 0.05),
    "payment_failure": (0.92, 0.06, 0.02),
    "login_problems": (0.85, 0.10, 0.05),
    "app_crashes": (0.88, 0.08, 0.04),
    "slow_performance": (0.75, 0.18, 0.07),
    "notification_issues": (0.65, 0.28, 0.07),
    "customer_support": (0.85, 0.10, 0.05),
    "delivery_delays": (0.82, 0.13, 0.05),
    "subscription_pricing": (0.78, 0.17, 0.05),
    "ui_praise": (0.02, 0.08, 0.90),
    "general_praise": (0.01, 0.06, 0.93),
    "dark_mode_request": (0.10, 0.70, 0.20),
    "neutral_mixed": (0.10, 0.80, 0.10),
    "search_broken": (0.80, 0.15, 0.05),
    "photo_upload": (0.78, 0.16, 0.06),
    "privacy_ads": (0.70, 0.22, 0.08),
}

PHRASES: dict[str, dict[str, list[str]]] = {
    "battery_drain": {
        "negative": [
            "the app drains my battery like crazy",
            "battery drains so fast since the last update",
            "my phone battery dies within a few hours when this app is installed",
            "this app is killing my battery even in the background",
            "battery usage went through the roof after updating",
            "phone gets hot and the battery drops 30% in an hour",
            "battery life is terrible because of this app",
            "the app eats battery nonstop, it shows 40% usage in settings",
            "my battery goes from full to empty by lunch",
            "background battery drain is insane now",
        ],
        "neutral": [
            "battery usage seems a bit higher since the update",
            "noticed the app uses more battery than before",
            "battery consumption is somewhat higher than other apps",
        ],
        "positive": [
            "love the app but it does drain the battery a little",
            "great features, just wish it used less battery",
        ],
        "detail": [
            "", "Checked battery settings and Nimbus is at the top.", "Started right after version 5.2.",
            "Happens even when I'm not using it.", "Never had this problem before.", "Phone is only a year old.",
        ],
    },
    "payment_failure": {
        "negative": [
            "payment keeps failing at checkout",
            "my card gets declined in the app but works everywhere else",
            "checkout fails with an error every time I try to pay",
            "I was charged twice and the order still failed",
            "payment page just spins and then says transaction failed",
            "cannot complete any purchase, payment error every time",
            "the wallet top-up fails and money disappears for days",
            "apple pay and google pay both fail at checkout now",
            "payment was taken but the app says it failed",
        ],
        "neutral": [
            "had one payment failure at checkout, second attempt worked",
            "payment took a few tries to go through",
        ],
        "positive": [
            "usually great, but checkout payment failed once today",
        ],
        "detail": [
            "", "Tried three different cards.", "Error code PAY-502 shows up.", "Started with the latest update.",
            "Bank says the charge is pending.", "Lost a flash sale because of this.",
        ],
    },
    "login_problems": {
        "negative": [
            "I can't log in to my account anymore",
            "login keeps failing even with the correct password",
            "the app logs me out every single time I open it",
            "stuck on the login screen forever",
            "password reset email never arrives so I can't sign in",
            "two factor code never comes through, locked out of my account",
            "sign in with google just loops back to the login page",
            "keeps saying invalid credentials when my password is right",
        ],
        "neutral": [
            "had to log in again a couple of times this week",
            "login was slow today but worked eventually",
        ],
        "positive": [
            "nice app, though I get logged out now and then",
        ],
        "detail": [
            "", "Reinstalled twice.", "Cleared cache, no luck.", "Works on the website though.",
            "Very frustrating.", "Happens on wifi and mobile data.",
        ],
    },
    "app_crashes": {
        "negative": [
            "the app crashes every time I open it",
            "keeps crashing when I try to view my cart",
            "app freezes and then force closes",
            "crashes constantly on startup",
            "the app closes itself randomly while browsing",
            "crash as soon as I tap on my orders",
            "it crashed three times while I was checking out",
            "app keeps crashing after the splash screen",
        ],
        "neutral": [
            "app crashed once today but reopened fine",
            "occasional crash when switching tabs",
        ],
        "positive": [
            "really like it, just crashes once in a while",
        ],
        "detail": [
            "", "Phone is fully updated.", "Reinstalling didn't help.", "Happens on both my phone and tablet.",
            "Please fix.", "Lost my cart twice.",
        ],
    },
    "slow_performance": {
        "negative": [
            "the app is so slow and laggy",
            "pages take forever to load",
            "search results take ages to show up",
            "scrolling is choppy and everything lags",
            "it takes a full minute just to open the app",
            "super sluggish compared to a few months ago",
            "images load really slowly even on fast wifi",
        ],
        "neutral": [
            "a bit slow sometimes but usable",
            "loading could be faster",
        ],
        "positive": [
            "good app overall, a little slow to load",
        ],
        "detail": ["", "My internet is fine.", "Other apps are fast.", "Getting worse every update.", "Very annoying."],
    },
    "notification_issues": {
        "negative": [
            "too many push notifications, it's spam",
            "notifications don't arrive for my delivery updates",
            "I turned off notifications but still get them",
            "getting promo notifications at 3am",
            "order status notifications never show up",
        ],
        "neutral": [
            "notifications are a bit frequent",
            "would like more control over notifications",
        ],
        "positive": [
            "handy order notifications, maybe a bit too many",
        ],
        "detail": ["", "Checked my settings twice.", "Please add a quiet mode.", "Annoying."],
    },
    "customer_support": {
        "negative": [
            "customer support never replies to my emails",
            "support chat is useless, just a bot",
            "waited on hold for an hour and nobody answered",
            "customer service closed my ticket without solving anything",
            "support keeps sending the same copy paste answer",
            "impossible to reach a real person in support",
        ],
        "neutral": [
            "support took a few days to reply but helped",
            "customer service response was slow",
        ],
        "positive": [
            "support was helpful in the end, just slow",
        ],
        "detail": ["", "Opened three tickets.", "Still waiting after a week.", "Very disappointing service."],
    },
    "delivery_delays": {
        "negative": [
            "my order is two weeks late",
            "delivery keeps getting pushed back",
            "package never arrived and tracking hasn't updated",
            "shipping takes way longer than promised",
            "delivery estimate is always wrong",
            "the courier marked it delivered but I got nothing",
        ],
        "neutral": [
            "delivery was a couple of days late",
            "shipping could be quicker",
        ],
        "positive": [
            "good products, delivery just took a bit long",
        ],
        "detail": ["", "Tracking says in transit for days.", "Needed it for a birthday.", "Not the first time."],
    },
    "subscription_pricing": {
        "negative": [
            "the premium subscription is way too expensive",
            "they raised the price of Nimbus Plus again",
            "got charged for a subscription I cancelled",
            "hidden fees at checkout are ridiculous",
            "delivery fees went up and prices keep rising",
            "cancelling the subscription is deliberately hard",
        ],
        "neutral": [
            "subscription price is a bit high for what you get",
            "not sure the premium plan is worth it",
        ],
        "positive": [
            "premium is nice but pricey",
        ],
        "detail": ["", "Found cheaper elsewhere.", "Feels like a cash grab.", "Considering cancelling."],
    },
    "ui_praise": {
        "negative": ["the new design looks nice but is confusing"],
        "neutral": ["the interface is fine, nothing special", "design is clean enough"],
        "positive": [
            "the new design is beautiful and easy to use",
            "love the clean interface",
            "super intuitive layout, finding products is easy",
            "the app looks great and navigation is smooth",
            "really nice UI, everything is where I expect it",
            "the redesign made shopping so much easier",
        ],
        "detail": ["", "Great job team.", "Best shopping app design.", "Keep it up."],
    },
    "general_praise": {
        "negative": ["used to love this app, not anymore"],
        "neutral": ["it does the job", "decent app overall"],
        "positive": [
            "love this app",
            "best shopping app I've used",
            "great app, highly recommend",
            "amazing deals and a great experience",
            "I use Nimbus every week, fantastic",
            "five stars, works perfectly for me",
            "excellent app, never had any issues",
            "so convenient, saves me a lot of time",
        ],
        "detail": ["", "Thank you!", "Recommended it to my friends.", "Keep up the good work.", "Perfect."],
    },
    "dark_mode_request": {
        "negative": ["no dark mode in 2026 is unacceptable"],
        "neutral": [
            "feature request only: add a dark mode setting",
            "neutral note: dark theme option would be useful someday",
            "curious whether a dark mode toggle is planned",
            "dark mode is a common request, no urgency stated",
        ],
        "positive": ["great app, dark mode would make it perfect"],
        "detail": ["", "Not a complaint, just a wishlist item.", "Most apps have it now."],
    },
    "neutral_mixed": {
        "negative": ["it's just meh, lots of small annoyances"],
        "neutral": [
            "mixed feelings, some parts okay some not",
            "neither impressed nor disappointed, just average",
            "neutral experience overall, nothing stood out",
            "so-so app, average in every way",
            "three-star experience, neither good nor bad",
            "balanced review: equal pros and cons",
            "indifferent about this app, it is fine",
            "okay but unremarkable, middle of the road",
            "neither recommend nor discourage, just okay",
            "mediocre but usable, no strong opinion",
        ],
        "positive": ["pretty good overall, a few rough edges"],
        "detail": ["", "Giving three stars.", "No strong opinion either way.", "Average for the category."],
    },
    "search_broken": {
        "negative": [
            "in-app search returns nothing useful",
            "search is broken, cannot find products I know exist",
            "the search bar never finds the right item",
            "product search shows irrelevant results every time",
            "search feature is useless after the update",
            "typing in search freezes the app",
        ],
        "neutral": [
            "search results are mixed, sometimes okay",
            "search works for popular items but not niche ones",
        ],
        "positive": [
            "search is usually fine, one miss today",
        ],
        "detail": ["", "Tried brand names and SKUs.", "Happens on both spelling variants.", "Filters make it worse."],
    },
    "photo_upload": {
        "negative": [
            "cannot upload photos to my listing",
            "image upload fails with a generic error",
            "photos get stuck at 99% when uploading",
            "camera upload crashes every time",
            "product pictures will not sync to the gallery",
        ],
        "neutral": [
            "photo upload is slow but eventually works",
            "had to retry image upload twice",
        ],
        "positive": [
            "photo upload usually works, failed once",
        ],
        "detail": ["", "Tried wifi and mobile data.", "HEIC and JPG both fail.", "Happens on the latest version."],
    },
    "privacy_ads": {
        "negative": [
            "too many personalised ads after I opted out",
            "privacy settings do nothing, still tracked for ads",
            "targeted ads feel invasive in this app",
            "I disabled ad personalisation but ads got worse",
            "ads follow me across screens, creepy tracking",
        ],
        "neutral": [
            "ads are frequent but expected for a free app",
            "privacy page is confusing about ad tracking",
        ],
        "positive": [
            "like the app, wish there were fewer ads",
        ],
        "detail": ["", "Checked privacy toggles twice.", "Started after the ads SDK update.", "No clear opt-out."],
    },
}

OPENERS = {
    "negative": ["", "", "Ugh.", "Really disappointed.", "Terrible experience.", "Not happy.", "Seriously,", "Honestly,", "Worst update ever."],
    "neutral": ["", "", "Neutral take:", "For the record,", "FYI,", "Quick note:"],
    "positive": ["", "", "Wow!", "Honestly,", "So happy.", "Great news:"],
}
CLOSERS = {
    "negative": ["", "", "Please fix this asap.", "Uninstalling until it's fixed.", "1 star until resolved.", "Very frustrating!!", "Fix it please."],
    "neutral": ["", "", "No strong feelings either way.", "Three stars.", "That's all."],
    "positive": ["", "", "Thanks!", "Love it.", "Keep it up!"],
}
EMOJI = {"negative": ["😡", "😤", "👎", "🔋💀", "😞"], "neutral": ["", "🤷"], "positive": ["❤️", "👍", "😍", "🎉"]}
RATING = {"negative": [1, 1, 2], "neutral": [3], "positive": [4, 5, 5]}

FIRST_NAMES = ["John", "Priya", "Maria", "Chen", "Aisha", "David", "Emma", "Rahul", "Sofia", "Liam", "Fatima", "Carlos"]
LAST_NAMES = ["Carter", "Sharma", "Garcia", "Wang", "Khan", "Miller", "Rossi", "Patel", "Novak", "Okafor", "Silva", "Kim"]


@dataclass
class SyntheticReview:
    review_id: str
    created_at: str
    rating: int
    platform: str
    app_version: str
    text: str | None
    source: str
    gt_theme: str
    gt_sentiment: str
    gt_pii: str


def _version_for_week(week: int) -> str:
    if week < 4:
        return "5.0"
    if week < 8:
        return "5.1"
    if week < 10:
        return "5.2"
    return "5.3"


def _pii_snippet(rng: np.random.Generator) -> tuple[str, str]:
    kind = rng.choice(["email", "phone", "order_id", "account_id", "card", "person", "url", "customer_id", "handle"])
    first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
    if kind == "email":
        return f"Email me at {first.lower()}.{last.lower()}{rng.integers(1, 99)}@example.com", kind
    if kind == "phone":
        fmt = rng.integers(0, 3)
        if fmt == 0:
            return f"Call me on +1 555-{rng.integers(100, 999)}-{rng.integers(1000, 9999)}", kind
        if fmt == 1:
            return f"my number is (555) {rng.integers(100, 999)}-{rng.integers(1000, 9999)}", kind
        return f"reach me at +44 7700 900{rng.integers(100, 999)}", kind
    if kind == "order_id":
        return f"Order ORD-{rng.integers(100000, 999999)} is the one", kind
    if kind == "account_id":
        return f"my account ID is ACC-{rng.integers(10000000, 99999999)}", kind
    if kind == "customer_id":
        return f"customer id CUST-{rng.integers(100000, 999999)}", kind
    if kind == "card":
        card = rng.choice(["4111 1111 1111 1111", "5555-5555-5555-4444", "4012888888881881", "3782 822463 10005"])
        return f"used card {card}", kind
    if kind == "person":
        return f"This is {first} {last} writing", kind
    if kind == "url":
        return f"screenshot here https://imgur.example/{rng.integers(10000, 99999)}", kind
    return f"@{first.lower()}_{last.lower()}{rng.integers(1, 999)} said the same", kind


def generate(seed: int = 42, pii_rate: float = 0.06, duplicate_rate: float = 0.008,
             scale: float = 2.0) -> list[SyntheticReview]:
    """Generate planted Nimbus reviews. scale=1 ≈ 10K rows; scale=2 (default) ≈ 20K richer rows."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    rng = np.random.default_rng(seed)
    records: list[tuple[datetime, str, str, str]] = []  # (timestamp, theme, sentiment, text-ish)
    reviews: list[SyntheticReview] = []

    for week in range(N_WEEKS):
        for theme, means in WEEKLY_MEANS.items():
            mean = means[week] * scale
            n = int(rng.poisson(mean)) if mean > 0 else 0
            for _ in range(n):
                ts = START_DATE + timedelta(weeks=week, seconds=int(rng.integers(0, 7 * 24 * 3600)))
                sentiment = str(rng.choice(["negative", "neutral", "positive"], p=SENTIMENT_MIX[theme]))
                records.append((ts, theme, sentiment, ""))

    records.sort(key=lambda r: r[0])
    for i, (ts, theme, sentiment, _) in enumerate(records, start=1):
        ph = PHRASES[theme]
        core = str(rng.choice(ph[sentiment]))
        if rng.random() < 0.5:
            core = core[0].upper() + core[1:]
        parts = [str(rng.choice(OPENERS[sentiment])), core + ("." if rng.random() < 0.7 else ""), str(rng.choice(ph["detail"]))]
        pii_types: list[str] = []
        if rng.random() < pii_rate:
            snippet, kind = _pii_snippet(rng)
            parts.append(snippet + ".")
            pii_types.append(kind)
        parts.append(str(rng.choice(CLOSERS[sentiment])))
        if rng.random() < 0.10:
            parts.append(str(rng.choice(EMOJI[sentiment])))
        text = " ".join(p for p in parts if p).strip()
        if rng.random() < 0.05:
            text = text.lower()
        if sentiment == "negative" and rng.random() < 0.04:
            text = text.rstrip(".!") + "!!!!!"
        week = (ts - START_DATE).days // 7
        platform = "android" if rng.random() < (0.72 if theme == "battery_drain" else 0.58) else "ios"
        reviews.append(SyntheticReview(
            review_id=f"R{i:05d}",
            created_at=ts.isoformat(),
            rating=int(rng.choice(RATING[sentiment])),
            platform=platform,
            app_version=_version_for_week(week),
            text=text,
            source="synthetic",
            gt_theme=theme,
            gt_sentiment=sentiment,
            gt_pii=";".join(pii_types),
        ))

    # Data-quality defects the preprocessing stage must handle explicitly.
    n = len(reviews)
    dup_src = rng.choice(n, size=int(n * duplicate_rate), replace=False)
    next_id = n + 1
    extra: list[SyntheticReview] = []
    for j in dup_src:
        src = reviews[int(j)]
        extra.append(SyntheticReview(**{**src.__dict__, "review_id": f"R{next_id:05d}"}))
        next_id += 1
    for k, bad in enumerate(["", "   ", None, None, "\n\t", "", None, "  "]):
        src = reviews[int(rng.integers(0, n))]
        extra.append(SyntheticReview(**{**src.__dict__, "review_id": f"R{next_id:05d}", "text": bad,
                                        "gt_theme": "invalid", "gt_pii": ""}))
        next_id += 1
    for k in range(6):
        src = reviews[int(rng.integers(0, n))]
        mojibake = (src.text or "").replace("'", "\u00e2\u20ac\u2122") + " donâ€™t"
        extra.append(SyntheticReview(**{**src.__dict__, "review_id": f"R{next_id:05d}", "text": mojibake}))
        next_id += 1
    reviews.extend(extra)
    reviews.sort(key=lambda r: r.created_at)
    return reviews


FIELDS = ["review_id", "created_at", "rating", "platform", "app_version", "text", "source", "gt_theme", "gt_sentiment", "gt_pii"]


def write_csv(reviews: list[SyntheticReview], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in reviews:
            row = dict(r.__dict__)
            row["text"] = "" if r.text is None else r.text
            w.writerow(row)
