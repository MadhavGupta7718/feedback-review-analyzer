import math
from datetime import datetime, timedelta

import pytest

from ml.complaints.radar import RadarParams, compute_radar, growth_pct

END = datetime(2026, 8, 23, 12, 0, 0)
THEMES = [{"theme_id": "t1", "name": "Battery Drain"}]


def make(n: int, days_ago_lo: float, days_ago_hi: float, theme="t1", sentiment="negative", start_id=0, **extra):
    out = []
    for i in range(n):
        frac = (i + 0.5) / max(n, 1)
        ago = days_ago_lo + (days_ago_hi - days_ago_lo) * frac
        out.append({"review_id": f"R{start_id + i:05d}", "created_at": END - timedelta(days=ago), "theme_id": theme,
                    "sentiment": sentiment, "confidence": 0.9, "similarity": 0.8, **extra})
    return out


def windows(prev: int, cur: int, neg=True, older: int = 20):
    """older = reviews before the previous window so the batch covers >= 2 full windows."""
    s = "negative" if neg else "positive"
    return (make(older, 29, 40, sentiment=s, start_id=0) + make(prev, 14.01, 27.99, sentiment=s, start_id=1000)
            + make(cur, 0.01, 13.99, sentiment=s, start_id=5000))


def item(reviews, **kw):
    return compute_radar(reviews, THEMES, RadarParams(**kw), as_of=END)["items"][0]


@pytest.mark.parametrize("prev,cur,expected", [(100, 200, 100.0), (100, 50, -50.0), (100, 100, 0.0)])
def test_growth_formula(prev, cur, expected):
    assert growth_pct(cur, prev) == expected
    it = item(windows(prev, cur))
    assert it["previous_mentions"] == prev and it["current_mentions"] == cur
    assert it["growth_pct"] == expected


def test_zero_previous_is_new_not_infinity():
    assert growth_pct(100, 0) is None
    it = item(windows(0, 100))
    assert it["status"] == "NEW"
    assert it["growth_pct"] is None and it["growth_label"] == "NEW"
    assert not any(isinstance(v, float) and (math.isinf(v) or math.isnan(v)) for v in it.values() if isinstance(v, float))


def test_negative_counts_rejected():
    with pytest.raises(ValueError):
        growth_pct(-1, 5)


def test_statuses_for_mandatory_growth_cases():
    assert item(windows(100, 200))["status"] == "EMERGING"
    assert item(windows(100, 50))["status"] == "DECLINING"
    assert item(windows(100, 100))["status"] == "STABLE"


def test_high_volume_low_growth_is_stable():
    it = item(windows(500, 540))
    assert it["status"] == "STABLE" and it["growth_pct"] == 8.0


def test_low_volume_high_growth_is_not_emerging():
    it = item(windows(2, 10))  # +400% but only 10 mentions
    assert it["growth_pct"] == 400.0
    assert it["status"] == "INSUFFICIENT_EVIDENCE"


def test_high_volume_high_negative_ratio_emerging_and_priority_math():
    it = item(windows(50, 142))
    assert it["status"] == "EMERGING"
    assert it["growth_pct"] == 184.0
    assert it["negative_ratio"] == 1.0
    assert it["priority"] == round(142 * (1 + min(1.84, 3.0)), 2)
    assert it["calculation"]["growth"] == "(142 - 50) / 50 x 100 = 184.0%"


def test_positive_theme_growth_is_not_a_complaint():
    assert item(windows(50, 142, neg=False))["status"] == "NOT_A_COMPLAINT"


def test_insufficient_negative_evidence():
    reviews = make(20, 29, 40) + make(40, 14.01, 27.99, start_id=1000)
    reviews += make(98, 0.01, 13.99, sentiment="neutral", start_id=5000) + make(2, 0.01, 13.99, start_id=9000)
    it = item(reviews, min_negative_ratio=0.0)
    assert it["status"] == "INSUFFICIENT_EVIDENCE"
    assert any("negative evidence" in r for r in it["reasons"])


def test_identical_periods():
    it = item(windows(60, 60))
    assert it["growth_pct"] == 0.0 and it["status"] == "STABLE"


def test_missing_previous_period():
    reviews = make(80, 0.01, 10)  # batch only covers 10 days
    out = compute_radar(reviews, THEMES, RadarParams(), as_of=END)
    assert out["window"]["has_previous_window"] is False
    assert out["items"][0]["status"] == "INSUFFICIENT_EVIDENCE"


def test_malformed_timestamps_are_counted_not_silently_used():
    reviews = windows(50, 142) + [{"review_id": "BAD1", "created_at": "not-a-date", "theme_id": "t1", "sentiment": "negative"},
                                  {"review_id": "BAD2", "created_at": None, "theme_id": "t1", "sentiment": "negative"}]
    out = compute_radar(reviews, THEMES, RadarParams(), as_of=END)
    assert out["malformed_timestamps"] == 2
    assert out["items"][0]["current_mentions"] == 142


def test_no_reviews():
    out = compute_radar([], THEMES, RadarParams())
    assert out["items"] == [] and out["window"] is None


def test_no_data_theme():
    out = compute_radar(windows(10, 10), THEMES + [{"theme_id": "t2", "name": "Empty"}], RadarParams(), as_of=END)
    t2 = next(i for i in out["items"] if i["theme_id"] == "t2")
    assert t2["status"] == "NO_DATA" and t2["growth_label"] == "n/a"


def test_evidence_ids_are_real_current_negative_reviews():
    reviews = windows(50, 142)
    it = item(reviews)
    by_id = {r["review_id"]: r for r in reviews}
    assert len(it["evidence_review_ids"]) == 5
    for rid in it["evidence_review_ids"]:
        r = by_id[rid]
        assert r["sentiment"] == "negative" and r["theme_id"] == "t1"
        assert r["created_at"] > END - timedelta(days=14)


def test_segment_association_uses_non_causal_language():
    reviews = (make(20, 29, 40) + make(50, 14.01, 27.99, start_id=1000, app_version="5.2")
               + make(142, 0.01, 13.99, start_id=5000, app_version="5.3")
               + make(300, 0.01, 13.99, theme="t9", sentiment="positive", start_id=20000, app_version="5.2"))
    it = item(reviews)
    assoc = it["associations"]
    assert assoc and assoc[0]["value"] == "5.3"
    assert "not evidence of cause" in assoc[0]["statement"]
    assert "caused" not in assoc[0]["statement"].lower()


def test_weekly_trend_and_acceleration():
    reviews = make(10, 21.01, 27.99) + make(20, 14.01, 20.99, start_id=100) + make(40, 7.01, 13.99, start_id=200) + make(80, 0.01, 6.99, start_id=400)
    it = item(reviews)
    assert it["trend"] == "rising"
    assert it["weekly_counts"][-4:] == [10, 20, 40, 80]
    assert it["acceleration_pp"] == 0.0  # +100% then +100%
