"""Per-batch evaluation stays scoped to the rows of that analytics DB."""
from ml.evaluation import batch_eval


def test_sentiment_macro_recall_from_planted_labels():
    rows = [
        {"gt_sentiment": "negative", "sentiment": "negative", "p_negative": 0.8, "p_neutral": 0.1, "p_positive": 0.1},
        {"gt_sentiment": "neutral", "sentiment": "neutral", "p_negative": 0.2, "p_neutral": 0.6, "p_positive": 0.2},
        {"gt_sentiment": "positive", "sentiment": "positive", "p_negative": 0.1, "p_neutral": 0.1, "p_positive": 0.8},
        {"gt_sentiment": "positive", "sentiment": "negative", "p_negative": 0.7, "p_neutral": 0.1, "p_positive": 0.2},
    ]
    rep = batch_eval.sentiment_report(rows, model="m", revision="r", device="cpu")
    assert rep is not None
    assert rep["dataset"] == "this analytics database only"
    assert rep["metrics"]["headline_accuracy"] == 0.75
    assert rep["metrics"]["headline_macro_recall"] == rep["metrics"]["three_class"]["macro_recall"]


def test_stars_weak_labels_used_when_no_gt():
    assert batch_eval.stars_to_sentiment(1) == "negative"
    assert batch_eval.stars_to_sentiment(3) == "neutral"
    assert batch_eval.stars_to_sentiment(5) == "positive"
    rows = [
        {"rating": 1, "sentiment": "negative"},
        {"rating": 5, "sentiment": "positive"},
    ]
    y_true, y_pred, source = batch_eval._labels(rows)
    assert source == "star_rating_weak_labels"
    assert len(y_true) == 2


def test_overall_recall_mean_skips_missing():
    out = batch_eval.build(
        [{"gt_sentiment": "negative", "sentiment": "negative", "rating": 1}],
        model="m",
        revision=None,
        device="cpu",
        theme_eval=None,
        themes=[],
        radar_items=[],
    )
    assert out["themes"] is None
    assert out["radar"] is None
    assert out["overall_recall"]["sentiment_macro_recall"] is not None
    assert out["overall_recall"]["mean_available_recalls"] == out["overall_recall"]["sentiment_macro_recall"]
