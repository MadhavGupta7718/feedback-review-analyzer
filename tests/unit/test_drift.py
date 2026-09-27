import numpy as np

from ml.drift.monitor import DriftThresholds, categorical_drift, drift_report, length_drift, psi, volume_drift

T = DriftThresholds()


def reviews(neg, neu, pos, theme_mix=None, length=60, seed=0):
    rng = np.random.default_rng(seed)
    sents = ["negative"] * neg + ["neutral"] * neu + ["positive"] * pos
    themes = theme_mix or {"a": 0.5, "b": 0.5}
    keys, probs = list(themes), list(themes.values())
    out = []
    for s in sents:
        L = max(5, int(rng.normal(length, 10)))
        out.append({"sentiment": s, "theme_id": str(rng.choice(keys, p=probs)), "text_redacted": "x" * L})
    return out


def test_identical_distributions_no_drift():
    a = reviews(400, 200, 400, seed=1)
    r = drift_report(a, a, 14, 14)
    assert r["metrics"]["sentiment"]["value"] == 0.0
    assert r["metrics"]["theme"]["value"] == 0.0
    assert r["metrics"]["volume"]["value"] == 0.0
    assert r["metrics"]["review_length"]["value"] == 0.0
    assert r["overall_status"] == "none"


def test_small_change_no_or_moderate_drift():
    ref = reviews(400, 200, 400, seed=1)
    cur = reviews(430, 190, 380, theme_mix={"a": 0.53, "b": 0.47}, length=61, seed=2)
    r = drift_report(ref, cur, 14, 14)
    assert r["metrics"]["sentiment"]["status"] == "none"
    assert r["metrics"]["theme"]["status"] == "none"
    assert r["metrics"]["volume"]["status"] == "none"
    assert r["metrics"]["review_length"]["status"] == "none"


def test_large_change_significant_drift():
    ref = reviews(200, 200, 600, seed=1)
    cur = reviews(1300, 100, 100, theme_mix={"a": 0.1, "b": 0.2, "c": 0.7}, length=120, seed=2)
    r = drift_report(ref, cur, 14, 14)
    m = r["metrics"]
    assert m["sentiment"]["status"] == "significant"
    assert m["theme"]["status"] == "significant"
    assert m["volume"]["status"] == "moderate" or m["volume"]["status"] == "significant"
    assert m["review_length"]["status"] == "significant"
    assert r["overall_status"] == "significant"


def test_psi_known_value():
    assert psi(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0.0
    v = psi(np.array([0.5, 0.5]), np.array([0.9, 0.1]))
    expected = (0.9 - 0.5) * np.log(0.9 / 0.5) + (0.1 - 0.5) * np.log(0.1 / 0.5)
    assert abs(v - expected) < 1e-3


def test_volume_thresholds_and_zero_reference():
    assert volume_drift(100, 10, 110, 10, T)["status"] == "none"
    assert volume_drift(100, 10, 130, 10, T)["status"] == "moderate"
    assert volume_drift(100, 10, 200, 10, T)["status"] == "significant"
    z = volume_drift(0, 10, 5, 10, T)
    assert z["value"] is None and z["status"] == "significant"


def test_length_insufficient_data():
    assert length_drift([10], [10, 12], T)["status"] == "insufficient_data"


def test_categorical_new_category_handled():
    r = categorical_drift(["a"] * 100, ["a"] * 50 + ["new"] * 50, T)
    assert r["status"] == "significant"
    assert r["top_changes"][0]["category"] in ("a", "new")
