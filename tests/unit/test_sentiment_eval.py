"""Binary-truth scoring helpers, threshold tuning and the leakage-free Sentiment140 split (no models needed)."""
from __future__ import annotations

import csv

import numpy as np
import pytest

from ml import config
from ml.evaluation import metrics as M
from ml.evaluation.s140_split import SPLIT_DIR, assign_split, group_key

Y = np.array([M.NEG, M.NEG, M.NEG, M.POS, M.POS, M.POS])
P = np.array([
    [0.8, 0.1, 0.1],   # neg, right
    [0.2, 0.7, 0.1],   # neutral, nearer neg
    [0.1, 0.2, 0.7],   # pos, wrong
    [0.1, 0.6, 0.3],   # neutral, nearer pos
    [0.1, 0.1, 0.8],   # pos, right
    [0.6, 0.3, 0.1],   # neg, wrong
])


def test_neutral_strategies_are_distinct_and_correct():
    s = M.neutral_strategies(Y, P)
    assert s["A_neutral_to_negative"]["accuracy"] == round(3 / 6, 4)   # rows 0,1,4
    assert s["B_neutral_to_positive"]["accuracy"] == round(3 / 6, 4)   # rows 0,3,4
    assert s["C_nearest_binary"]["accuracy"] == round(4 / 6, 4)        # rows 0,1,3,4
    d = s["D_exclude_neutral"]
    assert d["n"] == 4 and d["coverage"] == round(4 / 6, 4) and d["accuracy"] == 0.5
    assert all(v["neutral_prediction_rate"] == round(2 / 6, 4) for v in s.values())
    c = s["C_nearest_binary"]["confusion"]
    assert sum(sum(r.values()) for r in c.values()) == 6


def test_nearest_binary_equals_legacy_binary_forced():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet([1, 1, 1], size=500)
    y = rng.choice([M.NEG, M.POS], size=500)
    assert M.neutral_strategies(y, probs)["C_nearest_binary"]["accuracy"] == M.evaluate_binary_truth(y, probs)["binary_forced"]["accuracy"]


def test_positive_score_removes_neutral_mass():
    s = M.positive_score(np.array([[0.2, 0.6, 0.2], [0.1, 0.0, 0.3], [0.0, 1.0, 0.0]]))
    assert s[0] == pytest.approx(0.5) and s[1] == pytest.approx(0.75)
    assert np.isfinite(s).all()


def test_tune_threshold_finds_the_separating_cut_and_prefers_05_on_ties():
    y = np.array([M.NEG] * 4 + [M.POS] * 4)
    score = np.array([0.1, 0.3, 0.6, 0.65, 0.75, 0.8, 0.9, 0.95])
    t = M.tune_threshold(y, score)
    assert t["val_accuracy"] == 1.0 and 0.65 <= t["threshold"] < 0.75
    assert (M.apply_threshold(score, t["threshold"]) == y).all()
    flat = M.tune_threshold(np.array([M.NEG, M.POS]), np.array([0.0, 1.0]))
    assert flat["threshold"] == 0.5


def test_sentiment_binary_threshold_is_the_validation_value():
    assert 0.05 <= config.SENTIMENT_BINARY_THRESHOLD <= 0.95
    rep = config.ARTIFACTS_DIR / "reports" / "sentiment_study_val.json"
    if rep.exists():
        import json

        assert json.loads(rep.read_text(encoding="utf-8"))["threshold_tuning"]["threshold"] == config.SENTIMENT_BINARY_THRESHOLD


def test_group_key_normalises_trivial_reposts():
    a = group_key("@bob I   LOVE this &amp; that http://t.co/x")
    b = group_key("@alice i love this & that https://bit.ly/y")
    assert a == b == "@user i love this & that http"


def test_assign_split_is_deterministic_and_close_to_80_10_10():
    keys = [f"tweet number {i}" for i in range(20_000)]
    first = [assign_split(k) for k in keys]
    assert first == [assign_split(k) for k in keys]
    share = {s: first.count(s) / len(keys) for s in ("train", "val", "test")}
    assert abs(share["train"] - 0.8) < 0.01 and abs(share["val"] - 0.1) < 0.01 and abs(share["test"] - 0.1) < 0.01
    assert [assign_split(k, seed=7) for k in keys[:200]] != first[:200]


@pytest.mark.skipif(not (SPLIT_DIR / "val_index.csv").exists(), reason="split not built (scripts/build_s140_split.py)")
def test_built_split_has_no_line_overlap_and_is_balanced():
    lines = {}
    for s in ("train", "val", "test"):
        with open(SPLIT_DIR / f"{s}_index.csv", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        lines[s] = {int(r["line"]) for r in rows}
        pos = sum(r["target"] == "4" for r in rows) / len(rows)
        assert 0.48 < pos < 0.52
    assert not (lines["train"] & lines["val"]) and not (lines["train"] & lines["test"]) and not (lines["val"] & lines["test"])
    with open(SPLIT_DIR / "train_subset.csv", encoding="utf-8") as fh:
        sub = {int(r["line"]) for r in csv.DictReader(fh)}
    assert sub <= lines["train"]
