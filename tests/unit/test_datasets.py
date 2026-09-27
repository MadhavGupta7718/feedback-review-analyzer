import csv
from collections import Counter

from ml import config
from ml.data import synthetic
from ml.data.sentiment140 import parse_s140_date, pseudonymise_user


def test_parse_s140_date():
    dt = parse_s140_date("Mon Apr 06 22:19:45 PDT 2009")
    assert dt is not None and dt.year == 2009 and dt.month == 4 and dt.day == 6 and dt.hour == 22
    assert parse_s140_date("not a date") is None
    assert parse_s140_date("") is None


def test_pseudonymise_user_is_stable_and_hides_name():
    a = pseudonymise_user("_TheSpecialOne_")
    assert a == pseudonymise_user("_TheSpecialOne_")
    assert "Special" not in a and a.startswith("user_")


def test_s140_fixture_schema_and_labels():
    with open(config.FIXTURES_DIR / "s140_fixture.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 200
    assert Counter(r["target"] for r in rows) == {"0": 100, "4": 100}
    assert all(r["user"].startswith("user_") for r in rows)
    assert all(r["text"].strip() for r in rows)


def test_synthetic_generation_is_deterministic():
    a = synthetic.generate(seed=7)
    b = synthetic.generate(seed=7)
    assert [r.__dict__ for r in a] == [r.__dict__ for r in b]
    c = synthetic.generate(seed=8)
    assert [r.text for r in a] != [r.text for r in c]


def test_synthetic_planted_patterns():
    reviews = synthetic.generate(seed=42)
    by_week = Counter()
    for r in reviews:
        week = (int(r.created_at[8:10]) + {6: 0, 7: 30, 8: 61}[int(r.created_at[5:7])] - 1) // 7
        by_week[(r.gt_theme, week)] += 1
    assert sum(by_week[("payment_failure", w)] for w in range(10)) == 0
    assert sum(by_week[("payment_failure", w)] for w in (10, 11)) > 100
    assert sum(by_week[("battery_drain", w)] for w in (10, 11)) > 2 * sum(by_week[("battery_drain", w)] for w in (0, 1))
    assert sum(by_week[("login_problems", w)] for w in (10, 11)) < sum(by_week[("login_problems", w)] for w in (0, 1))


def test_synthetic_contains_defects_and_pii():
    reviews = synthetic.generate(seed=42)
    assert sum(1 for r in reviews if r.text is None or not r.text.strip()) == 8
    assert sum(1 for r in reviews if r.gt_pii) > 300
    assert all(r.source == "synthetic" for r in reviews)
