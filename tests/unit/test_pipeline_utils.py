from ml.pipeline import subsample


def test_subsample_is_deterministic_evenly_spaced_and_keeps_both_ends():
    rows = [{"i": i} for i in range(1000)]
    a, b = subsample(rows, 100), subsample(rows, 100)
    assert a == b and len(a) == 100
    assert a[0]["i"] == 0 and a[-1]["i"] == 999
    gaps = {y["i"] - x["i"] for x, y in zip(a, a[1:])}
    assert gaps <= {10, 11}


def test_subsample_without_limit_or_with_large_limit_returns_all():
    rows = [{"i": i} for i in range(5)]
    assert subsample(rows, None) == rows
    assert subsample(rows, 50) == rows
