from datetime import datetime

import pytest

from ml.pipeline import load_csv, load_source, parse_any_date, subsample


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


def test_parse_any_date_formats_and_timezones():
    assert parse_any_date("2026-06-01T10:00:00Z") == datetime(2026, 6, 1, 10)
    assert parse_any_date("2026-06-01T15:30:00+05:30") == datetime(2026, 6, 1, 10)
    assert parse_any_date("2026-06-01") == datetime(2026, 6, 1)
    assert parse_any_date("25/06/2026") == datetime(2026, 6, 25)
    assert parse_any_date("06/13/2026") == datetime(2026, 6, 13)
    assert parse_any_date("yesterday") is None and parse_any_date("") is None


def test_load_csv_maps_aliases_and_sanitises_ids(tmp_path):
    f = tmp_path / "reviews.csv"
    f.write_text("ID,Review,Date,Stars,Version\n"
                 "a1,App keeps crashing,2026-06-01,1,5.2\n"
                 "a1,Great app,2026-06-02,4.6,5.3\n"
                 "bad id!,Slow login,not a date,,\n"
                 ",,2026-06-03,2,\n", encoding="utf-8-sig")
    rows, info = load_csv(f)
    assert [r["review_id"] for r in rows] == ["a1", "U000002", "U000003", "U000004"]
    assert rows[0]["text"] == "App keeps crashing" and rows[0]["rating"] == 1 and rows[0]["app_version"] == "5.2"
    assert rows[1]["rating"] == 5 and rows[1]["created_at"] == "2026-06-02T00:00:00"
    assert rows[2]["created_at"] is None and rows[3]["text"] is None
    assert all(r["source"] == "csv" for r in rows)
    assert info["kind"] == "csv" and info["rows_without_date"] == 1 and info["path"] == "reviews.csv"
    assert str(tmp_path) not in str(info)


def test_load_csv_requires_text_and_dates(tmp_path):
    no_text = tmp_path / "a.csv"
    no_text.write_text("date,stars\n2026-06-01,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="text column"):
        load_csv(no_text)
    no_date = tmp_path / "b.csv"
    no_date.write_text("review\nfine\n", encoding="utf-8")
    with pytest.raises(ValueError, match="timestamp column"):
        load_csv(no_date)
    bad_dates = tmp_path / "c.csv"
    bad_dates.write_text("review,date\nfine,soon\n", encoding="utf-8")
    with pytest.raises(ValueError, match="parseable date"):
        load_csv(bad_dates)


def test_load_source_routes_csv_paths(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("text,created_at\nok,2026-06-01\n", encoding="utf-8")
    rows, info = load_source(str(f))
    assert len(rows) == 1 and info["kind"] == "csv"
    with pytest.raises(ValueError):
        load_source("unknown")
