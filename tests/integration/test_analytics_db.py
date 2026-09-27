"""Traceability + leakage tests against the generated analytics database (artifacts/analytics.db)."""
import json
import sqlite3

import pytest

from ml import config
from ml.pii.leak_scan import scan

pytestmark = pytest.mark.skipif(not config.ANALYTICS_DB.exists(), reason="run `python -m ml.pipeline` first")


@pytest.fixture(scope="module")
def con():
    c = sqlite3.connect(config.ANALYTICS_DB)
    yield c
    c.close()


def test_every_evidence_row_points_to_a_real_review_of_that_theme(con):
    rows = con.execute("SELECT e.theme_id, e.review_id, e.kind, r.theme_id, r.sentiment FROM evidence e "
                       "LEFT JOIN reviews r ON r.review_id = e.review_id").fetchall()
    assert rows, "no evidence rows"
    for theme_id, review_id, kind, r_theme, r_sent in rows:
        assert r_theme is not None, f"orphan evidence {review_id}"
        assert r_theme == theme_id, f"{review_id} evidence for {theme_id} but belongs to {r_theme}"
        if kind == "radar_evidence":
            assert r_sent == "negative"


def test_theme_counts_match_review_table(con):
    for tid, data in con.execute("SELECT theme_id, data FROM themes"):
        t = json.loads(data)
        n, neg = con.execute("SELECT COUNT(*), SUM(sentiment='negative') FROM reviews WHERE theme_id=?", (tid,)).fetchone()
        assert t["size"] == n
        assert t["negative_count"] == neg


def test_radar_counts_recomputable_from_reviews(con):
    meta = json.loads(con.execute("SELECT data FROM reports WHERE key='radar_meta'").fetchone()[0])
    w = meta["window"]
    for tid, data in con.execute("SELECT theme_id, data FROM issues"):
        it = json.loads(data)
        cur = con.execute("SELECT COUNT(*) FROM reviews WHERE theme_id=? AND created_at>? AND created_at<=?",
                          (tid, w["current_start"], w["end"])).fetchone()[0]
        prev = con.execute("SELECT COUNT(*) FROM reviews WHERE theme_id=? AND created_at>? AND created_at<=?",
                           (tid, w["previous_start"], w["current_start"])).fetchone()[0]
        assert (it["current_mentions"], it["previous_mentions"]) == (cur, prev), tid
        if prev:
            assert it["growth_pct"] == round((cur - prev) / prev * 100, 2)


def test_no_pii_in_database_text(con):
    leaks = [rid for rid, t in con.execute("SELECT review_id, text_redacted FROM reviews") if scan(t)]
    assert leaks == []


def test_no_raw_text_column(con):
    cols = [r[1] for r in con.execute("PRAGMA table_info(reviews)")]
    assert "text" not in cols and "text_redacted" in cols


def test_whole_database_dump_has_no_pii(con):
    dump = "\n".join(con.iterdump())
    # the synthetic generator only plants example.com emails, 555/7700 phones and ORD/ACC/CUST ids
    for needle in ("@example.com", "ORD-", "ACC-", "CUST-", "4111 1111", "555-", "(555)", "+44 7700"):
        assert needle not in dump, needle


def test_traceability_report_passed(con):
    rep = json.loads(con.execute("SELECT data FROM reports WHERE key='traceability'").fetchone()[0])
    assert rep["status"] == "PASS" and rep["problems"] == []
