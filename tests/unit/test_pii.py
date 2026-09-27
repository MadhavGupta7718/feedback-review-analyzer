import logging

import pytest

from ml.data import synthetic
from ml.pii import get_safe_logger, redact
from ml.pii.leak_scan import scan
from ml.preprocessing.clean import clean_batch, clean_one


@pytest.mark.parametrize(
    "text,kind,secret",
    [
        ("mail me at jane.doe42@example.com please", "EMAIL", "jane.doe42@example.com"),
        ("contact: bob+shop@mail.co.uk", "EMAIL", "bob+shop@mail.co.uk"),
        ("Call me on +1 555-123-4567 today", "PHONE", "555-123-4567"),
        ("my number is (555) 987-6543", "PHONE", "987-6543"),
        ("reach me at +44 7700 900123", "PHONE", "900123"),
        ("see https://imgur.example/abc123 for proof", "URL", "imgur.example"),
        ("pic at twitpic.com/2y1zl lol", "URL", "twitpic.com"),
        ("go to www.nimbus-shop.com/help", "URL", "nimbus-shop"),
        ("Order ORD-482913 never arrived", "ORDER_ID", "482913"),
        ("order number 77123456 is missing", "ORDER_ID", "77123456"),
        ("my account ID is ACC-12345678", "ACCOUNT_ID", "12345678"),
        ("account number: 99887766", "ACCOUNT_ID", "99887766"),
        ("customer id CUST-554433", "CUSTOMER_ID", "554433"),
        ("used card 4111 1111 1111 1111 twice", "CARD", "4111 1111 1111 1111"),
        ("card 5555-5555-5555-4444", "CARD", "5555-5555-5555-4444"),
        ("amex 3782 822463 10005", "CARD", "822463"),
        ("@john_smith99 said the same", "USER", "john_smith99"),
        ("This is Maria Garcia writing", "PERSON", "Maria Garcia"),
        ("my name is Oluwaseun Adeyemi and I am upset", "PERSON", "Oluwaseun"),
        ("this is priya sharma writing", "PERSON", "priya"),
        ("Regards, Tomasz", "PERSON", "Tomasz"),
        # regressions found by the Sentiment140 audit
        ("09166279004 -- my new number", "PHONE", "09166279004"),
        ("@alice@DevineNews what happened", "USER", "DevineNews"),
        ("wrong username for Mr Brooks", "PERSON", "Brooks"),
    ],
)
def test_each_pii_type_is_redacted(text, kind, secret):
    res = redact(text)
    assert secret not in res.text, res.text
    assert res.counts.get(kind, 0) >= 1, (res.counts, res.text)


@pytest.mark.parametrize(
    "text",
    [
        "Battery drains 30% in an hour since version 5.2",
        "This is terrible, the app crashes on 2026-08-01",
        "I'm so tired of waiting 3 weeks for delivery",
        "This is Nimbus at its worst",
        "Love the new design!",
        "Order arrived late but support helped",
        "I voted like 1000000 times",
        "awww.. that's sad",
    ],
)
def test_no_false_positives_on_normal_reviews(text):
    res = redact(text)
    assert res.total == 0, (res.counts, res.text)
    assert res.text == text


def test_leak_scanner_ignores_benign_patterns_but_flags_real_ones():
    assert scan("singing <333333333 awww.. till 05312009, voted 1000000 times") == []
    assert "long_digit_run" in scan("call 555 123 4567")
    assert "email" in scan("x@y.com")
    assert "url" in scan("see www.site.com")


def test_redaction_never_returns_raw_values():
    res = redact("email a@b.com and card 4111111111111111")
    assert set(res.counts) == {"EMAIL", "CARD"}
    assert all(isinstance(v, int) for v in res.counts.values())


def test_safe_logger_redacts(caplog):
    log = get_safe_logger("pii-test")
    with caplog.at_level(logging.INFO, logger="pii-test"):
        log.info("user wrote %s", "mail jane@example.com, call +1 555-222-3333")
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "jane@example.com" not in joined and "555-222-3333" not in joined
    assert "[EMAIL]" in joined and "[PHONE]" in joined


def test_clean_one_rejects_empty_and_null():
    assert clean_one("R1", None).reject_reason == "null_text"
    assert clean_one("R2", "   \n").reject_reason == "empty_text"
    assert clean_one("R3", "!!!").reject_reason == "no_content_after_cleaning"


def test_clean_repairs_mojibake_and_entities():
    rec = clean_one("R1", "I donâ€™t like it &amp; it crashes!!!!!!!")
    assert rec.text == "I don't like it & it crashes!!!"


def test_synthetic_ground_truth_pii_is_fully_redacted():
    """Recall on every planted PII snippet in the 10K synthetic batch + false-positive rate on clean rows."""
    reviews = synthetic.generate(seed=42)
    rows = [r.__dict__ for r in reviews]
    cleaned, rep = clean_batch(rows)
    by_id = {r["review_id"]: r for r in cleaned}
    names = {n.lower() for n in synthetic.FIRST_NAMES + synthetic.LAST_NAMES}
    leaks, fp = [], 0
    for r in reviews:
        c = by_id.get(r.review_id)
        if c is None:
            continue
        text = c["text_redacted"]
        if r.gt_pii:
            words = {w.strip(".,!").lower() for w in text.split()}
            if scan(text) or (r.gt_pii == "person" and words & names):
                leaks.append((r.review_id, r.gt_pii))
        elif c["pii_redaction_count"]:
            fp += 1
    assert leaks == [], leaks[:10]
    clean_rows = sum(1 for r in reviews if not r.gt_pii and r.review_id in by_id)
    assert fp / clean_rows < 0.005, f"false positive rows {fp}/{clean_rows}"
    assert rep.input_rows == rep.kept_rows + sum(rep.rejected.values()) + rep.ingestion_duplicates_removed
