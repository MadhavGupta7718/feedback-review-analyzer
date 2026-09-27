"""Product brief: validator (fact preservation, adversarial outputs), template, and fallback behaviour.

These tests need no GPU: model output is supplied as text. The live Qwen test is in tests/models/test_qwen_brief.py.
"""
from __future__ import annotations

import pytest

from ml.brief import service
from ml.brief.template import template_brief
from ml.brief.validator import parse_sections, validate

FACTS = {
    "dataset": {"kind": "synthetic"},
    "date_range": ["2026-06-01T00:00:00", "2026-08-23T00:00:00"],
    "slots": {
        "total_reviews": 842, "negative_pct": 41.2, "neutral_pct": 8.8, "positive_pct": 50.0, "n_themes": 6, "window_days": 14,
        "E1.name": "Battery Drain", "E1.current": 142, "E1.previous": 50, "E1.growth": "+184%", "E1.negative_pct": 81.0, "E1.status": "EMERGING",
        "C1.name": "App Crashes", "C1.negative": 120, "C1.size": 150, "C1.negative_pct": 80.0, "C1.growth": "+2%",
        "D1.name": "Login Errors", "D1.growth": "-40%", "D1.current": 30, "D1.previous": 50,
    },
    "overview": {"total_reviews": 842, "sentiment_pct": {"negative": 41.2, "neutral": 8.8, "positive": 50.0}, "n_themes": 6,
                 "pii_redactions_total": 3},
    "emerging": [{"key": "E1", "theme_id": "theme_002", "name": "Battery Drain", "status": "EMERGING", "current": 142, "previous": 50,
                  "growth_label": "+184%", "growth_pct": 184.0, "negative_pct": 81.0, "trend": "rising",
                  "evidence_review_ids": ["R01002", "R01044", "R01209"],
                  "quotes": [{"review_id": "R01002", "text": "battery dies by noon", "sentiment": "negative"}],
                  "associations": [], "keywords": ["battery", "drain"], "calculation": "(142 - 50) / 50 x 100 = 184.0%"}],
    "top_complaints": [{"key": "C1", "theme_id": "theme_001", "name": "App Crashes", "size": 150, "negative_count": 120,
                        "negative_pct": 80.0, "growth_label": "+2%", "status": "STABLE", "representative_ids": ["R00007"]}],
    "declining": [{"key": "D1", "theme_id": "theme_003", "name": "Login Errors", "growth_label": "-40%", "current": 30, "previous": 50}],
    "drift": {"theme": {"value": 0.3, "status": "significant"}},
    "drift_overall": "significant",
    "sentiment_accuracy": 0.7646,
    "allowed_review_ids": ["R00007", "R01002", "R01044", "R01209"],
    "allowed_theme_names": ["App Crashes", "Battery Drain", "Login Errors"],
}

GOOD = """SUMMARY:
Across {total_reviews} reviews, {negative_pct} are negative. {E1.name} is an emerging complaint, rising from {E1.previous} to {E1.current} mentions ({E1.growth}) in the last {window_days} days, with {E1.negative_pct} negative. {C1.name} remains the largest complaint theme with {C1.negative} negative reviews.

INVESTIGATE:
- Review the {E1.name} evidence (R01002, R01044) to confirm the pattern.
- Investigate possible causes of battery drain reports on recent versions.
- Monitor whether {D1.name} keeps declining.
"""


def test_valid_output_passes_and_numbers_are_substituted_exactly():
    v = validate(GOOD, FACTS)
    assert v["passed"], v["errors"]
    assert "Across 842 reviews, 41.2% are negative." in v["summary"]
    assert "from 50 to 142 mentions (+184%)" in v["summary"]
    assert "120 negative reviews" in v["summary"]
    assert v["investigate"][0].startswith("Review the Battery Drain evidence (R01002, R01044)")


def test_adversarial_invented_number_921_is_rejected():
    bad = GOOD.replace("Across {total_reviews} reviews", "Across 921 reviews")
    v = validate(bad, FACTS)
    assert not v["passed"]
    assert not v["checks"]["no_raw_digits"]
    assert any("921" in e for e in v["errors"])


def test_correct_number_written_as_digits_is_still_rejected():
    v = validate(GOOD.replace("{total_reviews}", "842"), FACTS)
    assert not v["passed"] and not v["checks"]["no_raw_digits"]


def test_reversed_direction_is_rejected():
    v = validate(GOOD.replace("from {E1.previous} to {E1.current}", "from {E1.current} to {E1.previous}"), FACTS)
    assert not v["passed"] and not v["checks"]["direction"]


def test_decrease_wording_for_emerging_theme_is_rejected():
    v = validate(GOOD.replace("is an emerging complaint, rising", "is a complaint that decreased"), FACTS)
    assert not v["passed"] and not v["checks"]["direction"]


def test_increase_wording_for_declining_theme_is_rejected():
    v = validate(GOOD.replace("Monitor whether {D1.name} keeps declining.", "{D1.name} complaints increased sharply."), FACTS)
    assert not v["passed"] and not v["checks"]["direction"]


@pytest.mark.parametrize("phrase", ["was caused by the latest release", "is due to the new update", "led to churn",
                                    "because of version 5.3", "is driven by the redesign"])
def test_causal_claims_are_rejected(phrase):
    v = validate(GOOD.replace("is an emerging complaint", f"is an emerging complaint that {phrase}"), FACTS)
    assert not v["passed"] and not v["checks"]["no_causal"]


def test_investigating_causes_is_allowed_only_in_investigation_items():
    assert validate(GOOD, FACTS)["checks"]["no_causal"]
    v = validate(GOOD.replace("{E1.name} is an emerging", "The root cause of {E1.name} is an emerging"), FACTS)
    assert not v["checks"]["no_causal"]


def test_certainty_language_is_rejected():
    v = validate(GOOD.replace("remains the largest", "definitely remains the largest"), FACTS)
    assert not v["passed"] and not v["checks"]["hedged"]


def test_unknown_slot_is_rejected():
    v = validate(GOOD.replace("{C1.negative}", "{C1.revenue}"), FACTS)
    assert not v["passed"] and not v["checks"]["known_slots"]


def test_other_themes_number_in_sentence_is_rejected():
    v = validate(GOOD.replace("with {C1.negative} negative reviews", "with {E1.current} negative reviews"), FACTS)
    assert not v["passed"] and not v["checks"]["slot_consistency"]


def test_unknown_review_id_is_rejected():
    v = validate(GOOD.replace("R01044", "R99999"), FACTS)
    assert not v["passed"] and not v["checks"]["known_review_ids"]


def test_invented_quoted_theme_name_is_rejected():
    v = validate(GOOD.replace("{C1.name} remains", "\"Checkout Freeze\" remains"), FACTS)
    assert not v["passed"] and not v["checks"]["known_theme_names"]


def test_missing_emerging_theme_is_rejected():
    no_e1 = GOOD.replace("{E1.name} is an emerging complaint, rising from {E1.previous} to {E1.current} mentions ({E1.growth}) in the last {window_days} days, with {E1.negative_pct} negative. ", "")
    v = validate(no_e1, FACTS)
    assert not v["passed"] and not v["checks"]["coverage"]


def test_pii_in_output_is_rejected():
    v = validate(GOOD.replace("to confirm the pattern", "and email jane.doe@example.com"), FACTS)
    assert not v["passed"] and not v["checks"]["no_pii"]


def test_malformed_output_is_rejected():
    assert not validate("Battery is bad.", FACTS)["checks"]["format"]
    assert parse_sections("") == ("", [])


def test_template_brief_uses_only_fact_values():
    b = template_brief(FACTS)
    assert b["generation_path"] == "template"
    assert "842" in b["executive_summary"] and "921" not in b["executive_summary"]
    assert "Battery Drain (+184%, 50 to 142 mentions)" in b["executive_summary"]
    assert all(e["review_id"] in FACTS["allowed_review_ids"] for e in b["evidence"])


class _FailingWriter:
    def write(self, facts):
        return {"ok": False, "brief": None, "validation": {"passed": False, "errors": ["raw numbers written instead of slots: 921"]}}


def _patch_service(monkeypatch, *, writer=None, live=True, pre=None):
    monkeypatch.setattr(service, "build_facts", lambda db: FACTS)
    monkeypatch.setattr(service, "qwen_status", lambda: {"brief_mode": "auto", "qwen_installed": live, "cuda_available": live,
                                                         "qwen_loaded": False, "last_error": None, "live_generation_possible": live})
    monkeypatch.setattr(service, "_get_writer", lambda: writer)
    monkeypatch.setattr(service, "precomputed", lambda db: pre)


def test_fallback_to_template_when_qwen_output_fails_validation(monkeypatch):
    _patch_service(monkeypatch, writer=_FailingWriter())
    b = service.generate_brief("unused.db", "auto")
    assert b["generation_path"] == "template"
    assert any("failed validation" in r for r in b["fallback_reasons"])
    assert "921" not in b["executive_summary"]


class _CrashingWriter:
    def write(self, facts):
        raise RuntimeError("CUDA out of memory")


def test_generation_exception_falls_back_instead_of_500(monkeypatch):
    _patch_service(monkeypatch, writer=_CrashingWriter())
    b = service.generate_brief("unused.db", "auto")
    assert b["generation_path"] == "template"
    assert any("raised RuntimeError" in r for r in b["fallback_reasons"])


def test_fallback_to_precomputed_when_no_gpu(monkeypatch):
    _patch_service(monkeypatch, live=False, pre={"executive_summary": "stored", "generated_at_utc": "x"})
    b = service.generate_brief("unused.db", "auto")
    assert b["generation_path"] == "qwen_precomputed"
    assert any("no CUDA GPU" in r for r in b["fallback_reasons"])


def test_template_engine_never_calls_qwen(monkeypatch):
    def boom():
        raise AssertionError("writer must not be loaded")

    _patch_service(monkeypatch)
    monkeypatch.setattr(service, "_get_writer", boom)
    b = service.generate_brief("unused.db", "template")
    assert b["generation_path"] == "template" and "fallback_reasons" not in b


def test_writer_load_failure_falls_back(monkeypatch):
    _patch_service(monkeypatch, writer=None)
    monkeypatch.setattr(service, "_writer_error", "ModelNotInstalledError: Qwen/Qwen2.5-3B-Instruct")
    b = service.generate_brief("unused.db", "qwen")
    assert b["generation_path"] == "template"
    assert any("could not be loaded" in r for r in b["fallback_reasons"])


def test_qwen_prompt_contains_no_numbers_from_facts_and_no_raw_quotes():
    from ml.brief.qwen_writer import build_prompt

    p = build_prompt(FACTS)
    for value in ("842", "142", "184", "81.0", "120"):
        assert value not in p
    assert "{E1.current}" in p and "Battery Drain" in p
