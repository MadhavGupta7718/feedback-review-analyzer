"""Text normalisation + validation + redaction for a review batch.

Order of operations (per review)
  1. validate   null / empty / whitespace-only texts are rejected with an explicit reason
  2. repair     mojibake (ftfy), HTML entities, Unicode NFC, control characters
  3. redact     PII -> placeholders (ml.pii). Raw text is discarded after this step.
  4. tidy       whitespace, runs of punctuation (keeps emphasis: "!!!!!" -> "!!!")
Emojis, hashtags and casing are kept: they carry sentiment and the models were trained on tweets.
"""
from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

import ftfy

from ml.pii import redact

_CONTROL = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f\u200b\ufeff]")
_WS = re.compile(r"\s+")
_PUNCT_RUN = re.compile(r"([!?.])\1{3,}")
_MIXED_PUNCT = re.compile(r"([!?]){4,}")


def repair(text: str) -> str:
    s = ftfy.fix_text(text)
    s = html.unescape(s)
    s = unicodedata.normalize("NFC", s)
    s = _CONTROL.sub(" ", s)
    return s


def tidy(text: str) -> str:
    s = _PUNCT_RUN.sub(lambda m: m.group(1) * 3, text)
    s = _MIXED_PUNCT.sub(lambda m: m.group(0)[:3], s)
    return _WS.sub(" ", s).strip()


def model_text(redacted: str) -> str:
    """Input form for the Twitter-trained sentiment model (its model card maps users -> @user, links -> http)."""
    s = redacted.replace("[USER]", "@user").replace("[URL]", "http")
    return s


@dataclass
class CleanRecord:
    review_id: str
    text: str | None
    reject_reason: str | None = None
    pii_counts: dict = field(default_factory=dict)


def clean_one(review_id: str, raw: str | None) -> CleanRecord:
    if raw is None or (isinstance(raw, float)):
        return CleanRecord(review_id, None, "null_text")
    if not raw.strip():
        return CleanRecord(review_id, None, "empty_text")
    repaired = repair(raw)
    red = redact(repaired)
    text = tidy(red.text)
    if not text or not re.search(r"\w", text):
        return CleanRecord(review_id, None, "no_content_after_cleaning", red.counts)
    return CleanRecord(review_id, text, None, red.counts)


@dataclass
class BatchCleanReport:
    input_rows: int = 0
    kept_rows: int = 0
    rejected: dict = field(default_factory=dict)
    ingestion_duplicates_removed: int = 0
    identical_text_rows_kept: int = 0
    pii_redactions: dict = field(default_factory=dict)
    rows_with_pii: int = 0
    mojibake_repaired: int = 0
    html_entities_decoded: int = 0


def dedupe_key(row: dict) -> tuple:
    """Ingestion duplicates = same text AND same timestamp AND same metadata (a re-delivered record).
    Different people writing the same short text ("love this app") are legitimate and kept."""
    return (str(row.get("text", "")).strip().lower(), row.get("created_at"), row.get("platform"), row.get("app_version"))


def clean_batch(rows: list[dict]) -> tuple[list[dict], BatchCleanReport]:
    """rows: dicts with at least review_id, text. Returns cleaned rows (raw text removed) + report."""
    rep = BatchCleanReport(input_rows=len(rows))
    rejected: Counter = Counter()
    pii: Counter = Counter()
    seen: set = set()
    text_counts: Counter = Counter()
    out: list[dict] = []
    for row in rows:
        raw = row.get("text")
        if isinstance(raw, str):
            key = dedupe_key(row)
            if raw.strip() and key in seen:
                rep.ingestion_duplicates_removed += 1
                continue
            seen.add(key)
            if ftfy.fix_text(raw) != raw:
                rep.mojibake_repaired += 1
            if html.unescape(raw) != raw:
                rep.html_entities_decoded += 1
        rec = clean_one(row["review_id"], raw)
        if rec.reject_reason:
            rejected[rec.reject_reason] += 1
            continue
        if rec.pii_counts:
            rep.rows_with_pii += 1
            pii.update(rec.pii_counts)
        text_counts[rec.text.lower()] += 1
        cleaned = {k: v for k, v in row.items() if k != "text"}
        cleaned["text_redacted"] = rec.text
        cleaned["pii_redaction_count"] = sum(rec.pii_counts.values())
        out.append(cleaned)
    rep.kept_rows = len(out)
    rep.rejected = dict(rejected)
    rep.pii_redactions = dict(pii)
    rep.identical_text_rows_kept = sum(c for c in text_counts.values() if c > 1)
    assert rep.input_rows == rep.kept_rows + sum(rejected.values()) + rep.ingestion_duplicates_removed, "silent row loss"
    return out, rep
