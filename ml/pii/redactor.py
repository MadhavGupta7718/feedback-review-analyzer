"""Deterministic PII redaction.

Design rules
  * Redaction returns only the redacted text and per-type COUNTS; raw matched values are never
    returned, stored or logged.
  * Patterns run in a fixed order so that more specific identifiers win (a card number is never
    half-consumed as a phone number, an email is never split into a handle + domain).
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field

PLACEHOLDERS = {
    "EMAIL": "[EMAIL]",
    "URL": "[URL]",
    "CARD": "[CARD]",
    "ORDER_ID": "[ORDER_ID]",
    "ACCOUNT_ID": "[ACCOUNT_ID]",
    "CUSTOMER_ID": "[CUSTOMER_ID]",
    "PHONE": "[PHONE]",
    "USER": "[USER]",
    "PERSON": "[PERSON]",
}

_EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")
_URL = re.compile(
    r"(?i)\b(?:https?://|www\.)[^\s<>\"']+"
    r"|\b[a-z0-9][a-z0-9-]{1,62}\.(?:com|net|org|io|ly|co|me|gl|us|uk|in|tv|fm|info|biz)(?:/[^\s<>\"']*)?(?![\w@])"
)
_CARD = re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])")
_ORDER = re.compile(r"(?i)\b(?:ORD(?:ER)?[-#]\s?\d{4,}|order\s*(?:#|no\.?|number|id)\s*[:#]?\s*[A-Z0-9-]*\d{4,}[A-Z0-9-]*)")
_ACCOUNT = re.compile(r"(?i)\b(?:ACC(?:T|OUNT)?[-#]\s?\d{4,}|account\s*(?:#|no\.?|number|id)\s*(?:is\s*)?[:#]?\s*[A-Z0-9-]*\d{4,}[A-Z0-9-]*)")
_CUSTOMER = re.compile(r"(?i)\b(?:CUST(?:OMER)?[-#]\s?\d{4,}|customer\s*(?:#|no\.?|number|id)\s*(?:is\s*)?[:#]?\s*[A-Z0-9-]*\d{4,}[A-Z0-9-]*)")
_PHONE = re.compile(r"(?<![\w])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{2,5}(?:[\s.-]\d{2,5}){1,3}(?![\w])")
_HANDLE = re.compile(r"(?<![\w@\[])@[A-Za-z0-9_]{1,30}")

_NAME = r"[A-Z][a-z]{1,20}(?:[-'][A-Z][a-z]+)?"
_ANY_WORD = r"[A-Za-z][a-z]{1,20}"
# After an explicit self-identification cue the next one or two words are a name regardless of case.
_PERSON_STRONG = re.compile(rf"(?i:\b(?:my name is|name's|signed|regards,?|sincerely,?)\s+)({_ANY_WORD}(?:\s{_ANY_WORD})?)")
# Weaker cues need either two capitalised words or a known first name.
_PERSON_WEAK = re.compile(rf"(?i:\b(?:this is|i am|i'm|im|call me)\s+)({_NAME}\s{_NAME})\b")
_PERSON_GAZ = re.compile(rf"(?i:\b(?:this is|i am|i'm|im|call me|it's|from)\s+)({_ANY_WORD}(?:\s{_ANY_WORD})?)\b")

# Small first-name gazetteer used only with a weak cue (catches lower-cased self-identification).
_FIRST_NAMES = {
    "james", "john", "robert", "michael", "william", "david", "richard", "joseph", "thomas", "charles", "daniel",
    "matthew", "anthony", "mark", "steven", "paul", "andrew", "joshua", "kevin", "brian", "george", "edward",
    "mary", "patricia", "jennifer", "linda", "elizabeth", "barbara", "susan", "jessica", "sarah", "karen",
    "nancy", "lisa", "emma", "olivia", "sophia", "sofia", "isabella", "mia", "amelia", "liam", "noah", "oliver",
    "priya", "rahul", "amit", "anita", "arjun", "deepa", "vikram", "neha", "madhav", "rohan", "aisha", "fatima",
    "omar", "ali", "hassan", "maria", "jose", "carlos", "juan", "luis", "ana", "chen", "wei", "li", "yuki",
    "hiroshi", "kim", "min", "ivan", "olga", "anna", "pierre", "marie", "hans", "lars", "sven", "chidi", "ngozi",
}

# Capitalised words that commonly follow "this is"/"I'm" but are not names.
_NOT_NAMES = {
    "the", "a", "an", "not", "so", "very", "really", "just", "such", "terrible", "great", "awful", "amazing",
    "good", "bad", "best", "worst", "ridiculous", "unacceptable", "fine", "okay", "ok", "going", "done", "sure",
    "happy", "sad", "sorry", "tired", "bored", "nimbus", "my", "your", "our", "their", "it", "what", "why",
    "how", "still", "never", "always", "here", "there", "back", "home", "off", "on", "in", "out", "at", "and",
    "stuck", "gonna", "getting", "trying", "literally", "totally", "absolutely", "honestly", "seriously",
    "writing", "and", "but", "from", "with", "again", "please", "is", "was", "i", "me", "you", "to", "for",
    "team", "all", "everyone", "support", "customer", "service", "app", "thanks", "thank",
}


def _digits(s: str) -> int:
    return sum(ch.isdigit() for ch in s)


@dataclass
class RedactionResult:
    text: str
    counts: dict = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


class PIIRedactor:
    def __init__(self, ner_pipeline=None):
        self._ner = ner_pipeline

    def redact(self, text: str | None) -> RedactionResult:
        if text is None:
            return RedactionResult("", {})
        counts: Counter = Counter()

        def sub(pattern: re.Pattern, kind: str, s: str, check=None) -> str:
            def repl(m: re.Match) -> str:
                if check is not None and not check(m.group(0)):
                    return m.group(0)
                counts[kind] += 1
                return PLACEHOLDERS[kind]
            return pattern.sub(repl, s)

        s = text
        s = sub(_EMAIL, "EMAIL", s)
        s = sub(_URL, "URL", s)
        s = sub(_ORDER, "ORDER_ID", s)
        s = sub(_ACCOUNT, "ACCOUNT_ID", s)
        s = sub(_CUSTOMER, "CUSTOMER_ID", s)
        s = sub(_CARD, "CARD", s, check=lambda v: 13 <= _digits(v) <= 19)
        s = sub(_PHONE, "PHONE", s, check=self._looks_like_phone)
        s = sub(_HANDLE, "USER", s)
        s = self._redact_person(s, counts)
        return RedactionResult(s, dict(counts))

    @staticmethod
    def _looks_like_phone(v: str) -> bool:
        d = _digits(v)
        if d < 7 or d > 15:
            return False
        if re.fullmatch(r"(?:19|20)\d{2}[-./]\d{1,2}[-./]\d{1,2}", v.strip()):
            return False  # ISO date
        if re.fullmatch(r"\d{1,2}[-./]\d{1,2}[-./](?:19|20)?\d{2}", v.strip()):
            return False  # d/m/y date
        return True

    def _redact_person(self, s: str, counts: Counter) -> str:
        def strong(m: re.Match) -> str:
            words = m.group(1).split()
            if words[0].lower() in _NOT_NAMES:
                return m.group(0)
            name = " ".join(words if len(words) == 2 and words[1].lower() not in _NOT_NAMES else words[:1])
            counts["PERSON"] += 1
            return m.group(0).replace(name, PLACEHOLDERS["PERSON"])

        def weak(m: re.Match) -> str:
            name = m.group(1)
            if any(w.lower() in _NOT_NAMES for w in name.split()):
                return m.group(0)
            counts["PERSON"] += 1
            return m.group(0).replace(name, PLACEHOLDERS["PERSON"])

        def gazetteer(m: re.Match) -> str:
            words = m.group(1).split()
            if words[0].lower() not in _FIRST_NAMES:
                return m.group(0)
            name = " ".join(words if len(words) == 2 and words[1].lower() not in _NOT_NAMES else words[:1])
            counts["PERSON"] += 1
            return m.group(0).replace(name, PLACEHOLDERS["PERSON"], 1)

        s = _PERSON_STRONG.sub(strong, s)
        s = _PERSON_WEAK.sub(weak, s)
        s = _PERSON_GAZ.sub(gazetteer, s)
        if self._ner is not None:
            spans = [e for e in self._ner(s) if e.get("entity_group") == "PER" and e.get("score", 0) >= 0.90]
            for e in sorted(spans, key=lambda e: e["start"], reverse=True):
                if s[e["start"]:e["end"]].startswith("["):
                    continue
                s = s[: e["start"]] + PLACEHOLDERS["PERSON"] + s[e["end"]:]
                counts["PERSON"] += 1
        return s


_default = PIIRedactor()


def redact(text: str | None) -> RedactionResult:
    return _default.redact(text)


class RedactingFilter(logging.Filter):
    """Logging filter that redacts PII from every record before it is emitted."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        record.msg = _default.redact(msg).text
        record.args = ()
        return True


def get_safe_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not any(isinstance(f, RedactingFilter) for f in logger.filters):
        logger.addFilter(RedactingFilter())
    return logger
