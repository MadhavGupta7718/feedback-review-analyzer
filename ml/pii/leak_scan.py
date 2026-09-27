"""Independent leak scanner used by tests and the pipeline audit.

Deliberately stricter/simpler than the redactor: it flags anything that *looks* like residual PII in
text that is about to leave the pipeline (DB rows, API payloads, logs, exports, LLM prompts).
"""
from __future__ import annotations

import re

LEAK_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}"),
    "url": re.compile(r"(?i)https?://|\bwww\.[a-z0-9]"),
    "long_digit_run": re.compile(r"\d(?:[ ()+.-]?\d){6,}"),
    "prefixed_id": re.compile(r"(?i)\b(?:ORD|ACC|CUST)-?\d{4,}"),
    "handle": re.compile(r"(?<![\w\[])@[A-Za-z0-9_]{2,}"),
}


def _benign_digits(run: str) -> bool:
    digits = re.sub(r"\D", "", run)
    if len(set(digits)) == 1:  # "<3333333" hearts, "1111111"
        return True
    if re.fullmatch(r"[1-9]\d{0,2}0{4,}", digits):  # round numbers like 1000000, 1600000
        return True
    if re.fullmatch(r"(19|20)\d{2}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])", run):  # ISO date only
        return True
    if re.fullmatch(r"(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(19|20)\d{2}", digits):  # MMDDYYYY
        return True
    return False


def scan(text: str) -> list[str]:
    hits = []
    for name, pat in LEAK_PATTERNS.items():
        if name == "long_digit_run":
            if any(not _benign_digits(m.group(0)) for m in pat.finditer(text or "")):
                hits.append(name)
        elif pat.search(text or ""):
            hits.append(name)
    return hits


def scan_many(texts) -> dict[str, int]:
    hits: dict[str, int] = {}
    for t in texts:
        for name in scan(t):
            hits[name] = hits.get(name, 0) + 1
    return hits
