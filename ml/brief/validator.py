"""Validation of LLM-written brief prose against the verified fact sheet.

The LLM writes prose containing slots such as {E1.current}; it must never write digits itself. This module
parses the model output, rejects anything that could change or invent a fact, and substitutes slot values.

Checks (all must pass):
  format            SUMMARY and INVESTIGATE sections present, lengths within bounds
  no_raw_digits     no digits outside slots (review IDs from the fact sheet excepted)
  known_slots       every {slot} exists in the fact sheet
  known_review_ids  every review ID mentioned is one of the fact sheet's evidence IDs
  slot_consistency  a sentence naming one theme does not use another theme's numbers
  direction         never "from {Ek.current} to {Ek.previous}"; no decrease wording for NEW/EMERGING themes,
                    no increase wording for DECLINING themes
  no_causal         no causal claims ("caused by", "due to", "led to", ...)
  hedged            no certainty claims ("proves", "definitely", ...)
  known_theme_names quoted Title-Case names must be themes from the fact sheet
  known_theme_names quoted Title-Case names must be themes from the fact sheet
  coverage          every emerging theme (E-keys) is mentioned in the summary
  no_pii            the final text passes the leak scanner
"""
from __future__ import annotations

import re

from ml.pii.leak_scan import scan

SLOT = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_]+)?)\}")
REVIEW_ID = re.compile(r"\bR\d{3,8}\b")
CAUSAL = re.compile(
    r"\b(caus(?:e|es|ed|ing)|because|due to|led to|leads? to|lead to|result(?:s|ed)? in|as a result|trigger(?:s|ed)?|"
    r"driven by|responsible for|root cause|attributable|stems? from|owing to|therefore|thus|hence)\b", re.I)
INVESTIGATE_CAUSES = re.compile(
    r"\b(?:investigat\w*|identify\w*|determin\w*|explor\w*|understand\w*|look into|find|assess\w*|diagnos\w*)\s+"
    r"(?:the\s+)?(?:possible\s+|potential\s+|underlying\s+|root\s+)*caus(?:e|es)\b", re.I)
CERTAIN = re.compile(r"\b(prove[sn]?|proof|definitely|certainly|undoubtedly|clearly shows|confirm(?:s|ed)|guarantee[sd]?)\b", re.I)
DECREASE = re.compile(r"\b(decreas\w*|declin\w*|drop(?:ped|s)?|fell|fall(?:ing|s)?|reduc\w*|shr[iu]nk\w*)\b", re.I)
INCREASE = re.compile(r"\b(increas\w*|ris(?:e|es|ing)|rose|grow\w*|grew|surg\w*|spik\w*|jump\w*|climb\w*)\b", re.I)
KEYED = re.compile(r"^([ECD]\d+)\.(\w+)$")


def _fmt(slot: str, value) -> str:
    if isinstance(value, bool) or value is None:
        return str(value)
    if slot.endswith("_pct") and isinstance(value, (int, float)):
        return f"{value:g}%"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def parse_sections(raw: str) -> tuple[str, list[str]]:
    text = raw.strip().replace("\r\n", "\n")
    m = re.search(r"SUMMARY:\s*(.*?)\s*(?:INVESTIGATE:\s*(.*))?$", text, re.S | re.I)
    if not m:
        return "", []
    summary = " ".join(m.group(1).split())
    items = []
    for line in (m.group(2) or "").split("\n"):
        line = line.strip()
        if re.match(r"^[-*•]\s+", line):
            items.append(" ".join(re.sub(r"^[-*•]\s+", "", line).split()))
    return summary, items


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def validate(raw: str, facts: dict) -> dict:
    slots: dict = facts["slots"]
    allowed_ids = set(facts.get("allowed_review_ids", []))
    status_by_key = {e["key"]: e["status"] for e in facts.get("emerging", [])}
    status_by_key.update({d["key"]: "DECLINING" for d in facts.get("declining", [])})
    errors: list[str] = []
    checks: dict[str, bool] = {}

    summary, items = parse_sections(raw)
    checks["format"] = bool(summary) and 40 <= len(summary) <= 1200 and 1 <= len(items) <= 6 and all(10 <= len(i) <= 400 for i in items)
    if not checks["format"]:
        errors.append("output must contain 'SUMMARY:' (40-1200 chars) and 'INVESTIGATE:' with 1-6 '- ' bullet lines")
    parts = [summary, *items]
    joined = "\n".join(parts)

    used = SLOT.findall(joined)
    unknown = sorted({s for s in used if s not in slots})
    checks["known_slots"] = not unknown
    if unknown:
        errors.append(f"unknown slots: {', '.join(unknown[:5])}")

    ids = set(REVIEW_ID.findall(joined))
    bad_ids = sorted(ids - allowed_ids)
    checks["known_review_ids"] = not bad_ids
    if bad_ids:
        errors.append(f"review IDs not in the fact sheet: {', '.join(bad_ids[:5])}")

    stripped = REVIEW_ID.sub(" ", SLOT.sub(" ", joined))
    digits = re.findall(r"\d[\d,.]*%?", stripped)
    checks["no_raw_digits"] = not digits
    if digits:
        errors.append(f"raw numbers written instead of slots: {', '.join(digits[:5])}")

    consistency_ok, direction_ok = True, True
    for sent in _sentences(joined):
        keys_named = {m.group(1) for s in SLOT.findall(sent) if (m := KEYED.match(s)) and m.group(2) == "name"}
        for s in SLOT.findall(sent):
            m = KEYED.match(s)
            if m and m.group(2) != "name" and keys_named and m.group(1) not in keys_named:
                consistency_ok = False
                errors.append(f"sentence uses {{{s}}} next to a different theme's name: '{sent[:120]}'")
                break
        for key in {m.group(1) for s in SLOT.findall(sent) if (m := KEYED.match(s))}:
            if re.search(r"\{%s\.current\}\W+(?:\w+\W+){0,3}?(?:to|->|→)\W+\{%s\.previous\}" % (re.escape(key), re.escape(key)), sent):
                direction_ok = False
                errors.append(f"reversed direction for {key}: '{sent[:120]}'")
            st = status_by_key.get(key)
            if len(keys_named | {key}) == 1:
                if st in ("NEW", "EMERGING") and DECREASE.search(sent):
                    direction_ok = False
                    errors.append(f"decrease wording for {st} theme {key}: '{sent[:120]}'")
                if st == "DECLINING" and INCREASE.search(sent):
                    direction_ok = False
                    errors.append(f"increase wording for DECLINING theme {key}: '{sent[:120]}'")
    checks["slot_consistency"] = consistency_ok
    checks["direction"] = direction_ok

    causal = CAUSAL.findall("\n".join([summary, *(INVESTIGATE_CAUSES.sub(" ", i) for i in items)]))
    checks["no_causal"] = not causal
    if causal:
        errors.append(f"causal language: {', '.join(sorted(set(c.lower() for c in causal)))}")
    certain = CERTAIN.findall(joined)
    checks["hedged"] = not certain
    if certain:
        errors.append(f"certainty language: {', '.join(sorted(set(c.lower() for c in certain)))}")

    allowed_names = {n.lower() for n in facts.get("allowed_theme_names", [])}
    quoted = [q.strip(" ,.") for q in re.findall(r"[\"“]([^\"”]{2,60})[\"”]", joined)]
    invented = sorted({q for q in quoted if q[:1].isupper() and q.lower() not in allowed_names})
    checks["known_theme_names"] = not invented
    if invented:
        errors.append(f"quoted names that are not themes in the fact sheet: {', '.join(invented[:5])}")

    allowed_names = {n.lower() for n in facts.get("allowed_theme_names", [])}
    quoted = [q.strip(" ,.") for q in re.findall(r"[\"“]([^\"”]{2,60})[\"”]", joined)]
    invented = sorted({q for q in quoted if q[:1].isupper() and q.lower() not in allowed_names})
    checks["known_theme_names"] = not invented
    if invented:
        errors.append(f"quoted names that are not themes in the fact sheet: {', '.join(invented[:5])}")

    missing = [e["key"] for e in facts.get("emerging", [])
               if f"{{{e['key']}.name}}" not in summary and e["name"].lower() not in summary.lower()]
    checks["coverage"] = not missing
    if missing:
        errors.append(f"summary does not mention emerging themes: {', '.join('{' + k + '.name}' for k in missing)}")

    def fill(t: str) -> str:
        return SLOT.sub(lambda m: _fmt(m.group(1), slots[m.group(1)]) if m.group(1) in slots else m.group(0), t)

    final_summary = fill(summary)
    final_items = [fill(i) for i in items]
    leaks = [h for t in [final_summary, *final_items] for h in scan(t)]
    checks["no_pii"] = not leaks
    if leaks:
        errors.append(f"leak scanner flagged: {', '.join(sorted(set(leaks)))}")

    return {"passed": not errors, "errors": errors, "checks": checks, "summary": final_summary, "investigate": final_items,
            "slots_used": sorted(set(used))}
