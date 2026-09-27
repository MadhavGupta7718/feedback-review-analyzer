"""Streaming validation and deterministic sampling for the raw Sentiment140 CSV.

The file is read with the stdlib csv module one row at a time so the 1.6M-row / ~240 MB file
never has to be held in memory as a DataFrame. Every malformed row is counted, never dropped
silently.
"""
from __future__ import annotations

import csv
import hashlib
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from ml import config

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

S140_DATE_FORMAT = "%a %b %d %H:%M:%S %Y"  # timezone token removed before parsing
LABEL_MAP = {"0": "negative", "2": "neutral", "4": "positive"}


def parse_s140_date(raw: str) -> datetime | None:
    parts = raw.split()
    if len(parts) != 6:
        return None
    try:
        return datetime.strptime(" ".join(parts[:4] + parts[5:]), S140_DATE_FORMAT)
    except ValueError:
        return None


def iter_rows(path: Path = config.S140_PATH):
    """Yield (line_number, fields) for every CSV record, including malformed ones."""
    with open(path, "r", encoding=config.S140_ENCODING, newline="") as fh:
        reader = csv.reader(fh)
        for i, fields in enumerate(reader, start=1):
            yield i, fields


@dataclass
class ValidationReport:
    path: str
    size_bytes: int
    encoding: str
    rows: int = 0
    well_formed_rows: int = 0
    malformed_rows: int = 0
    malformed_examples: list = field(default_factory=list)
    column_count_distribution: dict = field(default_factory=dict)
    label_distribution: dict = field(default_factory=dict)
    unexpected_labels: dict = field(default_factory=dict)
    flag_distribution: dict = field(default_factory=dict)
    missing_values: dict = field(default_factory=dict)
    empty_text: int = 0
    duplicate_ids: int = 0
    duplicate_id_conflicting_labels: int = 0
    excluded_conflicting_ids: int = 0
    duplicate_texts: int = 0
    unique_users: int = 0
    unparseable_dates: int = 0
    date_min: str | None = None
    date_max: str | None = None
    rows_per_month: dict = field(default_factory=dict)
    text_length: dict = field(default_factory=dict)
    replacement_char_rows: int = 0
    html_entity_rows: int = 0
    url_rows: int = 0
    mention_rows: int = 0


def validate(path: Path = config.S140_PATH, progress_every: int = 400_000) -> tuple[ValidationReport, dict]:
    """Single streaming pass. Returns the report and an index {label: [row_numbers]} of
    well-formed, non-empty rows with unique IDs for later deterministic sampling."""
    report = ValidationReport(path=str(path), size_bytes=path.stat().st_size, encoding=config.S140_ENCODING)
    col_counts: Counter = Counter()
    labels: Counter = Counter()
    flags: Counter = Counter()
    months: Counter = Counter()
    missing = Counter({c: 0 for c in config.S140_COLUMNS})
    id_label: dict[int, str] = {}
    text_hashes: set[bytes] = set()
    users: set[str] = set()
    lengths: list[int] = []
    dmin = dmax = None
    index: dict[str, list[int]] = {"0": [], "4": []}

    for line_no, fields in iter_rows(path):
        report.rows += 1
        col_counts[len(fields)] += 1
        if len(fields) != 6:
            report.malformed_rows += 1
            if len(report.malformed_examples) < 5:
                report.malformed_examples.append({"line": line_no, "fields": len(fields)})
            continue
        report.well_formed_rows += 1
        target, rid, date, flag, user, text = fields
        for col, val in zip(config.S140_COLUMNS, fields):
            if val is None or val.strip() == "":
                missing[col] += 1
        labels[target] += 1
        if target not in ("0", "4"):
            report.unexpected_labels[target] = report.unexpected_labels.get(target, 0) + 1
        flags[flag] += 1
        users.add(user)

        is_dup_id = False
        rid_int = None
        try:
            rid_int = int(rid)
            if rid_int in id_label:
                is_dup_id = True
                report.duplicate_ids += 1
                if id_label[rid_int] != target:
                    report.duplicate_id_conflicting_labels += 1
                    id_label[rid_int] = "CONFLICT"
            else:
                id_label[rid_int] = target
        except ValueError:
            missing["id"] += 1

        h = hashlib.blake2b(text.strip().lower().encode("utf-8", "replace"), digest_size=8).digest()
        if h in text_hashes:
            report.duplicate_texts += 1
        else:
            text_hashes.add(h)

        stripped = text.strip()
        if not stripped:
            report.empty_text += 1
        lengths.append(len(stripped))
        if "\ufffd" in text:
            report.replacement_char_rows += 1
        if "&quot;" in text or "&amp;" in text or "&lt;" in text or "&gt;" in text:
            report.html_entity_rows += 1
        if "http://" in text or "https://" in text or "www." in text:
            report.url_rows += 1
        if "@" in text:
            report.mention_rows += 1

        dt = parse_s140_date(date)
        if dt is None:
            report.unparseable_dates += 1
        else:
            months[dt.strftime("%Y-%m")] += 1
            dmin = dt if dmin is None or dt < dmin else dmin
            dmax = dt if dmax is None or dt > dmax else dmax

        if stripped and not is_dup_id and target in index and dt is not None:
            index[target].append((line_no, rid_int))

        if progress_every and report.rows % progress_every == 0:
            print(f"  ... {report.rows:,} rows scanned", flush=True)

    # A duplicated ID with conflicting labels means the same tweet is both "negative" and
    # "positive" in the ground truth; every copy is excluded from sampling.
    conflicting = {rid for rid, lab in id_label.items() if lab == "CONFLICT"}
    report.excluded_conflicting_ids = len(conflicting)
    index = {lab: [ln for ln, rid in rows if rid not in conflicting] for lab, rows in index.items()}

    arr = np.asarray(lengths)
    report.column_count_distribution = {str(k): v for k, v in sorted(col_counts.items())}
    report.label_distribution = dict(sorted(labels.items()))
    report.flag_distribution = dict(flags.most_common(5))
    report.missing_values = dict(missing)
    report.unique_users = len(users)
    report.date_min = dmin.isoformat() if dmin else None
    report.date_max = dmax.isoformat() if dmax else None
    report.rows_per_month = dict(sorted(months.items()))
    if arr.size:
        report.text_length = {
            "min": int(arr.min()), "max": int(arr.max()), "mean": round(float(arr.mean()), 2),
            "median": float(np.median(arr)), "p95": float(np.percentile(arr, 95)),
        }
    return report, index


def deterministic_sample(index: dict[str, list[int]], per_label: int, seed: int, exclude: set[int] | None = None) -> list[int]:
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for label in sorted(index):
        pool = np.asarray([r for r in index[label] if not exclude or r not in exclude])
        chosen.extend(rng.choice(pool, size=per_label, replace=False).tolist())
    return sorted(chosen)


def extract_rows(line_numbers: list[int], path: Path = config.S140_PATH) -> list[list[str]]:
    wanted = set(line_numbers)
    rows = []
    for line_no, fields in iter_rows(path):
        if line_no in wanted:
            rows.append([str(line_no)] + fields)
    return rows


def pseudonymise_user(user: str) -> str:
    return "user_" + hashlib.sha256(user.encode("utf-8")).hexdigest()[:10]
