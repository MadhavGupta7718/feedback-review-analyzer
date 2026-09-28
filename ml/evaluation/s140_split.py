"""Deterministic 80/10/10 train / validation / test split of the full Sentiment140 file.

Leakage control: rows are grouped by a normalised text key (lower-case, HTML entities decoded, @mentions -> @user,
links -> http, whitespace collapsed) and a whole group is assigned to one split by a seeded hash of the key, so the
same or trivially re-posted text can never appear in two splits. Label proportions stay ~50/50 because assignment is
independent of the label (checked and reported, not assumed).

Exclusions (every one is counted in the report, nothing is dropped silently):
  malformed rows, labels other than 0/4, empty text,
  every copy of a tweet ID that occurs with conflicting labels,
  repeated copies of a tweet ID with the same label (the first copy is kept),
  every row of a text group that carries both labels (the same text is both "negative" and "positive").

Split files hold line numbers and labels only; val/test/train-subset texts are extracted to data/interim (gitignored).
"""
from __future__ import annotations

import csv
import hashlib
import html
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ml import config
from ml.data.sentiment140 import iter_rows

SPLIT_DIR = config.DATA_DIR / "interim" / "s140_split"
FRACTIONS = {"train": 800, "val": 100, "test": 100}  # per mille
_MENTION = re.compile(r"@\w+")
_LINK = re.compile(r"(https?://\S+|www\.\S+)")
_WS = re.compile(r"\s+")


def group_key(text: str) -> str:
    s = html.unescape(text).lower()
    s = _MENTION.sub("@user", s)
    s = _LINK.sub("http", s)
    return _WS.sub(" ", s).strip()


def assign_split(key: str, seed: int = config.SEED) -> str:
    v = int.from_bytes(hashlib.blake2b(f"{seed}:{key}".encode("utf-8", "replace"), digest_size=8).digest(), "big") % 1000
    return "train" if v < FRACTIONS["train"] else "val" if v < FRACTIONS["train"] + FRACTIONS["val"] else "test"


def _khash(key: str) -> bytes:
    return hashlib.blake2b(key.encode("utf-8", "replace"), digest_size=8).digest()


def build(path: Path = config.S140_PATH, seed: int = config.SEED, train_subset_per_label: int = 50_000) -> dict:
    counts: Counter = Counter()
    id_labels: dict[str, set] = defaultdict(set)
    key_labels: dict[bytes, set] = defaultdict(set)
    rows: list[tuple[int, str, str, bytes, str]] = []  # line, label, id, key hash, split
    for line_no, f in iter_rows(path):
        counts["rows"] += 1
        if len(f) != 6:
            counts["malformed"] += 1
            continue
        target, rid, _date, _flag, _user, text = f
        if target not in ("0", "4"):
            counts["unexpected_label"] += 1
            continue
        if not text.strip():
            counts["empty_text"] += 1
            continue
        key = group_key(text)
        kh = _khash(key)
        id_labels[rid].add(target)
        key_labels[kh].add(target)
        rows.append((line_no, target, rid, kh, assign_split(key, seed)))

    conflict_ids = {rid for rid, labs in id_labels.items() if len(labs) > 1}
    conflict_keys = {kh for kh, labs in key_labels.items() if len(labs) > 1}
    seen_ids: set = set()
    kept: dict[str, list[tuple[int, str]]] = {"train": [], "val": [], "test": []}
    split_keys: dict[str, set] = {s: set() for s in kept}
    for line_no, target, rid, kh, split in rows:
        if rid in conflict_ids:
            counts["conflicting_id_rows"] += 1
            continue
        if rid in seen_ids:
            counts["repeated_id_rows"] += 1
            continue
        seen_ids.add(rid)
        if kh in conflict_keys:
            counts["conflicting_text_rows"] += 1
            continue
        kept[split].append((line_no, target))
        split_keys[split].add(kh)

    leaks = {f"{a}&{b}": len(split_keys[a] & split_keys[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    assert all(v == 0 for v in leaks.values()), f"text-group leakage across splits: {leaks}"

    rng = np.random.default_rng(seed)
    train_subset = []
    for lab in ("0", "4"):
        pool = [ln for ln, t in kept["train"] if t == lab]
        train_subset.extend(int(x) for x in rng.choice(pool, size=min(train_subset_per_label, len(pool)), replace=False))
    train_subset_set = set(train_subset)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for split, items in kept.items():
        with open(SPLIT_DIR / f"{split}_index.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["line", "target"])
            w.writerows(sorted(items))

    wanted = {ln: "val" for ln, _ in kept["val"]} | {ln: "test" for ln, _ in kept["test"]}
    wanted |= {ln: "train_subset" for ln in train_subset_set}
    out = {name: open(SPLIT_DIR / f"{name}.csv", "w", newline="", encoding="utf-8") for name in ("val", "test", "train_subset")}
    writers = {k: csv.writer(v) for k, v in out.items()}
    for w in writers.values():
        w.writerow(["line", "id", "target", "text"])
    for line_no, f in iter_rows(path):
        name = wanted.get(line_no)
        if name:
            writers[name].writerow([line_no, f[1], f[0], f[5]])
    for fh in out.values():
        fh.close()

    def dist(items):
        c = Counter(t for _, t in items)
        n = sum(c.values())
        return {"n": n, "negative": c["0"], "positive": c["4"], "positive_share": round(c["4"] / n, 4) if n else None}

    return {
        "source": path.name, "seed": seed, "fractions_per_mille": FRACTIONS,
        "method": "text-group hash split; see ml/evaluation/s140_split.py docstring",
        "counts": dict(counts),
        "conflicting_ids": len(conflict_ids), "conflicting_text_groups": len(conflict_keys),
        "splits": {s: dist(items) for s, items in kept.items()},
        "train_subset_for_finetuning": {"per_label": train_subset_per_label, "n": len(train_subset)},
        "cross_split_text_group_overlap": leaks,
    }
