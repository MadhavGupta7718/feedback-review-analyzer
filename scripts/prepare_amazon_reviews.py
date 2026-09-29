"""Turn an Amazon Reviews 2023 category file (McAuley-Lab/Amazon-Reviews-2023, raw/review_categories/<Category>.jsonl)
into a review CSV the pipeline accepts (`python -m ml.pipeline --source <csv>`).

Keeps the most recent `--weeks` weeks ending at the newest review, with real timestamps and star ratings. If that window
holds more than `--max-rows` reviews, an evenly spaced subsample (in time order) is kept, so weekly volume ratios are
preserved. User IDs are dropped; review IDs are sequential; Amazon markup ([[VIDEOID:...]], <br />) is removed. The CSV is written under data/raw (gitignored).

  python scripts/prepare_amazon_reviews.py data/raw/reviews/amazon2023/raw/review_categories/Software.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import config  # noqa: E402

# Amazon export markup, not review content: embedded-video tokens and HTML line breaks
_MARKUP = re.compile(r"\[\[VIDEOID:[0-9a-fA-F]+\]\]|<br\s*/?>", re.IGNORECASE)
_WS = re.compile(r"\s+")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--weeks", type=int, default=8)
    ap.add_argument("--max-rows", type=int, default=15_000)
    ap.add_argument("--end", default=None, help="window end date YYYY-MM-DD (UTC, inclusive); default: newest review. "
                                                "The dataset's final months are sparse because collection ended, which looks like a decline")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.end:
        newest = int((datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc) + timedelta(days=1)).timestamp() * 1000) - 1
    else:
        with open(args.jsonl, encoding="utf-8") as fh:
            newest = max(json.loads(line)["timestamp"] for line in fh)
    cutoff = newest - args.weeks * 7 * 24 * 3600 * 1000

    rows = []
    with open(args.jsonl, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if not cutoff < r["timestamp"] <= newest:
                continue
            text = " ".join(p for p in ((r.get("title") or "").strip(), (r.get("text") or "").strip()) if p)
            text = _WS.sub(" ", _MARKUP.sub(" ", text)).strip()
            if not text:
                continue
            rows.append((r["timestamp"], int(r["rating"]), text))
    rows.sort(key=lambda x: x[0])
    in_window = len(rows)
    if in_window > args.max_rows:
        keep = np.linspace(0, in_window - 1, args.max_rows).astype(int)
        rows = [rows[i] for i in keep]

    out = args.out or config.DATA_DIR / "raw" / "reviews" / f"amazon_{args.jsonl.stem.lower()}_recent.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["review_id", "date", "rating", "text"])
        for n, (ts, rating, text) in enumerate(rows, 1):
            dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            w.writerow([f"A{n:06d}", dt.strftime("%Y-%m-%dT%H:%M:%SZ"), rating, text])

    end = datetime.fromtimestamp(newest / 1000, tz=timezone.utc)
    weeks = Counter(min(args.weeks - 1, int((end - datetime.fromtimestamp(ts / 1000, tz=timezone.utc)) / timedelta(days=7))) for ts, _, _ in rows)
    print(json.dumps({"out": str(out), "newest_review_utc": end.isoformat(), "weeks": args.weeks,
                      "reviews_in_window": in_window, "rows_written": len(rows),
                      "ratings": dict(sorted(Counter(r for _, r, _ in rows).items())),
                      "rows_per_week_oldest_first": [weeks[w] for w in range(args.weeks - 1, -1, -1)]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
