"""Convert the Amazon clothing CSV (Review + Cons_rating) into interim train data.

Source (default):
  D:\\Downloads\\data_amazon.xlsx - Sheet1.csv\\data_amazon.xlsx - Sheet1.csv

Writes:
  data/interim/amazon_clothing/reviews.csv   (text, rating, created_at synthetic, review_id)
  data/interim/amazon_clothing/meta.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import config  # noqa: E402

DEFAULT_SRC = Path(r"D:\Downloads\data_amazon.xlsx - Sheet1.csv\data_amazon.xlsx - Sheet1.csv")
OUT_DIR = config.DATA_DIR / "interim" / "amazon_clothing"
START = datetime(2024, 1, 1)


def rating_to_sent(r: int) -> str:
    if r <= 2:
        return "negative"
    if r == 3:
        return "neutral"
    return "positive"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    if not args.src.exists():
        print(f"Source not found: {args.src}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    out_csv = args.out / "reviews.csv"

    n, skipped, counts = 0, 0, {"negative": 0, "neutral": 0, "positive": 0}
    with args.src.open("r", encoding="utf-8", errors="replace", newline="") as fh, \
            out_csv.open("w", encoding="utf-8", newline="") as out:
        reader = csv.DictReader(fh)
        fields = {h.lower(): h for h in (reader.fieldnames or [])}
        review_col = fields.get("review")
        rating_col = fields.get("cons_rating") or fields.get("rating")
        if not review_col or not rating_col:
            print(f"Need Review + Cons_rating columns; got {reader.fieldnames}", file=sys.stderr)
            return 1
        w = csv.DictWriter(out, fieldnames=["review_id", "text", "rating", "gt_sentiment", "created_at", "source"])
        w.writeheader()
        for i, row in enumerate(reader, start=1):
            text = (row.get(review_col) or "").strip()
            if not text:
                skipped += 1
                continue
            try:
                rating = int(round(float(row.get(rating_col))))
            except (TypeError, ValueError):
                skipped += 1
                continue
            if rating < 1 or rating > 5:
                skipped += 1
                continue
            sent = rating_to_sent(rating)
            # Deterministic synthetic date from hash (training corpus only; not for upload radar claims)
            h = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
            day = h % 360
            sec = h % 86400
            created = START + timedelta(days=day, seconds=sec)
            rid = f"AC{i:06d}"
            w.writerow({
                "review_id": rid,
                "text": text,
                "rating": rating,
                "gt_sentiment": sent,
                "created_at": created.isoformat(),
                "source": "amazon_clothing",
            })
            counts[sent] += 1
            n += 1

    meta = {
        "source_path": str(args.src),
        "rows": n,
        "skipped": skipped,
        "label_counts": counts,
        "note": "created_at is synthetic (hash-based) for training corpus only",
        "out_csv": str(out_csv),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
