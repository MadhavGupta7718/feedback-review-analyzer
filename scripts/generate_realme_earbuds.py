"""Generate ~15K synthetic Realme earbuds reviews → data/raw/reviews/realme_earbuds_reviews.csv"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.data.realme_earbuds import generate, write_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "raw" / "reviews" / "realme_earbuds_reviews.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--scale", type=float, default=1.55, help="1.55 ≈ 15K rows")
    args = ap.parse_args()
    rows = generate(seed=args.seed, scale=args.scale)
    path = write_csv(rows, args.out)
    themes = Counter(r.gt_theme for r in rows)
    sent = Counter(r.gt_sentiment for r in rows)
    print(f"Wrote {path}")
    print(f"rows={len(rows)} themes={len(themes)} sentiment={dict(sent)}")
    print("top themes:", themes.most_common(8))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
