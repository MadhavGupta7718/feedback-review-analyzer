"""Generate the deterministic synthetic Nimbus review dataset and a small test fixture."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import config  # noqa: E402
from ml.data import synthetic  # noqa: E402

OUT = config.DATA_DIR / "raw" / "synthetic_reviews.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    reviews = synthetic.generate(seed=args.seed)
    synthetic.write_csv(reviews, OUT)
    fixture = [r for i, r in enumerate(reviews) if i % 35 == 0]
    synthetic.write_csv(fixture, config.FIXTURES_DIR / "synthetic_fixture.csv")

    weeks = Counter((r.created_at[:10]) for r in reviews)
    summary = {
        "seed": args.seed,
        "rows": len(reviews),
        "fixture_rows": len(fixture),
        "themes": dict(Counter(r.gt_theme for r in reviews).most_common()),
        "sentiment": dict(Counter(r.gt_sentiment for r in reviews)),
        "pii_rows": sum(1 for r in reviews if r.gt_pii),
        "pii_types": dict(Counter(r.gt_pii for r in reviews if r.gt_pii)),
        "date_min": min(r.created_at for r in reviews),
        "date_max": max(r.created_at for r in reviews),
        "distinct_days": len(weeks),
    }
    dest = config.ARTIFACTS_DIR / "reports" / "synthetic_summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
