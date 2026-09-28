"""Build the deterministic Sentiment140 train/val/test split -> data/interim/s140_split/ + artifacts/reports/s140_split.json."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import config  # noqa: E402
from ml.evaluation.s140_split import build  # noqa: E402


def main() -> int:
    t0 = time.perf_counter()
    rep = build()
    rep["seconds"] = round(time.perf_counter() - t0, 1)
    dest = config.ARTIFACTS_DIR / "reports" / "s140_split.json"
    dest.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
