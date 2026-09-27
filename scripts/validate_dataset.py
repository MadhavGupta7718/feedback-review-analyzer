"""Phase 1: validate the raw Sentiment140 file and cut deterministic samples.

Outputs
  artifacts/reports/dataset_validation.json   measured statistics (committed)
  data/interim/s140_eval_5k.csv               balanced labelled sample for sentiment evaluation (gitignored)
  data/interim/s140_batch_10k.csv             balanced 10K batch, disjoint from the eval sample (gitignored)
  tests/fixtures/s140_fixture.csv             200-row fixture for automated tests (committed, usernames pseudonymised)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import config  # noqa: E402
from ml.data import sentiment140 as s140  # noqa: E402

HEADER = ["source_line", "target", "id", "date", "flag", "user", "text"]


def write_sample(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for r in rows:
            r = list(r)
            r[5] = s140.pseudonymise_user(r[5])
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-per-label", type=int, default=2500)
    ap.add_argument("--batch-per-label", type=int, default=5000)
    ap.add_argument("--fixture-per-label", type=int, default=100)
    args = ap.parse_args()

    if not config.S140_PATH.exists():
        print(f"FAILED: dataset not found at {config.S140_PATH}")
        return 1

    t0 = time.perf_counter()
    report, index = s140.validate(config.S140_PATH)
    scan_s = time.perf_counter() - t0

    eval_rows = s140.deterministic_sample(index, args.eval_per_label, seed=config.SEED)
    batch_rows = s140.deterministic_sample(index, args.batch_per_label, seed=config.SEED + 1, exclude=set(eval_rows))
    fixture_rows = s140.deterministic_sample(index, args.fixture_per_label, seed=config.SEED + 2)
    wanted = sorted(set(eval_rows) | set(batch_rows) | set(fixture_rows))
    extracted = {int(r[0]): r for r in s140.extract_rows(wanted)}

    write_sample(config.DATA_DIR / "interim" / "s140_eval_5k.csv", [extracted[i] for i in eval_rows])
    write_sample(config.DATA_DIR / "interim" / "s140_batch_10k.csv", [extracted[i] for i in batch_rows])
    write_sample(config.FIXTURES_DIR / "s140_fixture.csv", [extracted[i] for i in fixture_rows])

    checks = {
        "row_count_matches_expected": report.rows == config.S140_EXPECTED_ROWS,
        "all_rows_have_6_columns": report.malformed_rows == 0,
        "labels_only_0_and_4": set(report.label_distribution) <= {"0", "4"},
        "no_silent_loss": report.rows == report.well_formed_rows + report.malformed_rows,
        "timestamps_parseable": report.unparseable_dates == 0,
        "eval_batch_disjoint": not (set(eval_rows) & set(batch_rows)),
        "conflicting_label_ids_excluded_from_samples": report.excluded_conflicting_ids == report.duplicate_id_conflicting_labels,
    }
    out = {
        "report": asdict(report),
        "scan_seconds": round(scan_s, 1),
        "samples": {
            "seed": config.SEED,
            "eval": {"rows": len(eval_rows), "per_label": args.eval_per_label},
            "batch": {"rows": len(batch_rows), "per_label": args.batch_per_label},
            "fixture": {"rows": len(fixture_rows), "per_label": args.fixture_per_label},
            "eligible_rows_per_label": {k: len(v) for k, v in index.items()},
        },
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAILED",
    }
    dest = config.ARTIFACTS_DIR / "reports" / "dataset_validation.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if out["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
