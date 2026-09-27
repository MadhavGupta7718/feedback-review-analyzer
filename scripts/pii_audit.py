"""Measure PII redaction on the synthetic batch (planted ground truth) and on the Sentiment140 10K batch
(no ground truth: counts + residual-leak scan + a sample of redacted outputs for manual inspection).
Writes artifacts/reports/pii_audit.json. Only redacted text is written."""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import config  # noqa: E402
from ml.data import synthetic  # noqa: E402
from ml.pii.leak_scan import scan, scan_many  # noqa: E402
from ml.preprocessing.clean import clean_batch  # noqa: E402


def main() -> int:
    reviews = synthetic.generate(seed=config.SEED)
    cleaned, rep = clean_batch([r.__dict__ for r in reviews])
    by_id = {r["review_id"]: r for r in cleaned}
    names = {n.lower() for n in synthetic.FIRST_NAMES + synthetic.LAST_NAMES}
    per_type = Counter()
    caught = Counter()
    fp_rows = 0
    clean_rows = 0
    for r in reviews:
        c = by_id.get(r.review_id)
        if c is None:
            continue
        t = c["text_redacted"]
        if r.gt_pii:
            per_type[r.gt_pii] += 1
            words = {w.strip(".,!").lower() for w in t.split()}
            leaked = bool(scan(t)) or (r.gt_pii == "person" and bool(words & names))
            if not leaked:
                caught[r.gt_pii] += 1
        else:
            clean_rows += 1
            fp_rows += 1 if c["pii_redaction_count"] else 0
    synth = {
        "clean_report": rep.__dict__,
        "planted_pii_rows": sum(per_type.values()),
        "recall_by_type": {k: f"{caught[k]}/{v}" for k, v in sorted(per_type.items())},
        "overall_recall": round(sum(caught.values()) / max(1, sum(per_type.values())), 4),
        "false_positive_rows": fp_rows,
        "rows_without_planted_pii": clean_rows,
        "false_positive_row_rate": round(fp_rows / max(1, clean_rows), 5),
        "residual_leak_scan": scan_many(c["text_redacted"] for c in cleaned),
    }

    s140 = {}
    batch = config.DATA_DIR / "interim" / "s140_batch_10k.csv"
    if batch.exists():
        with open(batch, encoding="utf-8") as fh:
            rows = [{"review_id": f"S{r['id']}", "text": r["text"], "created_at": r["date"]} for r in csv.DictReader(fh)]
        s_clean, s_rep = clean_batch(rows)
        redacted_examples = [c["text_redacted"] for c in s_clean if c["pii_redaction_count"]][:25]
        s140 = {
            "clean_report": s_rep.__dict__,
            "residual_leak_scan": scan_many(c["text_redacted"] for c in s_clean),
            "redacted_examples_for_manual_review": redacted_examples,
        }

    out = {"synthetic": synth, "sentiment140_batch": s140}
    dest = config.ARTIFACTS_DIR / "reports" / "pii_audit.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
