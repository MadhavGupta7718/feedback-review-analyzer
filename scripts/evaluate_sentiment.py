"""Phase 4: evaluate RoBERTa sentiment against the deterministic Sentiment140 labelled sample, benchmark
GPU vs CPU, check reproducibility, and score the synthetic set's (weak) 3-class labels.
Writes artifacts/reports/sentiment_validation.json (served by GET /sentiment/validation)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import config  # noqa: E402
from ml.data import synthetic  # noqa: E402
from ml.evaluation import metrics  # noqa: E402
from ml.preprocessing.clean import clean_batch  # noqa: E402
from ml.sentiment.model import LABELS, SentimentModel  # noqa: E402

EVAL_PATH = config.DATA_DIR / "interim" / "s140_eval_5k.csv"


def load_eval() -> tuple[list[str], np.ndarray, dict]:
    with open(EVAL_PATH, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    recs = [{"review_id": r["id"], "text": r["text"], "created_at": r["date"], "target": r["target"]} for r in rows]
    cleaned, rep = clean_batch(recs)
    texts = [c["text_redacted"] for c in cleaned]
    y = np.array([metrics.NEG if c["target"] == "0" else metrics.POS for c in cleaned])
    return texts, y, rep.__dict__


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-sample", type=int, default=500)
    args = ap.parse_args()
    import torch

    texts, y, clean_rep = load_eval()
    report: dict = {
        "evaluation_date_utc": datetime.now(timezone.utc).isoformat(),
        "model": config.SENTIMENT_MODEL,
        "dataset": "Sentiment140 (training.1600000.processed.noemoticon.csv)",
        "sample": {"path": "data/interim/s140_eval_5k.csv", "seed": config.SEED, "size": int(len(y)),
                   "label_counts": {"negative": int((y == 0).sum()), "positive": int((y == 2).sum())},
                   "selection": "stratified random 2,500 per label; label-conflicting duplicate IDs excluded",
                   "preprocessing": clean_rep},
        "ground_truth_note": "Sentiment140 has binary labels only (0=negative, 4=positive). It contains NO neutral ground truth.",
        "methodology": metrics.__doc__.strip(),
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentimentModel(device)
    report["model_revision"] = model.revision
    report["device"] = device
    report["gpu"] = torch.cuda.get_device_name(0) if device == "cuda" else None

    model.predict(texts[:64], batch_size=64)  # warm-up
    out = model.predict(texts, batch_size=64)
    report["metrics"] = metrics.evaluate_binary_truth(y, out.probs)
    report["mean_confidence"] = round(float(out.confidences.mean()), 4)

    out2 = model.predict(texts, batch_size=64)
    report["reproducibility"] = {
        "identical_labels_between_runs": bool(out.labels == out2.labels),
        "max_abs_prob_diff": float(np.abs(out.probs - out2.probs).max()),
    }

    bench = []
    if device == "cuda":
        for bs in (16, 32, 64, 128, 256):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                o = model.predict(texts, batch_size=bs)
                bench.append({"device": "cuda", "dtype": "float16", "batch_size": bs, "n": len(texts),
                              "seconds": round(o.seconds, 2), "reviews_per_sec": round(len(texts) / o.seconds, 1),
                              "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3), "oom": False})
            except torch.cuda.OutOfMemoryError:
                bench.append({"device": "cuda", "batch_size": bs, "oom": True})
    del model
    torch.cuda.empty_cache()

    cpu = SentimentModel("cpu")
    # evenly spaced subset so both labels are represented (the eval file is ordered by source line)
    sub_idx = np.linspace(0, len(texts) - 1, args.cpu_sample).astype(int)
    sub = [texts[i] for i in sub_idx]
    cpu.predict(sub[:32], batch_size=32)
    o_cpu = cpu.predict(sub, batch_size=32)
    bench.append({"device": "cpu", "dtype": "float32", "batch_size": 32, "n": len(sub), "seconds": round(o_cpu.seconds, 2),
                  "reviews_per_sec": round(len(sub) / o_cpu.seconds, 1), "torch_threads": torch.get_num_threads()})
    agree = float(np.mean(np.array(o_cpu.labels) == np.array(out.labels)[sub_idx]))
    report["cpu_vs_gpu_label_agreement"] = round(agree, 4)
    report["cpu_metrics_on_subset"] = metrics.evaluate_binary_truth(y[sub_idx], o_cpu.probs)["binary_forced"]
    report["gpu_metrics_on_same_subset"] = metrics.evaluate_binary_truth(y[sub_idx], out.probs[sub_idx])["binary_forced"]
    report["benchmark"] = bench

    # Synthetic 3-class (weak labels written by templates, NOT human annotation)
    rev = synthetic.generate(seed=config.SEED)
    cleaned, _ = clean_batch([r.__dict__ for r in rev])
    gt = {r.review_id: r.gt_sentiment for r in rev}
    s_texts = [c["text_redacted"] for c in cleaned]
    del cpu
    model = SentimentModel(device)
    o_syn = model.predict(s_texts, batch_size=64)
    idx = {l: i for i, l in enumerate(LABELS)}
    y_syn = np.array([idx[gt[c["review_id"]]] for c in cleaned])
    y_hat = np.array([idx[l] for l in o_syn.labels])
    report["synthetic_3class"] = {
        "note": "Synthetic labels come from the template sentiment used to generate each review (weak labels, not human annotation).",
        **metrics.evaluate_3class(y_syn, y_hat),
        "seconds": round(o_syn.seconds, 2),
    }

    dest = config.ARTIFACTS_DIR / "reports" / "sentiment_validation.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("methodology",)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
