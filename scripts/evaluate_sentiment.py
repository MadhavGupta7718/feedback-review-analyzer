"""Evaluate the product RoBERTa sentiment model on the held-out Sentiment140 TEST split (scripts/build_s140_split.py),
benchmark GPU vs CPU, check reproducibility, and score the synthetic set's (weak) 3-class labels.

Scorings against the binary truth: the three legacy scorings (binary_forced / strict_3class / abstain), the four
neutral strategies (A-D), and the binary decision at the validation-tuned threshold (config.SENTIMENT_BINARY_THRESHOLD).
Nothing here is tuned: the threshold comes from the validation split. A summary of the accuracy study
(artifacts/reports/sentiment_study_*.json, sentiment_finetune.json) is attached when those reports exist.
Writes artifacts/reports/sentiment_validation.json (served by GET /sentiment/validation)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import config  # noqa: E402
from ml.data import synthetic  # noqa: E402
from ml.evaluation import metrics  # noqa: E402
from ml.preprocessing.clean import clean_batch  # noqa: E402
from ml.sentiment import experiment as X  # noqa: E402
from ml.sentiment.model import LABELS, SentimentModel  # noqa: E402

REPORTS = config.ARTIFACTS_DIR / "reports"
BENCH_N = 5000


def load_test() -> tuple[list[str], np.ndarray, dict]:
    test = X.load_split("test")
    texts = [X.prepare(r).current for r in test.raw]
    split = json.loads((REPORTS / "s140_split.json").read_text(encoding="utf-8"))
    return texts, test.y, split


def _json(name: str) -> dict | None:
    f = REPORTS / name
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def study_summary() -> dict | None:
    val, test, ft, cmp = (_json(n) for n in ("sentiment_study_val.json", "sentiment_study_test.json",
                                             "sentiment_finetune.json", "sentiment_study_compare.json"))
    if not (val and test):
        return None
    base = test["results"]["baseline_current"]
    rows = [
        {"config": "Pretrained, neutral -> nearest class (baseline)", "model": config.SENTIMENT_MODEL, "in_product": True,
         "val_accuracy": val["variants"]["current"]["strategies"]["C_nearest_binary"]["accuracy"],
         "val_macro_f1": val["variants"]["current"]["strategies"]["C_nearest_binary"]["macro_f1"],
         "test_accuracy": base["strategies"]["C_nearest_binary"]["accuracy"],
         "test_macro_f1": base["strategies"]["C_nearest_binary"]["macro_f1"]},
    ]
    if "tuned_threshold" in base:
        rows.append({"config": f"Pretrained, validation-tuned threshold {val['threshold_tuning']['threshold']}",
                     "model": config.SENTIMENT_MODEL, "in_product": True,
                     "val_accuracy": val["threshold_tuning"]["val_report_at_threshold"]["accuracy"],
                     "val_macro_f1": val["threshold_tuning"]["val_report_at_threshold"]["macro_f1"],
                     "test_accuracy": base["tuned_threshold"]["accuracy"], "test_macro_f1": base["tuned_threshold"]["macro_f1"]})
    if ft and "final" in ft:
        f = ft["final"]
        rows.append({"config": f"Fine-tuned binary RoBERTa ({f['experiment_id']}, experiment only)", "model": ft["output_dir"],
                     "in_product": False, "val_accuracy": f["val_full"]["accuracy"], "val_macro_f1": f["val_full"]["macro_f1"],
                     "test_accuracy": f["test"]["accuracy"], "test_macro_f1": f["test"]["macro_f1"]})
    for repo, m in (cmp or {}).get("models", {}).items():
        if "val" in m:
            rows.append({"config": "Comparison model (evaluation only)", "model": repo, "in_product": False,
                         "val_accuracy": m["val"]["strategies"]["C_nearest_binary"]["accuracy"],
                         "val_macro_f1": m["val"]["strategies"]["C_nearest_binary"]["macro_f1"],
                         "test_accuracy": m["test"]["strategies"]["C_nearest_binary"]["accuracy"],
                         "test_macro_f1": m["test"]["strategies"]["C_nearest_binary"]["macro_f1"]})
    return {"split": "Sentiment140 80/10/10 text-group split (val %d / test %d)" % (val["n"], test["n"]),
            "selected_preprocessing": val["selected_preprocessing"], "rows": rows,
            "note": "Selection used validation only; each test number comes from a single logged test run "
                    "(artifacts/reports/sentiment_test_log.jsonl)."}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-sample", type=int, default=500)
    args = ap.parse_args()
    import torch

    texts, y, split = load_test()
    thr = config.SENTIMENT_BINARY_THRESHOLD
    report: dict = {
        "evaluation_date_utc": datetime.now(timezone.utc).isoformat(),
        "model": config.SENTIMENT_MODEL,
        "dataset": "Sentiment140 (training.1600000.processed.noemoticon.csv)",
        "sample": {"path": "data/interim/s140_split/test.csv", "seed": split["seed"], "size": int(len(y)),
                   "label_counts": {"negative": int((y == metrics.NEG).sum()), "positive": int((y == metrics.POS).sum())},
                   "selection": "held-out 10% test split of all 1.6M tweets (text-group hash split, no train/val overlap)",
                   "preprocessing": "production: repair -> PII redaction -> tidy -> @user/http",
                   "split_exclusions": {"conflicting_id_rows": split["counts"].get("conflicting_id_rows", 0),
                                        "conflicting_text_rows": split["counts"].get("conflicting_text_rows", 0)}},
        "ground_truth_note": "Sentiment140 has binary labels only (0=negative, 4=positive). It contains NO neutral ground truth.",
        "methodology": metrics.__doc__.strip() + (
            f"\n\n  binary_threshold  positive iff P(pos) / (P(pos) + P(neg)) > {thr}; the threshold was tuned for accuracy on the"
            "\n                    VALIDATION split only and applied unchanged here."),
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentimentModel(device)
    report["model_revision"] = model.revision
    report["device"] = device
    report["gpu"] = torch.cuda.get_device_name(0) if device == "cuda" else None

    model.predict(texts[:64], batch_size=64)  # warm-up
    out = model.predict(texts, batch_size=128)
    report["metrics"] = metrics.evaluate_binary_truth(y, out.probs)
    tp = metrics.apply_threshold(metrics.positive_score(out.probs), thr)
    report["metrics"]["binary_threshold"] = {**metrics.binary_report(y, tp), "threshold": thr,
                                             "confusion": metrics.confusion_2x2(y, tp)}
    report["neutral_strategies"] = metrics.neutral_strategies(y, out.probs)
    report["mean_confidence"] = round(float(out.confidences.mean()), 4)

    sub_idx = np.linspace(0, len(texts) - 1, BENCH_N).astype(int)
    bench_texts = [texts[i] for i in sub_idx]
    r1 = model.predict(bench_texts, batch_size=64)
    r2 = model.predict(bench_texts, batch_size=64)
    report["reproducibility"] = {
        "identical_labels_between_runs": bool(r1.labels == r2.labels),
        "max_abs_prob_diff": float(np.abs(r1.probs - r2.probs).max()),
        "note": f"{BENCH_N} evenly spaced test tweets scored twice with identical settings",
        "batch_size_128_vs_64_label_agreement": round(float(np.mean(np.array(r1.labels) == np.array(out.labels)[sub_idx])), 5),
        "batch_size_128_vs_64_max_abs_prob_diff": float(np.abs(out.probs[sub_idx] - r1.probs).max()),
    }

    bench = []
    if device == "cuda":
        for bs in (16, 32, 64, 128, 256):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                o = model.predict(bench_texts, batch_size=bs)
                bench.append({"device": "cuda", "dtype": "float16", "batch_size": bs, "n": len(bench_texts),
                              "seconds": round(o.seconds, 2), "reviews_per_sec": round(len(bench_texts) / o.seconds, 1),
                              "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3), "oom": False})
            except torch.cuda.OutOfMemoryError:
                bench.append({"device": "cuda", "batch_size": bs, "oom": True})
    del model
    torch.cuda.empty_cache()

    cpu = SentimentModel("cpu")
    cpu_idx = np.linspace(0, len(texts) - 1, args.cpu_sample).astype(int)
    sub = [texts[i] for i in cpu_idx]
    cpu.predict(sub[:32], batch_size=32)
    o_cpu = cpu.predict(sub, batch_size=32)
    bench.append({"device": "cpu", "dtype": "float32", "batch_size": 32, "n": len(sub), "seconds": round(o_cpu.seconds, 2),
                  "reviews_per_sec": round(len(sub) / o_cpu.seconds, 1), "torch_threads": torch.get_num_threads()})
    agree = float(np.mean(np.array(o_cpu.labels) == np.array(out.labels)[cpu_idx]))
    report["cpu_vs_gpu_label_agreement"] = round(agree, 4)
    report["cpu_metrics_on_subset"] = metrics.evaluate_binary_truth(y[cpu_idx], o_cpu.probs)["binary_forced"]
    report["gpu_metrics_on_same_subset"] = metrics.evaluate_binary_truth(y[cpu_idx], out.probs[cpu_idx])["binary_forced"]
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
    report["previous_evaluation"] = {"sample": "data/interim/s140_eval_5k.csv (stratified 5,000 tweets)", "binary_forced_accuracy": 0.7646,
                                     "note": "earlier headline number; a different, smaller sample, so not directly comparable"}
    report["accuracy_study"] = study_summary()

    with open(REPORTS / "sentiment_test_log.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"utc": report["evaluation_date_utc"], "config": "evaluate_sentiment.py: product model, fixed threshold (re-run, no selection)"}) + "\n")
    dest = REPORTS / "sentiment_validation.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("methodology", "neutral_strategies")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
