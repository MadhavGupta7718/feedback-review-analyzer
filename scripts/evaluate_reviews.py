"""Out-of-domain check: sentiment accuracy on labelled PRODUCT / BUSINESS REVIEWS (the text type this project analyses),
not tweets. Nothing is tuned on these datasets: the product model's binary threshold comes from the Sentiment140
validation split and the fine-tuned model was trained on Sentiment140 only.

Datasets (public test splits, downloaded to data/raw/reviews, gitignored; review text is never written to reports):
  amazon_polarity  fancyzhx/amazon_polarity  label from star rating (1-2 = negative, 4-5 = positive; 3 excluded by the authors)
  yelp_polarity    fancyzhx/yelp_polarity    label from star rating (1-2 = negative, 3-4 = positive per the authors)

  python scripts/evaluate_reviews.py [--per-label 10000]
Writes artifacts/reports/review_eval.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import config  # noqa: E402
from ml.evaluation import metrics as M  # noqa: E402
from ml.sentiment import experiment as X  # noqa: E402

RAW = config.DATA_DIR / "raw" / "reviews"
DATASETS = {
    "amazon_polarity": {"file": RAW / "amazon_polarity" / "amazon_polarity" / "test-00000-of-00001.parquet",
                        "repo": "fancyzhx/amazon_polarity", "text": lambda d: (d["title"].str.strip() + ". " + d["content"].str.strip()).tolist(),
                        "labels": "star rating: 1-2 negative, 4-5 positive (3-star reviews not in the dataset)"},
    "yelp_polarity": {"file": RAW / "yelp_polarity" / "plain_text" / "test-00000-of-00001.parquet",
                      "repo": "fancyzhx/yelp_polarity", "text": lambda d: d["text"].str.strip().tolist(),
                      "labels": "star rating: 1-2 negative, 3-4 positive"},
}
MAX_LEN = 128


def sample(name: str, per_label: int) -> tuple[list[str], np.ndarray, dict]:
    import pandas as pd

    spec = DATASETS[name]
    d = pd.read_parquet(spec["file"])
    total = len(d)
    rng = np.random.default_rng(config.SEED)
    idx = np.concatenate([rng.choice(np.flatnonzero(d["label"].to_numpy() == lab), per_label, replace=False) for lab in (0, 1)])
    idx.sort()
    d = d.iloc[idx]
    raw = spec["text"](d)
    y = np.where(d["label"].to_numpy() == 1, M.POS, M.NEG)
    return raw, y, {"source": spec["repo"], "split": "test", "test_rows": total,
                    "sample": f"stratified random {per_label:,} per label, seed {config.SEED}", "n": int(len(y)), "labels": spec["labels"]}


def score(y: np.ndarray, probs: np.ndarray, binary_model: bool) -> dict:
    near = np.where(probs[:, M.POS] > probs[:, M.NEG], M.POS, M.NEG)
    out = {"nearest_class": {**M.binary_report(y, near), "confusion": M.confusion_2x2(y, near)}}
    if not binary_model:
        s = M.neutral_strategies(y, probs)
        t = M.apply_threshold(M.positive_score(probs), config.SENTIMENT_BINARY_THRESHOLD)
        out["s140_threshold"] = {**M.binary_report(y, t), "threshold": config.SENTIMENT_BINARY_THRESHOLD, "confusion": M.confusion_2x2(y, t)}
        out["strict_3class"] = M.binary_report(y, probs.argmax(1))
        out["exclude_neutral"] = {k: s["D_exclude_neutral"][k] for k in ("n", "accuracy", "macro_f1", "coverage")}
        out["neutral_prediction_rate"] = s["C_nearest_binary"]["neutral_prediction_rate"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-label", type=int, default=10_000)
    args = ap.parse_args()
    from ml.models import registry

    models = [("product: " + config.SENTIMENT_MODEL, X.sentiment_model_path(), False, None)]
    if (config.FINETUNED_SENTIMENT_DIR / "config.json").exists():
        models.append(("fine-tuned binary (Sentiment140, experiment)", config.FINETUNED_SENTIMENT_DIR, True, None))
    for repo in config.COMPARISON_MODELS:
        m = registry.locate(repo)
        if m.installed:
            note = ("trained on review corpora that include Amazon and Yelp reviews, so this is NOT an unseen-domain result"
                    if "siebert" in repo else "trained on SST-2 movie-review sentences")
            models.append((repo, m.path, True, note))

    data = {}
    for name in DATASETS:
        raw, y, meta = sample(name, args.per_label)
        texts = X.variant_texts("current", raw, [X.prepare(r) for r in raw])
        data[name] = (texts, y, meta)

    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "max_length": MAX_LEN,
              "preprocessing": "production (repair -> PII redaction -> tidy -> @user/http)",
              "tuning": "none on these datasets", "datasets": {}, "models": {}}
    import torch

    tok_lens = {}
    for label, path, binary, note in models:
        pred = X.Predictor(path, "cuda", max_length=MAX_LEN)
        pred.predict(["warm up"] * 8)
        rec = {"binary_model": binary, **({"caveat": note} if note else {}), "results": {}}
        for name, (texts, y, meta) in data.items():
            if name not in tok_lens:
                lens = [len(x) for x in pred.tok(texts[:5000], truncation=False)["input_ids"]]
                tok_lens[name] = {"mean_tokens_first5000": round(float(np.mean(lens)), 1),
                                  "share_over_128_tokens": round(float(np.mean([n > MAX_LEN for n in lens])), 4)}
            t0 = time.perf_counter()
            probs = pred.predict(texts, 64)
            r = score(y, probs, binary)
            r["seconds"] = round(time.perf_counter() - t0, 1)
            rec["results"][name] = r
            print(f"{label:70s} {name:16s} acc={r['nearest_class']['accuracy']:.4f} f1={r['nearest_class']['macro_f1']:.4f}"
                  + (f" thr={r['s140_threshold']['accuracy']:.4f}" if not binary else ""), flush=True)
        report["models"][label] = rec
        del pred
        torch.cuda.empty_cache()
    for name, (_, _, meta) in data.items():
        report["datasets"][name] = {**meta, **tok_lens.get(name, {})}
    (config.ARTIFACTS_DIR / "reports" / "review_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
