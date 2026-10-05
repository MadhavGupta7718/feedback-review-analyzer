"""Label Amazon clothing reviews with CardiffNLP RoBERTa (text only; ignore stars).

Reads:  data/interim/amazon_clothing/reviews.csv
Writes: data/interim/amazon_clothing/reviews_teacher_roberta.csv
        data/interim/amazon_clothing/teacher_label_meta.json

Uses the pretrained cardiffnlp checkpoint only (not the star-finetuned amazon_roberta_sentiment).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import config  # noqa: E402
from ml.models import registry  # noqa: E402
from ml.preprocessing.clean import model_text  # noqa: E402

SRC = config.DATA_DIR / "interim" / "amazon_clothing" / "reviews.csv"
OUT = config.DATA_DIR / "interim" / "amazon_clothing" / "reviews_teacher_roberta.csv"
META = config.DATA_DIR / "interim" / "amazon_clothing" / "teacher_label_meta.json"
LABELS = ("negative", "neutral", "positive")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=128)
    args = ap.parse_args()
    if not args.src.exists():
        print(f"Missing {args.src}", file=sys.stderr)
        return 1

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    local = registry.require(config.SENTIMENT_MODEL)
    print(f"teacher={local.repo_id} path={local.path} device={device}", flush=True)
    tok = AutoTokenizer.from_pretrained(str(local.path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(local.path), local_files_only=True)
    model.eval().to(device)
    if device == "cuda":
        model = model.to(torch.float16)

    id2label = {i: str(model.config.id2label[i]).lower() for i in range(model.config.num_labels)}
    order = []
    for lab in LABELS:
        order.append(next(i for i, l in id2label.items() if l == lab or l.startswith(lab[:3])))
    print(f"id2label={id2label} order={order}", flush=True)

    rows: list[dict] = []
    with args.src.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            text = (row.get("text") or "").strip()
            if not text:
                continue
            rows.append(row)
    print(f"rows={len(rows)}", flush=True)

    texts = [model_text(r["text"]) for r in rows]
    probs = np.zeros((len(texts), 3), dtype=np.float32)
    order_idx = np.argsort([len(t) for t in texts])
    t0 = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            idx = order_idx[start : start + args.batch_size]
            enc = tok(
                [texts[i] for i in idx],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits.float()
            p = torch.softmax(logits, dim=-1)[:, order].cpu().numpy()
            probs[idx] = p
            done = min(start + args.batch_size, len(texts))
            if done % 2048 < args.batch_size or done == len(texts):
                print(f"  labeled {done}/{len(texts)} ({time.perf_counter()-t0:.0f}s)", flush=True)
    if device == "cuda":
        torch.cuda.synchronize()
    secs = time.perf_counter() - t0

    pred = probs.argmax(1)
    conf = probs.max(1)
    counts = Counter(LABELS[i] for i in pred)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "review_id", "text", "rating", "star_sentiment", "gt_sentiment",
        "teacher_confidence", "p_negative", "p_neutral", "p_positive",
        "created_at", "source",
    ]
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r, yi, c, pr in zip(rows, pred, conf, probs):
            w.writerow({
                "review_id": r.get("review_id", ""),
                "text": r["text"],
                "rating": r.get("rating", ""),
                "star_sentiment": (r.get("gt_sentiment") or "").strip().lower(),
                "gt_sentiment": LABELS[int(yi)],
                "teacher_confidence": f"{float(c):.6f}",
                "p_negative": f"{float(pr[0]):.6f}",
                "p_neutral": f"{float(pr[1]):.6f}",
                "p_positive": f"{float(pr[2]):.6f}",
                "created_at": r.get("created_at", ""),
                "source": "amazon_clothing_teacher_roberta",
            })

    agree = sum(
        1 for r, yi in zip(rows, pred)
        if (r.get("gt_sentiment") or "").strip().lower() == LABELS[int(yi)]
    )
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "teacher_model": config.SENTIMENT_MODEL,
        "teacher_revision": local.revision,
        "src": str(args.src),
        "out": str(args.out),
        "n": len(rows),
        "seconds": round(secs, 1),
        "device": device,
        "batch_size": args.batch_size,
        "label_counts": dict(counts),
        "agreement_with_star_sentiment": round(agree / max(len(rows), 1), 4),
        "note": "gt_sentiment = teacher text label; star_sentiment = Cons_rating map (kept for comparison)",
    }
    META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
