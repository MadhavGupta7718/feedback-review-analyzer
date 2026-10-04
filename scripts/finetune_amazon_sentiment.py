"""Fine-tune 3-class RoBERTa on Amazon clothing reviews (Review + Cons_rating).

Holdout: deterministic 80/10/10 text-hash split. Metrics on TEST only go to
artifacts/reports/amazon_sentiment_eval.json. Model saved to
artifacts/models/amazon_roberta_sentiment/.

Recall boost (continue from existing checkpoint):
  python scripts/finetune_amazon_sentiment.py --resume --epochs 5 --lr 1e-5 \\
      --batch-size 16 --balanced-batches --focal-gamma 2
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import config  # noqa: E402
from ml.evaluation import metrics as M  # noqa: E402
from ml.models import registry  # noqa: E402
from ml.preprocessing.clean import model_text  # noqa: E402

DATA_CSV = config.DATA_DIR / "interim" / "amazon_clothing" / "reviews.csv"
OUT_DIR = config.ARTIFACTS_DIR / "models" / "amazon_roberta_sentiment"
REPORT = config.ARTIFACTS_DIR / "reports" / "amazon_sentiment_eval.json"
SEED = 42
MAX_LEN = 128
LABELS = ("negative", "neutral", "positive")
LAB2I = {k: i for i, k in enumerate(LABELS)}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_all(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_bucket(text: str) -> str:
    """Stable 80/10/10 by sha256 of text."""
    h = int(hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest(), 16) % 100
    if h < 80:
        return "train"
    if h < 90:
        return "val"
    return "test"


def load_rows(path: Path) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            text = (row.get("text") or "").strip()
            lab = (row.get("gt_sentiment") or "").strip().lower()
            if not text or lab not in LAB2I:
                continue
            buckets[split_bucket(text)].append({"text": text, "y": LAB2I[lab], "rating": row.get("rating")})
    return buckets


def encode(tok, texts: list[str], chunk: int = 2048) -> list[list[int]]:
    out: list[list[int]] = []
    cleaned = [model_text(t) for t in texts]
    for i in range(0, len(cleaned), chunk):
        batch = cleaned[i : i + chunk]
        out.extend(tok(batch, truncation=True, max_length=MAX_LEN)["input_ids"])
        if (i // chunk) % 5 == 0:
            print(f"    encode {min(i + chunk, len(cleaned))}/{len(cleaned)}", flush=True)
    return out


def pack_batch(ids, labels, idx, pad):
    import torch

    L = max(len(ids[i]) for i in idx)
    inp = torch.full((len(idx), L), pad, dtype=torch.long)
    att = torch.zeros((len(idx), L), dtype=torch.long)
    for r, i in enumerate(idx):
        inp[r, : len(ids[i])] = torch.tensor(ids[i])
        att[r, : len(ids[i])] = 1
    return idx, inp, att, torch.tensor(labels[idx], dtype=torch.long)


def batches(ids, labels, bs, pad, shuffle, rng):
    order = rng.permutation(len(ids)) if shuffle else np.argsort([len(x) for x in ids])
    for s in range(0, len(order), bs):
        yield pack_batch(ids, labels, order[s : s + bs], pad)


def balanced_batches(ids, labels, bs, pad, rng, steps_per_epoch: int):
    """Each batch draws ~equal counts of neg/neu/pos (with replacement for minorities)."""
    by_cls = [np.where(labels == c)[0] for c in range(3)]
    for _ in range(steps_per_epoch):
        per = max(1, bs // 3)
        rem = bs - 3 * per
        picks = []
        for c in range(3):
            n = per + (1 if c < rem else 0)
            picks.append(rng.choice(by_cls[c], size=n, replace=True))
        idx = np.concatenate(picks)
        rng.shuffle(idx)
        yield pack_batch(ids, labels, idx, pad)


def oversample_indices(labels: np.ndarray, neu_mult: int, neg_mult: int, rng) -> np.ndarray:
    """Expand train indices so neutral/negative appear more often under normal shuffling."""
    idx = np.arange(len(labels))
    parts = [idx]
    neu = idx[labels == 1]
    neg = idx[labels == 0]
    for _ in range(max(0, neu_mult - 1)):
        parts.append(neu)
    for _ in range(max(0, neg_mult - 1)):
        parts.append(neg)
    out = np.concatenate(parts)
    rng.shuffle(out)
    return out


def focal_ce(logits, targets, weight, gamma: float):
    """Weighted focal cross-entropy (gamma=0 → plain weighted CE)."""
    import torch
    import torch.nn.functional as F

    log_p = F.log_softmax(logits, dim=-1)
    p = log_p.exp()
    pt = p.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-8)
    ce = F.nll_loss(log_p, targets, weight=weight, reduction="none")
    return ((1.0 - pt) ** gamma * ce).mean()


def balanced_softmax_ce(logits, targets, prior, tau: float):
    """Balanced Softmax (long-tail): adjust logits by tau * log(class prior)."""
    import torch
    import torch.nn.functional as F

    adj = logits + tau * torch.log(prior.clamp(min=1e-8))
    return F.cross_entropy(adj, targets)


def predict_probs(model, ids, pad, device, bs=128) -> np.ndarray:
    import torch

    model.eval()
    probs = np.zeros((len(ids), 3), dtype=np.float32)
    id2 = {i: str(model.config.id2label[i]).lower() for i in range(model.config.num_labels)}
    order = [next(i for i, l in id2.items() if LABELS[j] in l or l.startswith(LABELS[j][:3])) for j in range(3)]
    with torch.inference_mode():
        for idx, inp, att, _ in batches(ids, np.zeros(len(ids), dtype=np.int64), bs, pad, False, np.random.default_rng(0)):
            logits = model(input_ids=inp.to(device), attention_mask=att.to(device)).logits.float()
            p = torch.softmax(logits, dim=-1)[:, order].cpu().numpy()
            probs[idx] = p
    return probs


def evaluate(model, ids, y, pad, device) -> dict:
    probs = predict_probs(model, ids, pad, device)
    pred = probs.argmax(1)
    rep = M.evaluate_3class(np.asarray(y), pred)
    recalls = [rep["per_class"][c]["recall"] for c in LABELS]
    rep["macro_recall"] = round(float(np.mean(recalls)), 4)
    return rep


def load_model(resume: bool, device: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if resume and (OUT_DIR / "config.json").exists():
        print(f"resuming from {OUT_DIR}", flush=True)
        tok = AutoTokenizer.from_pretrained(str(OUT_DIR), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(OUT_DIR), local_files_only=True)
        model.config.id2label = {0: "negative", 1: "neutral", 2: "positive"}
        model.config.label2id = {"negative": 0, "neutral": 1, "positive": 2}
        model.to(device)
        model = model.to(torch.float32)
        return tok, model, f"local:{OUT_DIR.name}"

    local = registry.require(config.SENTIMENT_MODEL)
    print(f"loading base {local.path}", flush=True)
    tok = AutoTokenizer.from_pretrained(str(local.path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(local.path), local_files_only=True)
    orig_id2 = {i: str(model.config.id2label[i]).lower() for i in range(model.config.num_labels)}
    src_order = [next(i for i, l in orig_id2.items() if LABELS[j] in l or l.startswith(LABELS[j][:3])) for j in range(3)]
    if src_order != [0, 1, 2] or model.config.num_labels != 3:
        print(f"remapping classifier head from {orig_id2} via {src_order}", flush=True)
        old = model.classifier.out_proj
        new = torch.nn.Linear(old.in_features, 3)
        with torch.no_grad():
            new.weight.copy_(old.weight[src_order])
            new.bias.copy_(old.bias[src_order])
        model.classifier.out_proj = new
        model.config.num_labels = model.num_labels = 3
    model.config.id2label = {0: "negative", 1: "neutral", 2: "positive"}
    model.config.label2id = {"negative": 0, "neutral": 1, "positive": 2}
    model.to(device)
    model = model.to(torch.float32)
    return tok, model, config.SENTIMENT_MODEL


def class_weights(train_y: np.ndarray, weight_power: float, neu_boost: float, neg_boost: float) -> np.ndarray:
    counts = np.bincount(train_y, minlength=3).astype(np.float32)
    w = (counts.sum() / (3 * np.maximum(counts, 1))) ** weight_power
    w[1] *= neu_boost  # neutral
    w[0] *= neg_boost  # negative
    w = w / w.mean()  # keep mean ~1 for stable LR
    return w


def train(args) -> dict:
    import torch
    from transformers import get_linear_schedule_with_warmup

    if not DATA_CSV.exists():
        raise SystemExit(f"Missing {DATA_CSV}; run scripts/prepare_amazon_clothing.py first")

    previous_headline = None
    if REPORT.exists():
        try:
            prev = json.loads(REPORT.read_text(encoding="utf-8"))
            previous_headline = {
                "accuracy": (prev.get("metrics") or {}).get("headline_accuracy"),
                "macro_recall": (prev.get("metrics") or {}).get("headline_macro_recall"),
                "evaluation_date_utc": prev.get("evaluation_date_utc"),
            }
        except Exception:
            previous_headline = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)
    seed_all(SEED)
    data = load_rows(DATA_CSV)
    for k, v in data.items():
        print(f"  {k}: {len(v)}", flush=True)

    tok, model, model_base = load_model(args.resume, device)
    print("model on device", flush=True)

    lr, bs, max_epochs = args.lr, args.batch_size, args.epochs
    print("tokenizing…", flush=True)
    train_ids = encode(tok, [r["text"] for r in data["train"]])
    print(f"  train tokens done ({len(train_ids)})", flush=True)
    val_ids = encode(tok, [r["text"] for r in data["val"]])
    print(f"  val tokens done ({len(val_ids)})", flush=True)
    test_ids = encode(tok, [r["text"] for r in data["test"]])
    print(f"  test tokens done ({len(test_ids)})", flush=True)
    train_y = np.array([r["y"] for r in data["train"]], dtype=np.int64)
    val_y = np.array([r["y"] for r in data["val"]], dtype=np.int64)
    test_y = np.array([r["y"] for r in data["test"]], dtype=np.int64)

    counts = np.bincount(train_y, minlength=3).astype(np.float32)
    prior = torch.tensor(counts / counts.sum(), dtype=torch.float32, device=device)
    weights = class_weights(train_y, args.weight_power, args.neu_boost, args.neg_boost)
    weight_t = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"class_weights={weights.tolist()} prior={prior.tolist()} balanced={args.balanced_batches} "
          f"focal_gamma={args.focal_gamma} balanced_softmax_tau={args.balanced_softmax_tau}", flush=True)
    print(f"starting {max_epochs} epoch(s) bs={bs} lr={lr}", flush=True)

    decay = [p for n, p in model.named_parameters() if p.requires_grad and not any(k in n for k in ("bias", "LayerNorm.weight"))]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in ("bias", "LayerNorm.weight"))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}], lr=lr)
    rng = np.random.default_rng(SEED)
    # Optional index oversampling (milder than equal-class batches)
    if args.oversample_neu > 1 or args.oversample_neg > 1:
        train_order = oversample_indices(train_y, args.oversample_neu, args.oversample_neg, rng)
        print(f"oversampled train size {len(train_y)} → {len(train_order)} "
              f"(neu_mult={args.oversample_neu}, neg_mult={args.oversample_neg})", flush=True)
    else:
        train_order = None

    n_train_examples = len(train_order) if train_order is not None else len(train_ids)
    steps_per_epoch = max(1, math.ceil(n_train_examples / bs))
    total = steps_per_epoch * max_epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    pad = tok.pad_token_id
    best, best_state, history = None, None, []
    t0 = time.perf_counter()
    model.train()
    print(f"train steps/epoch={steps_per_epoch}", flush=True)

    for epoch in range(1, max_epochs + 1):
        running, seen = 0.0, 0
        if args.balanced_batches:
            loader = balanced_batches(train_ids, train_y, bs, pad, rng, steps_per_epoch)
        elif train_order is not None:
            order = rng.permutation(train_order)
            def _oversampled():
                for s in range(0, len(order), bs):
                    yield pack_batch(train_ids, train_y, order[s : s + bs], pad)
            loader = _oversampled()
        else:
            loader = batches(train_ids, train_y, bs, pad, True, rng)
        for step, (_, inp, att, lab) in enumerate(loader, 1):
            out = model(input_ids=inp.to(device), attention_mask=att.to(device))
            lab_d = lab.to(device)
            if args.balanced_softmax_tau and args.balanced_softmax_tau > 0:
                loss = balanced_softmax_ce(out.logits, lab_d, prior, args.balanced_softmax_tau)
            elif args.focal_gamma and args.focal_gamma > 0:
                loss = focal_ce(out.logits, lab_d, weight_t, args.focal_gamma)
            else:
                loss = torch.nn.functional.cross_entropy(out.logits, lab_d, weight=weight_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            running += float(loss.detach()) * len(lab)
            seen += len(lab)
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step}/{steps_per_epoch} loss={running/seen:.4f} "
                      f"{time.perf_counter()-t0:.0f}s", flush=True)
        val_rep = evaluate(model, val_ids, val_y, pad, device)
        per_r = {c: val_rep["per_class"][c]["recall"] for c in LABELS}
        history.append({
            "epoch": epoch,
            "train_loss": round(running / max(seen, 1), 4),
            "val_accuracy": val_rep["accuracy"],
            "val_macro_recall": val_rep["macro_recall"],
            "val_per_class_recall": per_r,
            "elapsed_s": round(time.perf_counter() - t0, 1),
        })
        print(f"epoch {epoch}: loss={running/seen:.4f} val_acc={val_rep['accuracy']:.4f} "
              f"val_macro_recall={val_rep['macro_recall']:.4f} "
              f"recalls={{neg:{per_r['negative']:.3f}, neu:{per_r['neutral']:.3f}, pos:{per_r['positive']:.3f}}}",
              flush=True)
        score = val_rep["macro_recall"]
        if best is None or score > best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  * new best val_macro_recall={best:.4f}", flush=True)
        model.train()

    model.load_state_dict(best_state)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT_DIR))
    tok.save_pretrained(str(OUT_DIR))

    test_rep = evaluate(model, test_ids, test_y, pad, device)
    val_rep = evaluate(model, val_ids, val_y, pad, device)
    report = {
        "evaluation_date_utc": now(),
        "dataset": "Amazon clothing reviews (Review + Cons_rating)",
        "source_csv": str(DATA_CSV),
        "model_base": model_base,
        "model_dir": str(OUT_DIR),
        "split": {"train": len(train_y), "val": len(val_y), "test": len(test_y), "method": "sha256(text)%100 → 80/10/10"},
        "label_mapping": "1-2 negative, 3 neutral, 4-5 positive (Cons_rating)",
        "hyperparams": {
            "lr": lr, "batch_size": bs, "epochs": max_epochs, "max_length": MAX_LEN, "seed": SEED,
            "class_weights": weights.tolist(),
            "weight_power": args.weight_power,
            "neu_boost": args.neu_boost,
            "neg_boost": args.neg_boost,
            "balanced_batches": args.balanced_batches,
            "focal_gamma": args.focal_gamma,
            "oversample_neu": args.oversample_neu,
            "oversample_neg": args.oversample_neg,
            "balanced_softmax_tau": args.balanced_softmax_tau,
            "resume": args.resume,
        },
        "device": device,
        "training_seconds": round(time.perf_counter() - t0, 1),
        "history": history,
        "previous_headline": previous_headline,
        "best_val_macro_recall": best,
        "metrics": {
            "val": val_rep,
            "test": test_rep,
            "headline_accuracy": test_rep["accuracy"],
            "headline_macro_recall": test_rep["macro_recall"],
            "per_class": test_rep["per_class"],
            "confusion_3x3": test_rep["confusion"],
        },
        "ground_truth_note": "Labels from Cons_rating star ratings (weak labels). Metrics are on the held-out TEST split only.",
        "methodology": (
            "3-class fine-tune of RoBERTa on Amazon clothing reviews with recall-focused training "
            "(class-balanced batches, minority-boosted weights, focal loss). "
            "80/10/10 deterministic text-hash split with no train/test overlap. "
            "Macro recall = mean of per-class recall on TEST."
        ),
        "scope": "global model evaluation — not per uploaded batch",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT_DIR / "training_info.json").write_text(json.dumps({
        "base_model": model_base, "best_val_macro_recall": best,
        "test_accuracy": test_rep["accuracy"], "test_macro_recall": test_rep["macro_recall"],
        "previous_headline": previous_headline,
        "finished_utc": now(),
    }, indent=2), encoding="utf-8")
    print(json.dumps({
        "test_accuracy": test_rep["accuracy"],
        "test_macro_recall": test_rep["macro_recall"],
        "test_per_class_recall": {c: test_rep["per_class"][c]["recall"] for c in LABELS},
        "previous_headline": previous_headline,
        "model_dir": str(OUT_DIR),
        "report": str(REPORT),
    }, indent=2))
    return report


class _Tee:
    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8")

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return False

    def fileno(self):
        return self.streams[0].fileno()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--resume", action="store_true",
                    help="Continue from artifacts/models/amazon_roberta_sentiment if present")
    ap.add_argument("--balanced-batches", action="store_true",
                    help="Equal neg/neu/pos draws per batch (with replacement)")
    ap.add_argument("--focal-gamma", type=float, default=0.0,
                    help="Focal loss gamma (0 = plain weighted CE)")
    ap.add_argument("--weight-power", type=float, default=1.0,
                    help="Exponent on inverse-frequency class weights")
    ap.add_argument("--neu-boost", type=float, default=1.0, help="Extra multiplier for neutral weight")
    ap.add_argument("--neg-boost", type=float, default=1.0, help="Extra multiplier for negative weight")
    ap.add_argument("--oversample-neu", type=int, default=1, help="Duplicate neutral train rows this many times")
    ap.add_argument("--oversample-neg", type=int, default=1, help="Duplicate negative train rows this many times")
    ap.add_argument("--balanced-softmax-tau", type=float, default=0.0,
                    help="Balanced-softmax prior adjustment (0=off; try 1.0)")
    args = ap.parse_args()
    # Sensible recall defaults when using --resume without explicit boosts
    if args.resume and args.weight_power == 1.0 and args.neu_boost == 1.0 and args.oversample_neu == 1:
        args.weight_power = 1.25
        args.neu_boost = 1.5
        args.neg_boost = 1.2
    log_path = config.DATA_DIR / "finetune_amazon.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("w", encoding="utf-8", newline="\n")
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = _Tee(sys.__stderr__, log_fh)
    print(f"=== finetune start {now()} pid={os.getpid()} ===", flush=True)
    rc = 0
    try:
        train(args)
    except Exception:
        rc = 1
        import traceback

        traceback.print_exc()
    finally:
        print(f"=== finetune end {now()} rc={rc} ===", flush=True)
        try:
            log_fh.close()
        except Exception:
            pass
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
