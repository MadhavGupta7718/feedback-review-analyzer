"""Experimental fine-tune of a SEPARATE binary sentiment model on the Sentiment140 train split.

The product pipeline keeps the pretrained 3-class cardiffnlp model; this script never modifies it. The fine-tuned model
starts from the same checkpoint with a 2-class head built from the pretrained negative/positive output rows, and is
saved to artifacts/models/roberta-s140-binary (gitignored).

  python scripts/finetune_sentiment.py matrix     run the hyperparameter matrix (selection on a fixed validation subset)
  python scripts/finetune_sentiment.py final      full-validation metrics + threshold for the selected run, then ONE test run

Inputs are preprocessed exactly like production (repair -> PII redaction -> tidy -> @user/http), so no raw text is
used for training. Results: artifacts/reports/sentiment_finetune.json.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import config  # noqa: E402
from ml.evaluation import metrics as M  # noqa: E402
from ml.sentiment import experiment as X  # noqa: E402

REPORT = config.ARTIFACTS_DIR / "reports" / "sentiment_finetune.json"
CACHE = config.ARTIFACTS_DIR / "cache" / "sentiment_exp"
TEST_LOG = config.ARTIFACTS_DIR / "reports" / "sentiment_test_log.jsonl"
OUT_DIR = config.FINETUNED_SENTIMENT_DIR
SEED = 42
MAX_LEN = 64
VAL_SUBSET = 20_000
MATRIX = [  # (experiment_id, lr, batch, max_epochs)
    ("ft01", 1e-5, 32, 3),
    ("ft02", 2e-5, 32, 3),
    ("ft03", 3e-5, 32, 3),
    ("ft04", None, 16, 3),   # lr = best of ft01-ft03 on validation
    ("ft05", None, 32, 4),   # best lr, one more epoch allowed (only if the best run peaked at its last epoch)
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_all(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_report() -> dict:
    if REPORT.exists():
        return json.loads(REPORT.read_text(encoding="utf-8"))
    return {"runs": {}}


def save_report(rep: dict) -> None:
    REPORT.write_text(json.dumps(rep, indent=2), encoding="utf-8")


def production_texts(raw: list[str]) -> list[str]:
    return X.variant_texts("current", raw, [X.prepare(r) for r in raw])


def build_model(tok_path):
    import torch
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(str(tok_path), local_files_only=True)
    id2 = {i: l.lower() for i, l in model.config.id2label.items()}
    neg = next(i for i, l in id2.items() if l.startswith("neg"))
    pos = next(i for i, l in id2.items() if l.startswith("pos"))
    old = model.classifier.out_proj
    new = torch.nn.Linear(old.in_features, 2)
    with torch.no_grad():
        new.weight.copy_(old.weight[[neg, pos]])
        new.bias.copy_(old.bias[[neg, pos]])
    model.classifier.out_proj = new
    model.config.num_labels = model.num_labels = 2
    model.config.id2label = {0: "negative", 1: "positive"}
    model.config.label2id = {"negative": 0, "positive": 1}
    return model


def encode(tok, texts: list[str]) -> list[list[int]]:
    return tok(texts, truncation=True, max_length=MAX_LEN)["input_ids"]


def batches(ids: list[list[int]], labels: np.ndarray, bs: int, pad: int, shuffle: bool, rng: np.random.Generator):
    import torch

    order = rng.permutation(len(ids)) if shuffle else np.argsort([len(x) for x in ids])
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        L = max(len(ids[i]) for i in idx)
        inp = torch.full((len(idx), L), pad, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):
            inp[r, :len(ids[i])] = torch.tensor(ids[i])
            att[r, :len(ids[i])] = 1
        yield idx, inp, att, torch.tensor(labels[idx], dtype=torch.long)


def evaluate(model, ids, y01, pad, bs=256) -> tuple[dict, np.ndarray]:
    import torch

    model.eval()
    p_pos = np.zeros(len(ids), dtype=np.float32)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for idx, inp, att, _ in batches(ids, y01, bs, pad, False, np.random.default_rng(0)):
            logits = model(input_ids=inp.cuda(), attention_mask=att.cuda()).logits.float()
            p_pos[idx] = torch.softmax(logits, -1)[:, 1].cpu().numpy()
    model.train()
    y = np.where(y01 == 1, M.POS, M.NEG)
    return M.binary_report(y, np.where(p_pos > 0.5, M.POS, M.NEG)), p_pos


def train_run(exp_id: str, lr: float, bs: int, max_epochs: int, data: dict, rep: dict) -> dict:
    import torch
    from transformers import get_linear_schedule_with_warmup

    seed_all(SEED)
    model = build_model(data["path"]).cuda()
    model.train()
    decay = [p for n, p in model.named_parameters() if not any(k in n for k in ("bias", "LayerNorm.weight"))]
    no_decay = [p for n, p in model.named_parameters() if any(k in n for k in ("bias", "LayerNorm.weight"))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}], lr=lr)
    steps_per_epoch = math.ceil(len(data["train_ids"]) / bs)
    total = steps_per_epoch * max_epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    rng = np.random.default_rng(SEED)
    pad = data["pad"]
    history, best, best_state, bad = [], None, None, 0
    t0 = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        running, seen = 0.0, 0
        for step, (_, inp, att, lab) in enumerate(batches(data["train_ids"], data["train_y"], bs, pad, True, rng), 1):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=inp.cuda(), attention_mask=att.cuda(), labels=lab.cuda()).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            running += float(loss) * len(lab)
            seen += len(lab)
            if step % 500 == 0:
                print(f"  {exp_id} epoch {epoch} step {step}/{steps_per_epoch} loss {running / seen:.4f} "
                      f"{time.perf_counter() - t0:.0f}s", flush=True)
        vr, _ = evaluate(model, data["vsub_ids"], data["vsub_y"], pad)
        history.append({"epoch": epoch, "train_loss": round(running / seen, 4), "val_subset_accuracy": vr["accuracy"],
                        "val_subset_macro_f1": vr["macro_f1"], "elapsed_s": round(time.perf_counter() - t0, 1)})
        print(f"  {exp_id} epoch {epoch}: train_loss={running / seen:.4f} val_acc={vr['accuracy']:.4f} f1={vr['macro_f1']:.4f}", flush=True)
        if best is None or vr["accuracy"] > best["val_subset_accuracy"]:
            best = history[-1]
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= 1:  # early stopping, patience 1
                break
    run = {"experiment_id": exp_id, "lr": lr, "batch_size": bs, "max_epochs": max_epochs, "epochs_run": len(history),
           "best_epoch": best["epoch"], "val_subset_accuracy": best["val_subset_accuracy"],
           "val_subset_macro_f1": best["val_subset_macro_f1"], "training_seconds": round(time.perf_counter() - t0, 1),
           "history": history, "early_stopped": len(history) < max_epochs, "finished_utc": now()}
    overall = max((r["val_subset_accuracy"] for r in rep["runs"].values()), default=-1)
    if run["val_subset_accuracy"] > overall:
        model.load_state_dict(best_state)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(OUT_DIR))
        data["tok"].save_pretrained(str(OUT_DIR))
        (OUT_DIR / "training_info.json").write_text(json.dumps({**run, "base_model": config.SENTIMENT_MODEL,
                                                                "note": "experimental; not used by the product pipeline"}, indent=2))
        run["saved"] = True
    del model, opt, best_state
    torch.cuda.empty_cache()
    return run


def matrix(args) -> int:
    import torch
    from transformers import AutoTokenizer

    CACHE.mkdir(parents=True, exist_ok=True)
    path = X.sentiment_model_path()
    tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    t0 = time.perf_counter()
    train, val = X.load_split("train_subset"), X.load_split("val")
    rng = np.random.default_rng(SEED)
    vsub = np.concatenate([rng.choice(np.flatnonzero(val.y == c), VAL_SUBSET // 2, replace=False) for c in (M.NEG, M.POS)])
    vsub.sort()
    train_ids = encode(tok, production_texts(train.raw))
    vsub_ids = encode(tok, production_texts([val.raw[i] for i in vsub]))
    full_len = tok(production_texts(train.raw[:20000]), truncation=False)["input_ids"]
    data = {"path": path, "tok": tok, "pad": tok.pad_token_id, "train_ids": train_ids,
            "train_y": (train.y == M.POS).astype(np.int64), "vsub_ids": vsub_ids, "vsub_y": (val.y[vsub] == M.POS).astype(np.int64)}
    rep = load_report()
    rep.update({"base_model": config.SENTIMENT_MODEL, "output_dir": str(OUT_DIR.relative_to(config.REPO_ROOT)),
                "train_rows": len(train_ids), "train_class_counts": {"negative": int((train.y == M.NEG).sum()), "positive": int((train.y == M.POS).sum())},
                "val_subset_rows": int(len(vsub)), "val_subset_seed": SEED, "max_length": MAX_LEN,
                "share_over_max_length_first20k_train": round(float(np.mean([len(x) > MAX_LEN for x in full_len])), 5),
                "mean_tokens_first20k_train": round(float(np.mean([len(x) for x in full_len])), 1),
                "setup": {"optimizer": "AdamW", "weight_decay": 0.01, "warmup": "6% then linear decay", "grad_clip": 1.0,
                          "precision": "bf16 autocast", "early_stopping": "patience 1 on val-subset accuracy", "seed": SEED,
                          "head_init": "2-class head from pretrained negative/positive rows of the 3-class head",
                          "class_weighting": "none (train subset is exactly balanced)",
                          "gpu": torch.cuda.get_device_name(0)},
                "prep_seconds": round(time.perf_counter() - t0, 1)})
    save_report(rep)
    print(f"prepared in {rep['prep_seconds']}s; truncated share {rep['share_over_max_length_first20k_train']}", flush=True)
    for exp_id, lr, bs, ep in MATRIX:
        if exp_id in rep["runs"]:
            continue
        best_lr_run = max((rep["runs"][k] for k in ("ft01", "ft02", "ft03") if k in rep["runs"]), key=lambda r: r["val_subset_accuracy"], default=None)
        if lr is None:
            if best_lr_run is None:
                continue
            lr = best_lr_run["lr"]
        if exp_id == "ft05":
            best_run = max(rep["runs"].values(), key=lambda r: r["val_subset_accuracy"])
            if best_run["best_epoch"] < best_run["max_epochs"]:
                rep["runs"]["ft05"] = {"experiment_id": "ft05", "skipped": "best run peaked before its last epoch; a longer schedule is not indicated"}
                save_report(rep)
                continue
        print(f"=== {exp_id} lr={lr} bs={bs} max_epochs={ep}", flush=True)
        rep["runs"][exp_id] = train_run(exp_id, lr, bs, ep, data, rep)
        save_report(rep)
    done = {k: r for k, r in rep["runs"].items() if "val_subset_accuracy" in r}
    rep["selected_run"] = max(done, key=lambda k: done[k]["val_subset_accuracy"])
    rep["selection_rule"] = "highest val-subset accuracy (balanced subset, so accuracy and macro-F1 agree); test not used"
    save_report(rep)
    print(json.dumps({k: (r.get("val_subset_accuracy"), r.get("best_epoch"), r.get("training_seconds")) for k, r in rep["runs"].items()}, indent=1))
    print("selected", rep["selected_run"])
    return 0


def final(args) -> int:
    rep = load_report()
    info = json.loads((OUT_DIR / "training_info.json").read_text())
    assert info["experiment_id"] == rep["selected_run"], (info["experiment_id"], rep["selected_run"])
    pred = X.Predictor(OUT_DIR, "cuda", max_length=MAX_LEN)
    pred.predict(["warm up"] * 8)
    val = X.load_split("val")
    pv = pred.predict(production_texts(val.raw), 128)
    np.save(CACHE / "val_finetuned.npy", pv)
    val_rep = M.binary_report(val.y, np.where(pv[:, M.POS] > pv[:, M.NEG], M.POS, M.NEG))
    thr = M.tune_threshold(val.y, M.positive_score(pv))
    rep["final"] = {"experiment_id": info["experiment_id"], "val_full": {**val_rep, "confusion": M.confusion_2x2(val.y, np.where(pv[:, M.POS] > pv[:, M.NEG], M.POS, M.NEG))},
                    "threshold_tuning_val": thr}
    use_thr = thr["threshold"] if thr["val_accuracy"] >= val_rep["accuracy"] + 0.001 else 0.5
    rep["final"]["threshold_used"] = use_thr
    rep["final"]["threshold_rule"] = "tuned threshold only if it gains >= 0.1 pp on validation, else 0.5"
    test = X.load_split("test")
    vs, vn = pred.seconds, pred.inferred
    pt = pred.predict(production_texts(test.raw), 128)
    secs = pred.seconds - vs
    np.save(CACHE / "test_finetuned.npy", pt)
    p = M.apply_threshold(M.positive_score(pt), use_thr)
    rep["final"]["test"] = {**M.binary_report(test.y, p), "confusion": M.confusion_2x2(test.y, p),
                            "inference_seconds": round(secs, 1), "texts_per_second": round((pred.inferred - vn) / secs, 1)}
    with open(TEST_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"utc": now(), "config": f"fine-tuned binary {info['experiment_id']} (threshold {use_thr})"}) + "\n")
    base = CACHE / "test_current.npy"
    if base.exists():
        from scripts.sentiment_experiments import mcnemar, nearest  # noqa: E402

        pb = np.load(base)
        rep["final"]["mcnemar_vs_pretrained_nearest_test"] = mcnemar(test.y, nearest(pb), p)
    save_report(rep)
    print(json.dumps({"val": val_rep["accuracy"], "thr": thr, "test": rep["final"]["test"]["accuracy"],
                      "test_f1": rep["final"]["test"]["macro_f1"]}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["matrix", "final"])
    args = ap.parse_args()
    return {"matrix": matrix, "final": final}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
