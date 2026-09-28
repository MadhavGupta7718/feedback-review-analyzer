"""Sentiment140 accuracy study for cardiffnlp/twitter-roberta-base-sentiment-latest.

  python scripts/sentiment_experiments.py val-study      preprocessing / negation / neutral strategies / threshold on VALIDATION
  python scripts/sentiment_experiments.py test-eval      one-shot TEST evaluation of the baseline and the validation-selected config
  python scripts/sentiment_experiments.py compare        comparison models on validation (+ one test run each, reported only)
  python scripts/sentiment_experiments.py errors         error analysis on validation predictions

Split: data/interim/s140_split (scripts/build_s140_split.py). Every TEST evaluation is appended to
artifacts/reports/sentiment_test_log.jsonl so the number of test-set looks is auditable.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import config  # noqa: E402
from ml.evaluation import metrics as M  # noqa: E402
from ml.pii.leak_scan import scan  # noqa: E402
from ml.sentiment import experiment as X  # noqa: E402

REPORTS = config.ARTIFACTS_DIR / "reports"
CACHE = config.ARTIFACTS_DIR / "cache" / "sentiment_exp"
VAL_REPORT = REPORTS / "sentiment_study_val.json"
TEST_REPORT = REPORTS / "sentiment_study_test.json"
TEST_LOG = REPORTS / "sentiment_test_log.jsonl"
BATCH = 128


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def mcnemar(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Paired test on the same examples: b = A right & B wrong, c = A wrong & B right."""
    a_ok, b_ok = pred_a == y, pred_b == y
    b, c = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if b + c else 0.0
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "chi2": round(chi2, 3), "p_value": float(f"{math.erfc(math.sqrt(chi2 / 2)):.3g}")}


def nearest(probs: np.ndarray) -> np.ndarray:
    return np.where(probs[:, M.POS] > probs[:, M.NEG], M.POS, M.NEG)


def summary(y: np.ndarray, probs: np.ndarray) -> dict:
    s = M.neutral_strategies(y, probs)
    return {"strategies": s, "confusion_true2_pred3": M.confusion_2x3(y, probs.argmax(1))}


def confidence_analysis(y: np.ndarray, probs: np.ndarray) -> dict:
    pred3, conf = probs.argmax(1), probs.max(1)
    near = nearest(probs)
    score = M.positive_score(probs)
    bins = [0.33, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    by_conf = []
    for lo, hi in zip(bins, bins[1:]):
        m = (conf >= lo) & (conf < hi)
        by_conf.append({"confidence": f"[{lo:.2f}, {min(hi, 1):.2f})", "n": int(m.sum()),
                        "nearest_binary_accuracy": round(float((near[m] == y[m]).mean()), 4) if m.any() else None})
    neu = pred3 == M.NEU
    margin = np.abs(score - 0.5)
    by_margin = []
    for lo, hi in ((0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.501)):
        m = (margin >= lo) & (margin < hi)
        by_margin.append({"binary_margin": f"[{lo}, {min(hi, 0.5)})", "n": int(m.sum()), "share": round(float(m.mean()), 4),
                          "accuracy": round(float((near[m] == y[m]).mean()), 4) if m.any() else None})
    return {
        "mean_confidence_correct": round(float(conf[near == y].mean()), 4),
        "mean_confidence_wrong": round(float(conf[near != y].mean()), 4),
        "neutral_predictions": {"n": int(neu.sum()), "share": round(float(neu.mean()), 4),
                                "nearest_binary_accuracy_within_neutral": round(float((near[neu] == y[neu]).mean()), 4),
                                "true_negative_share_within_neutral": round(float((y[neu] == M.NEG).mean()), 4),
                                "mean_neutral_probability": round(float(probs[neu, M.NEU].mean()), 4)},
        "non_neutral_accuracy": round(float((near[~neu] == y[~neu]).mean()), 4),
        "accuracy_by_confidence": by_conf,
        "accuracy_by_binary_margin": by_margin,
    }


def log_test(entry: dict) -> None:
    with open(TEST_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"utc": now(), **entry}) + "\n")


# --------------------------------------------------------------------------- validation study
def val_study(args) -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    val = X.load_split("val")
    t0 = time.perf_counter()
    prepared = [X.prepare(r) for r in val.raw]
    prep_s = time.perf_counter() - t0
    pred = X.Predictor(X.sentiment_model_path(), "cuda")
    pred.predict(["warm up"] * 8)
    report: dict = {"generated_utc": now(), "split": "val", "n": int(len(val.y)),
                    "class_distribution": {"negative": int((val.y == M.NEG).sum()), "positive": int((val.y == M.POS).sum())},
                    "model": config.SENTIMENT_MODEL, "preprocess_seconds_current": round(prep_s, 1), "variants": {}}
    has_neg = np.array([X.negation_count(r) > 0 for r in val.raw])
    report["negation_rows"] = {"n": int(has_neg.sum()), "share": round(float(has_neg.mean()), 4)}
    base_texts = X.variant_texts("current", val.raw, prepared)
    probs_by = {}
    for name, (safe, desc) in X.VARIANTS.items():
        texts = base_texts if name == "current" else X.variant_texts(name, val.raw, prepared)
        before_s, before_n = pred.seconds, pred.inferred
        probs = pred.predict(texts, BATCH)
        probs_by[name] = probs
        np.save(CACHE / f"val_{name}.npy", probs)
        altered_neg = sum(1 for r, t in zip(val.raw, texts) if X.negation_count(r) != X.negation_count(t))
        empty = sum(1 for t in texts if not t.strip())
        s = summary(val.y, probs)
        rec = {"pii_safe": safe, "description": desc,
               "rows_differing_from_current": int(sum(a != b for a, b in zip(texts, base_texts))),
               "empty_inputs": empty, "negation_altered_rows": altered_neg,
               "new_unique_texts_inferred": pred.inferred - before_n, "inference_seconds": round(pred.seconds - before_s, 1),
               **s, "negation_subset_nearest_binary": M.binary_report(val.y[has_neg], nearest(probs)[has_neg])}
        if name != "current":
            rec["mcnemar_vs_current_nearest"] = mcnemar(val.y, nearest(probs_by["current"]), nearest(probs))
        report["variants"][name] = rec
        c = s["strategies"]["C_nearest_binary"]
        print(f"{name:22s} acc(C)={c['accuracy']:.4f} f1={c['macro_f1']:.4f} neutral={c['neutral_prediction_rate']:.4f} "
              f"neg-altered={altered_neg} diff-rows={rec['rows_differing_from_current']} t={rec['inference_seconds']}s", flush=True)
    report["throughput_unique_texts_per_s"] = round(pred.inferred / pred.seconds, 1)

    # selection: PII-safe variants only; switch from `current` only for a significant (McNemar p < 0.01) gain of >= 0.1 pp
    cur_acc = report["variants"]["current"]["strategies"]["C_nearest_binary"]["accuracy"]
    best, best_acc = "current", cur_acc
    for name, rec in report["variants"].items():
        if not rec["pii_safe"] or name == "current":
            continue
        acc = rec["strategies"]["C_nearest_binary"]["accuracy"]
        if acc >= cur_acc + 0.001 and rec["mcnemar_vs_current_nearest"]["p_value"] < 0.01 and acc > best_acc:
            best, best_acc = name, acc
    report["selected_preprocessing"] = best
    report["selection_rule"] = ("PII-safe variants only; keep 'current' unless a variant improves validation accuracy "
                                "(strategy C) by >= 0.1 pp with McNemar p < 0.01")

    probs = probs_by[best]
    score = M.positive_score(probs)
    thr = M.tune_threshold(val.y, score)
    report["threshold_tuning"] = {**thr, "score": "p_pos / (p_pos + p_neg)", "objective": "accuracy",
                                  "val_report_at_threshold": M.binary_report(val.y, M.apply_threshold(score, thr["threshold"]))}
    report["confidence_analysis"] = confidence_analysis(val.y, probs)
    VAL_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"selected": best, "threshold": report["threshold_tuning"]}, indent=2))
    return 0


# --------------------------------------------------------------------------- test evaluation
def eval_on(split, pred, texts, y, label, threshold=None) -> dict:
    before_s, before_n = pred.seconds, pred.inferred
    probs = pred.predict(texts, BATCH)
    secs = pred.seconds - before_s
    out = {"config": label, "n": int(len(y)), **summary(y, probs), "inference_seconds": round(secs, 1),
           "texts_per_second": round((pred.inferred - before_n) / secs, 1) if secs else None}
    if threshold is not None:
        p = M.apply_threshold(M.positive_score(probs), threshold)
        out["tuned_threshold"] = {"threshold": threshold, **M.binary_report(y, p), "confusion": M.confusion_2x2(y, p)}
    return out, probs


def test_eval(args) -> int:
    val_rep = json.loads(VAL_REPORT.read_text(encoding="utf-8"))
    sel, thr = val_rep["selected_preprocessing"], val_rep["threshold_tuning"]["threshold"]
    test = X.load_split("test")
    prepared = [X.prepare(r) for r in test.raw]
    pred = X.Predictor(X.sentiment_model_path(), "cuda")
    pred.predict(["warm up"] * 8)
    rep = {"generated_utc": now(), "split": "test", "n": int(len(test.y)),
           "class_distribution": {"negative": int((test.y == M.NEG).sum()), "positive": int((test.y == M.POS).sum())},
           "selected_preprocessing": sel, "validation_threshold": thr, "results": {}}
    texts = X.variant_texts("current", test.raw, prepared)
    base, p_base = eval_on("test", pred, texts, test.y, "baseline: current preprocessing", thr if sel == "current" else None)
    np.save(CACHE / "test_current.npy", p_base)
    rep["results"]["baseline_current"] = base
    log_test({"config": "pretrained current preprocessing (baseline + tuned threshold)" if sel == "current" else "baseline"})
    if sel != "current":
        t2 = X.variant_texts(sel, test.raw, prepared)
        r2, p2 = eval_on("test", pred, t2, test.y, f"selected preprocessing: {sel}", thr)
        np.save(CACHE / f"test_{sel}.npy", p2)
        rep["results"]["selected"] = r2
        log_test({"config": f"pretrained {sel} + tuned threshold"})
    sel_probs = p_base if sel == "current" else p2
    y = test.y
    tuned = M.apply_threshold(M.positive_score(sel_probs), thr)
    rep["mcnemar_tuned_vs_baseline_nearest"] = mcnemar(y, nearest(p_base), tuned)
    rep["mcnemar_tuned_vs_baseline_argmax_strict"] = mcnemar(y, p_base.argmax(1), tuned)
    old = old_eval_reference(pred)
    rep["reference_old_5k_eval_sample"] = old
    TEST_REPORT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    b = base["strategies"]["C_nearest_binary"]
    print(json.dumps({"baseline_C": b["accuracy"], "baseline_strict": base["strategies"], "tuned": rep["results"].get("selected", base).get("tuned_threshold")}, indent=1)[:3000])
    return 0


def old_eval_reference(pred) -> dict:
    """Continuity only: the earlier fixed 5K sample (not part of the new split design; rows may fall in any new split)."""
    import csv

    path = config.DATA_DIR / "interim" / "s140_eval_5k.csv"
    if not path.exists():
        return {"available": False}
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    y = np.array([M.NEG if r["target"] == "0" else M.POS for r in rows])
    texts = X.variant_texts("current", [r["text"] for r in rows], [X.prepare(r["text"]) for r in rows])
    probs = pred.predict(texts, BATCH)
    return {"n": int(len(y)), "C_nearest_binary_accuracy": M.binary_report(y, nearest(probs))["accuracy"],
            "note": "earlier evaluation sample; reported for continuity with previous results, not used for any selection"}


# --------------------------------------------------------------------------- comparison models
def compare(args) -> int:
    from ml.models import registry

    val, test = X.load_split("val"), X.load_split("test")
    v_texts = X.variant_texts("current", val.raw, [X.prepare(r) for r in val.raw])
    t_texts = X.variant_texts("current", test.raw, [X.prepare(r) for r in test.raw])
    out = {"generated_utc": now(), "preprocessing": "current (production)", "models": {}}
    for repo in config.COMPARISON_MODELS:
        m = registry.locate(repo)
        if not m.installed:
            out["models"][repo] = {"status": "NOT_INSTALLED"}
            continue
        import torch

        pred = X.Predictor(m.path, "cuda")
        pred.predict(["warm up"] * 8)
        rv, pv = eval_on("val", pred, v_texts, val.y, repo)
        np.save(CACHE / f"val_cmp_{repo.split('/')[-1]}.npy", pv)
        rt, pt = eval_on("test", pred, t_texts, test.y, repo)
        np.save(CACHE / f"test_cmp_{repo.split('/')[-1]}.npy", pt)
        log_test({"config": f"comparison model {repo} (reported only, not selected on test)"})
        out["models"][repo] = {"revision": m.revision, "val": rv, "test": rt,
                               "parameters_millions": round(sum(p.numel() for p in pred.model.parameters()) / 1e6, 1)}
        print(repo, "val", rv["strategies"]["C_nearest_binary"]["accuracy"], "test", rt["strategies"]["C_nearest_binary"]["accuracy"], flush=True)
        del pred
        torch.cuda.empty_cache()
    (REPORTS / "sentiment_study_compare.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------- error analysis
CATEGORIES = {
    "negation": lambda raw, txt: X.negation_count(raw) > 0,
    "short_text (<= 4 words)": lambda raw, txt: len(re.findall(r"\w+", txt.replace("@user", "").replace("http", ""))) <= 4,
    "slang / abbreviations": lambda raw, txt: bool(re.search(
        r"\b(lol|lmao|lmfao|omg|u|ur|gonna|wanna|gotta|ya|yea|nah|cuz|coz|tho|thx|pls|plz|idk|imo|smh|wtf|rofl|luv|bday|2day|2nite|b4|gr8|haha\w*|hehe\w*)\b", txt, re.I)),
    "elongated words (spelling proxy)": lambda raw, txt: bool(re.search(r"([a-z])\1{2,}", txt, re.I)),
    "emoji / emoticon remnants": lambda raw, txt: bool(re.search(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]|<3|[:;]-?[()DPp]", txt)),
    "contrast / mixed sentiment (but, though ...)": lambda raw, txt: bool(re.search(r"\b(but|though|although|however|yet)\b", txt, re.I)),
    "possible sarcasm (keyword proxy)": lambda raw, txt: bool(re.search(
        r"\b(yeah right|thanks a lot|just what i needed|love (it )?when|oh joy|oh great|so much fun|wonderful)\b", txt, re.I)),
    "question": lambda raw, txt: "?" in txt,
    "reply (starts with @user)": lambda raw, txt: txt.startswith("@user"),
}


def errors(args) -> int:
    val = X.load_split("val")
    prepared = [X.prepare(r) for r in val.raw]
    texts = X.variant_texts("current", val.raw, prepared)
    y = val.y
    sources = {"pretrained_nearest_binary": CACHE / "val_current.npy"}
    ft = CACHE / "val_finetuned.npy"
    if ft.exists():
        sources["finetuned"] = ft
    out = {"generated_utc": now(), "split": "val", "n": int(len(y)),
           "note": "Categories are automatic keyword/regex heuristics (proxies), not manual annotation; a tweet can be in several.",
           "models": {}}
    masks = {name: np.array([f(r, t) for r, t in zip(val.raw, texts)]) for name, f in CATEGORIES.items()}
    examples_out: dict = {}
    for key, path in sources.items():
        probs = np.load(path)
        pred = nearest(probs)
        wrong = pred != y
        conf = probs.max(1)
        cats = {}
        for name, m in masks.items():
            cats[name] = {"share_of_rows": round(float(m.mean()), 4), "error_rate": round(float(wrong[m].mean()), 4) if m.any() else None,
                          "share_of_errors": round(float((wrong & m).sum() / max(1, wrong.sum())), 4)}
        low = np.abs(M.positive_score(probs) - 0.5) < 0.1
        cats["low binary margin (|score-0.5| < 0.1)"] = {"share_of_rows": round(float(low.mean()), 4), "error_rate": round(float(wrong[low].mean()), 4),
                                                        "share_of_errors": round(float((wrong & low).sum() / max(1, wrong.sum())), 4)}
        cats["domain-specific language"] = {"note": "not measurable automatically without a domain lexicon"}

        def examples(mask, order, k=8):
            idx = [i for i in np.flatnonzero(mask)[order(np.flatnonzero(mask))]]
            res = []
            for i in idx:
                if len(res) >= k:
                    break
                if scan(texts[i]):
                    continue
                res.append({"line": int(val.lines[i]), "text_redacted": texts[i], "true": "positive" if y[i] == M.POS else "negative",
                            "p_negative": round(float(probs[i, 0]), 3), "p_neutral": round(float(probs[i, 1]), 3), "p_positive": round(float(probs[i, 2]), 3)})
            return res

        by_conf_desc = lambda ix: np.argsort(-conf[ix])  # noqa: E731
        by_margin = lambda ix: np.argsort(np.abs(M.positive_score(probs)[ix] - 0.5))  # noqa: E731
        out["models"][key] = {
            "overall_error_rate": round(float(wrong.mean()), 4),
            "false_positives": int((wrong & (y == M.NEG)).sum()), "false_negatives": int((wrong & (y == M.POS)).sum()),
            "categories": cats,
        }
        examples_out[key] = {
            "high_confidence_false_positives": examples(wrong & (y == M.NEG), by_conf_desc),
            "high_confidence_false_negatives": examples(wrong & (y == M.POS), by_conf_desc),
            "neutral_predictions": examples(probs.argmax(1) == M.NEU, by_conf_desc),
            "low_confidence": examples(np.ones_like(wrong), by_margin),
        }
    out["examples_file"] = "artifacts/cache/sentiment_exp/error_examples.json (gitignored: dataset text is never committed)"
    (REPORTS / "sentiment_error_analysis.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    (CACHE / "error_examples.json").write_text(json.dumps(examples_out, indent=2, ensure_ascii=False), encoding="utf-8")
    for key, v in out["models"].items():
        print(key, v["overall_error_rate"], {k: c.get("error_rate") for k, c in v["categories"].items()})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["val-study", "test-eval", "compare", "errors"])
    args = ap.parse_args()
    return {"val-study": val_study, "test-eval": test_eval, "compare": compare, "errors": errors}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
