"""Sentiment evaluation against BINARY ground truth (Sentiment140: 0=negative, 4=positive; no neutral class).

The model predicts three classes. Three explicitly different scorings are reported, never mixed:

  binary_forced   neutral is ignored: predict positive iff P(positive) > P(negative). Every example is scored.
                  This is the headline metric because it matches the binary label space.
  strict_3class   the model's argmax label is used; a "neutral" prediction is counted as WRONG for both
                  classes (the ground truth never says neutral). Lower bound.
  abstain         examples predicted neutral are excluded; accuracy is reported on the remainder together
                  with coverage (share of examples not abstained).
"""
from __future__ import annotations

import numpy as np

NEG, NEU, POS = 0, 1, 2


def _prf(y_true: np.ndarray, y_pred: np.ndarray, positive: int) -> dict:
    tp = int(((y_pred == positive) & (y_true == positive)).sum())
    fp = int(((y_pred == positive) & (y_true != positive)).sum())
    fn = int(((y_pred != positive) & (y_true == positive)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "support": int((y_true == positive).sum())}


def binary_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """y_true / y_pred in {NEG, POS} (y_pred may contain NEU, which counts as wrong)."""
    neg = _prf(y_true, y_pred, NEG)
    pos = _prf(y_true, y_pred, POS)
    return {
        "n": int(len(y_true)),
        "accuracy": round(float((y_true == y_pred).mean()), 4) if len(y_true) else 0.0,
        "negative": neg,
        "positive": pos,
        "macro_f1": round((neg["f1"] + pos["f1"]) / 2, 4),
    }


def confusion_2x3(y_true: np.ndarray, y_pred3: np.ndarray) -> dict:
    """Rows = true {negative, positive}; columns = predicted {negative, neutral, positive}."""
    out = {}
    for tname, t in (("negative", NEG), ("positive", POS)):
        out[tname] = {pname: int(((y_true == t) & (y_pred3 == p)).sum()) for pname, p in (("negative", NEG), ("neutral", NEU), ("positive", POS))}
    return out


def evaluate_binary_truth(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """y_true in {NEG, POS}; probs (n,3) ordered negative, neutral, positive."""
    y_true = np.asarray(y_true)
    pred3 = probs.argmax(1)
    forced = np.where(probs[:, POS] > probs[:, NEG], POS, NEG)
    keep = pred3 != NEU
    return {
        "binary_forced": binary_report(y_true, forced),
        "strict_3class": binary_report(y_true, pred3),
        "abstain": {**binary_report(y_true[keep], pred3[keep]), "coverage": round(float(keep.mean()), 4)},
        "neutral_prediction_rate": round(float((pred3 == NEU).mean()), 4),
        "confusion_true2_pred3": confusion_2x3(y_true, pred3),
    }


def evaluate_3class(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """For datasets that DO carry a neutral label (the synthetic set - weak template labels)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    per = {name: _prf(y_true, y_pred, c) for name, c in (("negative", NEG), ("neutral", NEU), ("positive", POS))}
    cm = {tn: {pn: int(((y_true == t) & (y_pred == p)).sum()) for pn, p in (("negative", NEG), ("neutral", NEU), ("positive", POS))}
          for tn, t in (("negative", NEG), ("neutral", NEU), ("positive", POS))}
    return {
        "n": int(len(y_true)),
        "accuracy": round(float((y_true == y_pred).mean()), 4),
        "per_class": per,
        "macro_f1": round(float(np.mean([v["f1"] for v in per.values()])), 4),
        "confusion": cm,
    }
