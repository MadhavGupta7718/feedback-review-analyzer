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


def confusion_2x2(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {tn: {pn: int(((y_true == t) & (y_pred == p)).sum()) for pn, p in (("negative", NEG), ("positive", POS))}
            for tn, t in (("negative", NEG), ("positive", POS))}


def positive_score(probs: np.ndarray) -> np.ndarray:
    """P(positive | not neutral) = p_pos / (p_pos + p_neg): the model's binary evidence with neutral mass removed."""
    return probs[:, POS] / np.clip(probs[:, POS] + probs[:, NEG], 1e-12, None)


def neutral_strategies(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """Four ways to score a 3-class model against binary truth.
    A neutral -> negative, B neutral -> positive (both inject a fixed bias by construction),
    C neutral -> nearest binary class by the model's own probabilities (= binary_forced),
    D neutral predictions excluded (reported with coverage; not comparable on accuracy alone)."""
    y_true = np.asarray(y_true)
    pred3 = probs.argmax(1)
    neu = pred3 == NEU
    preds = {"A_neutral_to_negative": np.where(neu, NEG, pred3), "B_neutral_to_positive": np.where(neu, POS, pred3),
             "C_nearest_binary": np.where(probs[:, POS] > probs[:, NEG], POS, NEG)}
    out = {k: {**binary_report(y_true, p), "confusion": confusion_2x2(y_true, p), "coverage": 1.0} for k, p in preds.items()}
    keep = ~neu
    out["D_exclude_neutral"] = {**binary_report(y_true[keep], pred3[keep]), "confusion": confusion_2x2(y_true[keep], pred3[keep]),
                                "coverage": round(float(keep.mean()), 4)}
    for v in out.values():
        v["neutral_prediction_rate"] = round(float(neu.mean()), 4)
    return out


def tune_threshold(y_true: np.ndarray, score: np.ndarray, grid: np.ndarray | None = None) -> dict:
    """Pick the positive-score threshold that maximises accuracy (ties -> closest to 0.5). Use on VALIDATION data only."""
    grid = np.round(np.arange(0.05, 0.9501, 0.005), 3) if grid is None else grid
    accs = np.array([float(((score > t).astype(int) * POS == y_true).mean()) for t in grid])
    best = np.flatnonzero(accs == accs.max())
    t = float(grid[best[np.argmin(np.abs(grid[best] - 0.5))]])
    return {"threshold": t, "val_accuracy": round(float(accs.max()), 4), "grid": [float(grid[0]), float(grid[-1]), 0.005],
            "accuracy_at_0.5": round(float(accs[np.argmin(np.abs(grid - 0.5))]), 4)}


def apply_threshold(score: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(score > threshold, POS, NEG)


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
