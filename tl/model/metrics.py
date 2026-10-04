"""Classification + calibration metrics in numpy (no scikit-learn dependency).

Convention: label 1 = AI, 0 = human; `probs` are P(AI).
"""

from __future__ import annotations

import math

import numpy as np


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def confusion(y_true, probs, threshold: float = 0.5) -> dict[str, int]:
    y = _arr(y_true).astype(int)
    pred = (_arr(probs) >= threshold).astype(int)
    return {
        "tp": int(((pred == 1) & (y == 1)).sum()),
        "fp": int(((pred == 1) & (y == 0)).sum()),
        "tn": int(((pred == 0) & (y == 0)).sum()),
        "fn": int(((pred == 0) & (y == 1)).sum()),
    }


def roc_auc(y_true, probs) -> float | None:
    """Mann-Whitney U formulation, ties get average rank. None if only one class present."""
    y = _arr(y_true).astype(int)
    p = _arr(probs)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    sorted_p = p[order]
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def brier(y_true, probs) -> float:
    return float(np.mean((_arr(probs) - _arr(y_true)) ** 2))


def reliability_bins(y_true, probs, n_bins: int = 10) -> list[dict]:
    y, p = _arr(y_true), _arr(probs)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        mask = (p >= lo) & ((p < hi) if k < n_bins - 1 else (p <= hi))
        n = int(mask.sum())
        out.append({
            "lo": float(lo), "hi": float(hi), "n": n,
            "mean_pred": float(p[mask].mean()) if n else None,
            "frac_pos": float(y[mask].mean()) if n else None,
        })
    return out


def ece(y_true, probs, n_bins: int = 10) -> float:
    """Expected calibration error: bin-size-weighted |accuracy - confidence| over P(AI) bins."""
    total = len(_arr(y_true))
    if total == 0:
        return float("nan")
    return float(sum(b["n"] / total * abs(b["frac_pos"] - b["mean_pred"])
                     for b in reliability_bins(y_true, probs, n_bins) if b["n"]))


def classification_report(y_true, probs, threshold: float = 0.5) -> dict:
    y = _arr(y_true)
    n = len(y)
    if n == 0:
        return {"n": 0}
    c = confusion(y, probs, threshold)
    precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else None
    recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else (0.0 if precision is not None and recall is not None else None)
    fpr = c["fp"] / (c["fp"] + c["tn"]) if (c["fp"] + c["tn"]) else None
    return {
        "n": n,
        "n_ai": int(y.sum()),
        "n_human": int(n - y.sum()),
        "threshold": threshold,
        "accuracy": (c["tp"] + c["tn"]) / n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "roc_auc": roc_auc(y, probs),
        "brier": brier(y, probs),
        "ece": ece(y, probs),
        "confusion": c,
        "reliability": reliability_bins(y, probs),
    }


# ---------------------------------------------------------------------- temperature scaling


def sigmoid(z):
    z = _arr(z)
    return 1.0 / (1.0 + np.exp(-z))


def nll(y_true, margins, temperature: float = 1.0) -> float:
    p = np.clip(sigmoid(_arr(margins) / temperature), 1e-12, 1 - 1e-12)
    y = _arr(y_true)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(y_true, margins) -> float:
    """Single temperature T minimizing NLL of sigmoid(margin / T); margin = logit_ai - logit_human.

    Coarse log-grid then golden-section refinement. Returns 1.0 if there's nothing to fit.
    """
    y, m = _arr(y_true), _arr(margins)
    if len(y) < 2 or y.min() == y.max():
        return 1.0
    grid = np.exp(np.linspace(math.log(0.05), math.log(20.0), 60))
    losses = [nll(y, m, t) for t in grid]
    k = int(np.argmin(losses))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(60):
        if nll(y, m, c) < nll(y, m, d):
            b = d
        else:
            a = c
        c, d = b - g * (b - a), a + g * (b - a)
    return float((a + b) / 2)
