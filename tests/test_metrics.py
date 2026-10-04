import numpy as np
import pytest

from tl.model import metrics as M


def test_confusion_and_report():
    y = [1, 1, 0, 0, 1, 0]
    p = [0.9, 0.4, 0.2, 0.6, 0.8, 0.1]
    r = M.classification_report(y, p, 0.5)
    assert r["confusion"] == {"tp": 2, "fp": 1, "tn": 2, "fn": 1}
    assert r["precision"] == pytest.approx(2 / 3) and r["recall"] == pytest.approx(2 / 3)
    assert r["f1"] == pytest.approx(2 / 3) and r["accuracy"] == pytest.approx(4 / 6)
    assert r["false_positive_rate"] == pytest.approx(1 / 3)


def test_roc_auc_known_values():
    assert M.roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert M.roc_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)  # ties
    assert M.roc_auc([1, 1], [0.2, 0.3]) is None
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    assert M.roc_auc(y, y + rng.normal(0, 0.01, 2000)) == pytest.approx(1.0)


def test_brier_ece_reliability():
    y = [1, 0, 1, 0]
    assert M.brier(y, [1, 0, 1, 0]) == 0
    assert M.ece(y, [1.0, 0.0, 1.0, 0.0]) == 0
    assert M.ece([1, 1, 1, 1], [0.6, 0.6, 0.6, 0.6]) == pytest.approx(0.4)
    bins = M.reliability_bins([1, 0], [0.95, 0.05])
    assert sum(b["n"] for b in bins) == 2 and bins[-1]["frac_pos"] == 1.0


def test_degenerate_cases():
    r = M.classification_report([0, 0], [0.1, 0.2])
    assert r["precision"] is None and r["recall"] is None and r["roc_auc"] is None
    assert M.classification_report([], [])["n"] == 0


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(1)
    true_margin = rng.normal(0, 2, 5000)
    y = (rng.random(5000) < M.sigmoid(true_margin)).astype(int)
    overconfident = true_margin * 4  # model 4x too sure of itself
    t = M.fit_temperature(y, overconfident)
    assert t == pytest.approx(4, rel=0.15)
    assert M.ece(y, M.sigmoid(overconfident / t)) < M.ece(y, M.sigmoid(overconfident)) / 3
    assert M.fit_temperature([1, 1], [2.0, 3.0]) == 1.0
