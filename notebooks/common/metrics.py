from __future__ import annotations
from typing import Tuple
from torch import Tensor
import numpy as np

from sklearn import metrics as _skm
import matplotlib.pyplot as plt

plt.style.use("ggplot")


def _fp_fn_curve(y_true, y_score, pos_label=None, sample_weight=None):
    """Compute errors for different probability thresholds.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True binary labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.

    y_score : ndarray of shape of (n_samples,)
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    pos_label : int or str, default=None
        The label of the positive class.
        When ``pos_label=None``, if `y_true` is in {-1, 1} or {0, 1},
        ``pos_label`` is set to 1, otherwise an error will be raised.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    fp  : ndarray of shape (n_thresholds,)
        False positive count (FP) such that element i is the false positive
        count of predictions with score >= thresholds[i].

    fn  : ndarray of shape (n_thresholds,)
        False negative count (FN) such that element i is the false negative
        count of predictions with score >= thresholds[i].

    thresholds : ndarray of shape (n_thresholds,)
        Decreasing score values.

    """
    if len(np.unique(y_true)) != 2:
        raise ValueError(
            "Only one class present in y_true. Detection error "
            "tradeoff curve is not defined in that case."
        )

    fps, tps, thresholds = _skm._ranking._binary_clf_curve(
        y_true, y_score, pos_label=pos_label, sample_weight=sample_weight
    )

    fns = tps[-1] - tps

    # start with false positives zero
    first_ind = (
        fps.searchsorted(fps[0], side="right") - 1
        if fps.searchsorted(fps[0], side="right") > 0
        else None
    )
    # stop with false negatives zero
    last_ind = tps.searchsorted(tps[-1]) + 1
    sl = slice(first_ind, last_ind)

    # reverse the output such that list of false positives is decreasing
    return (fps[sl][::-1], fns[sl][::-1], thresholds[sl][::-1])


def errors_curve(
    signals: Tensor, labels: Tensor, tolerance: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    signals = signals.cpu().flatten()

    full_labels = labels.cpu().clone()
    for x, y in (labels == 1).nonzero():
        for i in range(-tolerance, tolerance + 1):
            if 0 <= y + i < full_labels.size(1):
                full_labels[x, y + i] = 1
    full_labels = full_labels.flatten()

    fp, _, thrs = _fp_fn_curve(full_labels, signals, pos_label=1)
    _, fn, _ = _fp_fn_curve(1 - full_labels, -1 * signals, pos_label=0)

    return fp, fn, thrs


def average_precision_score(signals: Tensor, labels: Tensor) -> float:
    return _skm.average_precision_score(
        labels.cpu().flatten(), signals.cpu().flatten(), pos_label=0
    )


def precision_recall_curve(
    signals: Tensor, labels: Tensor
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _skm.precision_recall_curve(
        labels.cpu().flatten(), signals.cpu().flatten(), pos_label=0
    )


class HPCMetrics:
    def __init__(self, c_alarm: float, c_missed: float, tolerance: int) -> None:
        self.c_alarm = c_alarm
        self.c_missed = c_missed
        self.tolerance = tolerance

        self.false_positives = np.array([])
        self.false_negatives = np.array([])
        self.cost = np.array([])
        self.precision = np.array([])
        self.recall = np.array([])
        self.f1_score = np.array([])
        self.thresholds = np.array([])
        self.fitted = False

    def fit(self, signals: Tensor, labels: Tensor) -> HPCMetrics:
        fp, fn, th = errors_curve(signals, labels, self.tolerance)
        self.false_positives = fp
        self.false_negatives = fn
        self.thresholds = th

        fpc = self.c_alarm * self.false_positives
        fnc = self.c_missed * self.false_negatives
        self.cost = fpc + fnc

        tp = fn[-1] - fn
        self.precision = tp / (tp + fp)
        self.precision[np.isnan(self.precision)] = 0
        self.recall = tp / tp[-1]

        _pxr = self.precision * self.recall
        _ppr = self.precision + self.recall
        self.f1_score = 2 * _pxr / _ppr

        self.fitted = True
        return self

    def optimize(self) -> Tuple[float, float]:
        assert self.fitted
        cost = self.cost
        best_th = self.thresholds[np.argmin(cost)]
        best_cost = np.min(cost)
        return best_th, best_cost

    def plot(self, figsize: Tuple[int, int] = (15, 5)) -> None:
        best_th, best_cost = self.optimize()
        _, ax = plt.subplots(figsize=figsize)
        ax.set_title("Cost")
        ax.plot(self.thresholds, self.cost)
        ax.axvline(best_th)
        ax.axhline(best_cost)
        ax.set_xlabel("thresholds")
        plt.show()


###################################################
##                   PLOTS                       ##
###################################################


def plot_errors_curve(
    false_positives: np.ndarray,
    false_negatives: np.ndarray,
    thresholds: np.ndarray,
    figsize: Tuple[int, int] = (15, 5),
) -> None:
    axes: Tuple[plt.Axes, ...]
    _, axes = plt.subplots(1, 3, figsize=figsize)  # type: ignore
    fn_ax, fp_ax, c_ax = axes
    fn_ax.set_title("False Negatives")
    fn_ax.plot(thresholds, false_negatives)
    fn_ax.set_xlabel("thresholds")

    fp_ax.set_title("False Positives")
    fp_ax.plot(thresholds, false_positives)
    fp_ax.set_xlabel("thresholds")

    c_ax.set_title(f"Det curve")
    c_ax.plot(false_positives, false_negatives)
    c_ax.set_ylabel("precision")
    c_ax.set_xlabel("recall")
    plt.show()


def plot_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    thresholds: np.ndarray,
    figsize: Tuple[int, int] = (15, 15),
) -> None:
    auc_pr = _skm.auc(recall, precision)
    recall = recall[:-1]
    precision = precision[:-1]
    axes: Tuple[Tuple[plt.Axes, ...], ...]
    _, axes = plt.subplots(2, 2, figsize=figsize)  # type: ignore
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    ((prec_ax, rec_ax), (f1_ax, prc_ax)) = axes
    prec_ax.set_title("Precision")
    prec_ax.plot(thresholds, precision)
    prec_ax.set_xlabel("thresholds")

    rec_ax.set_title("Recall")
    rec_ax.plot(thresholds, recall)
    rec_ax.set_xlabel("thresholds")

    f1_ax.set_title("F1")
    f1_ax.plot(thresholds, f1)
    f1_ax.set_xlabel("thresholds")

    prc_ax.set_title(f"Precision - Recall (AUC={auc_pr:.3f})")
    prc_ax.plot(recall, precision)
    prc_ax.set_ylabel("precision")
    prc_ax.set_xlabel("recall")
    plt.show()
