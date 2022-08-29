from typing import Sequence, Tuple
from torch import Tensor
import numpy as np
import pandas as pd

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


def average_precision_score(signals: Tensor, labels: Tensor) -> float:
    return _skm.average_precision_score(
        labels.cpu().flatten(), signals.cpu().flatten(), pos_label=0
    )


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


def det_curve(
    signals: Tensor, labels: Tensor
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _skm.det_curve(labels.cpu().flatten(), signals.cpu().flatten(), pos_label=0)


def precision_recall_curve(
    signals: Tensor, labels: Tensor
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _skm.precision_recall_curve(
        labels.cpu().flatten(), signals.cpu().flatten(), pos_label=0
    )


def get_errors(signal: pd.Series, labels: pd.Series, thr: float, tolerance: int = 1):
    pred = signal[signal > thr].index
    anomalies = labels[labels != 0].index

    fp = set(pred)
    fn = set(anomalies)
    for lag in range(-tolerance, tolerance + 1):
        fp = fp - set(anomalies + lag)
        fn = fn - set(pred + lag)
    return fp, fn


class HPCMetrics:
    def __init__(self, c_alarm, c_missed, tolerance):
        self.c_alarm = c_alarm
        self.c_missed = c_missed
        self.tolerance = tolerance

    def cost(self, signal: Tensor, labels: Tensor, thr: float):
        assert signal.size() == labels.size()
        if len(signal.size()) == 1:
            return self._cost(signal, labels, thr)
        else:
            # Batched
            return sum(self._cost(s, l, thr) for s, l in zip(signal, labels)) / len(
                signal
            )

    def _cost(self, signal: Tensor, labels: Tensor, thr: float):
        # Obtain signals
        fp, fn = get_errors(pd.Series(signal), pd.Series(labels), thr, self.tolerance)

        # Compute the cost
        cost = self.c_alarm * len(fp) + self.c_missed * len(fn)

        return cost

    def opt_threshold(self, signal: Tensor, labels: Tensor, th_range: Sequence[float]):
        costs = [self.cost(signal, labels, th) for th in th_range]
        best_th = th_range[np.argmin(costs)]
        best_cost = np.min(costs)
        return best_th, best_cost


def plot_errors_curve(
    signals: Tensor, labels: Tensor, tolerance: int, figsize: Tuple[int, int] = (15, 5)
) -> None:
    fpr, fnr, thresholds = errors_curve(signals, labels, tolerance)
    axes: Tuple[plt.Axes, ...]
    _, axes = plt.subplots(1, 3, figsize=figsize)  # type: ignore
    fn_ax, fp_ax, c_ax = axes
    fn_ax.set_title("False Negatives")
    fn_ax.plot(thresholds, fnr)
    fn_ax.set_xlabel("thresholds")

    fp_ax.set_title("False Positives")
    fp_ax.plot(thresholds, fpr)
    fp_ax.set_xlabel("thresholds")

    c_ax.set_title(f"Det curve")
    c_ax.plot(fpr, fnr)
    c_ax.set_ylabel("precision")
    c_ax.set_xlabel("recall")
    plt.show()


def plot_det_curve(
    signals: Tensor, labels: Tensor, figsize: Tuple[int, int] = (15, 5)
) -> None:
    fpr, fnr, thresholds = det_curve(signals, labels)
    axes: Tuple[plt.Axes, ...]
    _, axes = plt.subplots(1, 3, figsize=figsize)  # type: ignore
    fn_ax, fp_ax, c_ax = axes
    fn_ax.set_title("FNR")
    fn_ax.plot(thresholds, fnr)
    fn_ax.set_xlabel("thresholds")

    fp_ax.set_title("FPR")
    fp_ax.plot(thresholds, fpr)
    fp_ax.set_xlabel("thresholds")

    c_ax.set_title(f"Det curve")
    c_ax.plot(fpr, fnr)
    c_ax.set_ylabel("precision")
    c_ax.set_xlabel("recall")
    plt.show()


def plot_precision_recall_curve(
    signals: Tensor, labels: Tensor, figsize: Tuple[int, int] = (15, 15)
) -> None:
    precision, recall, thresholds = precision_recall_curve(signals, labels)
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


# from typing import Dict, List
# import torch

# def _safe_divide(num: Tensor, denom: Tensor) -> Tensor:
#     """prevent zero division."""
#     denom[denom == 0.0] = 1
#     return num / denom


# def compute_metrics(preds: Tensor, target: Tensor) -> Dict[str, float]:
#     true_pred = target == preds
#     false_pred = target != preds
#     pos_pred = preds == 1
#     neg_pred = preds == 0

#     tp = (true_pred * pos_pred).sum()
#     fp = (false_pred * pos_pred).sum()
#     fn = (false_pred * neg_pred).sum()

#     precision = _safe_divide(tp.float(), tp + fp)
#     recall = _safe_divide(tp.float(), tp + fn)

#     f1 = 2 * _safe_divide(precision * recall, precision + recall)

#     return dict(
#         f1=f1.item(),
#         precision=precision.item(),
#         recall=recall.item(),
#         true_positive=tp.item(),
#         false_positive=fp.item(),
#         false_negative=fn.item(),
#     )


# def true_positive_rate(preds: Tensor, target: Tensor) -> float:
#     return compute_metrics(preds, target)["recall"]

# def evaluate_thresholds(
#     all_signals: Tensor, all_labels: Tensor
# ) -> Dict[str, List[float]]:
#     f1 = []
#     precision = []
#     recall = []
#     tp = []
#     fp = []
#     fn = []
#     thresholds = np.arange(all_signals.min(), all_signals.max(), 1e-3)
#     for at in thresholds:
#         labels = torch.tensor(all_signals > at, dtype=torch.int8)
#         metrics = compute_metrics(labels.flatten(), all_labels.flatten())
#         f1.append(metrics["f1"])
#         precision.append(metrics["precision"])
#         recall.append(metrics["recall"])
#         tp.append(metrics["true_positive"])
#         fp.append(metrics["false_positive"])
#         fn.append(metrics["false_negative"])

#     auc_pr = average_precision_score(all_labels, all_signals)

#     return dict(
#         thresholds=list(thresholds),
#         f1=f1,
#         precision=precision,
#         recall=recall,
#         true_positive=tp,
#         false_positive=fp,
#         false_negative=fn,
#         auc=[auc_pr],
#     )


# def plot_threshold_metrics(
#     metrics: Dict[str, List[float]], figsize: Tuple[int, int] = (20, 10)
# ) -> None:
#     axes: Tuple[Tuple[plt.Axes, ...], ...]
#     fig, axes = plt.subplots(2, 3, figsize=figsize)  # type: ignore
#     axes[0][0].set_title("F1")
#     axes[0][0].plot(metrics["thresholds"], metrics["f1"])
#     axes[0][0].set_xlabel("thresholds")

#     axes[0][1].set_title("Precision")
#     axes[0][1].plot(metrics["thresholds"], metrics["precision"])
#     axes[0][1].set_xlabel("thresholds")

#     axes[0][2].set_title("Recall")
#     axes[0][2].plot(metrics["thresholds"], metrics["recall"])
#     axes[0][2].set_xlabel("thresholds")

#     axes[1][0].set_title("True Positives")
#     axes[1][0].plot(metrics["thresholds"], metrics["true_positive"])
#     axes[1][0].set_xlabel("thresholds")

#     axes[1][1].set_title("False Positives")
#     axes[1][1].plot(metrics["thresholds"], metrics["false_positive"])
#     axes[1][1].set_xlabel("thresholds")

#     axes[1][2].set_title(f"Precision - Recall (AUC={metrics['auc']})")
#     axes[1][2].plot(metrics["precision"], metrics["recall"])
#     axes[1][2].set_xlabel("precision")
#     axes[1][2].set_xlabel("recall")
#     plt.show()
