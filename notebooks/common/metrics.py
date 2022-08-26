import re
from typing import Dict, List, Tuple
from torch import Tensor
import numpy as np
import torch

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt


def _safe_divide(num: Tensor, denom: Tensor) -> Tensor:
    """prevent zero division."""
    denom[denom == 0.0] = 1
    return num / denom


def compute_metrics(preds: Tensor, target: Tensor) -> Dict[str, Tensor]:
    true_pred = target == preds
    false_pred = target != preds
    pos_pred = preds == 1
    neg_pred = preds == 0

    tp = (true_pred * pos_pred).sum()
    fp = (false_pred * pos_pred).sum()
    fn = (false_pred * neg_pred).sum()

    precision = _safe_divide(tp.float(), tp + fp)
    recall = _safe_divide(tp.float(), tp + fn)

    f1 = 2 * _safe_divide(precision * recall, precision + recall)

    return dict(
        f1=f1,
        precision=precision,
        recall=recall,
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
    )


def true_positive_rate(preds: Tensor, target: Tensor) -> Tensor:
    return compute_metrics(preds, target)["recall"]


def evaluate_thresholds(
    all_errors: Tensor, all_labels: Tensor
) -> Dict[str, List[float]]:
    f1 = []
    precision = []
    recall = []
    tp = []
    fp = []
    fn = []
    thresholds = np.arange(all_errors.min(), all_errors.max(), 1e-3)
    for at in thresholds:
        labels = torch.tensor(all_errors > at, dtype=torch.int8)
        metrics = compute_metrics(labels.flatten(), all_labels.flatten())
        f1.append(metrics["f1"])
        precision.append(metrics["precision"])
        recall.append(metrics["recall"])
        tp.append(metrics["true_positive"])
        fp.append(metrics["false_positive"])
        fn.append(metrics["false_negative"])

    return dict(
        thresholds=list(thresholds),
        f1=f1,
        precision=precision,
        recall=recall,
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
    )


def plot_threshold_metrics(
    metrics: Dict[str, List[float]], figsize: Tuple[int, int] = (20, 10)
) -> None:
    axes: Tuple[Tuple[plt.Axes, ...], ...]
    fig, axes = plt.subplots(2, 3, figsize=figsize)  # type: ignore
    axes[0][0].set_title("F1")
    axes[0][0].plot(metrics["thresholds"], metrics["f1"])
    axes[0][0].set_xlabel("thresholds")

    axes[0][1].set_title("Precision")
    axes[0][1].plot(metrics["thresholds"], metrics["precision"])
    axes[0][1].set_xlabel("thresholds")

    axes[0][2].set_title("Recall")
    axes[0][2].plot(metrics["thresholds"], metrics["recall"])
    axes[0][2].set_xlabel("thresholds")

    axes[1][0].set_title("True Positives")
    axes[1][0].plot(metrics["thresholds"], metrics["true_positive"])
    axes[1][0].set_xlabel("thresholds")

    axes[1][1].set_title("False Positives")
    axes[1][1].plot(metrics["thresholds"], metrics["false_positive"])
    axes[1][1].set_xlabel("thresholds")

    axes[1][2].set_title("Precision - Recall")
    axes[1][2].plot(metrics["precision"], metrics["recall"])
    axes[1][2].set_xlabel("precision")
    axes[1][2].set_xlabel("recall")
    plt.show()


def plot_precision_recall_curve(
    errors: Tensor, labels: Tensor, figsize: Tuple[int, int] = (10, 10)
) -> None:
    precision, recall, thresholds = precision_recall_curve(
        labels.flatten(), errors.flatten()
    )
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

    prc_ax.set_title("Precision - Recall")
    prc_ax.plot(precision, recall)
    prc_ax.set_xlabel("precision")
    prc_ax.set_xlabel("recall")
    plt.show()
