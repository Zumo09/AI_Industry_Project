from __future__ import annotations
from typing import Tuple
from torch import Tensor
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import auc
import matplotlib.pyplot as plt

plt.style.use("ggplot")  # type: ignore


@torch.no_grad()
def fill_tolerance(labels: Tensor, tolerance: int) -> Tensor:
    ch = labels.size(1)
    ws = (ch, 1, 2 * tolerance + 1)
    ones = torch.ones(ws).to(labels)
    full = F.conv1d(labels, ones, padding="same", groups=ch)
    full = (full > 0).to(labels)
    return full


def errors_curve(
    y_score: Tensor, y_true: Tensor, tolerance: int, num_thrs: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert y_score.size() == y_true.size()
    y_true = y_true.unsqueeze(1).float()
    y_true_full = fill_tolerance(y_true, tolerance)

    mn, mx = y_score.min().item(), y_score.max().item()
    step = (mx - mn) / num_thrs
    thrs = np.arange(mn, mx, step)

    preds = torch.stack([y_score > th for th in thrs], dim=1).to(y_true)
    preds_full = fill_tolerance(preds, tolerance)
    fps = ((1 - y_true_full) * preds).sum(2).mean(0).cpu().numpy()
    fns = (y_true * (1 - preds_full)).sum(2).mean(0).cpu().numpy()

    return fps, fns, thrs
    # first_ind = fns.searchsorted(fns[0], side="right")
    # first_ind = first_ind - 1 if 0 < first_ind < num_thrs else None
    # # stop with false positives zero
    # tps = fps[-1] - fps
    # last_ind = tps.searchsorted(tps[-1]) + 1
    # sl = slice(first_ind, last_ind)

    # return fps[sl], fns[sl], thrs[sl]


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    msk = den == 0
    den[msk] = 9e-15
    # num[msk] = 0
    res = num / den
    res[np.isnan(res)] = 0
    return res


def precision_recall_f1(
    fps: np.ndarray, fns: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tps = fns[-1] - fns
    precision = _safe_divide(tps, tps + fps)
    recall = tps / tps[0] if tps[0] != 0 else tps * 0
    f1 = 2 * _safe_divide(precision * recall, precision + recall)
    return precision, recall, f1


class HPCMetrics:
    def __init__(
        self, c_alarm: float, c_missed: float, tolerance: int, n_thrs_levels: int = 100
    ) -> None:
        self.c_alarm = c_alarm
        self.c_missed = c_missed
        self.tolerance = tolerance
        self.n_ths = n_thrs_levels

        self.thresholds = np.array([])
        self.false_positives = np.array([])
        self.false_negatives = np.array([])
        self.cost = np.array([])
        # self.precision = np.array([])
        # self.recall = np.array([])
        # self.f1_score = np.array([])
        self.fitted = False

    def fit(self, signals: Tensor, labels: Tensor) -> HPCMetrics:
        fps, fns, ths = errors_curve(signals, labels, self.tolerance, self.n_ths)
        self.false_positives = fps
        self.false_negatives = fns
        self.thresholds = ths

        fpc = self.c_alarm * self.false_positives
        fnc = self.c_missed * self.false_negatives
        self.cost = fpc + fnc

        # pr, rc, f1 = precision_recall_f1(fps, fns)
        # self.precision = pr
        # self.recall = rc
        # self.f1_score = f1

        self.fitted = True
        return self

    def optimize(self) -> Tuple[float, float]:
        assert self.fitted, "Call .fit(...) first"
        best_th = self.thresholds[np.argmin(self.cost)]
        best_cost = np.min(self.cost)
        return best_th, best_cost


def default_cmodel() -> HPCMetrics:
    return HPCMetrics(1, 5, 12, 100)


###################################################
##                   PLOTS                       ##
###################################################


def plot_cost(cmodel: HPCMetrics, figsize: Tuple[int, int] = (15, 5)) -> None:
    best_th, best_cost = cmodel.optimize()
    _, ax = plt.subplots(figsize=figsize)
    ax.set_title("Cost")
    ax.plot(cmodel.thresholds, cmodel.cost)
    ax.axvline(best_th, c="gray", alpha=0.5)
    ax.axhline(best_cost, c="gray", alpha=0.5)
    ax.set_xlabel("thresholds")
    plt.show()


def plot_errors_curve(
    false_positives: np.ndarray,
    false_negatives: np.ndarray,
    thresholds: np.ndarray,
    figsize: Tuple[int, int] = (15, 5),
) -> None:
    # axes: Tuple[plt.Axes, ...]
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
    c_ax.set_ylabel("False Negatives")
    c_ax.set_xlabel("False Positives")
    plt.show()


def plot_precision_recall_f1_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    thresholds: np.ndarray,
    figsize: Tuple[int, int] = (15, 15),
) -> None:
    # precision = precision[:-1]
    # recall = recall[:-1]
    # f1 = f1[:-1]
    # thresholds = thresholds[:-1]
    auc_pr = auc(recall, precision)
    # axes: Tuple[Tuple[plt.Axes, ...], ...]
    _, axes = plt.subplots(2, 2, figsize=figsize)  # type: ignore
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
