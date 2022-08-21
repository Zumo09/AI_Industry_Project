from typing import Dict
from torch import Tensor


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
