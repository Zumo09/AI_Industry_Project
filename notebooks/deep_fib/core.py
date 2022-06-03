from typing import Dict, Optional
import random

import torch
from torch import Tensor
from torch.nn import Module

# from torchmetrics.functional import f1_score


def reconstruction_error(preds: Tensor, targets: Tensor) -> Tensor:
    num_cols = targets.size(-1)
    return torch.linalg.norm(preds - targets, ord=1, dim=-1) / num_cols


def residual_error(preds: Tensor, targets: Tensor) -> Tensor:
    return torch.mean(torch.abs(preds - targets), dim=-1)


def _safe_divide(num: Tensor, denom: Tensor) -> Tensor:
    """prevent zero division."""
    denom[denom == 0.0] = 1
    return num / denom


def f1_score(preds: Tensor, target: Tensor) -> Tensor:
    true_pred = target == preds
    false_pred = target != preds
    pos_pred = preds == 1
    neg_pred = preds == 0

    tp = (true_pred * pos_pred).sum()
    fp = (false_pred * pos_pred).sum()
    fn = (false_pred * neg_pred).sum()

    precision = _safe_divide(tp.float(), tp + fp)
    recall = _safe_divide(tp.float(), tp + fn)

    return 2 * _safe_divide(precision * recall, precision + recall)


def true_positive_rate(preds: Tensor, target: Tensor) -> Tensor:
    true_pred = target == preds
    false_pred = target != preds
    pos_pred = preds == 1
    neg_pred = preds == 0

    tp = (true_pred * pos_pred).sum()
    fn = (false_pred * neg_pred).sum()

    return _safe_divide(tp.float(), tp + fn)


class DeepFIBEngine:
    def __init__(
        self,
        anomaly_threshold: float,
        masks: Optional[Tensor] = None,
        mask_value: int = -1,
    ):
        self.anomaly_threshold = anomaly_threshold
        self.masks = masks
        self.mask_value = mask_value

    def train_step(self, model: Module, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        assert self.masks is not None, "Masks not initializes. Engine can't train'"
        inputs = batch["data"]
        batch_size = len(inputs)
        sample_masks = torch.stack(random.choices(self.masks, k=batch_size))

        targets = inputs.detach().clone()
        inputs[sample_masks == 0] = self.mask_value

        preds = model(inputs)

        loss = reconstruction_error(preds, targets).mean()

        return dict(loss=loss)

    @torch.no_grad()
    def val_step(self, model: Module, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        inputs = batch["data"]
        targets = inputs.detach().clone()
        gt_labels = batch["label"]

        preds = model(inputs)

        errors = residual_error(preds, targets)
        mre = errors.mean()
        labels = (errors > self.anomaly_threshold).to(torch.int)
        loss = reconstruction_error(preds, targets).mean()
        tpr = true_positive_rate(labels.flatten(), gt_labels.flatten())
        return dict(loss=loss, tpr=tpr, mre=mre)

    @torch.no_grad()
    def test_step(self, model: Module, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        inputs = batch["data"]

        targets = inputs.detach().clone()

        preds = model(inputs)

        errors = residual_error(preds, targets)
        mre = errors.detach().mean()
        labels = (errors.detach() > self.anomaly_threshold).to(torch.int)

        return dict(errors=errors, mre=mre, labels=labels)
