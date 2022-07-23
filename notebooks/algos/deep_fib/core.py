from typing import Dict, Optional
import random

import torch
from torch import Tensor
from torch.nn import Module

from utils.metrics import compute_metrics


def reconstruction_error(preds: Tensor, targets: Tensor) -> Tensor:
    num_cols = targets.size(-1)
    return torch.linalg.norm(preds - targets, ord=1, dim=-1) / num_cols


def residual_error(preds: Tensor, targets: Tensor) -> Tensor:
    return torch.mean(torch.abs(preds - targets), dim=-1)


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
        metrics = compute_metrics(labels.flatten(), gt_labels.flatten())
        metrics.update(dict(loss=loss, mre=mre))
        return metrics

    @torch.no_grad()
    def test_step(self, model: Module, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        inputs = batch["data"]

        targets = inputs.detach().clone()

        preds = model(inputs)

        errors = residual_error(preds, targets)
        mre = errors.detach().mean()
        labels = (errors.detach() > self.anomaly_threshold).to(torch.int)

        return dict(errors=errors, mre=mre, labels=labels)
