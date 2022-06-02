from typing import Dict, Optional
import random

import torch
from torch import Tensor
from torch.nn import Module
from torchmetrics.functional import f1_score


MASK = -1


def reconstruction_error(preds: Tensor, targets: Tensor) -> Tensor:
    num_cols = targets.size(-1)
    return torch.linalg.norm(preds - targets, ord=1, dim=-1) / num_cols


def residual_error(preds: Tensor, targets: Tensor) -> Tensor:
    return torch.sum(
        torch.abs(preds - targets), dim=-1
    )  # Do not reduce over time dimension, only columns dimension
    # return torch.sum(torch.abs(preds - targets))


class DeepFIBEngine:
    def __init__(self, anomaly_threshold: float, masks: Optional[Tensor] = None):
        self.anomaly_threshold = anomaly_threshold
        self.masks = masks

    def train_step(self, model: Module, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        assert self.masks is not None, "Masks not initializes. Engine can't train'"
        inputs = batch["data"]
        batch_size = len(inputs)
        sample_masks = torch.stack(random.choices(self.masks, k=batch_size))

        targets = inputs.detach().clone()
        print(inputs.sum())
        inputs[sample_masks == 0] = MASK
        print(inputs.sum())

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
        f1 = f1_score(labels, gt_labels, threshold=self.anomaly_threshold)

        return dict(loss=loss, f1=f1, mre=mre)

    @torch.no_grad()
    def test_step(self, model: Module, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        inputs = batch["data"]

        targets = inputs.detach().clone()

        preds = model(inputs)

        errors = residual_error(preds, targets)
        mre = errors.detach().mean()
        labels = (errors.detach() > self.anomaly_threshold).to(torch.int)

        return dict(errors=errors, mre=mre, labels=labels)
