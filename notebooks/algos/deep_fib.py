from typing import Dict, Optional
import random

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

import numpy as np

from common.data import NUM_FEATURES
from common import metrics


def get_masks(horizon: int, n_masks: int) -> torch.Tensor:
    """Pointwise non overlapping Masking"""
    shape = (horizon, NUM_FEATURES)
    masks = []
    prod = np.prod(shape)
    n_mask_point = int(prod / n_masks)
    # set are much more efficient at removing
    not_used = set(i for i in range(prod))

    while len(masks) < n_masks:
        mask = np.ones(prod)
        # choose from the aviable indices
        idxs = np.random.choice(tuple(not_used), n_mask_point, replace=False)
        # set to 0
        mask[idxs] = 0
        # mark as used
        not_used = not_used.difference(idxs)
        # reshape to the input shape
        mask = torch.tensor(mask).reshape(shape)
        masks.append(mask)

    return torch.stack(masks)


def reconstruction_error(
    preds: Tensor, targets: Tensor, loss_type: str = "l1"
) -> Tensor:
    if loss_type == "mse":
        return F.mse_loss(preds, targets, reduction="mean")
    elif loss_type == "l1":
        return F.l1_loss(preds, targets, reduction="mean")
    else:
        raise ValueError(f"{loss_type} not in [mse, l1]")


def residual_error(preds: Tensor, targets: Tensor) -> Tensor:
    return F.l1_loss(preds, targets, reduction="none").mean(-1)
    # return torch.mean(torch.abs(preds - targets), dim=-1)


class DeepFIBEngine:
    def __init__(
        self,
        anomaly_threshold: float,
        masks: Optional[Tensor] = None,
        mask_value: int = -1,
        loss_type: str = "l1",
    ):
        self.anomaly_threshold = anomaly_threshold
        self.masks = masks
        self.mask_value = mask_value
        self.loss_type = loss_type

    def train_step(self, model: Module, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        assert self.masks is not None, "Masks not initializes. Engine can't train'"
        model.train()
        inputs = batch["data"]
        batch_size = len(inputs)
        sample_masks = torch.stack(random.choices(self.masks, k=batch_size))

        targets = inputs.detach().clone()
        inputs[sample_masks == 0] = self.mask_value

        preds = model(inputs)

        loss = reconstruction_error(preds, targets, self.loss_type)

        return dict(loss=loss)

    @torch.no_grad()
    def val_step(self, model: Module, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        model.eval()
        inputs = batch["data"]
        targets = inputs.detach().clone()
        gt_labels = batch["label"]

        preds = model(inputs)

        errors = residual_error(preds, targets)
        # mre = errors.mean() mre == loss!!!
        labels = (errors > self.anomaly_threshold).to(torch.int)
        loss = reconstruction_error(preds, targets, self.loss_type)
        met = metrics.compute_metrics(labels.flatten(), gt_labels.flatten())
        met.update(dict(loss=loss))
        return met

    @torch.no_grad()
    def test_step(self, model: Module, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        model.eval()
        inputs = batch["data"]

        targets = inputs.detach().clone()

        preds = model(inputs)

        errors = residual_error(preds, targets)
        labels = (errors.detach() > self.anomaly_threshold).to(torch.int)

        return dict(errors=errors, labels=labels)
