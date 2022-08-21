from typing import Dict, Optional
import random

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

import numpy as np

from common.data import NUM_FEATURES
from common.metrics import compute_metrics


def get_masks(horizon: int, n: int) -> torch.Tensor:
    """Pointwise non overlapping Masking"""
    shape = (horizon, NUM_FEATURES)
    masks = []
    prod = np.prod(shape)
    n_mask = int(prod / n)
    # set are much more efficient at removing
    not_used = set(i for i in range(prod))

    while len(masks) < n:
        mask = np.ones(prod)
        # choose from the aviable indices
        idxs = np.random.choice(tuple(not_used), n_mask, replace=False)
        # set to 0
        mask[idxs] = 0
        # mark as used
        not_used = not_used.difference(idxs)
        # reshape to the input shape
        mask = torch.tensor(mask).reshape(shape)
        masks.append(mask)

    return torch.stack(masks)


def reconstruction_error(preds: Tensor, targets: Tensor) -> Tensor:
    # num_cols = targets.size(-1)
    # return torch.linalg.norm(preds - targets, ord=1, dim=-1) / num_cols
    return F.l1_loss(preds, targets, reduction="mean")


def residual_error(preds: Tensor, targets: Tensor) -> Tensor:
    return F.l1_loss(preds, targets, reduction="none").mean(-1)
    # return torch.mean(torch.abs(preds - targets), dim=-1)


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
        model.train()
        inputs = batch["data"]
        batch_size = len(inputs)
        sample_masks = torch.stack(random.choices(self.masks, k=batch_size))

        targets = inputs.detach().clone()
        inputs[sample_masks == 0] = self.mask_value

        preds = model(inputs)

        loss = reconstruction_error(preds, targets)

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
        loss = reconstruction_error(preds, targets)
        metrics = compute_metrics(labels.flatten(), gt_labels.flatten())
        metrics.update(dict(loss=loss))
        return metrics

    @torch.no_grad()
    def test_step(self, model: Module, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        model.eval()
        inputs = batch["data"]

        targets = inputs.detach().clone()

        preds = model(inputs)

        errors = residual_error(preds, targets)
        labels = (errors.detach() > self.anomaly_threshold).to(torch.int)

        return dict(errors=errors, labels=labels)
