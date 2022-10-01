from collections import OrderedDict
import os
from typing import Dict, Optional
import random

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import numpy as np

from common import metrics
from common.data import NUM_FEATURES
from common.models.modutils import save_model


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
        raise ValueError(f"{loss_type} not in ['mse', 'l1']")


def residual_error(preds: Tensor, targets: Tensor) -> Tensor:
    return F.l1_loss(preds, targets, reduction="none").mean(-1)
    # return torch.mean(torch.abs(preds - targets), dim=-1)


class DeepFIBEngine:
    def __init__(
        self,
        model: Module,
        device: Optional[torch.device] = None,
        masks: Optional[Tensor] = None,
        mask_value: Optional[float] = None,
        loss_type: str = "l1",
        optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[_LRScheduler] = None,
    ):
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)

        self.masks = masks
        self.mask_value = mask_value
        self.loss_type = loss_type
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.cmodel = metrics.default_cmodel()

        self._scores = []
        self._labels = []

    def _get_preds(self, inputs: Tensor) -> Tensor:
        out = self.model(inputs)
        out = torch.sigmoid(out)
        return out

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        assert self.optimizer is not None, "Optimizer is None. Engine can't train"
        assert self.masks is not None, "Masks are None. Engine can't train"
        self.model.train()
        self.optimizer.zero_grad()

        inputs = batch["data"].to(self.device)
        batch_size = len(inputs)
        sample_masks = torch.stack(random.choices(self.masks, k=batch_size))

        targets = inputs.detach().clone()

        if self.mask_value is not None:
            inputs[sample_masks == 0] = self.mask_value
        else:
            noise = torch.empty_like(inputs).normal_()
            msk = sample_masks == 0
            inputs[msk] = noise[msk]

        preds = self._get_preds(inputs)

        loss = reconstruction_error(preds, targets, self.loss_type)
        loss.backward()
        self.optimizer.step()

        return dict(loss=loss.item())

    @torch.no_grad()
    def val_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        self.model.eval()
        inputs = batch["data"].to(self.device)
        labels = batch["label"].to(self.device)
        targets = inputs.detach().clone()

        preds = self._get_preds(inputs)

        errors = residual_error(preds, targets)
        loss = reconstruction_error(preds, targets, self.loss_type)

        self._scores.append(errors.cpu().detach())
        self._labels.append(labels.cpu().detach())

        return dict(loss=loss.item())

    def end_epoch(self, epoch: int, save_path: Optional[str]) -> Dict[str, str]:
        log_dict = OrderedDict()
        scores = torch.concat(self._scores)
        labels = torch.concat(self._labels)

        thr, cost = self.cmodel.fit(scores, labels).optimize()

        self._scores.clear()
        self._labels.clear()

        log_dict["cost"] = f"{cost:.3f}"
        log_dict["threshold"] = f"{thr:.3f}"

        if self.lr_scheduler is not None:
            lrs = ", ".join(f"{lr:.2e}" for lr in self.lr_scheduler.get_last_lr())
            log_dict["lr"] = lrs
            self.lr_scheduler.step()

        if save_path is not None:
            sp = os.path.join(save_path, f"model_{epoch}.pth")
            save_model(self.model, sp)

        return log_dict

    @torch.no_grad()
    def predict(self, batch: Dict[str, Tensor]) -> Tensor:
        inputs = batch["data"]
        inputs = inputs.to(self.device)
        targets = inputs.detach().clone()

        self.model.eval()
        preds = self._get_preds(inputs)

        errors = residual_error(preds, targets)

        return errors
