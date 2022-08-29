import os
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import numpy as np

from common import metrics
from common.models.modutils import save_model


class CBLEngine:
    def __init__(
        self,
        model: Module,
        device: Optional[torch.device] = None,
        optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[_LRScheduler] = None,
    ):
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.metrics = ["auc"]

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        assert self.optimizer is not None, "Optimizer is None. Engine can't train'"
        self.model.train()
        self.optimizer.zero_grad()

        inputs = batch["data"].to(self.device)

        preds = self.model(inputs)

        loss = self.loss(preds)
        loss.backward()
        self.optimizer.step()

        return dict(loss=loss.item())

    def end_epoch(self, epoch: int, save_path: Optional[str]) -> str:
        log_str = ""
        if self.lr_scheduler is not None:
            lrs = ", ".join(f"{lr:.2e}" for lr in self.lr_scheduler.get_last_lr())
            log_str += f" - lr = {lrs}"
            self.lr_scheduler.step()

        if save_path is not None:
            sp = os.path.join(save_path, f"model_{epoch}.pth")
            save_model(self.model, sp)

        return log_str

    @torch.no_grad()
    def val_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        self.model.eval()
        inputs = batch["data"].to(self.device)
        gt_labels = batch["label"].to(self.device)
        targets = inputs.detach().clone()

        preds = self.model(inputs)

        errors = residual_error(preds, targets)
        loss = reconstruction_error(preds, targets, self.loss_type)

        auc = metrics.average_precision_score(errors, gt_labels)
        return dict(loss=loss.item(), auc=auc)
