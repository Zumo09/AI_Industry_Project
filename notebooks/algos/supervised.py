import os
from typing import Dict, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from common import metrics
from common.models.modutils import save_model
from common.models.deeplab import DeepLabNet


class SupervisedEngine:
    def __init__(
        self,
        model: DeepLabNet,
        device: torch.device,
        optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[_LRScheduler] = None,
    ):
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.loss = torch.nn.BCEWithLogitsLoss()
        self.cmodel = metrics.default_cmodel()
        self.metrics = ["cost", "threshold"]

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        assert self.optimizer is not None, "Optimizer is None. Engine can't train"
        self.model.train()
        self.optimizer.zero_grad()

        inputs = batch["data"].to(self.device)
        labels = batch["label"].to(self.device).float()

        outs = self.model(inputs).squeeze(-1)
        loss = self.loss(outs, labels)

        loss.backward()
        self.optimizer.step()
        return dict(loss=loss.item())

    @torch.no_grad()
    def val_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        self.model.eval()
        inputs = batch["data"].to(self.device)
        labels = batch["label"].to(self.device).float()

        outs = self.model(inputs).squeeze(-1)
        loss = self.loss(outs, labels)

        thr, cost = self.cmodel.fit(outs, labels).optimize()
        return dict(loss=loss.item(), cost=cost, threshold=thr)

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
    def predict(self, inputs: Tensor) -> Tensor:
        inputs = inputs.to(self.device)
        self.model.eval()
        outs = self.model(inputs)
        return outs.squeeze(-1)