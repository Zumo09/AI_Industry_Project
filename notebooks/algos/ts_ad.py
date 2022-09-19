import os
from typing import Callable, Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import numpy as np
from common.data import UnfoldedDataset, Dataset

from common.models.modutils import save_model
from common.models.deeplab import DeepLabNet

from common import metrics


class TSADEDataset(Dataset):
    def __init__(self, dataset: UnfoldedDataset, input_len: int) -> None:
        self.dataset = dataset
        self.input_len = input_len

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        dl = self.dataset[index]
        d, l = dl["data"], dl["label"]
        return {
            "data": d[: self.input_len],
            "target": d[self.input_len :],
            "label": l[self.input_len :],
        }


def rmse_errors():
    loss = torch.nn.MSELoss(reduction="none")
    return lambda y, x: torch.sqrt(loss(y, x).mean(-1))


def rmse_loss():
    loss = torch.nn.MSELoss()
    return lambda y, x: torch.sqrt(loss(y, x))


class TSADEngine:
    def __init__(
        self,
        model: DeepLabNet,
        optimizer: Optional[Optimizer] = None,
        device: Optional[torch.device] = None,
        lr_scheduler: Optional[_LRScheduler] = None,
    ):
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # self.loss = torch.nn.MSELoss()
        self.loss = rmse_loss()
        self.errors = rmse_errors()

        self.cmodel = metrics.default_cmodel()

        self._scores = []
        self._labels = []

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        assert self.optimizer is not None, "Optimizer is None. Engine can't train"
        self.model.train()
        self.optimizer.zero_grad()

        inputs = batch["data"].to(self.device)
        target = batch["target"].to(self.device)

        outs = self.model(inputs)

        loss = self.loss(outs, target)
        loss.backward()
        self.optimizer.step()
        return dict(loss=loss.item())

    @torch.no_grad()
    def val_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        self.model.eval()
        inputs = batch["data"].to(self.device)
        target = batch["target"].to(self.device)

        outs = self.model(inputs)
        loss = self.loss(outs, target)

        errors = self.errors(outs, target)

        self._scores.append(errors.cpu().detach())
        self._labels.append(batch["label"].cpu().detach())

        return dict(loss=loss.item())

    def end_epoch(self, epoch: int, save_path: Optional[str]) -> str:
        log_str = ""

        scores = torch.concat(self._scores)
        labels = torch.concat(self._labels)

        cost, thr = self.cmodel.fit(scores, labels).optimize()

        self._scores.clear()
        self._labels.clear()

        log_str += f" - cost = {cost:.3f} - threshold = {thr:.3f}"

        if self.lr_scheduler is not None:
            lrs = ", ".join(f"{lr:.2e}" for lr in self.lr_scheduler.get_last_lr())
            log_str += f" - lr = {lrs}"
            self.lr_scheduler.step()

        if save_path is not None:
            sp = os.path.join(save_path, f"model_{epoch}.pth")
            save_model(self.model, sp)

        return log_str

    @torch.no_grad()
    def predict(self, batch: Dict[str, Tensor]) -> Tensor:
        self.model.eval()
        inputs = batch["data"].to(self.device)
        target = batch["target"].to(self.device)

        outs = self.model(inputs)
        return self.errors(outs, target)
