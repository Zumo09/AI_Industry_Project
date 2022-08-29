from functools import reduce
import os
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import numpy as np

from common import metrics
from common.models.modutils import save_model
from common.models.resnet import ResNet, ResNetFeatures

AugmentFN = Callable[[Tensor], Tensor]


def identity() -> AugmentFN:
    return lambda x: x


def left_to_right_flipping(dim: int = 1) -> AugmentFN:
    return lambda x: x.flip(dim)


def pipeline(*augmentations: AugmentFN) -> AugmentFN:
    return reduce(lambda g, f: lambda x: f(g(x)), augmentations)


class CBLEngine:
    def __init__(
        self,
        model: ResNetFeatures,
        device: Optional[torch.device] = None,
        optimizer: Optional[Optimizer] = None,
        augmentation: Optional[AugmentFN] = None,
        lr_scheduler: Optional[_LRScheduler] = None,
    ):
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.augmentation = augmentation or identity()

        self.loss = lambda x, y: x - y
        self.metrics = []

    def augment(self, data: Tensor) -> Tuple[Tensor, Tensor]:
        head_1 = self.augmentation(data.clone())
        head_2 = self.augmentation(data.clone())
        return head_1, head_2

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        assert self.optimizer is not None, "Optimizer is None. Engine can't train'"
        self.model.train()
        self.optimizer.zero_grad()

        inputs = batch["data"].to(self.device)

        head_1_in, head_2_in = self.augment(inputs)

        head_1_out = self.model(head_1_in)
        head_2_out = self.model(head_2_in)

        loss = self.loss(head_1_out, head_2_out)
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

        return dict(loss=0)
