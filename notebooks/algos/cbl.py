from collections import OrderedDict
from functools import reduce
import os
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import numpy as np

from common.models.modutils import save_model
from common.models.resnet import ResNetFeatures

AugmentFN = Callable[[Tensor], Tensor]


def identity() -> AugmentFN:
    return lambda x: x


def left_to_right_flipping(dim: int = 1) -> AugmentFN:
    return lambda x: x.flip(dim)


def crop_and_resize(expantion_min: float = 1.5, expantion_max: float = 2) -> AugmentFN:
    if expantion_max < expantion_min:
        raise ValueError(
            f"Expantion max must be >= Expantion min, got {expantion_min} > {expantion_max}"
        )
    if expantion_min <= 1:
        raise ValueError(f"Expantion minimum must be > 1, got {expantion_min}")

    def cr(x: Tensor) -> Tensor:
        x = x.permute(0, 2, 1)
        in_len = x.shape[2]
        expantion = torch.empty(1).uniform_(expantion_min, expantion_max).item()
        size = int(expantion * in_len)
        x = F.interpolate(x, size=size, mode="linear")
        start_col = torch.randint(0, size - in_len, (1,)).item()
        end_col = start_col + in_len
        x = x[:, :, start_col:end_col]
        x = x.permute(0, 2, 1)
        return x

    return cr


def random_apply(fn: AugmentFN, prob: float = 0.5) -> AugmentFN:
    return lambda x: fn(x) if np.random.rand() < prob else x


def pipeline(*augmentations: AugmentFN) -> AugmentFN:
    return reduce(lambda g, f: lambda x: f(g(x)), augmentations)


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = torch.tensor(temperature)

    def forward(self, feats: Tensor) -> Tensor:
        """
        feats is batches of embeddings, where indices at distance batchsize//2 are pairs
        z_i, z_j as per SimCLR paper
        """
        # Compute pairwise cosine similarity on time dimention
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-2)
        # Mean value over channels
        cos_sim = cos_sim.mean(-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        return nll.mean()


class CBLFeatsEngine:
    def __init__(
        self,
        model: ResNetFeatures,
        optimizer: Optimizer,
        temperature: float = 0.5,
        device: Optional[torch.device] = None,
        aug_1: Optional[AugmentFN] = None,
        aug_2: Optional[AugmentFN] = None,
        lr_scheduler: Optional[_LRScheduler] = None,
    ):
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.aug_1 = aug_1 or identity()
        self.aug_2 = aug_2 or identity()

        self.loss = ContrastiveLoss(temperature)

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        loss = self._get_loss(batch)
        loss.backward()
        self.optimizer.step()
        return dict(loss=loss.item())

    @torch.no_grad()
    def val_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        self.model.eval()
        loss = self._get_loss(batch)
        return dict(loss=loss.item())

    def end_epoch(self, epoch: int, save_path: Optional[str]) -> Dict[str, str]:
        log_dict = OrderedDict()
        if self.lr_scheduler is not None:
            lrs = ", ".join(f"{lr:.2e}" for lr in self.lr_scheduler.get_last_lr())
            log_dict["lr"] = lrs
            self.lr_scheduler.step()

        if save_path is not None:
            sp = os.path.join(save_path, f"backbone_{epoch}.pth")
            save_model(self.model, sp)

        return log_dict

    def _get_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        inputs = batch["data"].to(self.device)
        head_1_in = self.aug_1(inputs)
        head_2_in = self.aug_2(inputs)

        inputs = torch.concat((head_1_in, head_2_in))
        inputs = inputs.permute(0, 2, 1)
        outs = self.model(inputs)[self.model.nodes[-1]]
        return self.loss(outs)
