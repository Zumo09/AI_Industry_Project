from typing import Callable, Dict, Optional, Protocol
from collections import defaultdict
import os

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

import numpy as np
from tqdm import tqdm

from .modutils import save_model


class Engine(Protocol):
    def train_step(self, model: Module, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        raise NotImplementedError()

    def test_step(self, model: Module, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        raise NotImplementedError()

    def val_step(self, model: Module, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        raise NotImplementedError()


class SummaryWriter(Protocol):
    def add_scalars(
        self, main_tag: str, tag_scalar_dict: Dict[str, float], global_step: int,
    ) -> None:
        """Add scalars to the buffer"""


def training_loop(
    *,
    model: Module,
    engine: Engine,
    num_epochs: int,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    lr_scheduler: Optional[_LRScheduler] = None,
    writer: Optional[SummaryWriter] = None,
    save_path: Optional[str] = None,
) -> None:
    if save_path is not None:
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

    log_step = 0
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_scalars = defaultdict(list)
        test_scalars = defaultdict(list)
        for batch in tqdm(train_dataloader, leave=False, desc=f"Train {epoch}"):
            batch = {
                k: d.to(device) for k, d in batch.items()
            }  # type: Dict[str, Tensor]

            optimizer.zero_grad()

            rets = engine.train_step(model, batch)

            rets["loss"].backward()
            optimizer.step()

            for tag, val in rets.items():
                train_scalars[tag].append(float(val))
                if writer is not None:
                    writer.add_scalars(tag, {"train": float(val)}, log_step)

            log_step += 1

        model.eval()
        for batch in tqdm(test_dataloader, leave=False, desc=f"Test {epoch}"):
            batch = {
                k: d.to(device) for k, d in batch.items()
            }  # type: Dict[str, Tensor]

            with torch.no_grad():
                rets = engine.val_step(model, batch)

            for tag, val in rets.items():
                test_scalars[tag].append(float(val))

        if writer is not None:
            for tag, val in train_scalars.items():
                writer.add_scalars(tag, {"train_mean": np.mean(val)}, log_step)
            for tag, val in test_scalars.items():
                writer.add_scalars(tag, {"test_mean": np.mean(val)}, log_step)

        train_loss = np.mean(train_scalars["loss"])
        test_loss = np.mean(test_scalars["loss"])
        log_str = f"Epoch {epoch} - train_loss = {train_loss:.3f} - test_loss = {test_loss:.3f}"
        if lr_scheduler is not None:
            lrs = ", ".join(f"{lr:.2e}" for lr in lr_scheduler.get_last_lr())
            log_str += f" - lr = {lrs}"
            lr_scheduler.step()

        print(log_str)

        if save_path is not None:
            save_model(model, os.path.join(save_path, f"model_{epoch}.pth"))
