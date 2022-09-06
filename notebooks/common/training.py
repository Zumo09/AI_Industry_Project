from typing import Callable, Dict, List, Optional, Protocol, Tuple
from collections import defaultdict
import os

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm


class Engine(Protocol):
    metrics: List[str]

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        raise NotImplementedError()

    def val_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        raise NotImplementedError()

    def end_epoch(self, epoch: int, save_path: Optional[str]) -> str:
        raise NotImplementedError()


class Writer(Protocol):
    def add_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, float],
        global_step: int,
    ) -> None:
        """Add scalars to the buffer"""


def training_loop(
    *,
    engine: Engine,
    num_epochs: int,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    writer: Optional[Writer] = None,
    save_path: Optional[str] = None,
) -> None:
    if save_path is not None:
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

    _train_loss = np.nan
    _smoothing = 0.8
    _log_step = 0
    for epoch in range(num_epochs):
        for batch in tqdm(train_dataloader, leave=False, desc=f"Train {epoch}"):
            rets = engine.train_step(batch)

            if np.isnan(_train_loss):
                _train_loss = rets["loss"]
            else:
                _train_loss *= _smoothing
                _train_loss += (1 - _smoothing) * rets["loss"]

            for tag, val in rets.items():
                if writer is not None:
                    writer.add_scalars(tag, {"train": val}, _log_step)

            _log_step += 1

        test_scalars = defaultdict(list)
        for batch in tqdm(test_dataloader, leave=False, desc=f"Test {epoch}"):
            rets = engine.val_step(batch)
            for tag, val in rets.items():
                test_scalars[tag].append(val)

        if writer is not None:
            for tag, val in test_scalars.items():
                writer.add_scalars(tag, {"test": np.mean(val)}, _log_step)

        test_loss = np.mean(test_scalars["loss"])
        log_str = f"Epoch {epoch} - train_loss = {_train_loss:.3f} - test_loss = {test_loss:.3f}"

        for key in engine.metrics:
            log_str += f" - test_{key}={np.mean(test_scalars[key]):.3f}"
        log_str += engine.end_epoch(epoch, save_path)

        print(log_str)


@torch.no_grad()
def get_predictions(
    prediction_fn: Callable[[Tensor], Tensor], test_loader: DataLoader
) -> Tuple[Tensor, Tensor]:
    scores_ = []
    labels_ = []

    for batch in tqdm(test_loader):
        errors = prediction_fn(batch["data"])
        scores_.append(errors.cpu())
        labels_.append(batch["label"])

    scores = torch.concat(scores_)
    labels = torch.concat(labels_).float()

    return scores, labels
