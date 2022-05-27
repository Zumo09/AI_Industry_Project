from typing import Callable, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

from tqdm import tqdm

from .summary import SummaryWriter
from .modutils import save_model

StepFunction = Callable[[Module, Tuple[Tensor, ...]], Tuple[Tensor, Tensor]]


def training_loop(
    *,
    model: Module,
    step_function: StepFunction,
    num_epochs: int,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    lr_scheduler: Optional[_LRScheduler] = None,
    writer: Optional[SummaryWriter] = None,
    save_path: Optional[str] = None,
) -> None:
    model.to(device)
    len_train = len(train_dataloader)
    len_test = len(test_dataloader)
    log_step = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_f1 = 0
        for batch in tqdm(train_dataloader, leave=False, desc=f"Train {epoch}"):
            batch = tuple(d.to(device) for d in batch)

            optimizer.zero_grad()

            f1, loss = step_function(model, batch)

            loss.backward()
            optimizer.step()

            train_loss += float(loss) / len_train
            train_f1 += float(f1) / len_train

            if writer is not None:
                writer.add_scalars("loss", {"train": float(loss)}, log_step)
                log_step += 1

        model.eval()
        test_loss = 0
        test_f1 = 0
        for batch in tqdm(test_dataloader, leave=False, desc=f"Test {epoch}"):
            batch = tuple(d.to(device) for d in batch)

            with torch.no_grad():
                f1, loss = step_function(model, batch)

            test_loss += float(loss) / len_test
            test_f1 += float(f1) / len_test

        if writer is not None:
            writer.add_scalars(
                "loss_mean", {"train": train_loss, "test": test_loss}, epoch
            )
            writer.add_scalars("f1_score", {"train": train_f1, "test": test_f1}, epoch)

        log_str = f"Epoch {epoch} - loss = {train_loss} ({test_loss}) - f1_score = {train_f1} ({test_f1})"
        if lr_scheduler is not None:
            lrs = ", ".join(f"{lr:.2e}" for lr in lr_scheduler.get_last_lr())
            log_str += f" - lr = {lrs}"
            lr_scheduler.step()

        print(log_str)

        if save_path is not None:
            save_model(model, save_path + f"model_{epoch}.pth")
