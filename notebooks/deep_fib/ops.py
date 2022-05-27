from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from torchmetrics.functional import f1_score

from . import sci_net

MASK = -1

def reconstruction_error(preds: Tensor, targets: Tensor) -> Tensor:
    return torch.norm(preds, p=1, dim=-1)  # type: ignore

def training_loop(
    model: sci_net.SCINet,
    num_epochs: int,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    error_threshold: float = 0.5,
    writer: Optional[SummaryWriter] = None,
) -> None:
    model.to(device)
    
    log_step = 0
    for epoch in range(1, num_epochs+1):
        model.train()
        for inputs, masks, _ in train_dataloader:
            inputs: Tensor = inputs.to(device)
            masks: Tensor = masks.to(device)
            targets = inputs.detach().clone()

            optimizer.zero_grad()

            inputs[masks == 0] = MASK
            preds = model(inputs)

            errors = reconstruction_error(preds, targets)

            loss = errors.mean()
            loss.backward()
            optimizer.step()

            if writer is not None:
                writer.add_scalars("error", {"train": float(loss)}, log_step)
                log_step += 1
        
        model.eval()
        for inputs, masks, labels in test_dataloader:
            inputs: Tensor = inputs.to(device)
            masks: Tensor = masks.to(device)
            labels: Tensor = labels.to(device)
            targets = inputs.detach().clone()

            inputs[masks == 0] = MASK

            with torch.no_grad():
                preds = model(inputs)

            errors = reconstruction_error(preds, targets)
            loss = errors.mean()

            if writer is not None:
                writer.add_scalars("error", {"train": float(loss)}, log_step)
                log_step += 1

