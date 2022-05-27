from typing import Tuple

import torch
from torch import Tensor

from torchmetrics.functional import f1_score

MASK = -1

def reconstruction_error(preds: Tensor, targets: Tensor) -> Tensor:
    return torch.norm(preds - targets, p=1, dim=-1)  # type: ignore

def step_function(model: torch.nn.Module, batch: Tuple[Tensor, ...]) -> Tuple[Tensor, Tensor]:
    inputs: Tensor = batch[0]
    masks: Tensor = batch[1]
    labels: Tensor = batch[2]
    targets = inputs.detach().clone()
    inputs[masks == 0] = MASK
    preds = model(inputs)

    errors = reconstruction_error(preds, targets)
    f1 = f1_score(errors.detach(), labels)
    loss = errors.mean()
    return f1,loss