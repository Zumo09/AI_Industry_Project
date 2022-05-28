from typing import Dict, Tuple

import torch
from torch import Tensor

from torchmetrics.functional import f1_score

MASK = -1


def reconstruction_error(preds: Tensor, targets: Tensor) -> Tensor:
    return torch.linalg.norm(preds - targets, ord=1, dim=-1)


def step_function(
    model: torch.nn.Module, batch: Dict[str, Tensor]
) -> Tuple[Tensor, Tensor]:
    inputs = batch["data"]
    masks = batch["mask"]
    labels = batch["label"]
    targets = inputs.detach().clone()
    inputs[masks == 0] = MASK

    preds = model(inputs)

    errors = reconstruction_error(preds, targets)
    f1 = f1_score(errors.detach(), labels.to(torch.int))
    loss = errors.mean()

    return f1, loss
