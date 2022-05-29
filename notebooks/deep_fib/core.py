from typing import Dict

import torch
from torch import Tensor

from torchmetrics.functional import f1_score

MASK = -1


def reconstruction_error(preds: Tensor, targets: Tensor) -> Tensor:
    num_cols = targets.size(-1)
    return torch.linalg.norm(preds - targets, ord=1, dim=-1) / num_cols


class DeepFIBEngine:
    def __init__(self, anomaly_threshold: float):
        self.anomaly_threshold = anomaly_threshold

    def train_step(
        self, model: torch.nn.Module, batch: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        inputs = batch["data"]
        masks = batch["mask"]
        targets = inputs.detach().clone()
        inputs[masks == 0] = MASK

        preds = model(inputs)

        errors = reconstruction_error(preds, targets)
        loss = errors.mean()

        return dict(loss=loss)

    @torch.no_grad()
    def test_step(
        self, model: torch.nn.Module, batch: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        inputs = batch["data"]
        labels = batch["label"]

        preds = self.detect_anomalies(model, inputs)
        loss = preds["errors"].mean()

        f1 = f1_score(preds["labels"], labels, threshold=self.anomaly_threshold)

        return dict(loss=loss, f1=f1)

    @torch.no_grad()
    def detect_anomalies(
        self, model: torch.nn.Module, inputs: Tensor
    ) -> Dict[str, Tensor]:
        targets = inputs.detach().clone()

        preds = model(inputs)

        errors = reconstruction_error(preds, targets)
        labels = (errors.detach() > self.anomaly_threshold).to(torch.int)

        return dict(errors=errors, labels=labels)
