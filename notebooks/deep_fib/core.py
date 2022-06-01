from typing import Dict

import torch
from torch import Tensor
import random

from torchmetrics.functional import f1_score

from .data import get_masks
from utils.data import NUM_FEATURES

MASK = -1


def reconstruction_error(preds: Tensor, targets: Tensor) -> Tensor:
    num_cols = targets.size(-1)
    return torch.linalg.norm(preds - targets, ord=1, dim=-1) / num_cols

def residual_error(preds: Tensor, targets: Tensor) -> Tensor:
    return torch.sum(torch.abs(preds - targets), dim=-1) # Do not reduce over time dimension, only columns dimension
    # return torch.sum(torch.abs(preds - targets))

class DeepFIBEngine:
    def __init__(self, anomaly_threshold: float, masks: Tensor):
        self.anomaly_threshold = anomaly_threshold
        self.masks = masks

    def train_step(
        self, model: torch.nn.Module, batch: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        inputs = batch["data"]
        # Sample a batch of masks with replacement
        # 32 should be replaced with the size of the batch (32)
        sample_masks = random.choices(self.masks, k = 32)
        
        masked_inputs = []
        for mask, elem in zip(sample_masks, batch["data"]):
            masked_input = elem
            masked_input[mask == 0] = MASK
            masked_inputs.append(masked_input)
            
        
        masked_inputs = np.array(masked_inputs)
        targets = inputs.detach().clone()

        preds = model(masked_inputs)

        errors = reconstruction_error(preds, targets)
        loss = errors.mean()

        return dict(loss=loss)

    @torch.no_grad()
    def validation_step(
        self, model: torch.nn.Module, batch: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        inputs = batch["data"]
        targets = inputs.detach().clone()
        labels = batch["label"]
        
        preds = self.detect_anomalies(model, inputs)
        mean_residual_error = preds["errors"].mean()
        reconstruction_errors = reconstruction_error(preds, targets)
        reconstruction_loss = reconstruction_errors.mean()
        
        f1 = f1_score(preds["labels"], labels, threshold=self.anomaly_threshold)

        return dict(reconstruction_loss=reconstruction_loss, f1=f1, mre=mean_residual_error)
    
    @torch.no_grad()
    def test_step(
        self, model: torch.nn.Module, batch: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        inputs = batch["data"]
        targets = inputs.detach().clone()
        
        preds = self.detect_anomalies(model, inputs)
        mean_residual_error = preds["errors"].mean()
        
        return dict(mre=mean_residual_error, preds=preds)
    
    @torch.no_grad()
    def detect_anomalies(
        self, model: torch.nn.Module, inputs: Tensor
    ) -> Dict[str, Tensor]:
        targets = inputs.detach().clone()

        preds = model(inputs)

        errors = residual_error(preds, targets)
        labels = (errors.detach() > self.anomaly_threshold).to(torch.int)

        return dict(errors=errors, labels=labels)
