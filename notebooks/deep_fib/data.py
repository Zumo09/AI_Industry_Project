from typing import Tuple

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from utils.data import Marconi100Dataset


class DeepFIBDataset(Dataset):
    def __init__(self, marconi_dataset: Marconi100Dataset, num_sample_per_data: int) -> None:
        self._dataset = marconi_dataset
        self.n = num_sample_per_data
        self.mask_tag = -1

    def __len__(self) -> int:
        return len(self._dataset) * self.n

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Tuple[pd.DataFrame, pd.Series]]:
        original, label = self._dataset[index // self.n]

        mask = self._mask(original.shape)
        original_tensor = torch.tensor(original.to_numpy())

        return original_tensor, mask, (original, label)

    def _mask(self, shape: Tuple[int, int]) -> torch.Tensor:
        """Pointwise Masking"""
        n_mask = int(np.prod(shape) / self.n)

        # TODO: make it works without replacements
        # get column indices for the features to mask
        col_idxs = np.random.choice(range(shape[1]), n_mask, replace=True)
        # get row indices for the samples to mask
        row_idxs = np.random.choice(range(shape[0]), n_mask, replace=True)

        mask = torch.ones(shape)
        mask[row_idxs, col_idxs] = 0
        return mask
