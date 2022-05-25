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
        
        mask = torch.ones(np.prod(shape))
    
        n_mask = int(np.prod(shape) / self.n)
    
        # make sure as many as n_mask samples are masked
        mask[:n_mask] = 0

        mask = torch.reshape(mask, shape)

        # permute rows
        mask = np.random.permutation(mask)

        # permute columns
        mask = mask[:, np.random.permutation(mask.shape[1])]
        
        return torch.tensor(mask)
