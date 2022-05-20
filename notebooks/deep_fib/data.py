from abc import ABC, abstractmethod
from typing import List, Tuple
import random

import torch
import pandas as pd
import numpy as np
from utils.data import Marconi100Dataset


class DeepFIBDataset(Marconi100Dataset, ABC):
    def __init__(self, paths: List[str], num_sample_per_data: int) -> None:
        super().__init__(paths)
        self.n = num_sample_per_data
        self.mask_tag = -1

    def __len__(self) -> int:
        return super().__len__() * self.n

    def __getitem__(self, index: int) -> Tuple[pd.DataFrame, torch.Tensor, pd.Series]:
        original, label = super().__getitem__(index // self.n)

        masked = self._mask(original, idx=index%self.n)

        return original, masked, label

    @abstractmethod
    def _mask(self, original: pd.DataFrame, idx: int) -> torch.Tensor:
        raise NotImplementedError()


class PointMaskDataset(DeepFIBDataset):
    def _mask(self, original: pd.DataFrame, idx: int) -> torch.Tensor:
        """Pointwise Masking"""
        n_mask = int(np.prod(original.shape) / self.n)

        # TODO: make it works without replacements
        # get column indices for the features to mask
        col_idxs = np.random.choice(range(original.shape[1]), n_mask, replace=True)
        # get row indices for the samples to mask
        row_idxs = np.random.choice(range(original.shape[0]), n_mask, replace=True)

        masked = torch.tensor(original.to_numpy())
        masked[row_idxs, col_idxs] = self.mask_tag
        return masked



class SequenceMaskDataset(DeepFIBDataset):
    def _mask(self, original: pd.DataFrame, idx: int) -> torch.Tensor:
        """Sequence Masking"""
        # n_mask = int(np.prod(original.shape) / self.n) ?

        # get row indices for the samples to mask
        # TODO: get start and stop of the sequence (based on idx??)
        start = 0
        stop = 10

        row_idxs = np.arange(start, stop)

        masked = torch.tensor(original.data)
        masked[row_idxs, :] = self.mask_tag
        return masked
