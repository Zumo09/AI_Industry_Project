from abc import ABC, abstractmethod
from typing import List, Tuple

import pandas as pd
from utils.data import Marconi100Dataset


class DeepFIBDataset(Marconi100Dataset, ABC):
    def __init__(self, paths: List[str], num_data_per_sample: int) -> None:
        super().__init__(paths)
        self.n = num_data_per_sample

    def __len__(self) -> int:
        return super().__len__() * self.n

    def __getitem__(self, index: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        original, label = super().__getitem__(index // self.n)
        masked = self._mask(original)

        return original, masked, label

    @abstractmethod
    def _mask(self, original: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()


class PDataset(DeepFIBDataset):
    def _mask(self, original: pd.DataFrame) -> pd.DataFrame:
        """Pointwise Masking"""
        return super()._mask(original)


class SDataset(DeepFIBDataset):
    def _mask(self, original: pd.DataFrame) -> pd.DataFrame:
        """Sequence Masking"""
        return super()._mask(original)
