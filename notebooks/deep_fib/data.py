from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd
from utils.data import Marconi100Dataset


class DeepFIBDataset(Marconi100Dataset, ABC):
    def __init__(self, num_data_per_sample: int) -> None:
        super().__init__()
        self.n = num_data_per_sample

    def __len__(self) -> int:
        return super().__len__() * self.n

    def __getitem__(self, index: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = super().__getitem__(index // self.n)
        masked = self._mask(df)

        return df, masked

    @abstractmethod
    def _mask(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()


class PDataset(DeepFIBDataset):
    def _mask(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pointwise Masking"""
        return super()._mask(df)


class SDataset(DeepFIBDataset):
    def _mask(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sequence Masking"""
        return super()._mask(df)
