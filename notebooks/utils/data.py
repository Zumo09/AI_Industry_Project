from typing import Generator
import os

import pandas as pd
from torch.utils.data import Dataset

DATASET_PATH = os.path.join("..", "data")


class Marconi100Dataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.names = []
        for name in os.listdir(DATASET_PATH):
            if "gzip" not in name:
                continue
            self.names.append(os.path.join(DATASET_PATH, name))

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> pd.DataFrame:
        path = self.names[index]
        return pd.read_parquet(path, engine="pyarrow")
    
    def __iter__(self) -> Generator[pd.DataFrame, None, None]:
        for i in range(len(self)):
            yield self[i]
