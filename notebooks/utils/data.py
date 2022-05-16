import os

import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

DATASET_PATH = os.path.join("..", "data")


class Marconi100Dataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = []
        for name in tqdm(os.listdir(DATASET_PATH)):
            if "gzip" not in name:
                continue
            path = os.path.join(DATASET_PATH, name)
            df = pd.read_parquet(path, engine="pyarrow")
            self.data.append(df)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> pd.DataFrame:
        return self.data[index]
