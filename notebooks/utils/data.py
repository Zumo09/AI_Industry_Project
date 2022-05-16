from typing import Optional, Tuple, List
import os

import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

DATASET_PATH = os.path.join("..", "data")


def get_dataset_paths() -> List[str]:
    paths = []  # type: List[str]
    for name in os.listdir(DATASET_PATH):
        if "gzip" not in name:
            continue
        paths.append(os.path.join(DATASET_PATH, name))
    return paths


def get_train_test_split(
    test_size: float, random_state: Optional[int] = None
) -> Tuple[List[str], List[str]]:
    paths = get_dataset_paths()
    train, test = train_test_split(
        paths, test_size=test_size, random_state=random_state
    )
    return train, test


class Marconi100Dataset(Dataset):
    def __init__(self, paths: List[str]) -> None:
        super().__init__()
        self.data = [
            pd.read_parquet(name, engine="pyarrow") for name in tqdm(paths)
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[pd.DataFrame, pd.Series]:
        df = self.data[index]
        timestamps = df["timestamp"]
        label = df["New_label"]
        label[label != 0] = 1 # labels were [0, 2]
        data = df.drop(["timestamp", "label", "New_label"], axis=1)
        return pd.DataFrame(data.values, index=timestamps, columns=data.columns), pd.Series(label.values, index=timestamps)
