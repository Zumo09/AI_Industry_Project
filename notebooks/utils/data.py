from typing import Optional, Tuple, List
import os

import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

DATASET_PATH = os.path.join(os.getcwd(), "data")
NUM_FEATURES = 460


def get_dataset_paths(dataset_base_path: str) -> List[str]:
    paths = []  # type: List[str]
    for name in os.listdir(dataset_base_path):
        if "gzip" not in name:
            continue
        paths.append(os.path.join(dataset_base_path, name))
    return paths


class Marconi100Dataset(Dataset):
    def __init__(self, paths: List[str], normalize: Optional[str] = None) -> None:
        super().__init__()
        self.data = [
            self._load(path, normalize) for path in tqdm(paths, desc="Loading")
        ]

    @staticmethod
    def _load(
        path: str, normalize: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        df = pd.read_parquet(path, engine="pyarrow")
        timestamps = df["timestamp"]
        label = df["New_label"].astype(int)
        label = label.replace(2, 1)  # labels were [0, 2], we want [0, 1]
        data = df.drop(["timestamp", "label", "New_label"], axis=1)
        if normalize is not None:
            if normalize == "normal":
                data = (data - data.mean()) / (data.std() + 1e-5)
            elif normalize == "minmax":
                # TODO
                raise NotImplementedError()
            else:
                raise ValueError(f"Normalization '{normalize}' not in (normal, minmax)")
        return (
            pd.DataFrame(data.values, index=timestamps, columns=data.columns),
            pd.Series(label.values, index=timestamps),
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[pd.DataFrame, pd.Series]:
        return self.data[index]
