from typing import Dict, Optional, Tuple, List
import os

import pandas as pd
import time
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from functools import partial
from multiprocessing import Pool


def get_dataset_paths(dataset_base_path: str) -> List[str]:
    paths = []  # type: List[str]
    for name in os.listdir(dataset_base_path):
        if "gzip" not in name:
            continue
        paths.append(os.path.join(dataset_base_path, name))
    return paths


def read(
    path: str, scaling: Optional[str] = None
) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    df = pd.read_parquet(path, engine="pyarrow")
    if len(df.index) == 0:
        return None
    timestamps = df["timestamp"]
    label = df["New_label"].astype(int)
    label = label.replace(2, 1)  # labels were [0, 2], we want [0, 1]
    data = df.drop(["timestamp", "label", "New_label"], axis=1)
    if scaling is not None:
        if scaling == "standard":
            data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-5)
            # data = (data - data.mean()) / (data.std() + 1e-5)
        elif scaling == "minmax":
            cols = data.columns
            data[cols] = MinMaxScaler().fit_transform(data[cols])
        else:
            raise ValueError(f"Scaling method '{scaling}' not in (standard, minmax)")
    return (
        pd.DataFrame(data.values, index=timestamps, columns=data.columns),
        pd.Series(label.values, index=timestamps),
    )


class Marconi100DatasetMulti(Dataset):
    def __init__(self, paths: List[str], scaling: Optional[str] = None) -> None:
        super().__init__()

        self.data = []
        with Pool() as pool:
            data = pool.imap_unordered(partial(read, scaling=scaling), paths)

            for d in tqdm(data, desc="Loading Multiprocessing", total=len(paths)):
                if d is not None:
                    self.data.append(d)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[pd.DataFrame, pd.Series]:
        return self.data[index]


class Marconi100Dataset(Dataset):
    def __init__(self, paths: List[str], scaling: Optional[str] = None) -> None:
        super().__init__()
        self.data = []
        for path in tqdm(paths, desc="Loading Single Process", total=len(paths)):
            d = read(path, scaling)
            if d is not None:
                self.data.append(d)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[pd.DataFrame, pd.Series]:
        return self.data[index]


if __name__ == "__main__":
    paths = get_dataset_paths("data")

    t1 = time.time()
    d1 = Marconi100Dataset(paths, "minmax")
    print(time.time() - t1)
    del d1

    t2 = time.time()
    d2 = Marconi100DatasetMulti(paths, "minmax")
    print(time.time() - t2)
