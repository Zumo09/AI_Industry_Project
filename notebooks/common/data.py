from enum import Enum
from typing import Dict, Optional, Tuple, List
import numpy
from pandas._typing import Dtype
import os
from functools import partial
from multiprocessing import Pool


import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

DATASET_PATH = os.path.join(os.getcwd(), "data")
NUM_FEATURES = 460


class Scaling(Enum):
    STANDARD = "Standard"
    MINMAX = "MinMax"
    NONE = "None"


def get_dataset_paths(dataset_base_path: str) -> List[str]:
    paths = []  # type: List[str]
    for name in os.listdir(dataset_base_path):
        if "gzip" not in name:
            continue
        paths.append(os.path.join(dataset_base_path, name))
    return paths


def read(
    path: str, scaling: Scaling = Scaling.NONE, dtype: Dtype = "float32"
) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    df = pd.read_parquet(path, engine="pyarrow")
    if len(df.index) == 0:
        return None
    timestamps = df["timestamp"]
    label = df["New_label"].astype(int)
    label = label.replace(2, 1)  # labels were [0, 2], we want [0, 1]
    data = df.drop(["timestamp", "label", "New_label"], axis=1)
    if scaling == Scaling.STANDARD:
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-5)
        # data = (data - data.mean()) / (data.std() + 1e-5)
    elif scaling == Scaling.MINMAX:
        cols = data.columns
        data[cols] = MinMaxScaler().fit_transform(data[cols])
    return (
        pd.DataFrame(data.values, index=timestamps, columns=data.columns, dtype=dtype),
        pd.Series(label.values, index=timestamps),
    )


class Marconi100Dataset(Dataset):
    def __init__(self, paths: List[str], scaling: Scaling = Scaling.NONE) -> None:
        super().__init__()

        self.data = []
        with Pool() as pool:
            data = pool.imap_unordered(partial(read, scaling=scaling), paths)

            for d in tqdm(data, desc="Loading", total=len(paths)):
                if d is not None:
                    self.data.append(d)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[pd.DataFrame, pd.Series]:
        return self.data[index]


def unfolded_indexes(
    dataset: Marconi100Dataset, horizon: int, stride: int
) -> List[Tuple[int, Tuple[int, int]]]:
    """Return a list of tuples containing:
    - the index of the dataframe in the marconi dataset
    - the starting and ending indexes of the selected window

    The dataframes with length less than horizon will be discarded
        TODO: padding? (may be a problem maybe for what regard the padding method...
        maybe a constant paddig might be raised as an anomaly)
    """
    indexes = []
    for idx in range(len(dataset)):
        df, _ = dataset[idx]
        length = len(df)
        if length < horizon:
            continue
        start = 0
        end = start + horizon
        while end < length:
            indexes.append((idx, (start, end)))
            start += stride
            end = start + horizon
        # keep last
        indexes.append((idx, (length - 1 - horizon, length - 1)))

    return indexes


class UnfoldedDataset(Dataset):
    def __init__(self, dataset: Marconi100Dataset, horizon: int, stride: int) -> None:
        self.dataset = dataset
        self.indexes = unfolded_indexes(dataset, horizon, stride)

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        df_idx, (start, end) = self.indexes[index]
        data, label = self.dataset[df_idx]
        data_t = torch.tensor(data.to_numpy())[start:end].float()
        label_t = torch.tensor(label.to_numpy())[start:end].int()

        return {"data": data_t, "label": label_t}


class StocDataset(Dataset):
    def __init__(self, dataset: Marconi100Dataset) -> None:
        self.dataset = dataset
        self.lens = [len(dataset[i]) for i in range(len(dataset))]
        self._len = sum(self.lens)
        self.cum_len = numpy.cumsum(self.lens)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        idx = 0  # max_i s.t. self.cumlen[i] < index
        df = self.dataset[idx]
        i = index - self.cum_len[idx]

        return df[i]


"""
Marconi:

1: 0....130000, 2: 0.....1231556, ....., 3: 0......123123

STOC:

1: 0....130000, 2: 130001.....130001 + 1231556, 130001 + 1231556 ..... 130001 + 1231556 + 123123





[2, 3, 6, 7, 1]

[2, 5, 11, 18, 19]

"""
