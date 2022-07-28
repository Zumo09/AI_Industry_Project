from typing import Dict, Optional, Tuple, List
import os

import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

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
    def __init__(self, paths: List[str], scaling: Optional[str] = None) -> None:
        super().__init__()
        self.data = []
        for path in tqdm(paths, desc="Loading"):
            df = pd.read_parquet(path, engine="pyarrow")
            if len(df.index) == 0:
                continue
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
                    raise ValueError(
                        f"Scaling method '{scaling}' not in (standard, minmax)"
                    )
            self.data.append(
                (
                    pd.DataFrame(data.values, index=timestamps, columns=data.columns),
                    pd.Series(label.values, index=timestamps),
                )
            )

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
