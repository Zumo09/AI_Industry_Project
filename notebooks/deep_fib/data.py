from typing import Dict, List, Tuple

import torch
import numpy as np
from torch.utils.data import Dataset
from utils.data import Marconi100Dataset, NUM_FEATURES


def get_masks(horizon: int, n: int) -> torch.Tensor:
    """Pointwise non overlapping Masking"""
    shape = (horizon, NUM_FEATURES)
    masks = []
    prod = np.prod(shape)
    n_mask = int(prod / n)
    # set are much more efficient at removing
    not_used = set(i for i in range(prod))

    while len(masks) < n:
        mask = np.ones(prod)
        # choose from the aviable indices
        idxs = np.random.choice(tuple(not_used), n_mask, replace=False)
        # set to 0
        mask[idxs] = 0
        # mark as used
        not_used = not_used.difference(idxs)
        # reshape to the input shape
        mask = torch.tensor(mask).reshape(shape)
        masks.append(mask)

    return torch.stack(masks)


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


class DeepFIBDataset(Dataset):
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

