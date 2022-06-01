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
    def __init__(
        self,
        marconi_dataset: Marconi100Dataset,
        *,
        horizon: int,
        stride: int,
    ) -> None:
        self.dataset = marconi_dataset
        self.indexes = unfolded_indexes(marconi_dataset, horizon, stride)
        self.win_len = horizon

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        df_idx, (start, end) = self.indexes[index]
        data, label = self.dataset[df_idx]
        data_t = torch.tensor(data.to_numpy())[start:end].float()
        label_t = torch.tensor(label.to_numpy())[start:end].int()

        return {"data": data_t, "label": label_t}

# class DeepFIBDataset(Dataset):
#     def __init__(
#         self,
#         marconi_dataset: Marconi100Dataset,
#         *,
#         horizon: int,
#         stride: int,
#         n_masks: int = 1,
#     ) -> None:
#         self.dataset = marconi_dataset
#         self.indexes = unfolded_indexes(marconi_dataset, horizon, stride)
#         self.n = n_masks
#         self.win_len = horizon
#         self.masks = masks((horizon, NUM_FEATURES), n_masks).float()

#     def __len__(self) -> int:
#         return len(self.indexes) * self.n

#     def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
#         data_idx = index // self.n
#         mask_idx = index % self.n

#         df_idx, (start, end) = self.indexes[data_idx]
#         data, label = self.dataset[df_idx]
#         data_t = torch.tensor(data.to_numpy())[start:end].float()
#         label_t = torch.tensor(label.to_numpy())[start:end].int()

#         mask = self.masks[mask_idx]
#         return {"data": data_t, "mask": mask, "label": label_t}


# import torch.nn.functional as F
# import pandas as pd
# from tqdm import tqdm


# def unfold(
#     marconi_dataset: Marconi100Dataset, horizon: int
# ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
#     ret = []
#     for i in tqdm(range(len(marconi_dataset)), desc="Unfolding"):
#         df, l = marconi_dataset[i]
#         df = torch.tensor(df.to_numpy())
#         l = torch.tensor(l.to_numpy()[..., np.newaxis])
#         if len(df) == 0:
#             continue
#         t = torch.cat([df, l], dim=1)
#         if len(t) > horizon:
#             t = t.unfold(0, horizon, horizon // 2).permute(0, 2, 1)
#         else:
#             t = pad(t, horizon, 1)
#         ret.extend((t[i, ..., :-1], t[i, ..., -1]) for i in range(len(t)))
#     return ret


# def pad(x: torch.Tensor, input_len: int, dim: int) -> torch.Tensor:
#     bottom = input_len - x.size(dim)
#     return F.pad(x, (0, 0, 0, bottom), "replicate")

# class DeepFIBDataset(Dataset):
#     def __init__(
#         self, marconi_dataset: Marconi100Dataset, horizon: int, num_sample_per_data: int
#     ) -> None:
#         self.dataset = unfold(marconi_dataset, horizon)
#         self.n = num_sample_per_data
#         self.masks = masks(self.dataset[0][0].shape, num_sample_per_data)

#     def __len__(self) -> int:
#         return len(self.dataset) * self.n

#     def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
#         data_idx = index // self.n
#         mask_idx = index % self.n
#         data, label = self.dataset[data_idx]
#         mask = self.masks[mask_idx]
#         return {"data": data.float(), "mask": mask.float(), "label": label}


# class DeepFIBDataset(Dataset):
#     def __init__(self, marconi_dataset: Marconi100Dataset, num_sample_per_data: int) -> None:
#         self._dataset = marconi_dataset
#         self.n = num_sample_per_data
#         self.mask_tag = -1

#     def __len__(self) -> int:
#         return len(self._dataset) * self.n

#     def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Tuple[pd.DataFrame, pd.Series]]:
#         original, label = self._dataset[index // self.n]

#         mask = self._mask(original.shape)
#         original_tensor = torch.tensor(original.to_numpy())

#         return original_tensor, mask, (original, label)

#     def _mask(self, shape: Tuple[int, int]) -> torch.Tensor:
#         """Pointwise Masking"""
#         mask = np.ones(np.prod(shape))
#         n_mask = int(np.prod(shape) / self.n)

#         # make sure as many as n_mask samples are masked
#         mask[:n_mask] = 0

#         # permute elements
#         mask = np.random.permutation(mask)

#         # reshape to the input shape
#         mask = torch.tensor(mask).reshape(shape)

#         return mask
