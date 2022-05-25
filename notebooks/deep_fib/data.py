from typing import List, Tuple

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from utils.data import Marconi100Dataset


def unfold(
    marconi_dataset: Marconi100Dataset, horizon: int
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    ret = []
    for i in tqdm(range(len(marconi_dataset)), desc="Unfolding"):
        df, l = marconi_dataset[i]
        df = torch.tensor(df.to_numpy())
        l = torch.tensor(l.to_numpy()[..., np.newaxis])
        if len(df) == 0:
            continue
        t = torch.cat([df, l], dim=1)
        if len(t) > horizon:
            t = t.unfold(0, horizon, horizon//2).permute(0, 2, 1)
        else:
            t = pad(t, horizon)
        ret.append((t[..., :-1], t[..., -1]))
    return ret


def pad(x: torch.Tensor, input_len: int) -> torch.Tensor:
    bottom = input_len - x.shape[1]
    return F.pad(x, (0, 0, 0, bottom), "replicate")


def masks(shape: Tuple[int, int], n: int) -> torch.Tensor:
    """Pointwise Masking"""
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


class DeepFIBDataset(Dataset):
    def __init__(
        self, marconi_dataset: Marconi100Dataset, horizon: int, num_sample_per_data: int
    ) -> None:
        self.dataset = unfold(marconi_dataset, horizon)
        self.n = num_sample_per_data
        self.masks = masks(self.dataset[0][0].shape[1:], num_sample_per_data)

    def __len__(self) -> int:
        return len(self.dataset) * self.n

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data_idx = index // self.n
        mask_idx = index % self.n
        data, label = self.dataset[data_idx]
        mask = self.masks[mask_idx]
        return data, mask, label


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
