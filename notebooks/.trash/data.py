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
