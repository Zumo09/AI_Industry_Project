from collections import defaultdict
import os
from random import random
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.nn import BCELoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Subset

import numpy as np

from common import metrics
from common.data import NUM_FEATURES, UnfoldedDataset
from common.models.modutils import save_model
from common.training import Writer
from common.kde import KernelDensity

from sklearn.model_selection import GridSearchCV
from algos import cbl
from tqdm import tqdm
from torch.utils.data import DataLoader

from .cbl import CBLEngine, AugmentFN

from typing import Dict, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class STOC:
    def __init__(
        self, engine: CBLEngine, gamma: float = 0.05, randomize_split: bool = True
    ):
        self.gamma = gamma  # has to be changed
        self.engine = engine
        self.randomize_split = randomize_split
        self.fitted_kde: Optional[KernelDensity] = None

    def _refine_data(self, dataset: UnfoldedDataset, k: int) -> Subset:
        kdes, features = self._fit_subsets(dataset, k)
        detections = np.array([self._find_detections(kde, features) for kde in kdes])

        # voting
        ensemble_output = np.logical_or.reduce(detections)

        # keep sample only when all kdes vote "False"
        idxs = list(np.arange(len(dataset), dtype=int)[~ensemble_output])

        if len(idxs) == 0:
            print("WARNING: all samples excluded in refinement! Recover all samples")
            idxs = list(np.arange(len(dataset), dtype=int))

        return Subset(dataset, idxs)

    def _find_detections(
        self, kde: KernelDensity, features: List[torch.Tensor]
    ) -> np.ndarray:
        # TODO: riceve una lista di features, deve predirre lo score distr per ogni sample e usare quel segnale per trovare le anomalie.
        # Bisogna decidere COSA è detection_max in questo caso. pero sicuramente dovrà essere len(detection_max) == len(features),
        # in pratica 1 detection per ogni sample del dataset

        # FEATURES : (N, T, F)
        score_distr = torch.stack(
            [
                kde.score_samples(f)
                for f in tqdm(features, leave=False, desc="score samples")
            ]
        )  # (N, T)
        thr_range = np.linspace(score_distr.min(), score_distr.max(), 100)

        detections_max = np.array([False for _ in range(len(features))])
        for thr in thr_range:
            detections: np.ndarray = (score_distr >= thr).numpy()  # (N, T)
            score_mean = detections.mean()
            if score_mean >= self.gamma:
                # considera Anomaly=True se la media sul singolo sample è >= 2 * gamma
                detections_max = detections.mean(1) >= 2 * self.gamma
                # # considera Anomaly=True se c'è almeno un timestamp anomalo del sample. Troppo???
                # detections_max = np.logical_or.reduce(detections.T)

        return detections_max  # (N,)

    def _fit_subsets(
        self, dataset: UnfoldedDataset, k: int
    ) -> Tuple[List[KernelDensity], List[torch.Tensor]]:
        # fit kdes on subsets (features) of the training set
        subsets = self._split_dataset_indices(len(dataset), k, self.randomize_split)
        kdes = []
        _features = []

        for sub in subsets:
            data_sub = Subset(dataset, sub)
            feats = self._extract_features(data_sub)
            kde = self._fit_kde(feats)

            _features.extend(zip(sub, feats))
            kdes.append(kde)

        # restore original ordering
        _features.sort(key=lambda x: x[0])
        features = [f[1] for f in _features]
        return kdes, features

    @torch.no_grad()
    def _extract_features(self, subset: Subset) -> List[torch.Tensor]:
        dataloader = DataLoader(subset, 32, shuffle=False, drop_last=False)

        res = []
        for batch in tqdm(dataloader, leave=False, desc="Extracting features"):
            inputs = batch["data"].to(self.engine.device)
            outs = self.engine.model(inputs)
            res.extend(outs.cpu())

        return res

    @staticmethod
    def _split_dataset_indices(
        n_data: int, k: int, randomize_split: bool
    ) -> List[List[int]]:
        idxs = list(range(n_data))

        if randomize_split:
            # randomly shuffle indices
            np.random.shuffle(idxs)
        else:
            # randomly roll indices
            start = np.random.randint(0, n_data - 1)
            idxs = list(np.roll(idxs, start))

        # get the len of each subset
        sub_len, extras = divmod(n_data, k)
        subsets_lens = [sub_len + 1 if i < extras else sub_len for i in range(k)]

        # get start and end indices of each "subset of indices"
        cum_lens = np.cumsum([0] + subsets_lens)
        return [idxs[s:e] for s, e in zip(cum_lens[:-1], cum_lens[1:])]

    def _fit_kde(self, features: List[torch.Tensor], cv: int = 0) -> KernelDensity:
        # TODO: Implement cross validation
        return KernelDensity(features, device=self.engine.device)

    def fit(
        self,
        train_dataset: UnfoldedDataset,
        val_dataset: UnfoldedDataset,
        batch_size: int,
        k: int = 5,
        epochs: List[int] = [1, 1, 1, 1],
        writer: Optional[Writer] = None,
        save_path: Optional[str] = None,
    ) -> None:
        self.fit_backbone(
            train_dataset, val_dataset, batch_size, k, epochs, writer, save_path
        )
        self.fit_kde(train_dataset, k)

    def fit_backbone(
        self,
        train_dataset: UnfoldedDataset,
        val_dataset: UnfoldedDataset,
        batch_size: int,
        k: int,
        epochs: List[int],
        writer: Optional[Writer] = None,
        save_path: Optional[str] = None,
    ):
        test_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        if save_path is not None:
            if not os.path.isdir(save_path):
                os.mkdir(save_path)

        _train_loss = np.nan
        _smoothing = 0.8
        _log_step = 0

        for i, num_epochs in enumerate(epochs):
            refined_data = self._refine_data(train_dataset, k)
            data_loader = DataLoader(
                refined_data,
                batch_size=batch_size,
                shuffle=True,
            )
            for e in range(num_epochs):
                for b in tqdm(
                    data_loader, desc=f"Train Iteration {i} - Epoch {e}", leave=False
                ):
                    rets = self.engine.train_step(b)
                    if np.isnan(_train_loss):
                        _train_loss = rets["loss"]
                    else:
                        _train_loss *= _smoothing
                        _train_loss += (1 - _smoothing) * rets["loss"]

                    if writer is not None:
                        for tag, val in rets.items():
                            writer.add_scalars(tag, {"train": val}, _log_step)

                    _log_step += 1

                test_scalars = defaultdict(list)
                for batch in tqdm(
                    test_dataloader, leave=False, desc=f"Test Iteration {i} - Epoch {e}"
                ):
                    rets = self.engine.val_step(batch)
                    for tag, val in rets.items():
                        test_scalars[tag].append(val)

                if writer is not None:
                    for tag, val in test_scalars.items():
                        writer.add_scalars(
                            tag, {"test": float(np.mean(val))}, _log_step
                        )

                log_str = f"Iteration {i} - Epoch {e} - train_loss={_train_loss:.3f}"

                for key, value in test_scalars.items():
                    log_str += f" - test_{key}={np.mean(value):.3f}"

                log_dict = self.engine.end_epoch(1000 * i + e, save_path)
                for key, value in log_dict.items():
                    log_str += f" - {key}={value}"

                print(log_str)

    def fit_kde(self, dataset: UnfoldedDataset, k: int) -> None:
        refined_data = self._refine_data(dataset, k)
        refined_features = self._extract_features(refined_data)
        self.fitted_kde = self._fit_kde(refined_features, 5)

    @torch.no_grad()
    def predict(self, batch: Dict[str, Tensor]) -> Tensor:
        assert self.fitted_kde is not None, "MODEL NOT FITTED"

        inputs = batch["data"].to(self.engine.device)  # (B, T, Fin)
        outs = self.engine.model(inputs)
        feats = outs.cpu()  # (B, T, F)

        score_distr = [self.fitted_kde.score_samples(f) for f in feats]  # (B, T)

        return torch.stack(score_distr)


###################################################################################

# def get_label(label: torch.Tensor) -> torch.Tensor:
#     return (label.sum() > 0).int()


# end = False
# step = 0
# start = random.randint(0, len(x) - 1)
# subsets = []
# l = list(range(len(x)))
# l_bool = [False for elem in l]

# while not end:
#     # if there are enough indexes to make up another subset, make it
#     if l_bool.count(False) >= int(len(l) / k):
#         l_sub = []
#         while len(l_sub) < int(len(l) / k):
#             idx = (start + step) % len(l)
#             l_sub.append(l[idx])
#             l_bool[idx] = True
#             step += 1
#         subsets.append(l_sub)
#     else:
#         # if there are still indexes that haven't been assigned
#         l_sub_last = []
#         while l_bool.count(False) > 0:
#             idx = (start + step) % len(l)
#             l_sub_last.append(l[idx])
#             l_bool[idx] = True
#             step += 1
#         subsets.append(l_sub_last)
#         if len(l_sub_last) < 100:
#             # if the last sublist is too short merge it with the last one
#             merged_sub = subsets[-2] + subsets[-1]
#             subsets.pop(-1)
#             subsets.pop(-2)
#             subsets.append(merged_sub)
#         end = True
# return subsets
