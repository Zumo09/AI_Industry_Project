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

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from algos import cbl
from tqdm import tqdm
from torch.utils.data import DataLoader

from .cbl import CBLEngine, AugmentFN

from common.models.deeplab import DeepLabNet

from typing import Dict, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CBLDeepLab(CBLEngine):
    def __init__(
        self,
        model: DeepLabNet,
        optimizer: Optimizer,
        temperature: float = 0.5,
        device: Optional[torch.device] = None,
        aug_1: Optional[AugmentFN] = None,
        aug_2: Optional[AugmentFN] = None,
        lr_scheduler: Optional[_LRScheduler] = None,
    ) -> None:
        super().__init__(
            model, optimizer, temperature, device, aug_1, aug_2, lr_scheduler
        )

    def _get_outs(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs).permute(0, 2, 1)


class STOC:
    def __init__(self, dataset: UnfoldedDataset, engine: CBLDeepLab, k: int):
        self.dataset = dataset
        self.k = k
        self.gamma = 0.05  # has to be changed
        self.engine = engine

        self.fitted_kde: Optional[KernelDensity] = None

    def _refine_data(self, dataset: UnfoldedDataset, k: int) -> Subset:
        kdes, features = self._fit_subsets(dataset, k)
        detections = np.array([self._find_detections(kde, features) for kde in kdes])

        # voting
        ensemble_output = np.logical_or.reduce(detections)

        # keep sample only when all kdes vote "False"
        idxs = np.arange(len(dataset))[~ensemble_output]

        return Subset(dataset, idxs)

    def _find_detections(
        self, kde: KernelDensity, features: List[np.ndarray]
    ) -> np.ndarray:
        # TODO: riceve una lista di features, deve predirre lo score distr per ogni sample e usare quel segnale per trovare le anomalie.
        # Bisogna decidere COSA è detection_max in questo caso. pero sicuramente dovrà essere len(detection_max) == len(features),
        # in pratica 1 detection per ogni sample del dataset

        # FEATURES : (N, T, F)
        score_distr = [kde.score_samples(f) for f in features]  # (N, T)
        thr_range = np.linspace(min(score_distr), max(score_distr), 100)

        detections_max = np.array([False for _ in range(len(features))])
        for thr in thr_range:
            detections = score_distr >= thr  # (N, T)
            score_mean = detections.mean()
            if score_mean >= self.gamma:
                detections_max = np.logical_or.reduce(
                    detections.T
                )  # considera Anomaly=True se c'è almeno un timestamp anomalo del sample. Troppo???

        return detections_max  # (N,)

    def _fit_subsets(
        self, dataset: UnfoldedDataset, k: int
    ) -> Tuple[List[KernelDensity], List[np.ndarray]]:
        # fit kdes on subsets (features) of the training set
        subsets = self._split_dataset_indices(len(dataset), k)
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
        features = [f[0] for f in _features]
        return kdes, features

    @torch.no_grad()
    def _extract_features(self, subset: Subset) -> List[np.ndarray]:
        dataloader = DataLoader(subset, 32, shuffle=False, drop_last=False)

        res = []
        for batch in tqdm(dataloader, leave=False, desc="Extracting features"):
            inputs = batch["data"].to(self.engine.device)
            outs = self.engine.model(inputs)
            res.extend(outs.cpu().numpy())

        return res

    @staticmethod
    def _split_dataset_indices(n_data: int, k: int) -> List[List[int]]:
        # randomly roll indices
        start = np.random.randint(0, n_data - 1)
        idxs = list(np.roll(list(range(n_data)), start))

        # get the len of each subset
        sub_len, extras = divmod(n_data, k)
        subsets_lens = [sub_len + 1 if i < extras else sub_len for i in range(k)]

        # get start and end indices of each "subset of indices"
        cum_lens = np.cumsum([0] + subsets_lens)
        return [idxs[s:e] for s, e in zip(cum_lens[:-1], cum_lens[1:])]

    def _fit_kde(self, features: List[np.ndarray]) -> KernelDensity:
        gs_kde = GridSearchCV(
            KernelDensity(kernel="gaussian"),
            {"bandwidth": np.linspace(0.01, 0.1, 20)},
            cv=5,
        )

        feats = np.concatenate(features)  # (NxT, F)

        gs_kde.fit(feats)
        h = gs_kde.best_params_["bandwidth"]
        kde = KernelDensity(kernel="gaussian", bandwidth=h)
        kde.fit(feats)

        return kde

    def train_iteration(self, num_epochs: int):
        refined_data = self._refine_data(self.dataset, self.k)
        data_loader = DataLoader(
            refined_data,
            8,
            shuffle=True,
        )
        for e in range(num_epochs):
            for b in tqdm(data_loader, desc=f"Epoch {e}"):
                cbl = self.engine.train_step(b)

    def fit(self, num_epochs: List[int]):
        for n in num_epochs:
            self.train_iteration(n)

        refined_data = self._refine_data(self.dataset, self.k)
        refined_features = self._extract_features(refined_data)
        self.fitted_kde = self._fit_kde(refined_features)

    @torch.no_grad()
    def predict(self, batch: Dict[str, Tensor]) -> Tensor:
        assert self.fitted_kde is not None, "MODEL NOT FITTED"

        inputs = batch["data"].to(self.engine.device)  # (B, T, Fin)
        outs = self.engine.model(inputs)
        feats = outs.cpu().numpy()  # (B, T, F)

        score_distr = [self.fitted_kde.score_samples(f) for f in feats]  # (B, T)

        return torch.as_tensor(score_distr)


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
