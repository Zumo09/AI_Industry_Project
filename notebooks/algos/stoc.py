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

from .cbl import CBLFeatsEngine, AugmentFN

from common.models.deeplab import DeepLabNet

from typing import  Dict, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

def get_label(label: torch.Tensor) -> torch.Tensor:
    return (label.sum() > 0).int()

class CBLDeepLab(CBLFeatsEngine):
    def __init__(
        self, 
        model: DeepLabNet, 
        optimizer: Optimizer, 
        temperature: float = 0.5, 
        device: Optional[torch.device] = None, 
        aug_1: Optional[AugmentFN] = None, 
        aug_2: Optional[AugmentFN] = None, 
        lr_scheduler: Optional[_LRScheduler] = None
    ) -> None:
        super().__init__(model, optimizer, temperature, device, aug_1, aug_2, lr_scheduler)  # type: ignore
        assert isinstance(self.model, DeepLabNet)
    
    def _get_outs(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs).permute(0, 2, 1)
        
    
class STOC:
    def __init__(
        self,
        dataset: UnfoldedDataset,
        engine: CBLDeepLab,
        k: int
    ):
        self.dataset = dataset
        self.k = k
        self.gamma = 0.05 # has to be changed
        self.engine = engine

        self.fitted_kde: Optional[KernelDensity] = None
    
    @torch.no_grad()
    def __extract_features(self, subset):
        dataloader = DataLoader(
            subset,
            32,
            shuffle=False,
            drop_last=False)
        
        res = []
        for batch in dataloader:
            inputs = batch["data"].to(self.engine.device)
            outs = self.engine.model(inputs)
            res.append(outs.cpu().unsqueeze(-1))
            
        return torch.concat(res, dim=0)
        
    def __fit_subsets(self, x: UnfoldedDataset, k: int) -> Tuple[List[KernelDensity], np.ndarray]:
        # fit kdes on subsets (features) of the training set
        subsets = self.__split_dataset_indices(len(x), k)
        kdes = []
        features_ = []
        
        for sub in subsets:
            data_sub = Subset(x, sub)
            kde, features = self._fit_kde(data_sub)
           
            # thr = self.__find_threshold(x, kde, features)
            # res = {"kde": kde, "thr": thr}
            
            features_.append(features)
            kdes.append(kde)
        
        return kdes, np.concatenate(features_)

    def _fit_kde(self, data_sub):
        features = self.__extract_features(data_sub) # D x F
            
        gs_kde = GridSearchCV(KernelDensity(kernel='gaussian'),
                                                {'bandwidth': np.linspace(0.01, 0.1, 20)}, cv=5)
        gs_kde.fit(features)
        h = gs_kde.best_params_['bandwidth']
        kde = KernelDensity(kernel='gaussian', bandwidth=h)
        kde.fit(features)
        return kde, features
    
    def __find_detections(self, kde: KernelDensity, features):
        score_distr: np.ndarray = kde.score_samples(features)
        thr_range = np.linspace(min(score_distr), max(score_distr), 100)
        
        detections_max = None
        for thr in thr_range:
            detections = score_distr >= thr
            score_mean = detections.mean()
            if score_mean >= self.gamma:
                detections_max = detections
      
        return detections_max
    
    def __refine_data(self):
        kdes, features = self.__fit_subsets(self.dataset, self.k)
        detections = np.array([self.__find_detections(kde, features) for kde in kdes])
        
        # voting
        ensemble_output = np.logical_or.reduce(detections)
        # keep sample only when all kdes vote "False"
        idxs = np.arange(len(self.dataset))[~ensemble_output]
        
        return Subset(self.dataset, idxs)
    
    # WORKING
    def __split_dataset_indices(self, n_data: int, k: int) -> List[List[int]]:
        # randomly roll indices
        start = np.random.randint(0, n_data - 1)
        idxs = list(np.roll(list(range(n_data)), start))

        # get the len of each subset
        sub_len, extras = divmod(n_data, k)
        subsets_lens = [sub_len + 1 if i < extras else sub_len for i in range(k)]

        # get start and end indices of each "subset of indices"
        cum_lens = np.cumsum([0] + subsets_lens)
        return [idxs[s:e] for s, e in zip(cum_lens[:-1], cum_lens[1:])]

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
    
    def train_epoch(self):
        refined_data = self.__refine_data()
        data_loader = DataLoader(
            refined_data,
            8,
            shuffle=True,
        )
        for e in range(num_epochs_per_refine):
            for b in tqdm(data_loader):
                cbl = self.engine.train_step(b)

    def fit(self, num_epoch):
        for _ in range(num_epoch):
            self.train_epoch()

        refined_data = self.__refine_data()
        self.fitted_kde, _ = self._fit_kde(refined_data)


    @torch.no_grad()
    def predict(self, batch: Dict[str, Tensor]) -> Tensor:
        assert self.fitted_kde is not None, "MODEL NOT FITTED"
        # Estrarre features del batch
        # predirre con fitted_kde

        return errors
        
        