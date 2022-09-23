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
from common.data import NUM_FEATURES
from common.models.modutils import save_model

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from algos import cbl
from tqdm import tqdm
from torch.utils.data import DataLoader
    
class STOC:
    def __init__(
        self,
        # should be the entire dataset
        dataset: UnfoldedDataset,
        engine: CBLFeatsEngine,
        k: int
    ):
        self.dataset = dataset
        self.k = k
        self.gamma = 0.05 # has to be changed
        
    def __extract_features(self, subset):
        dataloader = DataLoader(
            subset,
            32,
            shuffle=False,
            keep_last=True)
        
        res = []
        for batch in dataloader:
            inputs = batch["data"].to(self.engine.device)
            inputs = inputs.permute(0, 2, 1)
            outs = self.engine.model(inputs)[self.engine.model.nodes[-1]]
            res.append(outs.mean(-1))
            
        return torch.concat(res, dim=0).numpy()
        
    def __fit_subsets(self, x: UnfoldedDataset, k: int) -> List:
        # fit kdes on subsets (features) of the training set
        subsets = self.__split_dataset(x, k)
        kdes = []
        features_ = []
        
        for sub in subsets:
            data_sub = Subset(x, sub)
            features = self.__extract_features(data_sub)
            
            gs_kde = GridSearchCV(KernelDensity(kernel='gaussian'),
                                                {'bandwidth'=np.linspace(0.01, 0.1, 20)}, cv=5)
            gs_kde.fit(features)
            h = gs_kde.best_params_['bandwidth']
            kde = KernelDensity(kernel='gaussian', bandwidth=h)
            kde.fit(features)
           
            # thr = self.__find_threshold(x, kde, features)
            # res = {"kde": kde, "thr": thr}
            
            features_.append(features)
            kdes.append(kde)
        
        return kdes, np.concatenate(features)
    
    def __indicator_function(sample, kde, g, thr):
        base_model = kde["kde"]
        if base_model(g(sample)) >= thr:
            return 1
        return 0
    
    def __find_detections(self, kde, features):
        score_distr = kde["kde"](features)
        thr_range = np.linspace(min(score_distr), max(score_distr), 100)
        
        detections_max = None
        for thr in thr_range:
            detections = score_distr >= thr
            score_mean = detections.mean()
            if score_mean >= self.gamma:
                detections_max = detections
      
        return detections
    
    def __refine_data(self):
        kdes, features = self.__fit_subsets(self.dataset, self.k)
        detections = [self.__find_detections(kde, features) for kde in kdes]
        
        # voting
        ensemble_output = np.logical_or.reduce(detections)
        # keep sample only when all kdes vote "False"
        idxs = np.arange(len(self.dataset))[~ensemble_output]
        
        return Subset(self.dataset, idxs)
    
    # WORKING
    def __split_dataset(self, x: data.UnfoldedDataset, k: int) -> List:
        end = False
        step = 0
        start = random.randint(0, len(x) - 1)
        subsets = []
        l = list(range(len(x)))
        l_bool = [False for elem in l]

        while not end:
            # if there are enough indexes to make up another subset, make it
            if l_bool.count(False) >= int(len(l) / k):
                l_sub = []
                while len(l_sub) < int(len(l) / k):
                    idx = (start + step) % len(l)
                    l_sub.append(l[idx])
                    l_bool[idx] = True
                    step += 1
                subsets.append(l_sub)
            else:
                # if there are still indexes that haven't been assigned
                l_sub_last = []
                while l_bool.count(False) > 0:
                    idx = (start + step) % len(l)
                    l_sub_last.append(l[idx])
                    l_bool[idx] = True
                    step += 1
                subsets.append(l_sub_last)
                if len(l_sub_last) < 100:
                    # if the last sublist is too short merge it with the last one
                    merged_sub = subsets[-2] + subsets[-1]
                    subsets.pop(-1)
                    subsets.pop(-2)
                    subsets.append(merged_sub)
                end = True
        return subsets
    
    def train_epoch(self):
        refined_data = self.__refine_data()
        data_loader = DataLoader(
            refined_data,
            8,
            shuffle=True,
        )
        for b in tqdm(data_loader):
            cbl = self.engine.tran_step(b)