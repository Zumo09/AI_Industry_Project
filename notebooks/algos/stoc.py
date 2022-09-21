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
        device: Optional[torch.device] = None,
        k: int
    ):
        self.dataset = dataset
        self.device = device or torch.device("cpu")
        self.k = k
        self.gamma = 0.5 # has to be changed
        
    def __fit_subsets(self, x: UnfoldedDataset, k: int) -> List:
        subsets = self.__split_dataset(x, k)
        kdes = []
        
        for sub in subsets:
            data_sub = Subset(x, sub)
            gs_kde = GridSearchCV(KernelDensity(kernel='gaussian'),
                                                {'bandwidth'=np.linspace(0.01, 0.1, 20)}, cv=5)
            gs_kde.fit(data_sub)
            h = gs_kde.best_params_['bandwidth']
            kde = KernelDensity(kernel='gaussian', bandwidth=h)
            kde.fit(data_sub)
           
            g = self.engine.get_model
            thr = self.__find_threshold(x, kde, g)
            
            res = {"kde": kde, "thr": thr}
            kdes.append(res)
            
        return kdes
    
    def __indicator_function(sample, kde, g, thr):
        base_model = kde["kde"]
        if base_model(g(sample)) >= thr:
            return 1
        return 0
    
    def __find_threshold(self, x, kde, g):
        # determine for a given kde the best possible threshold as shown in Eq.2
        # predict the anomalies on the whole dataset to obtain the anomaly distribution
        # compute the threshold using the equation
        score_distr = kde["kde"](g(self.dataset))
        thr_range = np.linspace(0, 10, 100)
        
        thrs_evals = []
        dataloader = DataLoader(x, 1)
        generator = iter(dataloader)
        
        for thr in thr_range:
            sum_ = 0
            for i in range(len(x)):
                sample = next(generator)
                sum_ += self.__indicator_function(sample, kde, g, thr)
            thr_evals.append(sum_/len(x))
        
        return max(thr_evals)
    
    def __refine_data(self, x: UnfoldedDataset, g, k):
        kdes = self.__fit_subsets(x, k)
        
        new_x = x.detach().clone()
        mask = torch.ones(new_x.shape[0], new_x.shape[1])
        
        dataloader = DataLoader(new_x, 1)
        generator = iter(dataloader)
        for i in range(len(new_x)):
            sample = next(generator)
            for kde in kdes:
                if self.__indicator_function(sample, kde, g):
                    mask[i, :] = 0
        return new_x[mask != 0]
    
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
    
    def train_step(self, batch: Dict[str, Tensor]):
        refined_data = self.__refine_data(batch, self.engine.get_model, self.k)
        data_loader = DataLoader(
            refined_data,
            8,
            shuffle=True,
        )
        for b in tqdm(data_loader):
            cbl = self.engine.get_model
            rets = cbl.train_step(b)