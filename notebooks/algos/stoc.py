import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.nn import BCELoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import numpy as np

from common import metrics
from common.data import NUM_FEATURES
from common.models.modutils import save_model

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from common.models.resnet import ResNetFeatures 
    
class STOCEngine:
    def __init__(
        self,
        model: ResNetFeatures,
        optimizer: Optimizer,
        device: Optional[torch.device] = None,
        # k: int,
        # gamma: float, 
        lr_scheduler: Optional[_LRScheduler] = None,
    ):
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        self.loss = BCELoss()
        self.k = k
        self.gamma = gamma
        
        # self.base_estimators = []