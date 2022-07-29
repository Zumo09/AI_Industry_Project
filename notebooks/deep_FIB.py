import json
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard.writer import SummaryWriter

from sklearn.model_selection import train_test_split

from utils.data import get_dataset_paths
from utils.data import Marconi100Dataset
from utils.data import UnfoldedDataset
from utils.training import training_loop

from algos.deep_fib.core import get_masks
from algos.deep_fib.core import DeepFIBEngine
from algos.deep_fib.sci_net import SCINet

paths = get_dataset_paths("../data")
train, test = train_test_split(paths, test_size=0.1, random_state=42)

m_data_train = Marconi100Dataset(train, scaling="minmax")
m_data_test = Marconi100Dataset(test, scaling="minmax")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

horizon = 1024
stride = 512
n_masks = 1000

batch_size = 32

num_encoder_levels = 3

log_dir = "./outputs/deep_fib_even_lower_lr"
lr = 1e-4
num_epochs = 30

hidden = [512]
input_dim = 460
hidden_size = 2
kernel_size = 3
dropout = 0.5

anomaly_threshold = 0.1  # to be tuned

dataset_train = UnfoldedDataset(m_data_train, horizon=horizon, stride=stride)
dataset_test = UnfoldedDataset(m_data_test, horizon=horizon, stride=stride)

masks = get_masks(horizon, n_masks).float()

print(len(dataset_train), len(dataset_test), masks.size())


train_loader = DataLoader(
    dataset_train,
    batch_size,
    shuffle=True,
)
test_loader = DataLoader(
    dataset_test,
    batch_size,
    shuffle=False,
)
print(len(train_loader), len(test_loader))

cfg = dict(
    output_len=horizon,
    input_len=horizon,
    num_encoder_levels=num_encoder_levels,
    hidden_decoder_sizes=hidden,
    input_dim=input_dim,
    hidden_size=hidden_size,
    kernel_size=kernel_size,
    dropout=dropout,
)

model = SCINet(
    output_len=horizon,
    input_len=horizon,
    num_encoder_levels=num_encoder_levels,
    hidden_decoder_sizes=hidden,
    input_dim=input_dim,
    hidden_size=hidden_size,
    kernel_size=kernel_size,
    dropout=dropout,
).float()

engine = DeepFIBEngine(anomaly_threshold, masks)

optim = Adam(model.parameters(), lr=lr)
lr_sched = CosineAnnealingLR(optim, num_epochs)

with SummaryWriter(log_dir) as writer:
    with open(f"{log_dir}/config.json", "w") as f:
        json.dump(cfg, f)
        print(cfg)

    training_loop(
        model=model,
        engine=engine,
        num_epochs=num_epochs,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        device=device,
        optimizer=optim,
        lr_scheduler=lr_sched,
        writer=writer,
        save_path=log_dir + "/models",
    )
