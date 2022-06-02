import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from deep_fib.sci_net import SCIBlockCfg, SCINet
from deep_fib.data import DeepFIBDataset, get_masks
from deep_fib.core import DeepFIBEngine

from utils.data import Marconi100Dataset, get_dataset_paths
from utils.training import training_loop

paths = get_dataset_paths("data")
train, test = train_test_split(paths, test_size=0.1, random_state=42)

# train = train[:len(train)//30]
# test = test[:len(test)//10]
# len(train), len(test)

m_data_train = Marconi100Dataset(train, normalize="normal")
m_data_test = Marconi100Dataset(test, normalize="normal")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

horizon = 1024
stride = 1000
n_masks = 100

batch_size = 32
num_workers = 0

num_encoder_levels = 2

log_dir = "./trash"
lr = 1e-3
num_epochs = 3
step_size = 2

hidden = None
block_cfg = SCIBlockCfg(input_dim=460, hidden_size=4, kernel_size=3, dropout=0.5,)

anomaly_threshold = 0.7

dataset_train = DeepFIBDataset(m_data_train, horizon=horizon, stride=stride)
dataset_test = DeepFIBDataset(m_data_test, horizon=horizon, stride=stride)
masks = get_masks(horizon, n_masks).float()

print("Datasets")
print("Train: ", len(dataset_train))
print("Test : ", len(dataset_test))
print("Masks: ", len(masks))


train_loader = DataLoader(
    dataset_train,
    batch_size,
    shuffle=True,
    num_workers=num_workers,
    # persistent_workers=(num_workers != 0),
)
test_loader = DataLoader(
    dataset_test,
    batch_size,
    shuffle=False,
    num_workers=num_workers,
    # persistent_workers=(num_workers != 0),
)

print("Dataloaders")
print("Train: ", len(train_loader))
print("Test : ", len(test_loader))

model = SCINet(
    output_len=horizon,
    input_len=horizon,
    num_encoder_levels=num_encoder_levels,
    hidden_decoder_sizes=hidden,
    block_config=block_cfg,
).float()

engine = DeepFIBEngine(anomaly_threshold, masks)

optim = Adam(model.parameters(), lr=lr)
lr_sched = StepLR(optim, step_size)

training_loop(
    model=model,
    engine=engine,
    num_epochs=num_epochs,
    train_dataloader=train_loader,
    test_dataloader=test_loader,
    device=device,
    optimizer=optim,
    lr_scheduler=lr_sched,
)
