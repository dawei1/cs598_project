import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import Constants
from Dataloader_2 import *
from full_model import *
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader
from GPUtil import showUtilization as gpu_usage



full_train_dataset, sub_train_dataset, val_train_dataset = get_dataset()

#sub_train_dataset_len = sub_train_dataset.__len__()
#eval_set_size = int(sub_train_dataset_len * Constants.eval_set_ratio)
#train_set_size = sub_train_dataset_len - eval_set_size
#train_dataset, eval_dataset = torch.utils.data.random_split(sub_train_dataset, [train_set_size, eval_set_size])

full_train_dataset_len = full_train_dataset.__len__()
eval_set_size = int(full_train_dataset_len * Constants.eval_set_ratio)
train_set_size = full_train_dataset_len - eval_set_size
train_dataset, eval_dataset = torch.utils.data.random_split(full_train_dataset, [train_set_size, eval_set_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Constants.batch_size, shuffle=True, num_workers=Constants.num_of_workers)
val_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=Constants.batch_size, shuffle=True, num_workers=Constants.num_of_workers)

seed = 24
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
RANDOM_STATE = 545510477

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

start_model = PatchingModel(resnet_out_height_width, 6, c_prime).to(device)
start_optimizer = torch.optim.Adam(start_model.parameters(), lr=0.001, weight_decay=0.1)
trained_model = train_model(train_loader, n_epoch=30, model=start_model, optimizer=start_optimizer)
trained_model.eval()
y_hats = []
y_Preds = []
targets = []
for data, target in val_loader:
    data = data.to(device)
    y_hat = trained_model(data)
    y_hat = y_hat.to(torch.device('cpu'))
    y_hat = 1 - y_hat
    y_hat = (y_hat * 0.02) + 0.98
    y_hat = torch.prod(y_hat, 3)
    y_hat = torch.prod(y_hat, 2)
    y_hat = 1 - y_hat
    y_Pred = torch.where(y_hat>=0.5, torch.tensor([1]), torch.tensor([0]))

    y_hats.extend(y_hat.detach().numpy())
    y_Preds.extend(y_Pred.detach().numpy())
    targets.extend(target.detach().numpy())

    del data
    torch.cuda.empty_cache()

y_hats = np.array(y_hats)
y_Preds = np.array(y_Preds)
targets = np.array(targets)

acc = np.array([accuracy_score(targets[:, idx], y_Preds[:, idx]) for idx in range(targets.shape[1])])
print(acc)

auc = np.array([roc_auc_score(targets[:, idx], y_hats[:, idx]) for idx in range(targets.shape[1])])
print(auc)

del start_model
del trained_model
torch.cuda.empty_cache()
