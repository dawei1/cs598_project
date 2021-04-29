import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import Constants
from Dataloader import *
from full_model import *
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader



full_train_dataset, sub_train_dataset, val_train_dataset = get_dataset()

sub_train_dataset_len = sub_train_dataset.__len__()
eval_set_size = int(sub_train_dataset_len * Constants.eval_set_ratio)
train_set_size = sub_train_dataset_len - eval_set_size
train_dataset, eval_dataset = torch.utils.data.random_split(sub_train_dataset, [train_set_size, eval_set_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Constants.batch_size, shuffle=True, num_workers=Constants.num_of_workers)
val_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=Constants.batch_size, shuffle=True, num_workers=Constants.num_of_workers)

seed = 24
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
RANDOM_STATE = 545510477

acc_list = []
auc_list = []
X = None
Y = None

X = np.array([i[0].numpy() for i in sub_train_dataset])
Y = np.array([i[1].numpy() for i in sub_train_dataset])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

kf = KFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)
for train_index, test_index in kf.split(sub_train_dataset):
    train_image, test_image = X[train_index], X[test_index]
    train_label, test_label = Y[train_index], Y[test_index]
    tensor_x = torch.Tensor(train_image)
    tensor_y = torch.Tensor(train_label)
    train_dataset = TensorDataset(tensor_x, tensor_y)
    train_dataloder = DataLoader(train_dataset, batch_size=Constants.batch_size, shuffle=True, num_workers=Constants.num_of_workers)
    start_model = PatchingModel(resnet_out_height_width, P, c_prime).to(device)
    start_optimizer = torch.optim.Adam(start_model.parameters(), lr=0.001, weight_decay=0.1)
    trained_model = train_model(train_dataloder, n_epoch=20, model=start_model, optimizer=start_optimizer)
    trained_model.eval()
    y_hat = trained_model(torch.Tensor(test_image).to(device))
    y_hat = 1 - y_hat
    y_hat = (y_hat * 0.02) + 0.98
    y_hat = torch.prod(y_hat, 3)
    y_hat = torch.prod(y_hat, 2)
    y_hat = 1 - y_hat
    y_Pred = torch.where(y_hat>=0.5, torch.tensor([1]), torch.tensor([0]))
    
    acc = np.array([accuracy_score(test_label[:, idx], y_Pred[:, idx]) for idx in range(test_label.shape[1])])
    print(acc)
    acc_list.append(acc)
    
    auc = np.array([roc_auc_score(test_label[:, idx], y_hat[:, idx].detach().numpy()) for idx in range(test_label.shape[1])])
    print(auc)
    auc_list.append(auc)

acc_mean = np.mean(acc_list, axis=0)
auc_mean = np.mean(auc_list, axis=0)        
print(acc_mean, auc_mean)
print(np.mean(acc_mean), np.mean(auc_mean))
