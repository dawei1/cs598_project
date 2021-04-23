import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import Constants
from Dataloader import *
from full_model import *
from sklearn.metrics import accuracy_score


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

acc_list = []
auc_list = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index] 
    Y_train, Y_test = Y[train_index], Y[test_index]
    tensor_x = torch.Tensor(X_train)
    tensor_y = torch.Tensor(Y_train)
    train_dataset = TensorDataset(tensor_x, tensor_y)
    train_dataloder = DataLoader(train_dataset)
    model = train_model(train_dataloder, n_epoch=10)
    y_hat = model(X_test)
    y_Pred = torch.where(y_hat>=0.5, torch.tensor([1]), torch.tensor([0]))
    
    acc = accuracy_score(Y_test, Y_Pred)
    acc_list.append(acc)
    
    auc = roc_auc_score(Y_test, Y_Pred)
    auc_list.append(auc)
acc_mean = mean(acc_list)
auc_mean = mean(auc_list)        
print(acc_mean,auc_mean)