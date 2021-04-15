# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 01:25:29 2021

@author: ouyad
"""
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import Constants
from Dataloader import *
from ResNet import *
from sklearn.metrics import accuracy_score
import Constants


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

model = train_model(train_loader, n_epoch=10)

y_pred, y_true = eval_model(model, val_loader)
acc = accuracy_score(y_true, y_pred)
print(("Validation Accuracy: " + str(acc)))
