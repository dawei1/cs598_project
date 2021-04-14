# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 01:25:29 2021

@author: ouyad
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import Constants
from Dataloader import *
from ResNet import *


full_train_dataset, sub_train_dataset, val_train_dataset = get_dataset()

sub_train_dataset_len = sub_train_dataset.__len__()
eval_set_size = int(sub_train_dataset_len * Constants.eval_set_ratio)
train_set_size = sub_train_dataset_len - eval_set_size
train_dataset, eval_dataset = torch.utils.data.random_split(sub_train_dataset, [train_set_size, eval_set_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Constants.batch_size, shuffle=True)
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=Constants.batch_size, shuffle=True)

model = train_model(train_loader)