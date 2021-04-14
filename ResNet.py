# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 23:20:21 2021

@author: ouyad
"""

import torchvision.models as models
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import Constants
from Dataloader import *
import torch

model = models.resnet18()
#print(model)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
model.fc = None
model.avgpool = None
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
n_epochs = 10


def train_model(train_dataloader, model = model, n_epoch=n_epochs, optimizer=optimizer, criterion=criterion):
    model.train()
    for epoch in range(n_epoch):
        curr_epoch_loss = []
        for data, target in train_dataloader:
            #print(target)
            def closure():
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                curr_epoch_loss.append(loss.cpu().data.numpy())
                return loss
            optimizer.step(closure)
        print(f"Epoch {epoch}: curr_epoch_loss={np.mean(curr_epoch_loss)}")
    return model
    
    

