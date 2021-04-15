# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 23:20:21 2021

@author: ouyad
"""
import Constants
import torchvision.models as models
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import Constants
from Dataloader import *
import torch


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


model = models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, Constants.num_classes)
for param in model.named_parameters():
    print(param[0])
    if param[0] == 'conv1.weight':
        print('Keep the first conv2d layer trainable')
        continue
    if param[0] == 'fc.weight':
        print('Keep the last fc layer trainable')
        continue
    if param[0] == 'fc.bias':
        print('Keep the last fc layer trainable')
        continue
    # Freeze the pretrained model layers
    param[1].requires_grad = False

# Need to make this layer to return the input.
#model.fc = Identity()
# Need to make this layer to return the input.
#model.avgpool = Identity()
print(model)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
n_epochs = 1


def train_model(train_dataloader, model = model, n_epoch=n_epochs, optimizer=optimizer, criterion=criterion):
    print("Start training for model...")
    model.train()
    for epoch in range(n_epoch):
        print(f"Epoch {epoch}")
        curr_epoch_loss = []
        for data, target in train_dataloader:
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
    
    
def eval_model(model, dataloader):
    model.eval()
    Y_pred = []
    Y_test = []
    for data, target in dataloader:
        # your code here
        pred = model(data)
        _, predicted = torch.max(pred, 1)
        Y_pred.append(predicted.detach().numpy())
        Y_test.append(target.detach().numpy())
    Y_pred = np.concatenate(Y_pred, axis=0)
    Y_test = np.concatenate(Y_test, axis=0)
    print(len(Y_pred))
    print(len(Y_test))
    return Y_pred, Y_test



