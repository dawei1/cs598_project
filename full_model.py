import datetime

import numpy as np
import torch

import Patching as p
import Recognition as r
import ResNet as rn


class PatchingModel(torch.nn.Module):

    def __init__(self, height_width, P, c_prime):
        super(PatchingModel, self).__init__()
        self.resnet = rn.get_resnet_model()
        self.patching = p.Patching(height_width, P)
        self.recognition = r.Recognition(c_prime)

    def forward(self, x):
        output = self.resnet(x)
        output = self.patching(output)
        output = self.recognition(output)
        return output


resnet_out_height_width = 7
c_prime = 2048
P = 10
model = PatchingModel(resnet_out_height_width, P, c_prime)


criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.1)
n_epochs = 10


def train_model(train_dataloader, model = model, n_epoch=n_epochs, optimizer=optimizer, criterion=criterion):
    model.train()
    for epoch in range(n_epoch):
        print(f"Starting Epoch {epoch}")
        print(datetime.datetime.now())
        curr_epoch_loss = []
        for data, target in train_dataloader:
            #print(target)
            def closure():
                optimizer.zero_grad()
                output = model(data)
                output = 1 - output
                output = torch.prod(output, 3)
                output = torch.prod(output, 2)
                prediction = 1 - output
                loss = criterion(prediction, target)
                loss.backward()
                curr_epoch_loss.append(loss.cpu().data.numpy())
                return loss
            optimizer.step(closure)
        print(f"Epoch {epoch}: curr_epoch_loss={np.mean(curr_epoch_loss)}")
    return model
