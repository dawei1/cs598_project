import datetime

import numpy as np
import torch

import Patching as p
import Recognition as r
import ResNet as rn

from GPUtil import showUtilization as gpu_usage


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
patching_model = PatchingModel(resnet_out_height_width, P, c_prime)


criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(patching_model.parameters(), lr=0.001, weight_decay=0.1)
n_epochs = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def custom_loss(output, target):
    positive_prob_loss = torch.sum(target * torch.log(output))
    negative_prob_loss = torch.sum((1 - target) * torch.log(1 - output))
    total_loss = -positive_prob_loss - negative_prob_loss
    return total_loss

def train_model(train_dataloader, model = patching_model, n_epoch=n_epochs, optimizer=optimizer, criterion=criterion):
    model.train()
    for epoch in range(n_epoch):
        print(f"Starting Epoch {epoch}")
        print(datetime.datetime.now())
        curr_epoch_loss = []
        for data, target in train_dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            output = 1 - output
            output = (output * 0.02) + 0.98
            output = torch.prod(output, 3)
            output = torch.prod(output, 2)
            prediction = 1 - output
            loss = custom_loss(prediction, target)
            loss.backward()
            curr_epoch_loss.append(loss.cpu().data.numpy())
            optimizer.step()
            
            del loss
            del output
            del prediction
            del data
            del target
            torch.cuda.empty_cache()

        print(f"Epoch {epoch}: curr_epoch_loss={np.mean(curr_epoch_loss)}")
    return model
