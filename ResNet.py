# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 23:20:21 2021

@author: ouyad
"""

def get_resnet_model():
    import torchvision.models
    import torch.nn
    model = torchvision.models.resnet50(pretrained=True)

    '''
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
        #param[1].requires_grad = False
    '''
    model_list = list(model.children())[:-2]
    model = torch.nn.Sequential(*model_list)
    return model

'''
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
'''


