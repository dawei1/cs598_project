from sklearn.model_selection import KFold
from sklearn.metrics import *
import numpy as np
from numpy import mean
import torch
import utils
<<<<<<< HEAD
from full_model import *
from torch.utils.data.sampler import SubsetRandomSampler
=======
import Dataloader

>>>>>>> 2a518b15ddc88c77036b74a1a2a69049d1325c04

# Random Default from HW
RANDOM_STATE = 545510477

<<<<<<< HEAD
#input: X = training data and Y = corresponding labels, model = trained model
#output: accuracy, auc
def auc_eval(model, dataloader, k=5):
    
    acc_list = []
    auc_list = []
=======
def eval_model(model, val_dataloader):
>>>>>>> 2a518b15ddc88c77036b74a1a2a69049d1325c04
    
    model.eval()
    y_pred = torch.LongTensor()
    y_true = torch.LongTensor()

<<<<<<< HEAD
    for train_index, test_index in kf.split(X):
#         print("TRAIN:", train_index, "TEST:", test_index)
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
=======
    for image, label in val_dataloader:
        output = model.forward(image)
        #Are steps below needed here in EVAL as well?         
        output = 1 - output
        output = (output * 0.02) + 0.98
        output = torch.prod(output, 3)
        output = torch.prod(output, 2)
        prediction = 1 - output
        #############################################
    
        y_pred = torch.cat((y_pred,  prediction.detach().to('cpu')), dim=0)
        y_true = torch.cat((y_true, label.detach().to('cpu')), dim=0)   
>>>>>>> 2a518b15ddc88c77036b74a1a2a69049d1325c04
        
    auc = roc_auc_score(y_true, y_pred)
    
    return y_pred, y_true, auc


#input: X = training data and Y = corresponding labels, model = trained model
#output: accuracy, auc

# def auc_eval(model, k=5):
#     
# 
#     
#     auc_list = []
#     
#     kf = KFold(n_splits=k, random_state=RANDOM_STATE, shuffle=True)
# 
#     for train_index, test_index in kf.split(X):
# #         print("TRAIN:", train_index, "TEST:", test_index)
#         X_train, X_test = X[train_index], X[test_index] 
#         Y_train, Y_test = Y[train_index], Y[test_index] 
#         Y_Pred = model(X_train, Y_train, X_test)
#              
#         auc = roc_auc_score(Y_test, Y_Pred)
#         auc_list.append(auc)
#         
#     auc_mean = mean(auc_list)
#         
#     return auc_mean


