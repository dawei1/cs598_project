from sklearn.model_selection import KFold
from sklearn.metrics import *
import numpy as np
from numpy import mean
import torch
import utils
import Dataloader


# Random Default from HW
RANDOM_STATE = 545510477

def eval_model(model, val_dataloader):
    
    model.eval()
    y_pred = torch.LongTensor()
    y_true = torch.LongTensor()

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


