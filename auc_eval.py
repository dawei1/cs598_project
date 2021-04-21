from sklearn.model_selection import KFold
from sklearn.metrics import *
import numpy as np
from numpy import mean
import torch
import utils

# Random Default from HW
RANDOM_STATE = 545510477

#input: X = training data and Y = corresponding labels, model = trained model
#output: accuracy, auc
def auc_eval(model, X, Y, k=5):
    
    auc_list = []
    
    kf = KFold(n_splits=k, random_state=RANDOM_STATE, shuffle=True)

    for train_index, test_index in kf.split(X):
#         print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index] 
        Y_train, Y_test = Y[train_index], Y[test_index] 
        Y_Pred = model(X_train, Y_train, X_test)
             
        auc = roc_auc_score(Y_test, Y_Pred)
        auc_list.append(auc)
        
    auc_mean = mean(auc_list)
        
    return auc_mean