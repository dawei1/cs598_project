'''
This module performs an evaluation of the model with cross-validation, including
GPU integration, with the intent of determining the best hyperparameters.
The logic outputs metrics for accuracy and AUC. The import for the dataloader
can be adjusted to change which images are used (frontal vs frontal-lateral
concatenated). The code can also be adjusted below to use either the full dataset
or a portion of the dataset.
'''
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import Constants
# CHANGE HERE for using different style of images
from frontal.Dataloader import *
#from frontal_lateral_concat.Dataloader import *
from full_model import *
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader



full_train_dataset, sub_train_dataset, val_train_dataset = get_dataset()

# CHANGE HERE for using a subset of the data
sub_train_dataset_len = sub_train_dataset.__len__()
eval_set_size = int(sub_train_dataset_len * Constants.eval_set_ratio)
train_set_size = sub_train_dataset_len - eval_set_size
train_dataset, eval_dataset = torch.utils.data.random_split(sub_train_dataset, [train_set_size, eval_set_size])
#full_train_dataset_len = full_train_dataset.__len__()
#eval_set_size = int(full_train_dataset_len * Constants.eval_set_ratio)
#train_set_size = full_train_dataset_len - eval_set_size
#train_dataset, eval_dataset = torch.utils.data.random_split(full_train_dataset, [train_set_size, eval_set_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Constants.batch_size, shuffle=True, num_workers=Constants.num_of_workers)
val_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=Constants.batch_size, shuffle=True, num_workers=Constants.num_of_workers)

seed = 24
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
RANDOM_STATE = 545510477

X = None
Y = None

X = np.array([i[0].numpy() for i in sub_train_dataset])
Y = np.array([i[1].numpy() for i in sub_train_dataset])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
p_values = [16, 14, 12, 10, 8, 6, 4]
acc_list = {p:[] for p in p_values}
auc_list = {p:[] for p in p_values}

kf = KFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)
for train_index, test_index in kf.split(sub_train_dataset):
    print('starting next fold')
    train_image, test_image = X[train_index], X[test_index]
    train_label, test_label = Y[train_index], Y[test_index]
    tensor_x = torch.Tensor(train_image)
    tensor_y = torch.Tensor(train_label)
    train_dataset = TensorDataset(tensor_x, tensor_y)
    train_dataloder = DataLoader(train_dataset, batch_size=Constants.batch_size, shuffle=True, num_workers=Constants.num_of_workers)

    for p in p_values:
        print('starting next p value')
        start_model = PatchingModel(resnet_out_height_width, p, c_prime).to(device)
        start_optimizer = torch.optim.Adam(start_model.parameters(), lr=0.001, weight_decay=0.1)
        trained_model = train_model(train_dataloder, n_epoch=30, model=start_model, optimizer=start_optimizer)
        trained_model.eval()
        tensor_x_test = torch.Tensor(test_image)
        tensor_y_test = torch.Tensor(test_label)
        test_dataset = TensorDataset(tensor_x_test, tensor_y_test)
        test_dataloader = DataLoader(test_dataset, batch_size=Constants.batch_size, shuffle=True, num_workers=Constants.num_of_workers)
        y_hats = []
        y_Preds = []
        targets = []
        for data, target in test_dataloader:
            data = data.to(device)
            y_hat = trained_model(data)
            y_hat = y_hat.to(torch.device('cpu'))
            y_hat = 1 - y_hat
            y_hat = (y_hat * 0.02) + 0.98
            y_hat = torch.prod(y_hat, 3)
            y_hat = torch.prod(y_hat, 2)
            y_hat = 1 - y_hat
            y_Pred = torch.where(y_hat>=0.5, torch.tensor([1]), torch.tensor([0]))

            y_hats.extend(y_hat.detach().numpy())
            y_Preds.extend(y_Pred.detach().numpy())
            targets.extend(target.detach().numpy())

            del data
            torch.cuda.empty_cache()

        y_hats = np.array(y_hats)
        y_Preds = np.array(y_Preds)
        targets = np.array(targets)
        acc = np.array([accuracy_score(targets[:, idx], y_Preds[:, idx]) for idx in range(targets.shape[1])])
        print(acc)
        acc_list[p].append(acc)

        auc = np.array([roc_auc_score(targets[:, idx], y_hats[:, idx]) for idx in range(targets.shape[1])])
        print(auc)
        auc_list[p].append(auc)

        del start_model
        del trained_model
        torch.cuda.empty_cache()

acc_mean = {p: np.mean(acc_list, axis=0) for p, acc_list in acc_list.items()}
auc_mean = {p: np.mean(auc_list, axis=0) for p, auc_list in auc_list.items()}
print(acc_mean, auc_mean)

#print(np.mean(acc_mean), np.mean(auc_mean))
