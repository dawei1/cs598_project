# Please add a XXDatasetRootDir that point to the dataset path on your machine and assign it to DatasetRootDir if you
# want to generate other sub datasets.
import Constants
from Dataloader import *
import torch
full_train_dataset, sub_train_dataset, val_train_dataset = get_dataset()

# subset of the entire dataset, this one contains 8192 cases
sub_train_dataset_len = sub_train_dataset.__len__()
eval_set_size = int(sub_train_dataset_len * Constants.eval_set_ratio)
train_set_size = sub_train_dataset_len - eval_set_size
train_dataset, eval_dataset = torch.utils.data.random_split(sub_train_dataset, [train_set_size, eval_set_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Constants.batch_size, shuffle=True)
eval_dataset = torch.utils.data.DataLoader(eval_dataset, batch_size=Constants.batch_size, shuffle=True)