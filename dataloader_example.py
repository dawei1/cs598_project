'''
This module is an example of using the dataloader, and makes it easy to check
that the output of the dataloader is as expected.
'''
import Constants
from frontal_lateral_concat.Dataloader import *
import torch
import torchvision
import matplotlib.pyplot as plt

def imshow(img, title):
    npimg = img.numpy()
    plt.figure(figsize=(7, 7))
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

def show_batch_images(dataloader):
    images, labels = next(iter(dataloader))
    img = torchvision.utils.make_grid(images, padding=25)
    imshow(img, title=["NORMAL"])

full_train_dataset, sub_train_dataset, val_train_dataset = get_dataset()

# subset of the entire dataset, this one contains 8192 cases
sub_train_dataset_len = sub_train_dataset.__len__()
eval_set_size = int(sub_train_dataset_len * Constants.eval_set_ratio)
train_set_size = sub_train_dataset_len - eval_set_size
train_dataset, eval_dataset = torch.utils.data.random_split(sub_train_dataset, [train_set_size, eval_set_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Constants.batch_size, shuffle=True)
eval_dataset = torch.utils.data.DataLoader(eval_dataset, batch_size=Constants.batch_size, shuffle=True)
for i in range(2):
    show_batch_images(train_loader)
