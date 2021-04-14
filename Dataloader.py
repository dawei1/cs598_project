from torchvision import transforms, io
from torch.utils.data import Dataset
import torch
import pickle
import numpy as np


class XrayDataset(Dataset):

    def __init__(self, dataset, transform=None):
        self.data = dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_name = self.data[index]['imagePath']
        image = io.read_image(self.data[index]['imagePath'], io.image.ImageReadMode.UNCHANGED).float()
        label = torch.tensor(self.data[index]['label']).float()
        if self.transform:
            image = self.transform(image)
        return image, label


def generate_transform():
    import Constants
    transform_list = []
    if(Constants.ImageAugment):
        transform_list = [transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
                          transforms.RandomHorizontalFlip(p=0.5),
                          transforms.RandomCrop(224)]
    else:
        transform_list = [transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
                          transforms.CenterCrop(224)]
    return transforms.Compose(transform_list)


def get_dataset():
    import Constants
    transform_list = generate_transform()
    with open(Constants.ParsedDatasetPath, 'rb') as file_handler:
        full_train_dataset = pickle.load(file_handler)
    with open(Constants.ParsedSubsetPath, 'rb') as file_handler:
        sub_train_dataset = pickle.load(file_handler)
    with open(Constants.ParsedValidsetPath, 'rb') as file_handler:
        val_train_dataset = pickle.load(file_handler)
    full_train_dataset = XrayDataset(full_train_dataset, transform_list)
    sub_train_dataset = XrayDataset(sub_train_dataset, transform_list)
    val_train_dataset = XrayDataset(val_train_dataset, transform_list)
    return full_train_dataset, sub_train_dataset, val_train_dataset
