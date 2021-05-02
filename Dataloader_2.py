from torchvision import transforms, io
from torch.utils.data import Dataset
import torch
import pickle
import numpy as np

class XrayDataset(Dataset):

    def __init__(self, dataset, transform_list):
        import Constants
        self.data = dataset
        self.transform_list = transform_list
        self.blank_image = torch.zeros(3, Constants.image_crop_size, Constants.image_crop_size)
        self.image_size = Constants.image_crop_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.data[index]['Lateral_imagePath'] is not None:
            lateral_image = io.read_image(self.data[index]['Lateral_imagePath'],io.image.ImageReadMode.RGB).float() / 255.0
            lateral_image = self.transform_list[0](lateral_image)
        else:
            lateral_image = self.blank_image
        if self.data[index]['Frontal_imagePath'] is not None:
            frontal_image = io.read_image(self.data[index]['Lateral_imagePath'],io.image.ImageReadMode.RGB).float() / 255.0
            frontal_image = self.transform_list[0](frontal_image)
        else:
            frontal_image = self.blank_image
        image = torch.cat((frontal_image, lateral_image), -1)  # concat the image side by side
        image = self.transform_list[1](image)
        label = torch.tensor(self.data[index]['label']).int()
        return image, label


def generate_transform():
    import Constants
    transform_list = []
    if Constants.ImageAugment:
        transform_list = [transforms.Resize(Constants.image_resize_size, transforms.InterpolationMode.BICUBIC),
                          transforms.RandomHorizontalFlip(p=0.5),
                          transforms.RandomCrop(Constants.image_crop_size)]
    else:
        transform_list = [transforms.Resize(Constants.image_resize_size, transforms.InterpolationMode.BICUBIC),
                          transforms.CenterCrop(Constants.image_crop_size)]
    # Resize the image back to a square.
    final_transform_list = [transforms.Resize((Constants.image_crop_size, Constants.image_crop_size), transforms.InterpolationMode.BICUBIC),
                            # Pretrained model expects these mean and std values.
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(transform_list), transforms.Compose(final_transform_list)


def get_dataset():
    import Constants
    transform_list = generate_transform()
    with open(Constants.ParsedDatasetPath_wide, 'rb') as file_handler:
        full_train_dataset = pickle.load(file_handler)
    with open(Constants.ParsedSubsetPath_wide, 'rb') as file_handler:
        sub_train_dataset = pickle.load(file_handler)
    with open(Constants.ParsedValidsetPath_wide, 'rb') as file_handler:
        val_train_dataset = pickle.load(file_handler)
    full_train_dataset = XrayDataset(full_train_dataset, transform_list)
    sub_train_dataset = XrayDataset(sub_train_dataset, transform_list)
    val_train_dataset = XrayDataset(val_train_dataset, transform_list)
    return full_train_dataset, sub_train_dataset, val_train_dataset
