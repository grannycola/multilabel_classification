import os
import numpy as np

import torch
import yaml

from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from src.data.transformations import get_transform, get_val_transform

torch.manual_seed(42)


def get_default_from_yaml(param_name):
    with open('../../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    default_value = config.get(param_name, 0)
    return default_value


# Evaluate class disbalance to obtain inverse weights
def evaluate_disbalance(dataloader):
    total_samples = len(dataloader.dataset)
    class_frequencies = torch.zeros(3)

    for data_batch, labels_batch in dataloader:
        class_frequencies += labels_batch.sum(dim=0)

    class_weights = total_samples / (class_frequencies + 1e-10)
    return torch.Tensor(class_weights)


class SubsetTransformation(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        image, labels = self.subset[index]
        image = np.asarray(image).astype(np.uint8)
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed(image)
        return image, labels

    def __len__(self):
        return len(self.subset)


class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = os.path.join(self.image_dir, self.image_files[idx])

        image = Image.open(image_file).convert('RGB')
        image = np.asarray(image).astype(np.uint8)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        # Extract class labels from file name (assuming labels are encoded in the file name)
        file_name = self.image_files[idx]
        labels = file_name.split('_')[-1]
        labels = torch.tensor([int(label) for label in labels[:3]], dtype=torch.int64)

        return image, labels


def create_dataloaders(image_dir=None,
                       batch_size=None,
                       num_classes=None,
                       val_proportion=get_default_from_yaml('val_proportion'),
                       test_proportion=get_default_from_yaml('test_proportion'),
                       transform=get_transform()):

    if val_proportion + test_proportion >= 1:
        raise Exception("Sum of val and test proportions should be less than 1")

    print('Making new dataloader...')
    dataset = CustomDataset(image_dir, transform)

    test_size = int(test_proportion * len(dataset))
    val_size = int(val_proportion * len(dataset))
    train_size = len(dataset) - test_size - val_size

    train_dataset, test_val_dataset = random_split(dataset, [train_size, test_size + val_size])
    test_dataset, val_dataset = random_split(test_val_dataset, [test_size, val_size])

    val_dataset.dataset.transform = get_val_transform()
    test_dataset.dataset.transform = get_val_transform()

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  num_workers=get_default_from_yaml('num_workers'),
                                  shuffle=True)  # Shuffle the training data

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                num_workers=get_default_from_yaml('num_workers'))

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 num_workers=get_default_from_yaml('num_workers'))

    return train_dataloader, val_dataloader, test_dataloader
