import numpy as np
import pickle
import random
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import PIL
from PIL import ImageFilter


train_total = torch.load('C:/Users/user/PycharmProjects/SADE-MM/train_melanoma.pt')
valid_total = torch.load('C:/Users/user/PycharmProjects/SADE-MM/valid_melanoma.pt')
print("Data loaded")

train_image = train_total[:, 5:]
train_tab = train_total[:, 1:5]
train_data = train_total[:, 1:]
train_label = train_total[:, 0]
valid_image = valid_total[:, 5:]
valid_tab = valid_total[:, 1:5]
valid_data = valid_total[:, 1:]
valid_label = valid_total[:, 0]

train_min_cls_num = int(sum(train_label))
train_max_cls_num = len(train_label) - train_min_cls_num
train_min_pc = 100 * train_min_cls_num / (train_min_cls_num + train_max_cls_num)
train_max_pc = 100 * train_max_cls_num / (train_min_cls_num + train_max_cls_num)
valid_min_cls_num = int(sum(valid_label))
valid_max_cls_num = len(valid_label) - valid_min_cls_num
valid_min_pc = 100 * valid_min_cls_num / (valid_min_cls_num + valid_max_cls_num)
valid_max_pc = 100 * valid_max_cls_num / (valid_min_cls_num + valid_max_cls_num)

print("Train data = {} ({:.2f}%) : {} ({:.2f}%)".format(train_max_cls_num, train_max_pc, train_min_cls_num, train_min_pc))
print("Test data = {} ({:.2f}%) : {} ({:.2f}%)".format(valid_max_cls_num, valid_max_pc, valid_min_cls_num, valid_min_pc))
print("Train data imbalance ratio (max/min) = {:.2f}".format(train_max_cls_num / train_min_cls_num))
print("Test data imbalance ratio (max/min) = {:.2f}".format(valid_max_cls_num / valid_min_cls_num))


class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index: int):
        x = self.data[index]
        y = int(self.targets[index])
        return x, y

    def __len__(self):
        return len(self.targets)


class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class TwoCropsDataset(Dataset):
    def __init__(self, data1, data2, targets, transform):
        self.data1 = data1
        self.data2 = data2
        self.targets = targets
        self.transform = transform

        num_classes = len(np.unique(targets))
        assert num_classes == 2

        cls_num_list = [0] * num_classes
        for label in targets:
            cls_num_list[int(label)] += 1

        self.cls_num_list = cls_num_list

    def __getitem__(self, index: int):
        x1 = self.data1[index]
        x2 = self.data2[index]
        y = int(self.targets[index])
        if self.transform:
            x2 = self.transform(x2)
        return x1, x2, y

    def __len__(self):
        return len(self.targets)


class ImageTabular(Dataset):
    cls_num = 2

    def __init__(self, data, targets):
        super(ImageTabular, self).__init__()
        self.data = data
        self.targets = targets
        self.img_num_list = self.get_img_num_per_cls(self.cls_num)

    def __getitem__(self, index: int):
        x = self.data[index]
        y = int(self.targets[index])
        return x, y

    def __len__(self):
        return len(self.targets)

    def get_img_num_per_cls(self, cls_num):
        img_num_per_cls = []
        for i in range(cls_num):
            img_num_per_cls.append(int(sum(self.targets == i)))
        return img_num_per_cls


def imbalance_datasets():
    imbalance_train = ImageTabular(train_data, train_label)
    imbalance_test = ImageTabular(valid_data, valid_label)
    return imbalance_train, imbalance_test


def transform_types():
    test_trsfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    return test_trsfm

def transform_test_training():
    test_train_trsfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(128),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    return test_train_trsfm

def transform_image(dataset, trsfm_train, trsfm_test, train=True):
    tab = dataset.data[:, :4]
    img = dataset.data[:, 4:].reshape(-1, 3, 128, 128)
    if train:
        return TwoCropsDataset(tab, img, dataset.targets, transform=trsfm_train)
    else:
        return TwoCropsDataset(tab, img, dataset.targets, transform=trsfm_test)

def two_crops_test_training(dataset, test_train_trsfm):
    tab = dataset.data[:, :4]
    img = dataset.data[:, 4:].reshape(-1, 3, 128, 128)
    return TwoCropsDataset(tab, img, dataset.targets, transform=TwoCropsTransform(test_train_trsfm))