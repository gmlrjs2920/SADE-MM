import numpy as np
import random
import torch
from torchvision.datasets import MNIST, SVHN
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import PIL
from PIL import ImageFilter


aux_available = False

class SaltAndPepperNoise(object):
    def __init__(self,
                 threshold: float = 0.1,
                 imgType: str = "cv2",
                 lowerValue: int = 5,
                 upperValue: int = 250):
        self.threshold = threshold
        self.imgType = imgType
        self.lowerValue = lowerValue  # 255 would be too high
        self.upperValue = upperValue  # 0 would be too low
        super(SaltAndPepperNoise).__init__()

    def __call__(self, img):
        if self.imgType == "PIL":
            img = np.array(img)
        if type(img) != np.ndarray:
            raise TypeError("Image is not of type 'np.ndarray'!")

        if self.threshold != 0:
            random_matrix = np.random.random(img.shape)
            img[random_matrix >= (1 - self.threshold / 2)] = self.upperValue
            img[random_matrix <= self.threshold / 2] = self.lowerValue
        else:
            pass

        if self.imgType == "cv2":
            return img
        elif self.imgType == "PIL":
            return PIL.Image.fromarray(img)


mnist_train = MNIST('./MNIST', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = MNIST('./MNIST', train=False, transform=transforms.ToTensor(), download=True)

svhn_train = SVHN('./SVHN', split='train', transform=transforms.ToTensor(), download=True)
svhn_test = SVHN('./SVHN', split='test', transform=transforms.ToTensor(), download=True)

mnist_label = np.unique(mnist_train.targets)
svhn_label = np.unique(svhn_train.labels)


mnist_train_per_class = []
mnist_test_per_class = []
for i in range(len(mnist_label)):
    idx_train = np.where(mnist_train.targets == mnist_label[i])[0]
    idx_test = np.where(mnist_test.targets == mnist_label[i])[0]
    mnist_train_per_class.append(mnist_train.data[idx_train])
    mnist_test_per_class.append(mnist_test.data[idx_test])

svhn_train_per_class = []
svhn_test_per_class = []
for i in range(len(svhn_label)):
    idx_train = np.where(svhn_train.labels == svhn_label[i])[0]
    idx_test = np.where(svhn_test.labels == svhn_label[i])[0]
    svhn_train_per_class.append(svhn_train.data[idx_train])
    svhn_test_per_class.append(svhn_test.data[idx_test])

mnist_svhn_train_combined = []
mnist_svhn_test_combined = []
for i in range(len(mnist_label)):
    if mnist_train_per_class[i].shape[0] < svhn_train_per_class[i].shape[0]:
        idx_list = random.sample(range(svhn_train_per_class[i].shape[0]), mnist_train_per_class[i].shape[0])
        a = mnist_train_per_class[i].reshape([mnist_train_per_class[i].shape[0], -1])
        b = svhn_train_per_class[i].reshape([svhn_train_per_class[i].shape[0], -1])[idx_list]
        concat = torch.cat([a, torch.tensor(b)], dim=1)
        mnist_svhn_train_combined.append(concat)
    else:
        idx_list = random.sample(range(mnist_train_per_class[i].shape[0]), svhn_train_per_class[i].shape[0])
        a = mnist_train_per_class[i].reshape([mnist_train_per_class[i].shape[0], -1])[idx_list]
        b = svhn_train_per_class[i].reshape([svhn_train_per_class[i].shape[0], -1])
        concat = torch.cat([a, torch.tensor(b)], dim=1)
        mnist_svhn_train_combined.append(concat)

for i in range(len(mnist_label)):
    if mnist_test_per_class[i].shape[0] < svhn_test_per_class[i].shape[0]:
        idx_list = random.sample(range(svhn_test_per_class[i].shape[0]), mnist_test_per_class[i].shape[0])
        a = mnist_test_per_class[i].reshape([mnist_test_per_class[i].shape[0], -1])
        b = svhn_test_per_class[i].reshape([svhn_test_per_class[i].shape[0], -1])[idx_list]
        concat = torch.cat([a, torch.tensor(b)], dim=1)
        mnist_svhn_test_combined.append(concat)
    else:
        idx_list = random.sample(range(mnist_test_per_class[i].shape[0]), svhn_test_per_class[i].shape[0])
        a = mnist_test_per_class[i].reshape([mnist_test_per_class[i].shape[0], -1])[idx_list]
        b = svhn_test_per_class[i].reshape([svhn_test_per_class[i].shape[0], -1])
        concat = torch.cat([a, torch.tensor(b)], dim=1)
        mnist_svhn_test_combined.append(concat)

train_combined = torch.cat(mnist_svhn_train_combined)
test_combined = torch.cat(mnist_svhn_test_combined)

lst_train = []
lst_test = []
for i in range(len(mnist_svhn_train_combined)):
    lst_train += [i] * mnist_svhn_train_combined[i].shape[0]
    lst_test += [i] * mnist_svhn_test_combined[i].shape[0]

label_train_combined = torch.tensor(lst_train)
label_test_combined = torch.tensor(lst_test)

# if aux_available:
#     train_combined = np.array(train_combined)
#     test_combined = np.array(test_combined)
#
#     noise_train = []
#     noise_test = []
#     noise_label_train = []
#     noise_label_test = []
#     noise_level_train = []
#     noise_level_test = []
#
#     noise_level = [0, 0.1, 0.2, 0.3, 0.4]
#     for i in range(len(train_combined)):
#         for j in noise_level:
#             sp_noise = SaltAndPepperNoise(threshold=j)
#             img_mnist = train_combined[i][:784].reshape(28, 28)
#             img_svhn = train_combined[i][784:].reshape(3, 32, 32)
#             noise_mnist = sp_noise(img_mnist)
#             noise_svhn = sp_noise(img_svhn)
#             noise_combined = torch.cat([torch.tensor(noise_mnist.reshape(-1)), torch.tensor(noise_svhn.reshape(-1))])
#             noise_train.append(noise_combined)
#             noise_label_train.append(label_train_combined[i])
#             noise_level_train.append(j * 100)
#
#     for i in range(len(test_combined)):
#         for j in noise_level:
#             sp_noise = SaltAndPepperNoise(threshold=j)
#             img_mnist = test_combined[i][:784].reshape(28, 28)
#             img_svhn = test_combined[i][784:].reshape(3, 32, 32)
#             noise_mnist = sp_noise(img_mnist)
#             noise_svhn = sp_noise(img_svhn)
#             noise_combined = torch.cat([torch.tensor(noise_mnist.reshape(-1)), torch.tensor(noise_svhn.reshape(-1))])
#             noise_test.append(noise_combined)
#             noise_label_test.append(label_test_combined[i])
#             noise_level_test.append(j * 100)
#
#     noise_train = torch.stack(noise_train)
#     noise_test = torch.stack(noise_test)
#     noise_label_train = torch.stack(noise_label_train)
#     noise_label_test = torch.stack(noise_label_test)
#     noise_level_train = torch.tensor(noise_level_train)
#     noise_level_test = torch.tensor(noise_level_test)


class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


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


class AuxDataset(Dataset):
    def __init__(self, data, aux, targets):
        self.data = data
        self.aux = aux
        self.targets = targets

    def __getitem__(self, index: int):
        x = self.data[index]
        z = self.aux[index]
        y = int(self.targets[index])
        return x, z, y

    def __len__(self):
        return len(self.targets)


class TwoCropsDataset(Dataset):
    def __init__(self, data1, data2, targets, transform1, transform2):
        self.data1 = data1
        self.data2 = data2
        self.targets = targets
        self.transform1 = transform1
        self.transform2 = transform2

        num_classes = len(np.unique(targets))
        assert num_classes == 10

        cls_num_list = [0] * num_classes
        for label in targets:
            cls_num_list[label] += 1

        self.cls_num_list = cls_num_list

    def __getitem__(self, index: int):
        x1 = self.data1[index]
        x2 = self.data2[index]
        y = int(self.targets[index])
        if self.transform1:
            x1 = self.transform1(x1)
        if self.transform2:
            x2 = self.transform2(x2)
        return x1, x2, y

    def __len__(self):
        return len(self.targets)


# class TwoCropsDataset_aux(Dataset):
#     def __init__(self, data1, data2, aux, targets, transform1, transform2):
#         self.data1 = data1
#         self.data2 = data2
#         self.aux = aux
#         self.targets = targets
#         self.transform1 = transform1
#         self.transform2 = transform2
#
#         num_classes = len(np.unique(targets))
#         assert num_classes == 10
#
#         cls_num_list = [0] * num_classes
#         for label in targets:
#             cls_num_list[label] += 1
#
#         self.cls_num_list = cls_num_list
#
#     def __getitem__(self, index: int):
#         x1 = self.data1[index]
#         x2 = self.data2[index]
#         z = self.aux[index]
#         y = int(self.targets[index])
#         if self.transform1:
#             x1 = self.transform1(x1)
#         if self.transform2:
#             x2 = self.transform2(x2)
#         return x1, x2, z, y
#
#     def __len__(self):
#         return len(self.targets)


class IMBALANCE_MNIST_SVHN(Dataset):
    cls_num = 10

    def __init__(self, data, targets, imb_type='exp', imb_factor=1, rand_number=0, reverse=False):
        super(IMBALANCE_MNIST_SVHN, self).__init__()
        np.random.seed(rand_number)
        self.data = data
        self.targets = targets
        self.img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor, reverse)
        print(self.img_num_list)
        self.generate_imbalance(self.img_num_list)
        self.reverse = reverse

    def __getitem__(self, index: int):
        x = self.data[index]
        y = int(self.targets[index])
        return x, y

    def __len__(self):
        return len(self.targets)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor, reverse):
        # img_max = len(self.data) / cls_num
        img_max = len(np.where(self.targets == 0)[0])
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                if reverse:
                    num = img_max * (imb_factor ** ((cls_num - 1 - cls_idx) / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))
                else:
                    num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def generate_imbalance(self, img_num_per_cls):
        new_data = []
        new_targets = []
        classes = np.unique(self.targets)
        for i in range(len(classes)):
            idx = np.where(self.targets == classes[i])[0]
            np.random.shuffle(idx)
            selec_idx = idx[:img_num_per_cls[i]]
            new_data.append(self.data[selec_idx])
            new_targets += [i] * img_num_per_cls[i]
        new_data = torch.cat(new_data)
        new_targets = torch.tensor(new_targets)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


# class IMBALANCE_MNIST_SVHN_aux(Dataset):
#     cls_num = 10
#
#     def __init__(self, data, aux, targets, imb_type='exp', imb_factor=0.01, rand_number=0, reverse=False):
#         super(IMBALANCE_MNIST_SVHN_aux, self).__init__()
#         np.random.seed(rand_number)
#         self.data = data
#         self.aux = aux
#         self.targets = targets
#         self.img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor, reverse)
#         print(self.img_num_list)
#         self.generate_imbalance(self.img_num_list)
#         self.reverse = reverse
#
#     def __getitem__(self, index: int):
#         x = self.data[index]
#         z = self.aux[index]
#         y = int(self.targets[index])
#
#         return x, z, y
#
#     def __len__(self):
#         return len(self.targets)
#
#     def get_img_num_per_cls(self, cls_num, imb_type, imb_factor, reverse):
#         # img_max = len(self.data) / cls_num
#         img_max = len(np.where(self.targets == 0)[0])
#         img_num_per_cls = []
#         if imb_type == 'exp':
#             for cls_idx in range(cls_num):
#                 if reverse:
#                     num = img_max * (imb_factor ** ((cls_num - 1 - cls_idx) / (cls_num - 1.0)))
#                     img_num_per_cls.append(int(num))
#                 else:
#                     num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
#                     img_num_per_cls.append(int(num))
#         elif imb_type == 'step':
#             for cls_idx in range(cls_num // 2):
#                 img_num_per_cls.append(int(img_max))
#             for cls_idx in range(cls_num // 2):
#                 img_num_per_cls.append(int(img_max * imb_factor))
#         else:
#             img_num_per_cls.extend([int(img_max)] * cls_num)
#         return img_num_per_cls
#
#     def generate_imbalance(self, img_num_per_cls):
#         new_data = []
#         new_aux = []
#         new_targets = []
#         classes = np.unique(self.targets)
#         for i in range(len(classes)):
#             idx = np.where(self.targets == classes[i])[0]
#             np.random.shuffle(idx)
#             selec_idx = idx[:img_num_per_cls[i]]
#             new_data.append(self.data[selec_idx])
#             new_aux.append(self.aux[selec_idx])
#             new_targets += [i] * img_num_per_cls[i]
#         new_data = torch.cat(new_data)
#         new_aux = torch.cat(new_aux)
#         new_targets = torch.tensor(new_targets)
#         self.data = new_data
#         self.aux = new_aux
#         self.targets = new_targets
#
#     def get_cls_num_list(self):
#         cls_num_list = []
#         for i in range(self.cls_num):
#             cls_num_list.append(self.num_per_cls_dict[i])
#         return cls_num_list


def imbalance_datasets(imb_factor, reverse):
    imbalance_train = IMBALANCE_MNIST_SVHN(train_combined, label_train_combined)
    imbalance_test = IMBALANCE_MNIST_SVHN(test_combined, label_test_combined, imb_factor=imb_factor, reverse=reverse)
    balance_train = CustomDataset(train_combined, label_train_combined)
    balance_test = CustomDataset(test_combined, label_test_combined)
    return imbalance_train, imbalance_test, balance_train, balance_test


# def imbalance_datasets_aux(imb_factor, reverse):
#     imbalance_train = IMBALANCE_MNIST_SVHN_aux(noise_train, noise_level_train, noise_label_train)
#     imbalance_test = IMBALANCE_MNIST_SVHN_aux(noise_test, noise_level_test, noise_label_test, imb_factor=imb_factor, reverse=reverse)
#     balance_test = AuxDataset(noise_test, noise_level_test, noise_label_test)
#     return imbalance_train, imbalance_test, balance_test


def transform_types():
    train_trsfm_mnist = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_train_trsfm_mnist = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(28),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.8),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_trsfm_mnist = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_trsfm_svhn = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4380, 0.4440, 0.4730), (0.1980, 0.2010, 0.1970))
    ])
    test_train_trsfm_svhn = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4380, 0.4440, 0.4730), (0.1980, 0.2010, 0.1970))
    ])
    test_trsfm_svhn = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4380, 0.4440, 0.4730), (0.1980, 0.2010, 0.1970))
    ])
    return train_trsfm_mnist, test_train_trsfm_mnist, test_trsfm_mnist, train_trsfm_svhn, test_train_trsfm_svhn, test_trsfm_svhn


def transform_mnist_svhn(dataset, train_trsfm_mnist, test_trsfm_mnist, train_trsfm_svhn, test_trsfm_svhn, train=True):
    x_mnist = dataset.data[:, :784].reshape(-1, 28, 28)
    x_svhn = dataset.data[:, 784:].reshape(-1, 3, 32, 32)
    if train:
        return TwoCropsDataset(x_mnist, x_svhn, dataset.targets, transform1=train_trsfm_mnist, transform2=train_trsfm_svhn)
    else:
        return TwoCropsDataset(x_mnist, x_svhn, dataset.targets, transform1=test_trsfm_mnist, transform2=test_trsfm_svhn)


def two_crops_test_training(dataset, test_train_trsfm_mnist, test_train_trsfm_svhn):
    x_mnist = dataset.data[:, :784].reshape(-1, 28, 28)
    x_svhn = dataset.data[:, 784:].reshape(-1, 3, 32, 32)
    return TwoCropsDataset(x_mnist, x_svhn, dataset.targets, transform1=TwoCropsTransform(test_train_trsfm_mnist), transform2=TwoCropsTransform(test_train_trsfm_svhn))


# def transform_mnist_svhn_aux(dataset, train_trsfm_mnist, test_trsfm_mnist, train_trsfm_svhn, test_trsfm_svhn, train=True):
#     x_mnist = dataset.data[:, :784].reshape(-1, 28, 28)
#     x_svhn = dataset.data[:, 784:].reshape(-1, 3, 32, 32)
#     if train:
#         return TwoCropsDataset_aux(x_mnist, x_svhn, dataset.aux, dataset.targets, transform1=train_trsfm_mnist, transform2=train_trsfm_svhn)
#     else:
#         return TwoCropsDataset_aux(x_mnist, x_svhn, dataset.aux, dataset.targets, transform1=test_trsfm_mnist, transform2=test_trsfm_svhn)


# def two_crops_test_training_aux(dataset, test_train_trsfm_mnist, test_train_trsfm_svhn):
#     x_mnist = dataset.data[:, :784].reshape(-1, 28, 28)
#     x_svhn = dataset.data[:, 784:].reshape(-1, 3, 32, 32)
#     return TwoCropsDataset_aux(x_mnist, x_svhn, dataset.aux, dataset.targets, transform1=TwoCropsTransform(test_train_trsfm_mnist), transform2=TwoCropsTransform(test_train_trsfm_svhn))


# class IMBALANCE_MNIST_SVHN_Dataloader(DataLoader):
#     def __init__(self, batch_size, shuffle=True, training=True, test_time=False, imb_type='exp', imb_factor=0.01):
#         train_trsfm_mnist = transforms.Compose([
#             transforms.CenterCrop(26),
#             transforms.Resize((28, 28)),
#             transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
#             transforms.RandomRotation(10),
#             transforms.RandomAffine(5),
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])
#         test_trsfm_mnist = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])
#
#         train_trsfm_svhn = transforms.Compose([
#             transforms.Pad(padding=2),
#             transforms.RandomCrop(size=(32, 32)),
#             transforms.ColorJitter(brightness=63. / 255., saturation=[0.5, 1.5], contrast=[0.2, 1.8]),
#             transforms.ToTensor(),
#             transforms.Normalize((0.4380, 0.4440, 0.4730), (0.1980, 0.2010, 0.1970))
#         ])
#
#         test_trsfm_svhn = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.4380, 0.4440, 0.4730), (0.1980, 0.2010, 0.1970))
#         ])
#
#         assert training != test_time
#         if training:
#             dataset = IMBALANCE_MNIST_SVHN(
#                 train_combined, label_train_combined, transform_mnist=train_trsfm_mnist, transform_svhn=train_trsfm_svhn,
#                 imb_type=imb_type, imb_factor=imb_factor)
#             val_dataset = MyDataset(
#                 test_combined, label_test_combined, transform_mnist=test_trsfm_mnist, transform_svhn=test_trsfm_svhn)
#             self.dataset = dataset
#             self.train_dataset = dataset
#             self.val_dataset = val_dataset
#         if test_time:
#             dataset = IMBALANCE_MNIST_SVHN(
#                 train_combined, label_train_combined, transform_mnist=train_trsfm_mnist, transform_svhn=train_trsfm_svhn,
#                 imb_type=imb_type, imb_factor=imb_factor)
#             train_dataset = IMBALANCE_MNIST_SVHN(
#                 train_combined, label_train_combined, transform_mnist=train_trsfm_mnist, transform_svhn=train_trsfm_svhn,
#                 imb_type=imb_type, imb_factor=imb_factor, two_crops_trsfm=True)
#             val_dataset = IMBALANCE_MNIST_SVHN(
#                 test_combined, label_test_combined, transform_mnist=test_trsfm_mnist, transform_svhn=test_trsfm_svhn,
#                 imb_type=imb_type, imb_factor=imb_factor)
#             self.dataset = dataset
#             self.train_dataset = train_dataset
#             self.val_dataset = val_dataset
#
#         num_classes = len(np.unique(dataset.targets))
#         assert num_classes == 10
#
#         cls_num_list = [0] * num_classes
#         for label in dataset.targets:
#             cls_num_list[label] += 1
#
#         self.cls_num_list = cls_num_list
#
#         self.shuffle = shuffle
#         self.init_kwargs = {
#             'batch_size': batch_size,
#         }
#
#         super().__init__(dataset=self.dataset, **self.init_kwargs)
#
#     def train_set(self):
#         return DataLoader(dataset=self.train_dataset, shuffle=True, **self.init_kwargs)
#
#     def test_set(self):
#         return DataLoader(dataset=self.val_dataset, shuffle=False, **self.init_kwargs)