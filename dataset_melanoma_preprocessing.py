import torch
import numpy as np
import pandas as pd
import torchvision.transforms.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import fastai
from fastai.vision import *
from fastai.tabular import *
from image_tabular.dataset import *


data_path = "./data/siim-isic-melanoma-classification/"
train_df = pd.read_csv(data_path + "train.csv")
test_df = pd.read_csv(data_path + "test.csv")

print(len(train_df), len(test_df))
print(train_df.head())
print(train_df["target"].value_counts(normalize=True))

def get_valid_index(df, valid_pct:float=0.2, seed:int=0):
    np.random.seed(seed)
    rand_idx = np.random.permutation(len(df))
    cut = int(valid_pct * len(df))
    val_idx = rand_idx[:cut]
    return val_idx

val_idx = get_valid_index(train_df)
print(len(val_idx))

# image data
tfms = get_transforms(flip_vert=True)
size = 128
image_data = (ImageList.from_df(train_df, path=data_path, cols='image_name', folder='train_128', suffix='.jpg')
              .split_by_idx(val_idx)
              .label_from_df(cols='target')
              .transform(tfms, size=size))
test_image_data = ImageList.from_df(test_df, path=data_path, cols='image_name', folder='test_128', suffix='.jpg')
image_data.add_test(test_image_data)

# tabular data
dep_var = 'target'
cat_names = ['sex', 'anatom_site_general_challenge']
cont_names = ['age_approx']
procs = [FillMissing, Categorify, Normalize]

tab_data = (TabularList.from_df(train_df, path=data_path, cat_names=cat_names, cont_names=cont_names, procs=procs)
            .split_by_idx(val_idx)
            .label_from_df(cols=dep_var))
tab_data.add_test(TabularList.from_df(test_df, cat_names=cat_names, cont_names=cont_names, processor=tab_data.train.x.processor))

# integrated dataset
integrate_train, integrate_valid, integrate_test = get_imagetabdatasets(image_data, tab_data)
train_image = list(integrate_train.images)
train_tab = list(integrate_train.tabs)
print("train list generated")
valid_image = list(integrate_valid.images)
valid_tab = list(integrate_valid.tabs)
print("valid list generated")
test_image = list(integrate_test.images)
test_tab = list(integrate_test.tabs)
print("test list generated")

train_list = []
valid_list = []
test_list = []
for i in range(len(train_image)):
    a = torch.tensor(train_tab[i][1].data).unsqueeze(0)
    b = torch.cat(train_tab[i][0].data)
    c = train_image[i][0].data.reshape(-1)
    train_list.append(torch.cat((a, b, c)))
print("train fin generated")
for i in range(len(valid_image)):
    a = torch.tensor(valid_tab[i][1].data).unsqueeze(0)
    b = torch.cat(valid_tab[i][0].data)
    c = valid_image[i][0].data.reshape(-1)
    valid_list.append(torch.cat((a, b, c)))
print("valid fin generated")
for i in range(len(test_image)):
    a = torch.tensor(test_tab[i][1].data).unsqueeze(0)
    b = torch.cat(test_tab[i][0].data)
    c = test_image[i][0].data.reshape(-1)
    test_list.append(torch.cat((a, b, c)))
print("test fin generated")
train_fin = torch.stack(train_list)
valid_fin = torch.stack(valid_list)
test_fin = torch.stack(test_list)
torch.save(train_fin, 'train_melanoma.pt')
torch.save(valid_fin, 'valid_melanoma.pt')
torch.save(test_fin, 'test_melanoma.pt')


# train_total = torch.load('C:/Users/user/PycharmProjects/image_tabular-master/train_melanoma.pt')
# valid_total = torch.load('C:/Users/user/PycharmProjects/image_tabular-master/valid_melanoma.pt')
# test_total = torch.load('C:/Users/user/PycharmProjects/image_tabular-master/test_melanoma.pt')
# print("Data loaded")
#
# train_tab = train_total[:, 1:5]
# train_label = train_total[:, 0]
# valid_tab = valid_total[:, 1:5]
# valid_label = valid_total[:, 0]
# test_tab = test_total[:, 1:5]
# test_label = test_total[:, 0]
#
#
# class CustomDataset(Dataset):
#     def __init__(self, data, targets):
#         self.data = data
#         self.targets = targets
#
#     def __getitem__(self, index: int):
#         x = self.data[index]
#         y = int(self.targets[index])
#         return x, y
#
#     def __len__(self):
#         return len(self.targets)
#
#
# # tabular model
# class TabularModel(nn.Module):
#     def __init__(self, emb_size, n_cont, emb_p=0, fc_p=0.2):
#         super().__init__()
#         self.embed = nn.ModuleList([nn.Embedding(num_emb, emb_dim) for num_emb, emb_dim in emb_size])
#         self.emb_drop = nn.Dropout(emb_p)
#         self.bn_cont = nn.BatchNorm1d(n_cont)
#         n_emb = sum(e.embedding_dim for e in self.embed)
#         self.n_emb, self.n_cont = n_emb, n_cont
#         self.layers = nn.Sequential(
#             nn.Linear(in_features=12, out_features=8, bias=True),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm1d(8),
#             nn.Dropout(fc_p),
#             nn.Linear(in_features=8, out_features=8, bias=True),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm1d(8),
#             nn.Dropout(fc_p),
#             nn.Linear(in_features=8, out_features=2, bias=True)
#         )
#
#     def forward(self, x_cat, x_cont):
#         if self.n_emb != 0:
#             x = [e(x_cat[:, i]) for i, e in enumerate(self.embed)]
#             x = torch.cat(x, 1)
#             x = self.emb_drop(x)
#         if self.n_cont != 0:
#             x_cont = self.bn_cont(x_cont.reshape(-1, 1))
#             out = torch.cat([x, x_cont], 1)
#         return self.layers(out)
#
#
# def train_epoch(model, train_loader, criterion, optimizer, epoch):
#     model.train()
#     train_loss = 0
#     for batch_idx, data in enumerate(train_loader):
#         data, target = data
#         if torch.cuda.is_available():
#             data, target = data.cuda(), target.cuda()
#         optimizer.zero_grad()
#         out = model(data[:, :3].long(), data[:, 3])
#         loss = criterion(out, target)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#
#     avg_loss = train_loss / (batch_idx + 1)
#     print("Epoch {}: Train Loss = {:.3f}".format(epoch + 1, avg_loss))
#
#
# def test_epoch(model, test_loader, criterion, epoch):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#
#     with torch.no_grad():
#         for batch_idx, (data, target) in enumerate(test_loader):
#             data, target = data.cuda(), target.cuda()
#             out = model(data[:, :3].long(), data[:, 3])
#             loss = criterion(out, target)
#             test_loss += loss.item()
#             _, predicted = out.max(1)
#             total += target.size(0)
#             correct += predicted.eq(target).sum().item()
#
#     acc = 100 * correct / total
#     print("Epoch {}: Accuracy = {:.3f}".format(epoch + 1, acc))
#
#
# def main():
#     batch_size = 128
#     lr = 1e-3
#     n_epochs = 100
#
#     train_dataset = CustomDataset(train_tab, train_label)
#     test_dataset = CustomDataset(test_tab, test_label)
#
#     train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
#     test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
#
#     model = TabularModel(emb_size=[(3, 3), (7, 5), (3, 3)], n_cont=1)
#     model = model.cuda()
#
#     criterion = nn.CrossEntropyLoss().cuda()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#
#     for epoch in range(n_epochs):
#         train_epoch(model, train_loader, criterion, optimizer, epoch)
#         test_epoch(model, test_loader, criterion, epoch)
#
#
# if __name__ == "__main__":
#     main()
