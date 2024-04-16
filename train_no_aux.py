import argparse
import os
import numpy as np
import torch
import time
from torch.optim import SGD, Adam
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from model_no_aux import ConvNet_MM
from loss import DiverseExpertLoss, BalancedLoss
from dataset import imbalance_datasets, transform_types, transform_mnist_svhn


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_classes", default=10, type=int, help="number of classes")
    parser.add_argument("--num_mod", default=2, type=int, help="number of modalities")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--imb_factor", default=1, type=float)
    parser.add_argument("--num_experts", default=3, type=int, help="number of skilled experts")
    parser.add_argument("--tau", default=2.0, type=float, help="weight of inverse prior in the loss")

    parser.add_argument("--n_epochs", default=100, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--warmup_epochs", default=5, type=int)

    parser.add_argument("--lr_decay_rate", type=float, default=0.1, help="learning rate decay rate")
    parser.add_argument("--lr_decay_epochs", type=list, default=[30, 40, 50], help="at what epoch to decay lr with lr_decay_rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay for SGD")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum for SGD")
    parser.add_argument("--num_workers", default=4, type=int, help="number of workers for Dataloader")

    parser.add_argument("--save_freq", default=10, type=int, help="save frequency")

    args = parser.parse_args()

    return args


def adjust_learning_rate(optimizer, epoch, args): # learning rate decayed by a factor of 10 at the 160th and 180th epoch
    lr = args.lr
    warmup_epochs = args.warmup_epochs

    n_steps_passed = np.sum(epoch > np.asarray(args.lr_decay_epochs))
    if n_steps_passed > 0:
        lr = lr * (args.lr_decay_rate ** n_steps_passed)

    if epoch < warmup_epochs:
        lr = lr * float(1 + epoch) / warmup_epochs

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train_epoch(train_loader, model, optimizer, criterion, cls_num_list):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data1, data2, target = data
        if torch.cuda.is_available():
            data1, data2, target = data1.cuda(), data2.cuda(), target.cuda()
        optimizer.zero_grad()
        extra_info = {}
        out = model(data1, data2, cls_num_list, target)
        if isinstance(out, dict):
            logits = out["logits"]
            extra_info.update({"logits": logits.transpose(0, 1)})

            output = out["output"]
            final_loss = out["loss"]

        loss_cls = criterion(output_logits=output, target=target, extra_info=extra_info)
        loss = loss_cls + final_loss
        loss = torch.mean(loss)

        loss.backward()
        optimizer.step()


def test_epoch(valid_loader, model, epoch):
    model.eval()
    acc = 0
    f1_weighted = 0
    f1_macro = 0
    with torch.no_grad():
        for batch_idx, (data1, data2, target) in enumerate(valid_loader):
            if torch.cuda.is_available():
                data1, data2 = data1.cuda(), data2.cuda()
            logit = model.infer(data1, data2)
            prob = F.softmax(logit, dim=1).data.cpu().numpy()
            acc += accuracy_score(target, prob.argmax(1))
            f1_weighted += f1_score(target, prob.argmax(1), average='weighted')
            f1_macro += f1_score(target, prob.argmax(1), average='macro')

    avg_acc = acc / (batch_idx + 1)
    avg_f1_weighted = f1_weighted / (batch_idx + 1)
    avg_f1_macro = f1_macro / (batch_idx + 1)
    print("Test ACC: {:.5f}".format(avg_acc))
    print("Test F1 weighted: {:.5f}".format(avg_f1_weighted))
    print("Test F1 macro: {:.5f}".format(avg_f1_macro))
    return prob


def main():
    args = parse_args()
    imb = False
    imbalance_train, imbalance_test, balance_train, balance_test = imbalance_datasets(args.imb_factor, False)
    train_trsfm_mnist, _, test_trsfm_mnist, train_trsfm_svhn, _, test_trsfm_svhn = transform_types()
    if imb == True:
        train_dataset = transform_mnist_svhn(imbalance_train, test_trsfm_mnist, test_trsfm_mnist, test_trsfm_svhn, test_trsfm_svhn, train=True)
    if imb == False:
        train_dataset = transform_mnist_svhn(balance_train, test_trsfm_mnist, test_trsfm_mnist, test_trsfm_svhn, test_trsfm_svhn, train=True)
    valid_dataset = transform_mnist_svhn(balance_test, test_trsfm_mnist, test_trsfm_mnist, test_trsfm_svhn, test_trsfm_svhn, train=False)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, drop_last=True)
    valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size, drop_last=True)

    model = ConvNet_MM(num_mod=args.num_mod, num_experts=args.num_experts, num_classes=args.num_classes)

    criterion = DiverseExpertLoss(cls_num_list=train_dataset.cls_num_list, tau=args.tau)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    print("\nTraining...")
    for epoch in range(args.n_epochs):
        start_time = time.time()
        train_epoch(train_loader, model, optimizer, criterion, cls_num_list=train_dataset.cls_num_list)
        print('Time for one epoch: {:.4f}sec'.format(time.time() - start_time))
        print("\nEpoch {:d}".format(epoch + 1))
        test_epoch(valid_loader, model, epoch)

        if (epoch + 1) % args.save_freq == 0:
            if not os.path.isdir("checkpoint"):
                os.mkdir("checkpoint")
            torch.save(model.state_dict(), "./checkpoint/train_{}_{}.pth".format(args.imb_factor, epoch + 1))


if __name__ == '__main__':
    main()