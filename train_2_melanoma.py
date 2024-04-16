import argparse
import os
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score

from model_melanoma import ConvNet_MM
from dataset_melanoma import imbalance_datasets, transform_types, transform_image

from PIL import ImageFilter


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_experts", default=3, type=int, help="number of skilled experts")
    parser.add_argument("--num_mod", default=2, type=int, help="number of modalities")
    parser.add_argument("--num_classes", default=2, type=int, help="number of classes")

    parser.add_argument("--n_epochs", default=30, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)

    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay for SGD")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum for SGD")
    parser.add_argument("--num_workers", default=4, type=int, help="number of workers for Dataloader")

    parser.add_argument("--print_freq", default=10, type=int, help="print frequency")

    args = parser.parse_args()

    return args


def test_training(model, train_loader, aggregation_weight, optimizer, epoch, args):
    model.eval()
    train_loss = 0

    for idx, data in enumerate(train_loader):
        data1, data2, target = data
        if torch.cuda.is_available():
            data1, data2, target = data1.cuda(), data2.cuda(), target.cuda()

        optimizer.zero_grad()

        output = model.test_training(data1, data2)
        expert1_logits_output = output[:,0,:]
        expert2_logits_output = output[:,1,:]
        expert3_logits_output = output[:,2,:]

        aggregation_softmax = torch.nn.functional.softmax(aggregation_weight) # softmax for normalization
        aggregation_output = aggregation_softmax[0].cuda() * expert1_logits_output + \
                              aggregation_softmax[1].cuda() * expert2_logits_output + \
                              aggregation_softmax[2].cuda() * expert3_logits_output

        # output_label = aggregation_output.argmax(dim=1)
        # softmax_aggregation_output = F.softmax(aggregation_output, dim=1)

        criterion = nn.CrossEntropyLoss().cuda()
        loss = criterion(aggregation_output, target)

        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        if idx % args.print_freq == 0:
            current = idx
            total = len(train_loader)
            ratio = 100.0 * current / total
            print('Train Stage 2 Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.4f}'.format(epoch + 1, current, total, ratio, loss.item()))

    aggregation_softmax = torch.nn.functional.softmax(aggregation_weight, dim=0).detach().cpu().numpy()

    return np.round(aggregation_softmax[0], decimals=2), \
           np.round(aggregation_softmax[1], decimals=2), \
           np.round(aggregation_softmax[2], decimals=2)


def test_validation(model, valid_loader, num_classes, aggregation_weight):
    model.eval()
    # aggregation_weight.requires_grad = False
    conf_matrix = torch.zeros(num_classes, num_classes).cuda()

    targets = []
    outputs = []
    with torch.no_grad():
        for idx, (data1, data2, target) in enumerate(valid_loader):
            if torch.cuda.is_available():
                data1, data2, target = data1.cuda(), data2.cuda(), target.cuda()

            output = model.test_training(data1, data2)

            expert1_logits_output = output[:,0,:]
            expert2_logits_output = output[:,1,:]
            expert3_logits_output = output[:,2,:]
            aggregation_softmax = torch.nn.functional.softmax(aggregation_weight) # softmax for normalization
            aggregation_output = aggregation_softmax[0] * expert1_logits_output + \
                                 aggregation_softmax[1] * expert2_logits_output + \
                                 aggregation_softmax[2] * expert3_logits_output

            output_label = aggregation_output.argmax(dim=1)
            targets.append(target.cpu())
            outputs.append(output_label.cpu())
            for t, p in zip(target.view(-1), output_label.view(-1)):
                conf_matrix[t.long(), p.long()] += 1

    acc_per_class = conf_matrix.diag().sum() / conf_matrix.sum()
    acc = acc_per_class.cpu().numpy()
    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs)

    return np.round(acc * 100, decimals=2),\
           roc_auc_score(targets, outputs),\
           f1_score(targets, outputs, average='weighted'),\
           f1_score(targets, outputs, average='macro')


def main():
    args = parse_args()
    model = ConvNet_MM(num_mod=args.num_mod, num_experts=args.num_experts, num_classes=args.num_classes,
                       hidden_dim_tab=8, hidden_dim_img=256, hidden_dim_clf=32, p=0.2)
    n_train_epochs = [60, 80, 100]
    opt = "Adam"

    if not os.path.isdir("result_melanoma"):
        os.mkdir("result_melanoma")
    # f1 = open('result_melanoma/aggregation_weight_melanoma_v2.txt', 'w')
    # f1.close()
    # f2 = open('result_melanoma/accuracy_melanoma_v2.txt', 'w')
    # f2.close()
    # f3 = open('result_melanoma/train_info_melanoma_v2.txt', 'w')
    # f3.close()

    f4 = open('result_melanoma/std.txt', 'w')
    f4.close()

    for n_train_epoch in n_train_epochs:
        print("Loading checkpoint..")
        assert os.path.isdir("checkpoint_melanoma"), "Error: no checkpoint directory found!"
        checkpoint_model = torch.load("checkpoint_melanoma/train_melanoma_v4_{}.pth".format(n_train_epoch))
        model.load_state_dict(checkpoint_model)

        if torch.cuda.is_available():
            model = model.cuda()

        acc_list = []
        f1_list = []
        w1_list = []
        w2_list = []
        w3_list = []

        for _ in range(5):
            imbalance_train, imbalance_test = imbalance_datasets()
            trsfm = transform_types()
            train_dataset = transform_image(imbalance_train, trsfm, trsfm, train=True)
            valid_dataset = transform_image(imbalance_test, trsfm, trsfm, train=False)
            train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, drop_last=True)
            valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size, drop_last=True)

            aggregation_weight = torch.nn.Parameter(torch.FloatTensor(3), requires_grad=True)
            aggregation_weight.data.fill_(1/3)

            if opt == "SGD":
                optimizer = SGD([aggregation_weight], lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum,
                            nesterov=True)
            if opt == 'Adam':
                optimizer = Adam([aggregation_weight], lr=args.lr, weight_decay=args.weight_decay)

            line_break = "\n" + "-----------------------------------------------------------------------------\n"
            # with open('result_melanoma/aggregation_weight_melanoma_v2.txt', 'a') as f:
            #     f.write(line_break)
            # f.close()
            # with open('result_melanoma/accuracy_melanoma_v2.txt', 'a') as f:
            #     f.write(line_break)
            # f.close()
            # with open('result_melanoma/train_info_melanoma_v2.txt', 'a') as f:
            #     f.write(line_break)
            # f.close()

            for epoch in range(args.n_epochs):
                start_time = time.time()
                weight = test_training(model, train_loader, aggregation_weight, optimizer, epoch, args)
                print('Time for one epoch: {:.4f}sec'.format(time.time() - start_time))
                if weight[0] < 0.05 or weight[1] < 0.05 or weight[2] < 0.05:
                    break

                # if (epoch + 1) % 30 == 0:
            weight_result = "\n" + \
                            " - Aggregation weight: Expert 1 is {0:.2f}, Expert 2 is {1:.2f}, Expert 3 is {2:.2f}\n".\
                                format(weight[0], weight[1], weight[2])
            print(weight_result)
            # with open('result_melanoma/aggregation_weight_melanoma_v2.txt', 'a') as f:
            #     f.write(weight_result)
            # f.close()
            w1_list.append(weight[0])
            w2_list.append(weight[1])
            w3_list.append(weight[2])

            acc, roc_auc, f1_weighted, f1_macro = test_validation(model, valid_loader, args.num_classes,
                                                                  aggregation_weight)
            print(acc)
            acc_result = "\n" + \
                         " - Overall accuracy: {:.5f}, Overall ROC-AUC: {:.5f}, " \
                         "Overall F1 weighted: {:.5f}, Overall F1 macro: {:.5f}\n".format(acc, roc_auc, f1_weighted,
                                                                                          f1_macro)
            print(acc_result)
            # with open('result_melanoma/accuracy_melanoma_v2.txt', 'a') as f:
            #     f.write(acc_result)
            # f.close()
            acc_list.append(acc)
            f1_list.append(f1_macro)

            train_info = "\n" + \
                         " - Stage 1: Epoch {} / Stage 2: Epoch {}, {} lr {}\n".format(n_train_epoch, epoch,
                                                                                       opt, args.lr)
            # with open('result_melanoma/train_info_melanoma_v2.txt', 'a') as f:
            #     f.write(train_info)
            # f.close()

        fin_result = "\n" + " - {:.4f}, {:.4f}, {:.2f}, {:.2f}, {:.2f}\n".format(
            np.std(acc_list), np.std(f1_list), np.std(w1_list), np.std(w2_list), np.std(w3_list))
        with open('result_melanoma/std.txt', 'a') as f:
            f.write(fin_result)
        f.close()


if __name__ == '__main__':
    main()