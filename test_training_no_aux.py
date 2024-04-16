import argparse
import os
import numpy as np
import random
import time
import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from model_no_aux import ConvNet_MM
from dataset import imbalance_datasets, transform_types, transform_mnist_svhn, two_crops_test_training

from PIL import ImageFilter


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--imb_factor", default=1, type=float)
    parser.add_argument("--num_experts", default=3, type=int, help="number of skilled experts")
    parser.add_argument("--num_mod", default=2, type=int, help="number of modalities")
    parser.add_argument("--num_classes", default=10, type=int, help="number of classes")

    parser.add_argument("--n_epochs", default=10, type=int)
    parser.add_argument("--lr", default=0.1, type=float)

    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay for SGD")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum for SGD")
    parser.add_argument("--num_workers", default=4, type=int, help="number of workers for Dataloader")

    parser.add_argument("--print_freq", default=10, type=int, help="print frequency")

    args = parser.parse_args()

    return args


def test_training(model, train_loader, aggregation_weight, optimizer, epoch, args):
    model.eval()
    train_loss = 0

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    for idx, (data1, data2, _) in enumerate(train_loader):
        if torch.cuda.is_available():
            data1[0] = data1[0].cuda()
            data1[1] = data1[1].cuda()
            data2[0] = data2[0].cuda()
            data2[1] = data2[1].cuda()

        optimizer.zero_grad()

        output0 = model.test_training(data1[0], data2[0])
        output1 = model.test_training(data1[1], data2[1])
        expert1_logits_output0 = output0[:,0,:]
        expert2_logits_output0 = output0[:,1,:]
        expert3_logits_output0 = output0[:,2,:]
        expert1_logits_output1 = output1[:,0,:]
        expert2_logits_output1 = output1[:,1,:]
        expert3_logits_output1 = output1[:,2,:]

        aggregation_softmax = torch.nn.functional.softmax(aggregation_weight) # softmax for normalization
        aggregation_output0 = aggregation_softmax[0].cuda() * expert1_logits_output0 + \
                              aggregation_softmax[1].cuda() * expert2_logits_output0 + \
                              aggregation_softmax[2].cuda() * expert3_logits_output0
        aggregation_output1 = aggregation_softmax[0].cuda() * expert1_logits_output1 + \
                              aggregation_softmax[1].cuda() * expert2_logits_output1 + \
                              aggregation_softmax[2].cuda() * expert3_logits_output1
        softmax_aggregation_output0 = F.softmax(aggregation_output0, dim=1)
        softmax_aggregation_output1 = F.softmax(aggregation_output1, dim=1)

        # SSL loss: similarity maxmization
        cos_similarity = cos(softmax_aggregation_output0, softmax_aggregation_output1).mean()
        loss = -cos_similarity

        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        if idx % args.print_freq == 0:
            current = idx
            total = len(train_loader)
            ratio = 100.0 * current / total
            print('Test-train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(epoch + 1, current, total, ratio, loss.item()))

    aggregation_softmax = torch.nn.functional.softmax(aggregation_weight, dim=0).detach().cpu().numpy()

    return np.round(aggregation_softmax[0], decimals=2), \
           np.round(aggregation_softmax[1], decimals=2), \
           np.round(aggregation_softmax[2], decimals=2)


def test_validation(model, valid_loader, num_classes, aggregation_weight):
    model.eval()
    aggregation_weight.requires_grad = False
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

    return np.round(acc * 100, decimals=2), f1_score(targets, outputs, average='weighted'), f1_score(targets, outputs, average='macro')


def main():
    args = parse_args()
    model = ConvNet_MM(num_mod=args.num_mod, num_experts=args.num_experts, num_classes=args.num_classes)
    n_train_epochs = 100
    distribution = {
        'Uniform': (1, False),
        'Forward50': (0.02, False),
        'Forward25': (0.04, False),
        'Forward10': (0.1, False),
        'Forward5': (0.2, False),
        'Forward2': (0.5, False),
        'Backward50': (0.02, True),
        'Backward25': (0.04, True),
        'Backward10': (0.1, True),
        'Backward5': (0.2, True),
        'Backward2': (0.5, True),
    }

    print("Loading checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    checkpoint_model = torch.load("checkpoint/train_{}_{}.pth".format(args.imb_factor, n_train_epochs))
    model.load_state_dict(checkpoint_model)

    if torch.cuda.is_available():
        model = model.cuda()

    if not os.path.isdir("result"):
        os.mkdir("result")
    # f1 = open('result/aggregation_weight_' + str(args.imb_factor) + '.txt', 'w')
    # f1.close()
    # f2 = open('result/accuracy_' + str(args.imb_factor) + '.txt', 'w')
    # f2.close()
    test_distribution_set = ["Forward50", "Forward25", "Forward10", "Forward5", "Forward2", "Uniform", "Backward2",
                             "Backward5", "Backward10", "Backward25", "Backward50"]

    f3 = open('result/std_{}.txt'.format(args.imb_factor), 'w')
    f3.close()

    for test_distribution in test_distribution_set:
        print(test_distribution)
        acc_list = []
        f1_list = []
        w1_list = []
        w2_list = []
        w3_list = []
        for _ in range(10):
            imbalance_train, imbalance_test, _, balance_test = imbalance_datasets(distribution[test_distribution][0], distribution[test_distribution][1])
            _, test_train_trsfm_mnist, test_trsfm_mnist, _, test_train_trsfm_svhn, test_trsfm_svhn = transform_types()

            if test_distribution == "Uniform":
                dataset = balance_test
            else:
                dataset = imbalance_test

            train_dataset = two_crops_test_training(dataset, test_train_trsfm_mnist, test_train_trsfm_svhn)
            valid_dataset = transform_mnist_svhn(dataset, test_trsfm_mnist, test_trsfm_mnist, test_trsfm_svhn,
                                                 test_trsfm_svhn, train=False)

            train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, drop_last=True)
            valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size, drop_last=True)

            aggregation_weight = torch.nn.Parameter(torch.FloatTensor(3), requires_grad=True)
            aggregation_weight.data.fill_(1/3)

            optimizer = SGD([aggregation_weight], lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum,
                            nesterov=True)

            for epoch in range(args.n_epochs):
                start_time = time.time()
                weight = test_training(model, train_loader, aggregation_weight, optimizer, epoch, args)
                print('Time for one epoch: {:.4f}sec'.format(time.time() - start_time))
                if weight[0] < 0.05 or weight[1] < 0.05 or weight[2] < 0.05:
                    break

            weight_result = "\n" + test_distribution + \
                            " - Aggregation weight: Expert 1 is {0:.2f}, Expert 2 is {1:.2f}, Expert 3 is {2:.2f}\n".format(
                                weight[0], weight[1], weight[2])
            print(weight_result)
            # with open('result/aggregation_weight_' + str(args.imb_factor) + '.txt', 'a') as f:
            #     f.write(weight_result)
            # f.close()
            w1_list.append(weight[0])
            w2_list.append(weight[1])
            w3_list.append(weight[2])
            acc, f1_weighted, f1_macro = test_validation(model, valid_loader, args.num_classes, aggregation_weight)
            print(acc)
            acc_result = "\n" + test_distribution + \
                         " - Accuracy: {:.5f}, F1 weighted: {:.5f}, F1 macro: {:.5f}\n".format(acc, f1_weighted, f1_macro)
            print(acc_result)
            # with open('result/accuracy_' + str(args.imb_factor) + '.txt', 'a') as f:
            #     f.write(acc_result)
            # f.close()
            acc_list.append(acc)
            f1_list.append(f1_macro)
        fin_result = "\n" + test_distribution + " - {:.4f}, {:.4f}, {:.2f}, {:.2f}, {:.2f}\n".format(
            np.std(acc_list), np.std(f1_list), np.std(w1_list), np.std(w2_list), np.std(w3_list))
        with open('result/std_' + str(args.imb_factor) + '.txt', 'a') as f:
            f.write(fin_result)
        f.close()


if __name__ == '__main__':
    main()