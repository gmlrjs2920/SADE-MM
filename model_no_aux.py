import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter
import random
from loss import BalancedLoss


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class ConvMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.clf1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.clf2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        out = self.clf1(x)
        out = self.clf2(out)
        return out


class ConvSVHN(nn.Module):
    def __init__(self):
        super().__init__()
        self.clf1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.clf2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        out = self.clf1(x)
        out = self.clf2(out)
        return out


class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = self.fc(x)
        return x


class Classifier(nn.Module):
    def __init__(self, num_class, hidden_dim, dropout):
        super().__init__()
        self.clf = nn.Sequential(
            LinearLayer(7 * 7 * 64 + 8 * 8 * 64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            LinearLayer(hidden_dim, num_class)
        )

    def forward(self, x):
        x = self.clf(x)
        return x


class ConvNet_MM(nn.Module):
    def __init__(self, num_mod, num_experts, num_classes=10, hidden_dim=625, dropout=0.5):
        super(ConvNet_MM, self).__init__()

        self.num_mod = num_mod
        self.num_experts = num_experts

        # m1 = MNIST, m2 = SVHN
        self.layer1s = nn.ModuleList([ConvMNIST() for _ in range(num_experts + 1)])
        self.layer2s = nn.ModuleList([ConvSVHN() for _ in range(num_experts + 1)])

        self.m1_classifier = LinearLayer(7 * 7 * 64, num_classes)
        self.m2_classifier = LinearLayer(8 * 8 * 64, num_classes)
        self.m1_confidence = LinearLayer(7 * 7 * 64, 1)
        self.m2_confidence = LinearLayer(8 * 8 * 64, 1)

        self.linears = nn.ModuleList([Classifier(num_classes, hidden_dim, dropout) for _ in range(num_experts)])

        # self.apply(_weights_init)

    def cls_module(self, x1, x2):
        # classifier for each modality
        TCPlogit1 = self.m1_classifier(x1)
        TCPlogit2 = self.m2_classifier(x2)
        # confidence value for each modality
        TCPconf1 = self.m1_confidence(x1)
        TCPconf2 = self.m2_confidence(x2)
        logit = [TCPlogit1, TCPlogit2]
        conf = [TCPconf1, TCPconf2]
        return logit, conf

    def _separate_part(self, x1, x2, ind):
        out1, out2 = x1, x2

        out1 = (self.layer1s[ind])(out1)
        out1 = out1.view(out1.size(0), -1)

        out2 = (self.layer2s[ind])(out2)
        out2 = out2.view(out2.size(0), -1)

        if ind == self.num_experts:
            logit, conf = self.cls_module(out1, out2)
            return logit, conf
        else:
            return out1, out2

    def forward(self, x1, x2, cls_num_list=None, label=None, infer=False, test_training=False):
        outs = []
        logit, conf = self._separate_part(x1, x2, self.num_experts)

        for ind in range(self.num_experts):
            out1, out2 = self._separate_part(x1, x2, ind)
            out = torch.cat([out1 * conf[0], out2 * conf[1]], dim=1)
            out = (self.linears[ind])(out)
            outs.append(out)

        final_out = torch.stack(outs, dim=1).mean(dim=1)
        if infer:
            return final_out
        if test_training:
            return torch.stack(outs, dim=1)

        # criterion = torch.nn.CrossEntropyLoss(reduction='none')
        criterion = BalancedLoss(cls_num_list=cls_num_list)

        final_loss = 0
        pred = [F.softmax(logit[0], dim=1), F.softmax(logit[1], dim=1)]
        for i in range(self.num_mod):
            p_target = torch.gather(input=pred[i], dim=1, index=label.unsqueeze(dim=1)).view(-1)
            confidence_loss = torch.mean(F.mse_loss(conf[i].view(-1), p_target)
                                         + criterion(logit[i], label))  # mse loss + modality classification loss
            final_loss += confidence_loss

        return {
            "output": final_out,
            "logits": torch.stack(outs, dim=1),
            "logits_cls": logit,
            "conf": conf,
            "loss": final_loss
        }

    def infer(self, x1, x2):
        final_out = self.forward(x1, x2, infer=True)
        return final_out

    def test_training(self, x1, x2):
        logits = self.forward(x1, x2, test_training=True)
        return logits