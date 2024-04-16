import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models.resnet as resnet
from torch.nn import Parameter
import random
from loss import BalancedLoss


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class TabularModel(nn.Module):
    def __init__(self, emb_size, n_cont, emb_p=0, fc_p=0.2):
        super().__init__()
        self.embed = nn.ModuleList([nn.Embedding(num_emb, emb_dim) for num_emb, emb_dim in emb_size])
        self.emb_drop = nn.Dropout(emb_p)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        n_emb = sum(e.embedding_dim for e in self.embed)
        self.n_emb, self.n_cont = n_emb, n_cont
        self.layers = nn.Sequential(
            nn.Linear(in_features=12, out_features=8, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(8),
            nn.Dropout(fc_p),
            nn.Linear(in_features=8, out_features=8, bias=True),
        )

    def forward(self, x_cat, x_cont):
        if self.n_emb != 0:
            x = [e(x_cat[:, i]) for i, e in enumerate(self.embed)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x_cont = self.bn_cont(x_cont.reshape(-1, 1))
            out = torch.cat([x, x_cont], 1)
        return self.layers(out)


conv1x1 = resnet.conv1x1
Bottleneck = resnet.Bottleneck
BasicBlock = resnet.BasicBlock


class ResNet(nn.Module):

    def __init__(self, block, layers, hidden_dim=256, zero_init_residual=True):
        super(ResNet, self).__init__()
        self.inplanes = 32  # conv1에서 나올 채널의 차원 -> 이미지넷보다 작은 데이터이므로 32로 조정

        # inputs = 3x224x224 -> 3x128x128로 바뀜
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)  # 마찬가지로 전부 사이즈 조정
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 32, layers[0], stride=1)  # 3 반복
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)  # 4 반복
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)  # 6 반복
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)  # 3 반복

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.BatchNorm1d(256 * block.expansion),
            nn.Dropout(0.1),
            nn.Linear(256 * block.expansion, hidden_dim)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):  # planes -> 입력되는 채널 수
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # input [32, 128, 128] -> [C ,H, W]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # x.shape =[32, 64, 64]

        x = self.layer1(x)
        # x.shape =[128, 64, 64]
        x = self.layer2(x)
        # x.shape =[256, 32, 32]
        x = self.layer3(x)
        # x.shape =[512, 16, 16]
        x = self.layer4(x)
        # x.shape =[1024, 8, 8]

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ConvFBM(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.clf1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), # [-, 3, 128, 128] -> [-, 32, 128, 128]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # [-, 32, 128, 128] -> [-, 32, 64, 64]
        )
        self.clf2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # [-, 32, 64, 64] -> [-, 64, 64, 64]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # [-, 64, 64, 64] -> [-, 64, 32, 32]
        )
        self.clf3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # [-, 64, 32, 32] -> [-, 128, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # [-, 128, 32, 32] -> [-, 128, 16, 16]
        )
        self.clf4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # [-, 128, 16, 16] -> [-, 256, 16, 16]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # [-, 256, 16, 16] -> [-, 256, 8, 8]
        )
        self.clf5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # [-, 256, 8, 8] -> [-, 512, 8, 8]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # [-, 512, 8, 8] -> [-, 512, 4, 4]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.Linear(512, hidden_dim)
        )

    def forward(self, x):
        out = self.clf1(x)
        out = self.clf2(out)
        out = self.clf3(out)
        out = self.clf4(out)
        out = self.clf5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
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
    def __init__(self, num_class, hidden_dim_tab, hidden_dim_img, hidden_dim_clf, p):
        super().__init__()
        self.clf = nn.Sequential(
            LinearLayer(hidden_dim_img + hidden_dim_tab, hidden_dim_clf),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim_clf),
            nn.Dropout(p),
            LinearLayer(hidden_dim_clf, num_class)
        )

    def forward(self, x):
        x = self.clf(x)
        return x


class ConvNet_MM(nn.Module):
    def __init__(self, num_mod, num_experts, num_classes=2, hidden_dim_tab=8, hidden_dim_img=256, hidden_dim_clf=32, p=0.2):
        super(ConvNet_MM, self).__init__()

        self.num_mod = num_mod
        self.num_experts = num_experts

        # m1 = measure, m2 = fbm
        self.layer1s = nn.ModuleList([TabularModel(emb_size=[(3, 3), (7, 5), (3, 3)], n_cont=1) for _ in range(num_experts + 1)])
        self.layer2s = nn.ModuleList([ConvFBM(hidden_dim=hidden_dim_img) for _ in range(num_experts + 1)])

        self.m1_classifier = LinearLayer(hidden_dim_tab, num_classes)
        self.m2_classifier = LinearLayer(hidden_dim_img, num_classes)
        self.m1_confidence = LinearLayer(hidden_dim_tab, 1)
        self.m2_confidence = LinearLayer(hidden_dim_img, 1)

        self.linears = nn.ModuleList(
            [Classifier(num_classes, hidden_dim_tab, hidden_dim_img, hidden_dim_clf, p) for _ in range(num_experts)])

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

        out1 = (self.layer1s[ind])(out1[:, :3].long(), out1[:, 3])
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