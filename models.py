# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 17:12:21 2019

@author: msq96
"""

import math
import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, embed_dim):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(embed_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.pool = nn.AdaptiveMaxPool1d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0]  * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.fc = nn.Sequential(
                                nn.Linear(512 * block.expansion, 2048),
                                nn.ReLU(),
                                nn.BatchNorm1d(2048),
                                nn.Dropout(p=0.5),

                                nn.Linear(2048, 2048),
                                nn.ReLU(),
                                nn.BatchNorm1d(2048),
                                nn.Dropout(p=0.5),

                                nn.Linear(2048, num_classes)
                                )



    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50(num_classes, embed_dim):
    """Constructs a ResNet-50 model. [3, 4, 6, 3] [3, 8, 36, 3]

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, embed_dim=embed_dim)
    return model


class Sent2Class(nn.Module):

    def __init__(self, num_classes, embed_dim, path_embed_mat, freeze=False):
        super().__init__()

        embed_mat = pickle.load(open(path_embed_mat, 'rb')).cuda()
        self.embedding = nn.Embedding.from_pretrained(embed_mat, freeze=freeze)

        self.cnn = resnet50(num_classes, embed_dim)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = self.cnn(x)
        return x


