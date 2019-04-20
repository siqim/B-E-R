# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 01:56:39 2019

@author: msq96
"""


import math
import torch
from torch import nn
import pickle
from collections import OrderedDict
from torch.autograd import Variable
import pretrainedmodels


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """
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

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes * 2)
        self.conv2 = nn.Conv1d(planes * 2, planes * 4, kernel_size=3,
                               stride=stride, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(planes * 4)
        self.conv3 = nn.Conv1d(planes * 4, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv1d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm1d(width)
        self.conv2 = nn.Conv1d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm1d(width)
        self.conv3 = nn.Conv1d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000, embed_dim=300):

        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv1d(embed_dim, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm1d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv1d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm1d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv1d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm1d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv1d(embed_dim, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm1d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool1d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None


        self.last_linear = nn.Sequential(
                                nn.Linear(512 * block.expansion, 512 * block.expansion),
                                nn.ReLU(),
                                nn.BatchNorm1d(512 * block.expansion),
                                nn.Dropout(p=0.5),

                                nn.Linear(512 * block.expansion, num_classes)
                                )

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def se_resnet50(num_classes, embed_dim):
    model = SENet(SEResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes, embed_dim=embed_dim)
    return model

def se_resnext50_32x4d(num_classes, embed_dim):
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes, embed_dim=embed_dim)
    return model

def se_resnext101_32x4d(num_classes, embed_dim):
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes, embed_dim=embed_dim)
    return model

class SeNet2Class(nn.Module):

    def __init__(self, num_classes, embed_dim, path_embed_mat, freeze=False):
        super().__init__()

        embed_mat = pickle.load(open(path_embed_mat, 'rb')).cuda()
        self.embedding = nn.Embedding.from_pretrained(embed_mat, freeze=freeze)

        self.cnn = se_resnext50_32x4d(num_classes, embed_dim)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = self.cnn(x)
        return x





class ConvolutionalBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, first_stride=1):
        super(ConvolutionalBlock, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=first_stride, padding=1),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, x):

        return self.sequential(x)

class KMaxPool(nn.Module):

    def __init__(self, k='half'):
        super(KMaxPool, self).__init__()

        self.k = k
    def forward(self, x):
        # x : batch_size, channel, time_steps
        if self.k == 'half':
            time_steps = x.shape(2)
            self.k = time_steps//2
        kmax, kargmax = x.topk(self.k, dim=2)
        return kmax

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, downsample=False, downsample_type='resnet', optional_shortcut=True):
        super(ResidualBlock, self).__init__()
        self.optional_shortcut = optional_shortcut
        self.downsample = downsample

        if self.downsample:
            if downsample_type == 'resnet':
                self.pool = None
                first_stride = 2
            elif downsample_type == 'kmaxpool':
                self.pool = KMaxPool(k='half')
                first_stride = 1
            elif downsample_type == 'vgg':
                self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
                first_stride = 1
            else:
                raise NotImplementedError()
        else:
            first_stride = 1

        self.convolutional_block = ConvolutionalBlock(in_channels, out_channels, first_stride=first_stride)

        if self.optional_shortcut and self.downsample:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x):

        residual = x
        if self.downsample and self.pool:
            x = self.pool(x)
        x = self.convolutional_block(x)

        if self.optional_shortcut and self.downsample:
            residual = self.shortcut(residual)

        if self.optional_shortcut:
            x = x + residual

        return x

class VDCNN(nn.Module):

    def __init__(self, n_classes, path_embed_mat):
        super(VDCNN, self).__init__()
        embed_mat = pickle.load(open(path_embed_mat, 'rb'))


        vocabulary_size = embed_mat.shape[0]

        depth = 29
        embed_dim = 16
        optional_shortcut = True
        k = 8

        if depth == 9:
            n_conv_layers = {'conv_block_512':2, 'conv_block_256':2,
                             'conv_block_128':2, 'conv_block_64':2}
        elif depth == 17:
            n_conv_layers = {'conv_block_512':2, 'conv_block_256':2,
                             'conv_block_128':2, 'conv_block_64':2}
        elif depth == 29:
            n_conv_layers = {'conv_block_512':4, 'conv_block_256':4,
                             'conv_block_128':10, 'conv_block_64':10}
        elif depth == 49:
            n_conv_layers = {'conv_block_512':6, 'conv_block_256':10,
                             'conv_block_128':16, 'conv_block_64':16}

        # quantization
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=16, padding_idx=0)

        conv_layers = []
        conv_layers.append(nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1))

        for i in range(n_conv_layers['conv_block_64']):
            conv_layers.append(ResidualBlock(64, 64, optional_shortcut=optional_shortcut))

        for i in range(n_conv_layers['conv_block_128']):
            if i == 0:
                conv_layers.append(ResidualBlock(64, 128, downsample=True, optional_shortcut=optional_shortcut))
            conv_layers.append(ResidualBlock(128, 128, optional_shortcut=optional_shortcut))

        for i in range(n_conv_layers['conv_block_256']):
            if i == 0:
                conv_layers.append(ResidualBlock(128, 256, downsample=True, optional_shortcut=optional_shortcut))
            conv_layers.append(ResidualBlock(256, 256, optional_shortcut=optional_shortcut))

        for i in range(n_conv_layers['conv_block_512']):
            if i == 0:
                conv_layers.append(ResidualBlock(256, 512, downsample=True, optional_shortcut=optional_shortcut))
            conv_layers.append(ResidualBlock(512, 512, optional_shortcut=optional_shortcut))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.kmax_pooling = KMaxPool(k=k)

        linear_layers = []

        linear_layers.append(nn.Linear(512 * k, 2048))
        linear_layers.append(nn.Linear(2048, 2048))
        linear_layers.append(nn.Linear(2048, n_classes))

        self.linear_layers = nn.Sequential(*linear_layers)

    def forward(self, sentences):

        x = self.embedding(sentences)
        x = x.transpose(1,2) # (batch_size, sequence_length, embed_size) -> (batch_size, embed_size, sequence_length)
        x = self.conv_layers(x)
        x = self.kmax_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)

        return x