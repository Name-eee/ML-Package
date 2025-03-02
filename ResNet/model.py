import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchsummary import summary


# resnet
class ResBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, out_channels, stride=1, dropout_prob=0.3):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, stride=stride)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.convres = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.convres = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout_prob)

    def forward(self, x):
        res = self.convres(x)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)

        x = self.bn3(self.conv3(x))
        x = self.dropout(x)

        return self.relu(x + res)

class resnet(nn.Module):
    def __init__(self, channels=[64, 256, 512, 1024, 2048], output_channel=102):
        super(resnet, self).__init__()
        self.conv1 = nn.Conv2d(3, channels[0], 7, stride=2, padding=3)
        self.mp = nn.MaxPool2d(3, stride=2, padding=1)
        self.relu = nn.ReLU()

        self.layers = nn.ModuleList()
        for in_channels, bottleneck_channels, out_channels, stride, num_blocks in [
            (channels[0], 64, channels[1], 1, 3),
            (channels[1], 128, channels[2], 2, 4),
            (channels[2], 256, channels[3], 2, 6),
            (channels[3], 512, channels[4], 2, 3),
        ]:
            layer = [ResBlock(in_channels, bottleneck_channels, out_channels, stride=stride)]
            layer += [ResBlock(out_channels, bottleneck_channels, out_channels) for _ in range(num_blocks - 1)]
            self.layers.append(nn.Sequential(*layer))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], output_channel)

    def forward(self, x):
        x = self.relu(self.mp(self.conv1(x)))
        for layer in self.layers:
            x = layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
