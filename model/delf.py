import numpy as np
from PIL import Image

import torch as th
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms


class ResNetBase(nn.Module):

    def __init__(self, resnet_type=50, target_layer=3, num_classes=1000, freeze=True, full=False):
        # Call the constructor of superclass.
        super(self.__class__, self).__init__()
        # Initialize modules.
        if not resnet_type in [50, 101]:
            raise ValueError('Invalid ResNet type: {}'.format(resnet_type))
        if not target_layer in [1, 2, 3, 4]:
            raise ValueError('Invalid target layer: {}'.format(target_layer))
        if full and num_classes != 1000:
            raise ValueError('Full resnet is only supported for 1000 output classes.')
        # Store the number of classes.
        self.num_classes = num_classes
        # Store target layer name.
        self.target = 'layer{}'.format(target_layer)
        # Add ResNet convolutional layers.
        resnet = getattr(models, "resnet{}".format(resnet_type))(pretrained=True)
        self.target_channels, self.fc_size = {
            1: (256, 256 * 29 * 29),
            2: (512, 512 * 15 * 15),
            3: (1024, 1024 * 8 * 8),
            4: (2048, 2048 * 4 * 4)
        }[target_layer]
        for name, module in resnet.named_children():
            setattr(self, name, module)
            if name == self.target and not full:
                break
        if not full:
            # Add average-pooling and fully-connected layers.
            self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
            self.fc = nn.Linear(self.fc_size, num_classes)
        # Freeze the parameters.
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        for name, module in self.named_children():
            x = module(x)
            if name == self.target:
                break
        return x


class DELF(nn.Module):

    def __init__(self, base_model, target_channels, num_classes):
        super(self.__class__, self).__init__()
        # Feature extractor module.
        self.base_model = base_model
        self.target_channels = target_channels
        # Attention modules.
        self.conv1 = nn.Conv2d(self.target_channels, 1, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=1)
        # FC module for classification based on feature vector.
        self.fc = nn.Linear(self.target_channels, num_classes)

    def forward(self, batch):
        # Calculate feature maps.
        feature_maps = self.base_model(batch)
        # Calculate attention scores.
        interim = self.conv1(feature_maps)
        interim = F.relu(self.bn1(interim))
        interim = self.conv2(interim)
        attention = F.softplus(interim)
        weight = (attention - attention.min())/(attention.max() - attention.min())
        return weight*feature_maps
