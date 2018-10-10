"""Weakly-supervised attention-based segmentation model."""

import torch as th
from torch import nn
from torch.nn import functional as F
from torchvision import models

from torch.nn.modules import Upsample



class SegmentTrainer(nn.Module):

    def __init__(self, model="resnet50", target_layer="layer3"):
        super().__init__()

        if model not in ["resnet50", "resnet101"]: # TODO: Add vgg
            raise ValueError("Unsupported model: `{}`.".format(model))

        targets = {
            # for each allowed target layer keep the module index, feature map size, and
            # number of required transposed convolutions.
            "layer1": (4, 256, 2),
            "layer2": (5, 512, 3),
            "layer3": (6, 1024, 4),
            "layer4": (7, 2048, 5)
        }

        index, out_channels, num_tconvs = targets[target_layer]

        # Feature extraction modules.
        self.resnet = getattr(models, model)(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False

        resnet_convs = list(self.resnet.children())[:index+1]
        self.resnet_convs = nn.Sequential(*resnet_convs)

        # Probably not needed but just in case!
        for param in self.resnet_convs.parameters():
            param.requires_grad = False

        # Attention modules.
        self.conv1 = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=1)

        # # Upsampling modules for resizing attention to 224x224.
        # self.upsample = nn.Sequential(*[
        #     nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)
        #     for _ in range(num_tconvs)
        # ])
        self.upsample = Upsample(size=(224, 224), mode="bilinear", align_corners=False)


    def forward(self, inp):
        # Save a copy of feature maps.
        feature_maps = self.resnet_convs(inp)

        # Calculate attention scores.
        attention = self.conv1(feature_maps)
        attention = F.relu(self.bn1(attention))
        attention = self.conv2(attention)
        attention = F.softplus(attention)

        # Resize the attention to the same size as that of input.
        upsampled_attention = self.upsample(attention)

        # Calculate the mask.
        min_val = upsampled_attention.min()
        max_val = upsampled_attention.max()
        mask = (upsampled_attention - min_val) / (max_val - min_val)
        mask[mask < 0.5] = 0.

        # Apply the attention to the feature maps
        masked_inp = mask * inp

        log_probabilities = F.log_softmax(self.resnet(masked_inp), dim=1)
        return log_probabilities, mask



class Segment(nn.Module):

    def __init__(self, segment_trainer):
        super().__init__()

        self.resnet = segment_trainer.resnet
        self.resnet_convs = segment_trainer.resnet_convs
        self.conv1 = segment_trainer.conv1
        self.bn1 = segment_trainer.bn1
        self.conv2 = segment_trainer.conv2
        # self.upsample = segment_trainer.upsample
        self.upsample = Upsample(size=(224, 224), mode="bilinear", align_corners=False)


    def forward(self, inp, ret_all=False):
        # Save a copy of feature maps.
        feature_maps = self.resnet_convs(inp)

        # Calculate attention scores.
        attention = self.conv1(feature_maps)
        attention = F.relu(self.bn1(attention))
        attention = self.conv2(attention)
        attention = F.softplus(attention)

        # min_val = attention.min()
        # max_val = attention.max()
        # mask = (attention - min_val) / (max_val - min_val)
        # mask[mask < 0.5] = 0.
        mask = F.sigmoid(100 * (attention - 0.85))

        masked_fmaps = mask * feature_maps
        if ret_all:
            return attention, mask, feature_maps, masked_fmaps
        return masked_fmaps
