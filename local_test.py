import torch
from torchvision import models
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
import warnings
from efficientnet_pytorch1 import EfficientNet


##context_path --> resnet18
class resnet18(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.features = models.resnet18(pretrained=pretrained)
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)  # ---> 1 / 4
        feature2 = self.layer2(feature1)  # ---> 1 / 8
        feature3 = self.layer3(feature2)  # ---> 1 / 16 ---> 32 x 32 x
        feature4 = self.layer4(feature3)  # ---> 1 / 32 ---> 16 x 16 x
        # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature3, feature4, tail

a = resnet18()
print(a)
