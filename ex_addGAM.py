# from ex_modules import ASPP, NEW_FeatureFusionModule
import torch
from efficientnet_pytorch2.model import EfficientNet
import torch.nn as nn
from torch.nn import functional as F
import warnings


warnings.filterwarnings(action='ignore')
torch.backends.cudnn.benchmark = True




# ARM module
class AttentionRefinementModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input):
        # global average pooling
        x = self.avgpool(input)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        # x = self.sigmoid(self.bn(x))
        x = self.sigmoid(x)
        # channels of input and x should be same
        x = torch.mul(input, x)
        return x


# base network
class addGamNet(nn.Module):
    def __init__(self, depth):
        super(addGamNet, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')

        self.arm4 = AttentionRefinementModule(112, 112)
        self.arm5 = AttentionRefinementModule(1280, 1280)
        self.conv5 = nn.Conv2d(1280, 112, 1, 1)

        self.conv = nn.Conv2d(112 + 112, 6, 1, 1)
        self.bn = nn.BatchNorm2d(6)
        self.relu = nn.ReLU()
        self.fina_conv = nn.Conv2d(6, 6, 1, 1)




    def forward(self, x):
        endpoints = self.backbone.extract_endpoints(x)
        result4 = endpoints['reduction_4']
        result5 = endpoints['reduction_5']
        result4 = self.arm4(result4)
        result5 = self.arm5(result5)
        result5 = self.conv5(result5)
        result5 = F.interpolate(result5, scale_factor=2, mode='bilinear')
        result = torch.cat((result5, result4), dim=1)

        result = self.conv(result)
        result = self.bn(result)
        result = self.relu(result)

        result = F.interpolate(result, scale_factor=2, mode='bilinear')
        result = F.interpolate(result, scale_factor=2, mode='bilinear')
        result = F.interpolate(result, scale_factor=2, mode='bilinear')
        result = F.interpolate(result, scale_factor=2, mode='bilinear')

        result = self.fina_conv(result)


        return result