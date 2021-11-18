# from ex_modules import ASPP, NEW_FeatureFusionModule
import torch
from efficientnet_pytorch2.model import EfficientNet
import torch.nn as nn
from torch.nn import functional as F
import warnings


warnings.filterwarnings(action='ignore')
torch.backends.cudnn.benchmark = True
# base network
class BaseNet(nn.Module):
    def __init__(self, depth):
        super(BaseNet, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        self.conv = nn.Conv2d(112, 6, 1, 1)
        self.fina_conv = nn.Conv2d(6, 6, 1, 1)




    def forward(self, x):
        endpoints = self.backbone.extract_endpoints(x)
        result = endpoints['reduction_4']
        result = self.conv(result)
        result = F.interpolate(result, scale_factor=2, mode='bilinear')
        result = F.interpolate(result, scale_factor=2, mode='bilinear')
        result = F.interpolate(result, scale_factor=2, mode='bilinear')
        result = F.interpolate(result, scale_factor=2, mode='bilinear')
        result = self.fina_conv(result)


        return result