import torch
from torchvision import models
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
import warnings
from efficientnet_pytorch1 import EfficientNet

warnings.filterwarnings(action='ignore')
torch.backends.cudnn.benchmark = True
__all__ = ['ResNet', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))

# CC_module
def INF(B, H, W):
    device_ids = [0, 1, 2]
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)
    # return tensor()
# CC_module
class CC_module(nn.Module):
    # pytorch1.0 +
    def __init__(self, in_dim):
        super(CC_module, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,
                                                                                                     height,
                                                                                                     height).permute(0,
                                                                                                                     2,
                                                                                                                     1,
                                                                                                                    3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        # concate = concate * (concate>torch.mean(concate,dim=3,keepdim=True)).float()
        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + x

#NEW+ FFM module
class NEW_FeatureFusionModule(torch.nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        # self.in_channels = input_1.channels + input_2.channels
        # resnet101 3328 = 256(from context path) + 1024(from spatial path) + 2048(from spatial path)
        # resnet18  1024 = 256(from context path) + 256(from spatial path) + 512(from spatial path)
        self.in_channels = in_channels

        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1)
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))


    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)

        x = self.avgpool(feature)
        up_x = nn.Upsample((feature.shape[2], feature.shape[3]), mode='bilinear',
                         align_corners=True)(x)
        feature_one = feature + up_x

        x = self.avgpool(feature)
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature_one)

        return x

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
        feature3 = self.layer3(feature2)  # ---> 1 / 16 ---> 32 x 32 x 256
        feature4 = self.layer4(feature3)  # ---> 1 / 32 ---> 16 x 16 x 512
        # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature3, feature4, tail

## Spation path
class Spatial_path(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock1 = ConvBlock(in_channels=3, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, out_channels=128)
        self.convblock3 = ConvBlock(in_channels=128, out_channels=256)

    def forward(self, input):
        x1 = self.convblock1(input)  # ---> 256 x 256 x 64
        x2 = self.convblock2(x1)   # ---> 128 x 128 x 128
        x3 = self.convblock3(x2)   # ---> 64 x 64 x 256

        return x3   # ---> 64 x 64 x 256

class ZyjNet(nn.Module):
    def __init__(self, depth):
        super(ZyjNet, self).__init__()
        self.spatial_path = Spatial_path()
        self.resnet18 = resnet18()

        self.feature_fusion_module = NEW_FeatureFusionModule(64, 256 + 256 + 512)

        self.CC_module1 = CC_module(64)
        self.CC_module2 = CC_module(64)

        # build final convolution
        self.conv = nn.Conv2d(64, 6, kernel_size=1)

    def forward(self, x):
        # spatial_path:input:B X C X W X H   output：B X 256 X W/8 X H/8
        sx = self.spatial_path(x) # ---> 64 x 64 x 256
        cx_f3, cx_f4, cx_tail = self.resnet18(x) # ---> 32 x 32 x 256, 16 x 16 x 512, 1 x 1 x 512

        # get cx
        cx_f4 = torch.mul(cx_f4, cx_tail)
        f3 = torch.nn.functional.interpolate(cx_f3, size=sx.size()[-2:], mode='bilinear')
        f4 = torch.nn.functional.interpolate(cx_f4, size=sx.size()[-2:], mode='bilinear')
        cx = torch.cat((f3, f4), dim=1) # ----> 64 x 64 x (256 + 512)

        # feature fusion ---> input: cx and sx
        # resnet18 256 + (256 + 512)
        sx_cx = self.feature_fusion_module(sx, cx) # ----> 64 x 64 x 6

        # criss-cross attention
        result = self.CC_module1(sx_cx)
        result = self.CC_module2(result) # ----> 64 x 64 x 6

        # upsampling
        result = torch.nn.functional.interpolate(result, scale_factor=2, mode='bilinear')
        result = torch.nn.functional.interpolate(result, scale_factor=2, mode='bilinear')
        result = torch.nn.functional.interpolate(result, scale_factor=2, mode='bilinear')

        # final conv
        result = self.conv(result)

        return result