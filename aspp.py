import torch
import torch.nn as nn
from torch.nn import functional as F

# without bn version
class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP, self).__init__()

        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        self.mean2 = nn.AdaptiveAvgPool2d((2, 2))
        self.mean3 = nn.AdaptiveAvgPool2d((3, 3))
        self.mean4 = nn.AdaptiveAvgPool2d((4, 4))

        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.conv2 = nn.Conv2d(in_channel, depth, 1, 1)
        self.conv3 = nn.Conv2d(in_channel, depth, 1, 1)
        self.conv4 = nn.Conv2d(in_channel, depth, 1, 1)

        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)

        self.conv_1x1_output = nn.Conv2d(depth * 8, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode='bilinear')

        image_features2 = self.mean2(x)
        image_features2 = self.conv2(image_features2)
        image_features2 = F.interpolate(image_features2, size=size, mode='bilinear')

        image_features3 = self.mean3(x)
        image_features3 = self.conv3(image_features3)
        image_features3 = F.interpolate(image_features3, size=size, mode='bilinear')

        image_features4 = self.mean4(x)
        image_features4 = self.conv4(image_features4)
        image_features4 = F.interpolate(image_features4, size=size, mode='bilinear')


        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, image_features2, image_features3, image_features4,
                                              atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net



zhangyijie = ASPP(512, 256)
# print(zhangyijie)
a = torch.rand(1, 512, 64, 64)
result = zhangyijie(a)
print(result.shape)
