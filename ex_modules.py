import torch
from efficientnet_pytorch2.model import EfficientNet
import torch.nn as nn
from torch.nn import functional as F

inputs = torch.rand(1, 3, 224, 224)
model = EfficientNet.from_pretrained('efficientnet-b0')
endpoints = model.extract_endpoints(inputs)
print(endpoints['reduction_1'].shape)  # torch.Size([1, 16, 112, 112])
print(endpoints['reduction_2'].shape)  # torch.Size([1, 24, 56, 56])
print(endpoints['reduction_3'].shape)  # torch.Size([1, 40, 28, 28])
print(endpoints['reduction_4'].shape)  # torch.Size([1, 112, 14, 14])
print(endpoints['reduction_5'].shape)  # torch.Size([1, 1280, 7, 7])
# print(model)


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