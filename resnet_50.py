import torch
import torch.nn as nn
from torchsummary import summary


class ResBlock_1(nn.Module):

    def __init__(self, down_sample, in_channels, min_channels, out_channels):
        super(ResBlock_1, self).__init__()

        self.down_sample = down_sample

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, min_channels, 1, 1, 0),
            nn.BatchNorm2d(min_channels),
            nn.ReLU(),

            nn.Conv2d(min_channels, min_channels, 3, 1, 1),
            nn.BatchNorm2d(min_channels),
            nn.ReLU(),

            nn.Conv2d(min_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()

        )

        self.shortcut_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.shortcut_bn = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        if self.down_sample is True:
            shortcut = self.shortcut_bn(self.shortcut_conv(x))
            x = self.conv_1(x)
            x = x + shortcut
            x = self.relu(x)

        else:
            shortcut = x
            x = self.conv_1(x)
            x = x + shortcut
            x = self.relu(x)

        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ResBlock 实例化
resblock = ResBlock_1(True, 64, 64, 256).to(device)

# 打印模型参数, 可视化
summary(resblock, (64, 56, 56))


class ResBlock_2(nn.Module):

    def __init__(self, down_sample, in_channels, min_channels, out_channels):
        super(ResBlock_2, self).__init__()

        self.down_sample = down_sample

        if self.down_sample is True:
            s = 2
        else:
            s = 1
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels, min_channels, 1, 1, 0),
            nn.BatchNorm2d(min_channels),
            nn.ReLU(),

            nn.Conv2d(min_channels, min_channels, 3, s, 1),
            nn.BatchNorm2d(min_channels),
            nn.ReLU(),

            nn.Conv2d(min_channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.shortcut_conv = nn.Conv2d(in_channels, out_channels, 1, 2, 0)
        self.shortcut_bn = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        if self.down_sample is True:
            shortcut = self.shortcut_bn(self.shortcut_conv(x))
            x = self.conv_2(x)
            x = x + shortcut
            x = self.relu(x)

        else:
            shortcut = x
            x = self.conv_2(x)
            x = x + shortcut
            x = self.relu(x)

        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ResBlock 实例化
resblock = ResBlock_2(True, 256, 128, 512).to(device)

# 打印模型参数, 可视化
summary(resblock, (256, 56, 56))


class Resnet_50(nn.Module):
    """
    搭建一个简单的残差网络: RESNET 18
    输入: 224*224*3
    输出: 1000类
    """
    def __init__(self):
        super(Resnet_50, self).__init__()

        # Layer 0
        self.layer_0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer_1 = nn.Sequential(
            # 需要下采样
            ResBlock_1(True, 64, 64, 256),

            # 无需下采样
            ResBlock_1(False, 256, 64, 256),

            # 无需下采样
            ResBlock_1(False, 256, 64, 256),
        )

        self.layer_2 = nn.Sequential(
            # 需要下采样
            ResBlock_2(True, 256, 128, 512),

            # 无需下采样
            ResBlock_2(False, 512, 128, 512),

            # 无需下采样
            ResBlock_2(False, 512, 128, 512),

            # 无需下采样
            ResBlock_2(False, 512, 128, 512),
        )

        self.layer_3 = nn.Sequential(
            # 需要下采样
            ResBlock_2(True, 512, 256, 1024),

            # 无需下采样
            ResBlock_2(False, 1024, 256, 1024),

            # 无需下采样
            ResBlock_2(False, 1024, 256, 1024),

            # 无需下采样
            ResBlock_2(False, 1024, 256, 1024),

            # 无需下采样
            ResBlock_2(False, 1024, 256, 1024),

            # 无需下采样
            ResBlock_2(False, 1024, 256, 1024),
        )

        self.layer_4 = nn.Sequential(
            # 需要下采样
            ResBlock_2(True, 1024, 512, 2048),

            # 无需下采样
            ResBlock_2(False, 2048, 512, 2048),

            # 无需下采样
            ResBlock_2(False, 2048, 512, 2048),

        )

        # AdaptiveAvgPool2d
        self.app = nn.AdaptiveAvgPool2d(1)

        self.flatten = nn.Flatten()

        self.linear = nn.Linear(2048, 1000)

    def forward(self, x):

        x = self.layer_0(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.app(x)
        x = self.flatten(x)
        x = self.linear(x)

        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ResBlock 实例化
resblock = Resnet_50().to(device)

# 打印模型参数, 可视化
summary(resblock, (3, 224, 224))















































