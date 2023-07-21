import torch
import torch.nn as nn
from torchsummary import summary


class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.Relu = nn.ReLU()

    def forward(self, x):

        x = self.Relu(self.conv1(x))
        x = self.Relu(self.conv2(x))

        return x


class Down_conv(nn.Module):
    def __init__(self, input_channel, out_channel):
        super(Down_conv, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(input_channel, out_channel, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)

        self.Relu = nn.ReLU()

    def forward(self, x):

        x = self.pool(x)
        x = self.Relu(self.conv1(x))
        x = self.Relu(self.conv2(x))

        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 模型实例化
conv_block = Conv_Block(1, 64).to(device)

# 查看输出大小
summary(conv_block, (1, 256, 256))


# 定义 Unet 网络架构
class Unet_model(nn.Module):

    def __init__(self):
        super(Unet_model, self).__init__()
        # 定义左侧编码器
        self.layer1 = Conv_Block(1, 64)
        self.layer2 = Conv_Block(64, 128)
        self.layer3 = Conv_Block(128, 256)
        self.layer4 = Conv_Block(256, 512)                             # 32*32*512

        # 定义右侧的编码器
        self.layer5 = Conv_Block(256+512, 256)
        self.layer6 = Conv_Block(128+256, 128)
        self.layer7 = Conv_Block(64+128, 64)

        # 最后一个卷积
        self.layer8 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

        # 定义一下其他操作
        # 池化操作
        self.maxpool = nn.MaxPool2d(2, 2)

        # 上采样, bilinear: 双线性插值
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        # 最后要加一个激活层
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # 左侧编码器前传
        # 定义下采样部分
        # input: 1*256*256    output: 64*256*256
        x1 = self.layer1(x)
        # input: 64*256*256   output: 64*128*128
        x1_p = self.maxpool(x1)
        # input: 64*128*128   output: 128*128*128
        x2 = self.layer2(x1_p)
        # input: 128*128*128  output: 128*64*64
        x2_p = self.maxpool(x2)
        # input: 128*64*64    output: 256*64*64
        x3 = self.layer3(x2_p)
        # input: 256*64*64    output: 256*32*32
        x3_p = self.maxpool(x3)
        # input: 256*32*32    output: 512*32*32
        x4 = self.layer4(x3_p)

        # 右侧编码器前传
        # 定义上采样部分
        # input: 512*32*32    output: 512*64*64
        x5 = self.upsample(x4)
        # 通道拼接, 此时拼接的 dim=1   output: 768*64*64
        x5 = torch.cat([x5, x3], dim=1)
        x5 = self.layer5(x5)

        # 上采样, 拼接
        x6 = self.upsample(x5)
        x6 = torch.cat([x6, x2], dim=1)
        x6 = self.layer6(x6)

        # 上采样, 拼接
        x7 = self.upsample(x6)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.layer7(x7)

        # 最后一次卷积
        x8 = self.layer8(x7)

        # sigmoid
        x_out = self.sigmoid(x8)

        return x_out


unet_model = Unet_model().to(device)
summary(unet_model, (1, 256, 256))






















































