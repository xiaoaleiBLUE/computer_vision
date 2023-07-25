"""
训练 Unet 模型
1. 搭建 Unet 模型
2. 自定义 loss 函数
3. 开始训练
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from torch.utils.tensorboard import SummaryWriter                    # 使用 tensorboard 记录参数


"""
自定义数据集类
"""


class SegmentDataset(Dataset):

    def __init__(self, where='train', seq=None):

        # 获取数据
        self.img_list = glob.glob('./processed/{}/*/img_*'.format(where))

    def __len__(self):
        # 返回数据大小
        return len(self.img_list)

    def __getitem__(self, idx):
        # 获取具体的数据(图片, mask)

        # 索引获取图片 和 mask
        img_file = self.img_list[idx]
        mask_file = img_file.replace('img', 'label')

        # 进行加载 图片 mask
        img = np.load(img_file)
        mask = np.load(mask_file)

        # 是否进行数据增强
        if self.seq:

            segmap = SegmentationMapsOnImage(mask, shape=mask.shape)

            img, mask = self.seq(image=img, segmentation=segmap)

            # 直接获取数据内容
            mask = mask.get_arr()

        # 灰度图扩张维度变成张量
        return np.expand_dims(img, 0), np.expand_dims(mask, 0)


# 数据增强处理流程
seq = iaa.Sequential([
    iaa.Affine(
        scale=(0.8, 1.2),                     # 缩放
        rotate=(-45, 45)),                    # 旋转
    iaa.ElasticTransformation()               # 变换

])

# 加载 train test 的 dataset
train_dataset = SegmentDataset(where='train', seq=seq)
test_dataset = SegmentDataset(where='test', seq=None)

# 加载 dataloader
train_dataloader = DataLoader(train_dataset, batch_size=12, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=12, shuffle=False)

"""
构建 Unet 模型
"""


# 定义两次卷积操作
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.step = torch.nn.Sequential(
            # 第一次卷积
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            # ReLU
            torch.nn.ReLU(),
            # 第二次卷积
            torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            # ReLU
            torch.nn.ReLU()
        )

    def forward(self, x):

        return self.step(x)


# 定义 Unet 整体架构
class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义左侧编码器的操作
        self.layer1 = ConvBlock(1, 64)
        self.layer2 = ConvBlock(64, 128)
        self.layer3 = ConvBlock(128, 256)
        self.layer4 = ConvBlock(256, 512)

        # 定义右侧解码器的操作
        self.layer5 = ConvBlock(256 + 512, 256)
        self.layer6 = ConvBlock(128 + 256, 128)
        self.layer7 = ConvBlock(64 + 128, 64)

        # 最后一个卷积
        self.layer8 = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, padding=0, stride=1)

        # 定一些其他操作
        # 池化
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2)
        # 上采样
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        # sigmoid
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # 对输入数据进行处理

        # 定义下采样部分

        # input:1X256x256, output: 64x256x256
        x1 = self.layer1(x)
        # input:64x256x256, output: 64 x 128 x 128
        x1_p = self.maxpool(x1)

        # input:  64 x 128 x 128 , output: 128 x 128 x 128
        x2 = self.layer2(x1_p)
        # input:128 x 128 x 128 , output: 128 x 64 x 64
        x2_p = self.maxpool(x2)

        # input: 128 x 64 x 64, output: 256 x 64 x 64
        x3 = self.layer3(x2_p)
        # input:256 x 64 x 64, output: 256 x 32 x 32
        x3_p = self.maxpool(x3)

        # input: 256 x 32 x 32, output: 512 x 32 x 32
        x4 = self.layer4(x3_p)

        # 定义上采样
        # input: 512 x 32 x 32，output: 512 x 64 x 64
        x5 = self.upsample(x4)
        # 拼接,output: 768x 64 x 64
        x5 = torch.cat([x5, x3], dim=1)
        # input: 768x 64 x 64,output: 256 x 64 x 64
        x5 = self.layer5(x5)

        # input: 256 x 64 x 64,output: 256 x 128 x 128
        x6 = self.upsample(x5)
        # 拼接,output: 384 x 128 x 128
        x6 = torch.cat([x6, x2], dim=1)
        # input: 384 x 128 x 128, output: 128 x 128 x 128
        x6 = self.layer6(x6)

        # input:128 x 128 x 128, output: 128 x 256 x 256
        x7 = self.upsample(x6)
        # 拼接, output: 192 x 256 x256
        x7 = torch.cat([x7, x1], dim=1)
        # input: 192 x 256 x256, output: 64 x 256 x 256
        x7 = self.layer7(x7)

        # 最后一次卷积,input: 64 x 256 x 256, output: 1 x 256 x 256
        x8 = self.layer8(x7)

        # sigmoid
        # x9= self.sigmoid(x8)

        return x8


"""
测试模型
"""
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = UNet().to(device)

summary(model, (1, 256, 256))

"""
模拟输入
"""
random_input = torch.randn(1, 1, 256, 256).to(device)
output = model(random_input)

"""
准备训练
"""
# 定义损失
loss_fn = torch.nn.BCEWithLogitsLoss()

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 动态减少 LR
schedule = ReduceLROnPlateau(optimizer, 'min')

# 记录变量存放位置
writer = SummaryWriter(log_dir='./log')

"""
定义测试集的 loss 损害函数
"""


def check_test_loss(test_dataloader, model):

    test_loss = 0

    # 不记录梯度
    with torch.no_grad():

        for i, (x, y) in enumerate(test_dataloader):

            # 读取数据图片
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            # 预测值
            y_pred = model(x)

            # 计算损失
            loss_batch = loss_fn(y_pred, y)

            # 计算损失累加
            test_loss += loss_batch

    # 计算 1个epoch loss
    test_loss = test_loss / len(test_dataloader)

    # 返回 test_loss
    return test_loss


"""
定义测试集的 loss 损害函数
"""
# 训练 100 个 epoch
EPOCH_NUM = 200

# 初始化最好的loss, 后面会更新计算 test_loss
best_test_loss = 100

for epoch in range(EPOCH_NUM):

    # 获取批次图像
    start_time = time.time()

    loss = 0

    for i, (x, y) in enumerate(train_dataloader):

        # 每次更新前清空梯度
        model.zero_grad()

        # 获取数据图片
        x = x.to(device, dtype=torch.float32)

        # 获取数据标签
        y = y.to(device, dtype=torch.float32)

        # 预测值(对输入x, 进行预测)
        y_pred = model(x)

        # 计算损失
        loss_batch = loss_fn(y_pred, y)

        # 计算梯度
        loss_batch.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 记录计算迭代 整个dataloader 的loss和
        loss += loss_batch

    # 整个dataloader 的loss和进行平均 = 每个 epoch 的 loss
    loss = loss / len(train_dataloader)

    # 如果降低 LR,
    schedule.state_dict(loss)

    # 计算测试集的 loss
    test_loss = check_test_loss(test_dataloader, model)

    # tensorboard 记录 Loss/train(训练集)
    writer.add_scalar('Loss/train', loss, epoch)

    # tensorboard 记录 Loss/test(测试集)
    writer.add_scalar('Loss/test', test_loss, epoch)

    #  记录最好的测试集loss, 并保存模型
    if best_test_loss > test_loss:

        best_test_loss = test_loss

        best_pt_dir = './save_model'

        # 检查文件是否存在, 不存在则创建文件夹
        if not os.path.exists(best_pt_dir):
            os.makedirs(best_pt_dir)

        # 保存模型
        torch.save(model.state_dict(), '{}/unet_best.pt'.format(best_pt_dir))

        # 打印信息最好训练结果的 epoch
        print('第{}个EPOCH达到最低的测试loss:{}'.format(epoch, best_test_loss))

    # 打印训练信息
    print('第{}个epoch执行时间：{}s，train loss为：{}，test loss为：{}'.format(
        epoch,
        time.time() - start_time,
        loss,
        test_loss
    ))




















