"""
训练模型
1. 导入 Unet 模型
2. 自定义 dice Loss 函数
3. 开始训练
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import glob
from celluloid import Camera
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import time


"""
自定义数据集类
"""


class SegmentDataset(Dataset):

    def __init__(self, where='train', seq=None):
        """
        :param where:  默认为 train, 读取train文件夹下的数据, 如果为test, 则读取test文件夹下的数据
        :param seq:   是否数据增强处理, 默认为 None
        """
        # 获取 numpy 数据
        # 获取所有 图片 numpy数据
        self.img_list = glob.glob('./processed/{}/*/img_*'.format(where))

        # 数据增强的处理流程
        self.seq = seq

    def __len__(self):
        # 获取数据集大小的
        return len(self.img_list)

    def __getitem__(self, idx):
        """
        :param idx:  索引
        :return:  主要获取某一个数据
        """
        # 获取图片文件名, 标签路径
        img_file = self.img_list[idx]

        # 获取标签文件名, img_file 已经是索引后的图片, 所以对应后的标签也是 索引 后
        mask_file = img_file.replace('img', 'label')            # 将 img_file 中的 img 用 label 代替

        # 进行加载图片, 因为上述是路径
        img = np.load(img_file)
        mask = np.load(mask_file)

        # 数据增强
        if self.seq:

            segmap = SegmentationMapsOnImage(mask, shape=mask.shape)
            img, mask = self.seq(image=img, segmentation_maps=segmap)

            # 获取数组内容
            mask = mask.get_arr()

        # 传入模型, 需要扩张一个维度, 变成张量, 返回形式是张量
        return np.expand_dims(img, 0), np.expand_dims(mask, 0)


# 数据增强处理流程
seq = iaa.Sequential([
    iaa.Affine(
        scale=(0.8, 1.2),            # 缩放
        rotate=(-45, 45)             # 旋转
    ),
    iaa.ElasticTransformation()      # 弹性形变
])


# 使用dataloader加载数据
batch_size = 12

train_dataset = SegmentDataset(where='train', seq=seq)
test_dataset = SegmentDataset(where='test', seq=seq)

# dataloader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


"""
定义 Unet 网络架构, 基本结构 Conv_Block
"""


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


# 定义设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 模型实例化
model = Unet_model().to(device)

# 打印模型
summary(model, (1, 256, 256))

"""
模拟输入输出 (1, 1, 256, 256): 1代表批次数量  1: 通道数量  256,256: 代表图像大小
"""
random_input = torch.randn((1, 1, 256, 256)).to(device)

output = model(random_input)

print(output)
print(output.shape)                                      # torch.Size([1, 1, 256, 256]

"""
准备训练
定义 Dice loss 函数, 类: 类损失函数
"""


class dice_loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, mask):
        # 把数组拉直成向量
        perd = torch.flatten(pred)
        mask = torch.flatten(mask)

        # 计算交集
        overlap = (perd*mask).sum()

        # 求解分母: 前面的分母 + 后面的分母
        deum = perd.sum() + mask.sum() + 1e-8

        # 计算损失
        dice = (2 * overlap) / deum

        return 1-dice


# 定义损失函数
loss_fn = dice_loss()

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 动态调整优化器, 跟踪它, 检测越来越小
scheduler = ReduceLROnPlateau(optimizer, 'min')

# 使用 tensorboard 可视化, 写入路径 ./log
writer = SummaryWriter(log_dir='./log')


"""
计算测试集的 Loss
"""


# 测试集不需要计算梯度
def check_test_loss(test_dataloader, model):
    # 记录 test_loss
    test_loss = 0

    # 不记录梯度
    with torch.no_grad():
        # 遍历测试数据
        for i, (x, y) in enumerate(test_dataloader):
            # 获取图像
            x = x.to(device, dtype=torch.float32)

            # 获取标注
            y = y.to(device, dtype=torch.float32)

            # 获取预测值
            test_y_pre = model(x)

            # 计算损失
            test_loss_batch = loss_fn(test_y_pre, y)

            # 损失累加
            test_loss += test_loss_batch

    # 计算平均损失
    test_loss = test_loss / len(test_dataloader)

    return test_loss


"""
开始训练, 训练集进行训练
"""
# 设置训练批次
EPOCH_NUM = 200

# 记录最小的测试 loss
best_test_loss = 100

for epoch in range(EPOCH_NUM):

    # 整批数据的loss, loss_batch是计算每个 batch 的损失
    loss = 0

    # 记录一个 epoch 运行的时间
    start_time = time.time()

    # 获取每一批次图像信息
    for i, (x, y) in enumerate(train_dataloader):
        # 每次更新之前清空梯度
        model.zero_grad()

        # 获取图像
        x = x.to(device, dtype=torch.float32)

        # 获取标注
        y = y.to(device, dtype=torch.float32)

        # 获取预测值
        y_pre = model(x)

        # 计算 loss, 计算 loss 是在一个批次上来计算损失的
        loss_batch = loss_fn(y_pre, y)

        # 计算梯度
        loss_batch.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 获取每个batch的训练 loss, 在 gpu 上, 需要放在 cpu 上进行查看
        loss_batch = loss_batch.detach().cpu().item()

        # 计算训练一次, 所有 batch 上累计的loss
        loss += loss_batch

    # 计算训练一次, 训练集的: 平均损失 loss, 此时 loss 才是训练一个 epoch 的 loss
    loss = loss / len(train_dataloader)

    # 降低LR, 如果Loss连续10个epoch不再下降, 则降低LR
    scheduler.step(loss)

    # 计算测试集Loss
    test_loss = check_test_loss(test_dataloader, model)

    # 记录到 tensorboard 可视化, LOSS/train 传入 train 的 loss
    writer.add_scalar('LOSS/train', loss, epoch)
    writer.add_scalar('LOSS/tset', test_loss, epoch)

    # 保存最佳模型, 使用test_loss 进行评估
    if best_test_loss > test_loss:

        best_test_loss = test_loss

        # 保存模型
        torch.save(model.state_dict(), './save_model/unet_best_model.pt')

        # 输出信息
        print("第{}个epoch达到最低测试loss".format(epoch))

    # 打印输出信息
    print("第{}个epoch执行时间为{}s,train_loss={}, test_loss={}".format(
        epoch,
        time.time() - start_time,
        loss,
        test_loss
    ))

    #



































