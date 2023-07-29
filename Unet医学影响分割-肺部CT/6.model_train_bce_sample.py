"""
带加权采样的训练流程
1. 训练 Unet 模型
2. 自定义 loss 函数
3. 开始训练
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import glob
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from torch.utils.tensorboard import SummaryWriter


"""
定义自定义数据集
"""


class SegmentDataset(Dataset):

    def __init__(self, where='train', seq=None):
        # 获取数据
        self.img_list = glob.glob('processed/{}/*/img_*'.format(where))
        self.mask_list = glob.glob('processed/{}/*/img_*')
        # 数据增强pipeline
        self.seq = seq

    def __len__(self):
        # 返回数据大小
        return len(self.img_list)

    def __getitem__(self, idx):
        # 获取具体每一个数据

        # 获取图片
        img_file = self.img_list[idx]
        mask_file = img_file.replace('img', 'label')
        img = np.load(img_file)

        # 获取mask
        mask = np.load(mask_file)

        # 如果需要数据增强
        if self.seq:
            segmap = SegmentationMapsOnImage(mask, shape=mask.shape)
            img, mask = seq(image=img, segmentation_maps=segmap)
            # 直接获取数组内容
            mask = mask.get_arr()

        # 灰度图扩张维度成张量
        return np.expand_dims(img, 0), np.expand_dims(mask, 0)


# 数据增强处理
seq = iaa.Sequential([
    iaa.Affine(
        scale=(0.8, 1.2),           # 缩放
        rotate=(-45, 45)            # 旋转
    ),
    iaa.ElasticTransformation()     # 变换
])


# 获取训练集 此时 len(train_dataset) = 16194
train_dataset = SegmentDataset('train', seq=seq)
test_dataset = SegmentDataset('test', seq=None)

"""
大部分图片是不大带有肿瘤, 标签很少, 其实模型在学习没有在标注的信息(不是我们所需要的)
"""

# 检查一下有tumor(标注里面带有肿瘤的)的样本有多少
target_list = []

for img, label in tqdm.tqdm(train_dataset):

    # 带tumor(标注里面带有肿瘤)的赋值为1，否则为0
    if np.any(label):

        target_list.append(1)

    else:
        target_list.append(0)

# 统计
statistics = np.unique(target_list, return_counts=True)

# 查看样本中不带tumor的和带tumor的比例。可以看到大部分图片都是不带的(大部分图片都是不带肿瘤的)
ratio = statistics[1][0] / statistics[1][1]

# 构造权重列表，为不带tumor的样本设置权重1，带tumor的权重为ratio=9.69
# target_list中应该是一堆 0, 含有 1 的标签很少
weight_list = []

for target in target_list:

    if target == 0:
        weight_list.append(1)

    else:
        weight_list.append(ratio)

# 此时就会造成了原来不带 tumor的样本设置权重1, 带tumor的权重为ratio=9.69(被采样的次数更多)

# 统计
statistics = np.unique(weight_list, return_counts=True)

# 加权采样
sampler = torch.utils.data.sampler.WeightedRandomSampler(weight_list, len(weight_list))

# 使用dataloader加载
batch_size = 12

# 训练集 dataloader, (带有加权采样的)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

# 测试集 dataloader
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)


"""
定义 Unet 网络模型
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


# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 模型实例化
model = UNet().to(device)

# 损失函数
loss_fn = torch.nn.BCEWithLogitsLoss()

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 动态减少 LR
scheduler = ReduceLROnPlateau(optimizer, 'min')


# 计算测试集的loss
def check_test_loss(test_dataloader, model):

    loss = 0

    # 不记录梯度
    with torch.no_grad():

        for i, (x, y) in enumerate(test_dataloader):

            # 图片
            x = x.to(device, dtype=torch.float32)

            # 标签
            y = y.to(device, dtype=torch.float32)

            # 预测值
            y_pred = model(x)

            # 计算损失
            loss_batch = loss_fn(y_pred, y)

            loss += loss_batch

    return loss / len(test_dataloader)


"""
开始训练
"""

# 记录变量
writer = SummaryWriter(log_dir='./log')

# 训练100个epoch
EPOCH_NUM = 200
# 记录最好的测试acc
best_test_loss = 100

for epoch in range(EPOCH_NUM):
    # 获取批次图像
    start_time = time.time()
    loss = 0
    for i, (x, y) in enumerate(train_dataloader):

        # ！！！每次update前清空梯度
        model.zero_grad()

        # 获取数据
        # 图片
        x = x.to(device, dtype=torch.float32)

        # 标签
        y = y.to(device, dtype=torch.float32)

        # 预测值
        y_pred = model(x)

        # 计算损失
        loss_batch = loss_fn(y_pred, y)

        # 计算梯度
        loss_batch.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 记录每个batch的train loss
        loss_batch = loss_batch.detach().cpu()

        # 累计一个批次所有 batch 上的累计和
        loss += loss_batch

    # 每个epoch的loss
    loss = loss / len(train_dataloader)

    # 如果降低LR：如果loss连续10个epoch不再下降，就减少LR
    scheduler.step(loss)

    # 计算测试集的loss
    test_loss = check_test_loss(test_dataloader, model)

    # tensorboard 记录 Loss/train
    writer.add_scalar('Loss/train', loss, epoch)

    # tensorboard 记录 Loss/test
    writer.add_scalar('Loss/test', test_loss, epoch)

    # 记录最好的测试loss，并保存模型
    if best_test_loss > test_loss:

        best_test_loss = test_loss

        # 保存模型
        torch.save(model.state_dict(), './save_model/unet_best.pt')
        print('第{}个EPOCH达到最低的测试loss:{}'.format(epoch, best_test_loss))

    # 打印信息
    print('第{}个epoch执行时间：{}s，train loss为：{}，test loss为：{}'.format(
        epoch,
        time.time() - start_time,
        loss,
        test_loss
    ))
















