"""
测试模型
# 1.从权重中加载
# 2.对test_dataset做测试，GD和prediction分别显示出来，以作对比
# 3.生成动画
# 4.对imagesTs下的文件也测试一下(从未见过)
"""
import torch
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import glob
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import tqdm
from celluloid import Camera
from IPython.display import HTML
import glob
import os
import nibabel as nib
import cv2


class DoubleConv(torch.nn.Module):
    """
    Helper Class which implements the intermediate Convolutions
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.step = torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
                                        torch.nn.ReLU())

    def forward(self, x):

        return self.step(x)


class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = DoubleConv(1, 64)
        self.layer2 = DoubleConv(64, 128)
        self.layer3 = DoubleConv(128, 256)
        self.layer4 = DoubleConv(256, 512)

        self.layer5 = DoubleConv(512 + 256, 256)
        self.layer6 = DoubleConv(256 + 128, 128)
        self.layer7 = DoubleConv(128 + 64, 64)
        self.layer8 = torch.nn.Conv2d(64, 1, 1)

        self.maxpool = torch.nn.MaxPool2d(2)
        # self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x1 = self.layer1(x)
        x1m = self.maxpool(x1)

        x2 = self.layer2(x1m)
        x2m = self.maxpool(x2)

        x3 = self.layer3(x2m)
        x3m = self.maxpool(x3)

        x4 = self.layer4(x3m)

        x5 = torch.nn.Upsample(scale_factor=2, mode="bilinear")(x4)
        x5 = torch.cat([x5, x3], dim=1)
        x5 = self.layer5(x5)

        x6 = torch.nn.Upsample(scale_factor=2, mode="bilinear")(x5)
        x6 = torch.cat([x6, x2], dim=1)
        x6 = self.layer6(x6)

        x7 = torch.nn.Upsample(scale_factor=2, mode="bilinear")(x6)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.layer7(x7)

        ret = self.layer8(x7)

        # ret = self.sigmoid(ret)

        return ret


# 定义设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 模型实例化
model = UNet().to(device)

# 从权重中恢复
model.load_state_dict(torch.load('./save_model/no_weight_sample/unet_best.pt'))

# 模型验证
model.eval()


"""
自定义数据集
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


# 数据增强处理流程
seq = iaa.Sequential([
    iaa.Affine(
        scale=(0.8, 1.2),                     # 缩放
        rotate=(-45, 45)),                    # 旋转
    iaa.ElasticTransformation()               # 变换

])

# 使用dataloader加载
batch_size = 12
num_workers = 0

test_dataset = SegmentDataset('test', seq=None)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


"""
预测图片和真实图片进行绘制, 捕捉成视频
"""
# 将每层画面制作成视频
fig = plt.figure(figsize=(10, 10))
camera = Camera(fig)

# 遍历所有数据
index = 0
for x, y in tqdm.tqdm(test_dataset):

    # 输出输入
    input = torch.tensor([x]).to(device, dtype=torch.float32)

    # 推理
    y_pred = model(input)

    # 获取mask
    mask_data = (y_pred.detach().cpu().numpy()[0][0] > 0.5)

    plt.subplot(1, 2, 1)
    plt.imshow(x[0], cmap='bone')
    mask_ = np.ma.masked_where(y[0] == 0, y[0])
    plt.imshow(mask_, alpha=0.8, cmap="spring")
    plt.title('truth')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(x[0], cmap='bone')
    mask_ = np.ma.masked_where(mask_data == 0, mask_data)
    plt.imshow(mask_, alpha=0.8, cmap="spring")
    plt.title('prediction')
    plt.axis('off')

    camera.snap()

    index += 1
    if index > 500:
        break

animation = camera.animate()


"""
对未知的图片进行预测 mask, 需要对未知的图片 和 训练的图片进行一样的数据处理 
"""
test_file_list = glob.glob('./data/imagesTs/lun*')


# 标准化
def standardize(data):

    # 计算均值
    mean = data.mean()

    # 计算标准差
    std = np.std(data)

    # 计算结果
    standardized = (data - mean) / std

    return standardized


# 归一化
def normalize(data):

    # 计算最大最小值
    max_val = data.max()
    min_val = data.min()

    normalized = (data - min_val) / (max_val - min_val)

    return normalized


# 挑选一个文件测试
# 读取文件
# 裁剪边缘
# 标准化和归一化
file = test_file_list[3]

# 读取
img = nib.load(file)
img_data = img.get_fdata()

# 标准化和归一化
std = standardize(img_data)
normalized = normalize(std)

# 将每层画面制作成视频
fig = plt.figure()
camera = Camera(fig)

# 去图片的通道数
layer_num = normalized.shape[-1]

for i in tqdm.tqdm(range(layer_num)):

    # 获取该层图像
    layer = normalized[:, :, i]

    # 缩放
    layer = cv2.resize(layer, (256, 256))

    # 输出输入
    input = torch.tensor([[layer]]).to(device, dtype=torch.float32)

    # 推理
    y_pred = model(input)

    # 获取mask
    mask_data = (y_pred.detach().cpu().numpy()[0][0] > 0.5)

    plt.imshow(layer, cmap='bone')
    mask_ = np.ma.masked_where(mask_data == 0, mask_data)
    plt.imshow(mask_, alpha=0.8, cmap="spring")
    plt.title('prediction')
    plt.axis('off')

    camera.snap()

animation = camera.animate()

HTML(animation.to_html5_video())














