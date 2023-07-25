"""
自定义数据集
1. 构造 torch 自定义的 dataset
2. 数据增强
3. dataloader 加载
4. 测试一组数据
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
from torch.utils.data import Dataset, DataLoader
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

""""
自定义数据集 类函数(无数据增强)
"""


class SegmentDataset(Dataset):

    def __init__(self, where='train'):
        self.img_list = glob.glob('./processed/{}/*/img_*'.format(where))

    def __len__(self):
        # 返回数据大小
        return len(self.img_list)

    def __getitem__(self, idx):
        # 获取具体某一个数据的 图片, mask

        # 获取图片
        img_file = self.img_list[idx]
        mask_file = img_file.replace('img', 'label')

        # 获取图片
        img = np.load(img_file)

        # 获取 mask
        mask = np.load(mask_file)

        # 对灰度图扩张维度变成张量
        return np.expand_dims(img, 0), np.expand_dims(mask, 0)


"""
测试获取一张图
"""
dataset = SegmentDataset(where='train')

# 随机显示 16 张图
fig = plt.figure(figsize=(10, 10))

# 对照片进行迭代
for i in range(16):

    plt.subplot(4, 4, i+1)

    img, mask = dataset[i+100]

    plt.imshow(img[0], cmap='bone')

    mask = np.ma.masked_where(mask[0] == 0, mask[0])

    plt.imshow(mask, alpha=0.8, cmap='spring')

    plt.axis('off')

plt.show()


"""
自定义数据集 类函数
包含数据增强
"""


class SegmentDataset(Dataset):

    def __init__(self, where='train', seq=None):

        # 获取数据
        self.img_list = glob.glob('./processed/{}/*/img_*'.format(where))

    def __len__(self):

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
        scale=(0.8, 1.2),             # 缩放
        rotate=(-45, 45)              # 旋转
    ),
    iaa.ElasticTransformation()       # 变换
])


dataset = SegmentDataset('train', seq)

# 对同一个图片显示多次
fig = plt.figure(figsize=(12, 12))

for i in range(16):

    plt.subplot(4, 4, i+1)

    img, mask = dataset[300]

    plt.imshow(img[0], cmap='bone')

    mask = np.ma.masked_where(mask[0] == 0, mask[0])

    plt.imshow(mask, alpha=0.8, cmap='spring')

    plt.axis('off')

plt.show()


# 加载 dataset
train_dataset = SegmentDataset(where='train', seq=seq)
test_dataset = SegmentDataset(where='test', seq=None)

# 加载 dataloader
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 显示一个批次
for x in train_dataloader:

    for i in range(16):

        plt.subplot(4, 4, i+1)

        img, mask = x[0][i], x[1][i]

        plt.imshow(img[0], cmap='bone')

        mask = np.ma.masked_where(mask[0] == 0, mask[0])

        plt.imshow(mask, alpha=0.8, cmap='spring')

        plt.axis('off')

    plt.show()
    break
















