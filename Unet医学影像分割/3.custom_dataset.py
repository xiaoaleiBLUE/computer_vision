"""
自定义数据集(dataset)
1. 构造 torch 自定义的 dataset
2. 数据增强
3. 测试一下 dataLoader 加载
4. 测试显示一些图片
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import glob
from celluloid import Camera
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


"""
1. 构造 torch 自定义的 dataset
"""


# 构造自定义的 dataset
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


# 数据增强处理就成
seq = iaa.Sequential([
    iaa.Affine(
        scale=(0.8, 1.2),               # 缩放
        rotate=(-45, 45)                # 旋转
    ),
    iaa.ElasticTransformation()         # 弹性形变
])

"""
自定义dataset数据类, 只读取 train 文件夹下的数据
"""
# 加载训练集所有图片
dataset = SegmentDataset(where='train', seq=seq)
print(len(dataset))

# 显示某一个批次的图片
fig = plt.figure(figsize=(10, 10))
for i in range(16):

    plt.subplot(4, 4, i+1)

    # 获取数据
    img, mask = dataset[i]

    # 进行绘制在一起, 获取张量的第一个元素
    img_display = np.rot90(img[0])
    mask_display = np.rot90(mask[0])

    # 去除像素为 0
    mask = np.ma.masked_where(mask_display == 0, mask_display)

    plt.imshow(img_display, cmap='bone')
    plt.imshow(mask, alpha=0.8, cmap='spring')

plt.show()


"""
对同一张图进行变换处理后查看
"""
fig = plt.figure(figsize=(10, 10))
for i in range(16):

    plt.subplot(4, 4, i+1)

    # 获取数据
    img, mask = dataset[100]

    # 进行绘制在一起, 获取张量的第一个元素
    img_display = img[0]
    mask_display = mask[0]

    # 去除像素为 0
    mask = np.ma.masked_where(mask_display == 0, mask_display)

    plt.imshow(img_display, cmap='bone')
    plt.imshow(mask, alpha=0.8, cmap='spring')

plt.show()


"""
使用 dataloader 加载数据, 分为train_dataset, test_dataset(无需做数据增强)
"""
batch_size = 16

# 训练集 dataset
train_dataset = SegmentDataset(where='train', seq=seq)

# 测试集 dataset(无需做数据增强)
test_dataset = SegmentDataset(where='test', seq=None)

# 定义dataloader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=False)

# 查看一个批次, train_dataloader = 1932 / 16, 向上取整rain_dataloader=121
for x in train_dataloader:

    # len(x) = 2, 前半部分返回 img, 后半部分返回 mask

    plt.figure(figsize=(12, 12))

    # batch_size = 16, 一批有 16 张, 所以迭代 range(16)
    for i in range(16):

        plt.subplot(4, 4, i+1)

        # 获取图片及标注, 张量获取第一个维度
        img, mask = x[0][i][0], x[1][i][0]

        # 去除像素为 0
        mask = np.ma.masked_where(mask == 0, mask)

        plt.imshow(img, cmap='bone')
        plt.imshow(mask, alpha=0.8, cmap='spring')
    plt.show()
    break





















































