"""
预处理步骤
1. 读取 NIFIT 格式文件, 加载 图片 与 mask
2. 显示一层出来(包含 mask)
3. 动态显示多层(整个扫描)
4. 构造归一化, 标准化 函数
5. 处理所有文件, 保存为 np 文件
6. 检查文件
"""
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from celluloid import Camera
from IPython.display import HTML
import tqdm
import cv2
import os
from natsort import natsorted
import glob


# 读取 csv 文件
data = pd.read_csv('./data/metadata.csv')

# 查看前 5 个 文件
print(data.head(5))

# 获取第一行文件真实路径(data下面)
# 获取原始图像路径
ct_scan_sample_file = data.loc[0, 'ct_scan'].replace('../input/covid19-ct-scans', 'data')

# 获取肺部 mask 路径
lung_mask_sample_file = data.loc[0, 'lung_mask'].replace('../input/covid19-ct-scans', 'data')

# 获取感染的 mask 路径
infection_mask_sample_sample_file = data.loc[0, 'infection_mask'].replace('../input/covid19-ct-scans', 'data')

# 肺部和感染 mask
lung_and_infection_mask_sample_sample_file = data.loc[0, 'lung_and_infection_mask'].replace('../input/covid19-ct-scans', 'data')

print(ct_scan_sample_file)               # 'data/ct_scans/coronacases_org_001.nii'


# 读取 nifti 文件函数, 输入是: 路径
def read_nii_file(fileName):

    img = nib.load(fileName)
    img_data = img.get_fdata()

    img_data = np.rot90(np.array(img_data))

    return img_data


# 对上述四个路径进行读取
ct_scan_imgs = read_nii_file(ct_scan_sample_file)

lung_mas_imgs = read_nii_file(lung_mask_sample_file)

infection_mask_imgs = read_nii_file(infection_mask_sample_sample_file)

lung_and_infection_mas_imgs = read_nii_file(lung_and_infection_mask_sample_sample_file)

# 查看大小
print(ct_scan_imgs.shape, lung_mas_imgs.shape)        # (512, 512, 301), (512, 512, 301)

"""
对上述进行绘制四个路径读取的图片进行绘制
"""
# 绘制的具体通道数
layer_index = 180

fig = plt.figure(figsize=(20, 4))

plt.subplot(1, 4, 1)
plt.imshow(ct_scan_imgs[:, :, layer_index], cmap='bone')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(ct_scan_imgs[:, :, layer_index], cmap='bone')
mask_ = np.ma.masked_where(lung_mas_imgs[:, :, layer_index] == 0, lung_mas_imgs[:, :, layer_index])
plt.imshow(mask_, alpha=0.8, cmap='spring')
plt.title('Lung Mask')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(ct_scan_imgs[:, :, layer_index], cmap='bone')
mask_ = np.ma.masked_where(infection_mask_imgs[:, :, layer_index] == 0, infection_mask_imgs[:, :, layer_index])
plt.imshow(mask_, alpha=0.8, cmap='spring')
plt.title('Infection Mask')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(ct_scan_imgs[:, :, layer_index], cmap='bone')
mask_ = np.ma.masked_where(lung_and_infection_mas_imgs[:, :, layer_index] == 0, lung_and_infection_mas_imgs[:, :, layer_index])
plt.imshow(mask_, alpha=0.8, cmap='spring')
plt.title('Lung and Infection Mask')
plt.axis('off')

plt.show()


"""
将每一个通道捕捉成动态视频
"""
fig = plt.figure(figsize=(10, 10))
camera = Camera(fig)

for layer_index in tqdm.tqdm(range(ct_scan_imgs.shape[-1])):

    plt.subplot(1, 2, 1)
    plt.imshow(ct_scan_imgs[:, :, layer_index], cmap='bone')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(ct_scan_imgs[:, :, layer_index], cmap='bone')
    mask_ = np.ma.masked_where(lung_and_infection_mas_imgs[:, :, layer_index] == 0, lung_and_infection_mas_imgs[:, :, layer_index])
    plt.imshow(mask_, alpha=0.8, cmap='spring')
    plt.title('Lung and Infection Mask')
    plt.axis('off')

    camera.snap()

animation = camera.animate()

# 显示动画
HTML(animation.to_html5_video())
plt.show()

"""
标准化, 归一化
"""


# 标准化
def standardize(data):
    # 计算均值
    mean = data.mean()

    # 计算标准度
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


# 读取训练集文件路径列表
train_file_list = [file_path.replace('../input/covid19-ct-scans', 'data') for file_path in data.loc[:, 'ct_scan']]

# 读取训练集文件标签列表
train_label_list = [file_path.replace('../input/covid19-ct-scans', 'data') for file_path in data.loc[:, 'infection_mask']]

# 查看文件数量大小(每个文件中都包含多层的图片)
print(len(train_file_list))                   # 20

for index, file in tqdm.tqdm(enumerate(train_file_list)):
    """
    读取文件和label
    标准化和归一化
    存入文件夹
    缩放至模型所需大小256
    """

    # 读取
    img = nib.load(file)
    mask = nib.load(train_label_list[index])

    img_data = img.get_fdata()
    mask_data = mask.get_fdata().astype(np.uint8)

    # 标准化和归一化
    std = standardize(img_data)
    normalized = normalize(std)

    # 分为训练数据和测试数据
    if index < 17:
        save_dir = 'processed/train/'
    else:
        save_dir = 'processed/test/'

    # 遍历所有(通道层)层，分层存入文件夹，存储路径格式：'processed/train/0/img_0.npy'，'processed/train/0/label_0.npy'，
    layer_num = normalized.shape[-1]

    for i in range(layer_num):

        layer = normalized[:, :, i]

        mask = mask_data[:, :, i]

        # 缩放
        layer = cv2.resize(layer, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        # 创建文件夹
        img_dir = save_dir + str(index)

        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        # 保存为npy文件
        np.save(img_dir + '/img_' + str(i), layer)
        np.save(img_dir + '/label_' + str(i), mask)


"""
对数据抽取一组进行测试
"""
img_test = natsorted(glob.glob('processed/train/10/img*'))

label_test = natsorted(glob.glob('processed/train/10/label*'))


"""
进行捕捉动画
"""
fig = plt.figure()

camera = Camera(fig)

for index, img_file in enumerate(img_test):

    img_data = np.load(img_file)
    mask_data = np.load(label_test[index])

    plt.imshow(img_data, cmap='bone')
    mask_ = np.ma.masked_where(mask_data == 0, mask_data)

    plt.imshow(mask_, alpha=0.8, cmap="spring")
    plt.axis('off')

    camera.snap()

animation = camera.animate()

# 显示动画
HTML(animation.to_html5_video())
plt.show()















