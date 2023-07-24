"""
预处理步骤:
1. 读取 NIFTI 格式文件, 加载图片与 mask
2. 显示一层出来(mask)
3. 动态显示整个扫描(多层)
4. 构造归一化, 标准化函数
5. 处理所有文件, 保存为 np 文件
6. 检查 np 文件
"""
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera
from IPython.display import HTML
import tqdm
import glob
import cv2
import os
from natsort import natsorted


# 挑选一个文件进行测试
test_file = './data/imagesTr/lung_014.nii.gz'

test_mask = './data/labelsTr/lung_014.nii.gz'

# nibabel 读取信息
img = nib.load(test_file)
mask = nib.load(test_mask)

# 获取其中数据
img_data = img.get_fdata()
mask_data = mask.get_fdata()

# 查看形状
print(img_data.shape)

# 绘制某一通道的图片
plt.imshow(img_data[:, :, 380], cmap='bone')
plt.show()

# 绘制某一通道的mask
plt.imshow(mask_data[:, :, 380], cmap='bone')
plt.show()


# 将 mask 和 图片 绘制在一起
img_display = img_data[:, :, 380]
mask_display = mask_data[:, :, 380]

# 将 mask_display 像素值为 0 处遮挡起来
mask = np.ma.masked_where(mask_display == 0, mask_display)
plt.imshow(img_display, cmap='bone')
plt.imshow(mask, alpha=0.8, cmap='spring')
plt.show()

"""
遍历每一层画面和mask
"""


fig = plt.figure()
camera = Camera(fig)

for i in tqdm.tqdm(range(img_data.shape[-1])):

    # 将 mask 和 图片 绘制在一起
    img_display = img_data[:, :, i]
    mask_display = mask_data[:, :, i]

    # 将 mask_display 像素值为 0 处遮挡起来
    mask = np.ma.masked_where(mask_display == 0, mask_display)
    plt.imshow(img_display, cmap='bone')
    plt.imshow(mask, alpha=0.8, cmap='spring')

    plt.axis('off')

    camera.snap()

# 显示动画
animation = camera.animate()
HTML(animation.to_html5_video())
plt.show()


# 标椎化
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

    # 计算最大值与最小值
    max_val = data.max()
    min_val = data.min()

    # 计算结果
    normalized = (data - min_val) / (max_val - min_val)

    return normalized


"""
处理所有文件
"""
train_file_list = glob.glob('./data/imagesTr/lung*')
train_label_list = glob.glob('./data/labelsTr/lung*')


for index, file in tqdm.tqdm(enumerate(train_file_list)):
    """
    读取文件和 label
    标准化 和 归一化
    缩放至 256*256
    存入文件夹
    """
    img = nib.load(file)
    mask = nib.load(train_label_list[index])

    img_data = img.get_fdata()
    mask_data = mask.get_fdata().astype(np.uint8)

    # 标准化和归一化
    std = standardize(img_data)
    normalized = normalize(std)

    # 分为训练数据和测试数据
    if index < 57:
        save_dir = './processed/train/'

    else:
        save_dir = './processed/test/'

    # 查看本次迭代图片的通道层数图片的数值, 返回的是通道数
    layer_num = normalized.shape[-1]

    # 对图片的所有层(通道数)进行迭代
    for i in range(layer_num):

        # 查看该通道图片数据(1通道的灰度图)
        img_layer = normalized[:, :, i]

        # 查看该通道mask数据
        mask_layer = mask_data[:, :, i]

        # 进行缩放至 256*256
        img_layer = cv2.resize(img_layer, (256, 256))
        mask_layer = cv2.resize(mask_layer, (256, 256), interpolation=cv2.INTER_NEAREST)

        # 创建文件夹
        img_dir = save_dir + str(index)

        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        # 保存为 npz 文件
        np.save(img_dir + '/img_' + str(i), img_layer)
        np.save(img_dir + '/label_' + str(i), mask_layer)
print('done')

"""
测试一组数据
解决排序问题
"""
img_test = natsorted(glob.glob('./processed/train/15/img*'))
label_test = natsorted(glob.glob('./processed/train/15/label*'))

# 进行测试
fig = plt.figure()
camera = Camera(fig)

for index, img_file in enumerate(img_test):

    # 直接进行加载就行
    img_data = np.load(img_file)
    mask_data = np.load(label_test[index])

    mask_data = np.ma.masked_where(mask_data == 0, mask_data)

    # 进行绘制
    plt.imshow(img_data, cmap='bone')
    plt.imshow(mask_data, alpha=0.8, cmap='spring')
    plt.axis('off')

    camera.snap()

animation = camera.animate()

# 显示动画
HTML(animation.to_html5_video())
plt.show()











































