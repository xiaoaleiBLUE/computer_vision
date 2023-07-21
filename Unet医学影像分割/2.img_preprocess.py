"""
图片预处理
预处理步骤:
1. 读取 NIFIT 格式文件, 加载其中的图片和 mask
2. 显示一层图片(包含 mask)
3. 动态显示整个扫描(显示多层信息)
4. 构造归一化, 标准化函数
5. 预处理所有文件, 保存为 npz 文件, 存在磁盘
6. 检查一下存储的 npz 文件是否合格
"""
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera
from IPython.display import HTML
import tqdm
from matplotlib import animation
import glob
import os
from natsort import natsorted                               # 解决列表排序问题


"""
扫描其中的第 80 层, 总共有 122 层
"""
# 挑一个文件进行测试
test_file = './data/imagesTr/la_018.nii.gz'

# 对应的 mask 标注文件
test_mask = './data/labelsTr/la_018.nii.gz'

# nibabel 读取图像信息和 mask
img = nib.load(test_file)
mask = nib.load(test_mask)

# 获取其中数据
img_data = img.get_fdata()
mask_data = mask.get_fdata()

# 查看大小, 是个三维的
print(img_data.shape, mask_data.shape)                       # (320, 320, 122) (320, 320, 122)

# img_data是三维的, 显示图片, 三维的
plt.imshow(img_data[:, :, 80], cmap='bone')
plt.show()

# 进行旋转 90 度, 取第 80 层
plt.imshow(np.rot90(img_data[:, :, 80]), cmap='bone')
plt.show()

# 绘制 mask
plt.imshow(np.rot90(mask_data[:, :, 80]), cmap='bone')
plt.show()

# 将 mask 和图片绘制在一起
img_display = np.rot90(img_data[:, :, 80])
mask_display = np.rot90(mask_data[:, :, 80])

# 将 mask_display 像素值为 0 处, 遮挡起来
mask = np.ma.masked_where(mask_display == 0, mask_display)

plt.imshow(img_display, cmap='bone')

# 颜色改为泛红
plt.imshow(mask, alpha=0.8, cmap='spring')
plt.show()


"""
将 122 层画面合称为动画
3. 动态显示整个扫描(显示多层信息)
"""
fig = plt.figure()
camera = Camera(fig)
for i in tqdm.tqdm(range(img_data.shape[-1])):

    # 显示
    # 将 mask 和图片绘制在一起
    img_display = np.rot90(img_data[:, :, i])
    mask_display = np.rot90(mask_data[:, :, i])

    # 将 mask_display 像素值为 0 处遮挡起来
    mask = np.ma.masked_where(mask_display == 0, mask_display)

    plt.imshow(img_display, cmap='bone')
    plt.imshow(mask, alpha=0.8, cmap='spring')

    plt.axis('off')

    camera.snap()

animation = camera.animate()

# 显示动画
HTML(animation.to_html5_video())             # conda install -c conda-forget ffmpeg
plt.show()


"""
归一化, 标准化
"""
# 标准化


def standardize(data):
    # 计算均值
    mean = data.mean()
    # 计算标准差
    std = np.std(data)

    # 计算结果
    standardize = (data - mean) / std

    return standardize


# 归一化
def normalize(data):
    # 计算最大值和最小值
    max_val = data.max()
    min_val = data.min()

    # 计算结果
    normalize = (data - min_val) / (max_val - min_val)

    return normalize


# 处理第 80 层
test_data = img_data[:, :, 80]

# 进行标准化
std = standardize(test_data)

# 进行归一化
nor = normalize(std)

"""
处理所有的文件, 分别为 train, test 
"""
# 获取文件列表(图片, 标注)
train_file_list = glob.glob('./data/imagesTr/la*')
train_label_list = glob.glob('./data/labelsTr/la*')

# 遍历每个文件, 每个文件包含多层图像, 保存为 np 文件
# file: 为遍历文件列表地址路径: ./data/imagesTr\la_003.nii.gz
for index, file in tqdm.tqdm(enumerate(train_file_list)):
    """
    对图片的边缘进行裁剪, 最终输出 256*256
    标准化, 归一化
    存入文件夹
    """
    # 读取
    img = nib.load(file)
    mask = nib.load(train_label_list[index])

    img_data = img.get_fdata()
    mask_data = mask.get_fdata()

    # 原图像为 320*320*122, 裁剪边缘后形成: 256*256
    img_data_crop = img_data[32:-32, 32:-32]
    mask_data_crop = mask_data[32:-32, 32:-32]

    # 标准化, 归一化
    std = standardize(img_data_crop)

    # 此时遍历当前的维度: 256*256*130
    normalized = normalize(std)

    # 分成训练集和测试集
    if index < 17:
        # 当成训练集
        save_dir = './processed/train/'
        pass

    else:
        # 当成测试集
        save_dir = './processed/test/'
        pass

    # 遍历文件的所有层, 每一层单独挑出来存储进文件夹,
    # 路径格式: 'processed/train/0/img_0.npy', 'processed/test/0/label_0.npy'
    # 0: 代表序号, 总共 index(20) 个文件, 分成序号: 0-19
    # img_0.npy, label_0.npy: 分别代表该文件的第一层图像以及对应的标注

    # 获取层数, shape的最后一个维度
    layer_num = normalized.shape[-1]

    for i in range(layer_num):

        # 获取第 i 层的图像及标注
        layer = normalized[:, :, i]
        mask = mask_data_crop[:, :, i]

        # 创建文件夹
        img_dir = save_dir + str(index)

        # 判断是否有文件夹, 没有就立即创建文件夹
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        # 保存为 numpy 文件
        np.save(img_dir + '/img_' + str(i), layer)
        np.save(img_dir + '/label_' + str(i), mask)


# 检查一组数据是否合格
img_test = natsorted(glob.glob('./processed/train/15/img*'))
label_test = natsorted(glob.glob('./processed/train/15/label*'))

print(len(img_test), len(label_test))


"""
把这些图片合成动画
"""
fig = plt.figure()
camera = Camera(fig)

for index, img_file in enumerate(img_test):

    # 加载图片
    img_data = np.load(img_file)
    mask_data = np.load(label_test[index])

    # 将 mask 和图片绘制在一起
    img_display = np.rot90(img_data)

    mask_display = np.rot90(mask_data)

    # 将 mask 的像素值为 0 处遮盖
    mask = np.ma.masked_where(mask_display == 0, mask_display)

    plt.imshow(img_display, cmap='bone')
    plt.imshow(mask, alpha=0.8, cmap='spring')

    plt.axis('off')

    camera.snap()

animation = camera.animate()

# 显示动画
HTML(animation.to_html5_video())             # conda install -c conda-forget ffmpeg
plt.show()





























































