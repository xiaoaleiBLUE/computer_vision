"""
任务: 将书法图片进行分成 5 类

1. 读取数据
2. 提取 HOG 特征
3. 送至 SVM 训练
4. 评估模型
5. 保存模型
6. 可视化看一下训练效果
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
import os
import glob
import random
import tqdm
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


"""
读取数据
"""
# cv2.imread()不支持中文路径的读取
# 定义函数能读取中文路径的, 解决读取中文路径的读取
def readImg(filePath):

    raw_data = np.fromfile(filePath, dtype=np.uint8)
    img = cv2.imdecode(raw_data, -1)                          # 也可查看图片的 shape
    return img


# 重新进行读取后, 可直接显示
img_1 = readImg('./images/行书/丙/敬世江_5945d30c02c1e30a89de9dc3920a0011adc9aa46.jpg')

# 进行缩放
img_1 = cv2.resize(img_1, (200, 200))

# 转为 gary
img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)

plt.imshow(img_1_gray)
plt.show()


"""
提取 hog 特征, 输入是 gray 图像
"""
fd, hog_image = hog(image=img_1_gray, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True)

hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 8), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(img_1_gray, cmap='gray')

ax2.axis('off')
ax2.imshow(hog_image, cmap='gray')

ax3.axis('off')
ax3.imshow(hog_image_rescaled, cmap='gray')

plt.show()


"""
批量读取文件数据
提取 HOG 特征
"""
list_1 = glob.glob('./images/行书/*/*')

# 打乱数据
a = [1, 2, 3, 4, 5]
random.shuffle(a)                  # 打印输出的a, a = [2, 1, 5, 4, 3]


"""
定义函数, 读取中文图片图片路径, 进行缩放, 转为灰度图
"""
def imageReader(filename):
    # 进行中文路径读取
    raw_data = np.fromfile(filename, dtype=np.uint8)
    img = cv2.imdecode(raw_data, -1)

    # 进行缩放
    img = cv2.resize(img, (200, 200))

    # 转为灰度图
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img_gray


# 对应类别列表
style_list = ['篆书', '隶书', '草书', '行书', '楷书']

# 建立特征列表
feature_list = []

# 建立标签列表
label_list = []

for style in style_list:
    # 加一些中文
    print('开始遍历: {style}'.format(style=style))
    # 列出该风格下所有的文件
    file_list = glob.glob('./images/{}/*/*'.format(style))

    # 对 每一个类别的 file_list 进行随机打乱 文件路径名
    random.shuffle(file_list)

    # 进行选择前 1000 个
    selected_files = file_list[:1000]

    # 进行解析, 遍历文件名列表
    for file_item in tqdm.tqdm(selected_files):
        # 打开图片, 进行缩放, 转为灰度图
        img_gray = imageReader(file_item)

        fd = hog(image=img_gray, orientations=4, pixels_per_cell=(6, 6),
                 cells_per_block=(2, 2))
        # 特征列表
        feature_list.append(fd)

        # 提取图片对应的类别标签, 类别标签是中文, 可以用索引代表类别
        label = style_list.index(style)

        # 标签列表
        label_list.append(label)


print(len(label_list), len(feature_list[0]))                      # 5000 5000


"""
导入 SVM 模型
"""

# 将样本分为训练样本和测试样本, 传入特征列表 和 标签列表, 测试样本 25%
# 返回四个变量: x_train: 训练样本特征 x_test: 测试样本特征 y_train: 训练样本标签 y_test: 测试样本标签
x_train, x_test, y_train, y_test = train_test_split(feature_list, label_list, test_size=0.25, random_state=42)

print(len(x_train), len(y_train))

print(len(x_test), len(y_test))

# 训练
cls = svm.SVC()

# x_train: 训练样本特征  y_train: 训练样本标签
cls.fit(x_train, y_train)

# 根据训练结果, 输入 x_test, 可预测 x_test 对应的 样本标签
# predicted_labels: 预测的样本标签
predicted_labels = cls.predict(x_test)


"""
评估准确率
"""
acc_score = accuracy_score(y_test, predicted_labels)
print(acc_score)                                              # 0.7336


"""
更换 核 函数
"""
cls = svm.SVC(kernel='linear')
cls.fit(x_train, y_train)

predicted_labels = cls.predict(x_test)

acc_score = accuracy_score(y_test, predicted_labels)
print(acc_score)                                               # 0.6848


cls = svm.SVC(kernel='poly')
cls.fit(x_train, y_train)

predicted_labels = cls.predict(x_test)

acc_score = accuracy_score(y_test, predicted_labels)
print(acc_score)                                               # 0.6992


"""
保存训练好的模型
"""
dump(cls, './models/poly.joblib')


"""
加载模型
"""
new_cls = load('./models/poly.joblib')








