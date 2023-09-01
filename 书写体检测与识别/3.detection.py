"""
author: xiaoalei
检测书法文字
步骤:
1. 读取图片, 灰度, 二值化处理
2. 侵蚀去噪点
3. 膨胀连接
4. 闭合孔洞
5. 边缘检测
6. 画检测框
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


# 读取
img = cv2.imread('./test_imgs/shufa.jpg')

# 灰度化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 显示灰度图
plt.imshow(gray, cmap='gray')
plt.show()


"""
二值化处理
进行二值化处理, 在灰度图中 --> 0 越黑    --> 255 越白
参数含义:
参数1: 输入图片: gray
参数2: 比较阈值: 100
参数3: 超出阈值被设定的值: 超过100, 被设定为 255
参数4: 模式: cv2.THRESH_BINARY
作用: 将画面像素与比较阈值对比, 小于它则设为0(黑色), 大于它则设为目标值:255(白色)
"""
r, black_img = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

plt.imshow(black_img, cmap='gray')
plt.show()


"""
边缘检测
"""
edges = cv2.Canny(black_img, 30, 200)

plt.imshow(edges, cmap='gray')
plt.show()


"""
找轮廓, 遍历轮廓
"""
coutours, h = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

img_copy = img.copy()

for c in coutours:
    # 获取坐标框
    x, y, w, h = cv2.boundingRect(c)

    # 进行绘制
    cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 3)

plt.imshow(img_copy)
plt.show()


"""
进行形态学变换, 基于二值化后的图片进行处理
先侵蚀, 去除噪点
"""
kernel = np.ones((3, 3), dtype=np.int8)
erosion_1 = cv2.erode(black_img, kernel, iterations=1)

plt.imshow(erosion_1, cmap='gray')
plt.show()


"""
在膨胀
"""
kernel = np.ones((10, 10), dtype=np.int8)
dilation = cv2.dilate(erosion_1, kernel, iterations=2)

plt.imshow(dilation, cmap='gray')
plt.show()


"""
膨胀后的结果进行闭合
"""
kernel = np.ones((10, 10), dtype=np.int8)
closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

plt.imshow(closing, cmap='gray')
plt.show()


"""
闭合后的边缘检测
"""
edges_1 = cv2.Canny(closing, 30, 200)

plt.imshow(edges_1, cmap='gray')
plt.show()


"""
再次找轮廓, 遍历轮廓
"""
coutours, h = cv2.findContours(edges_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

img_copy = img.copy()

for c in coutours:
    # 获取坐标框
    x, y, w, h = cv2.boundingRect(c)

    # 进行绘制
    cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 3)

plt.imshow(img_copy)
plt.show()



