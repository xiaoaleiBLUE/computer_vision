"""
author: xiaoalei
canny 边缘检测算法
主要步骤:
1. 高斯模糊降噪
2. 使用 Sobel filter 计算图像像素梯度
3. NMS 非最大值抑制计算局部最大值
4. Hysteresis thresholding 滞后阈值法过滤

canny 的两个参数T_lower, T_upper
注意两个值的选择
A 高于阈值 maxVal，所以是真正的边界点
C虽然低于 maxVal, 但高于 minVal, 并且与A相连, 所以也被认为真正的边界点
B则会被抛弃, 不仅低于 maxVal， 而且不与真正的边界点相连
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('./test_imgs/pumpkin.jpg')

img_fixed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

edges_1 = cv2.Canny(img.copy(), 100, 200)
edges_2 = cv2.Canny(img.copy(), 50, 200)
edges_3 = cv2.Canny(img.copy(), 50, 100)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 8), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(img.copy(), cmap='gray')

ax2.axis('off')
ax2.imshow(edges_1, cmap='gray')

ax3.axis('off')
ax3.imshow(edges_2, cmap='gray')

ax4.axis('off')
ax4.imshow(edges_3, cmap='gray')

plt.show()







