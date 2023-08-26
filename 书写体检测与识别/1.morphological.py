"""
author: xiaoalei
1. opencv 读取图片并显示
2. 对读取的图片进行侵蚀操作
3. 对读取的图片进行膨胀操作
4. 对图片进行 opening 操作
5. 对图片进行 closing 操作
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


"""
1. opencv 读取图片并显示
"""
img = cv2.imread('./test_imgs/j.png')
print(img.shape)

# 灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.shape)

# 进行显示
plt.imshow(gray)
plt.show()


"""
2. 对读取的图片进行侵蚀操作
侵蚀: 去除小的白色噪点, 将两个连起来的形状打散
"""
kernel = np.ones((3, 3), dtype=np.int8)                            # 定义核的大小: 3*3
ersion_1 = cv2.erode(gray.copy(), kernel, iterations=1)            # iterations: 迭代次数

plt.imshow(ersion_1, cmap='gray')
plt.show()

kernel = np.ones((5, 5), dtype=np.int8)                            # 定义核的大小: 5*5
ersion_2 = cv2.erode(gray.copy(), kernel, iterations=1)            # iterations: 迭代次数

plt.imshow(ersion_2, cmap='gray')
plt.show()

kernel = np.ones((5, 5), dtype=np.int8)                            # 定义核的大小: 5*5
ersion_3 = cv2.erode(gray.copy(), kernel, iterations=2)            # iterations: 迭代次数

plt.imshow(ersion_3, cmap='gray')
plt.show()

"""
多张图片绘制在一起
"""
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 8), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(gray, cmap='gray')
ax1.set_title('orginal image')

ax2.axis('off')
ax2.imshow(ersion_1, cmap='gray')
ax2.set_title('3*3, 1')

ax3.axis('off')
ax3.imshow(ersion_2, cmap='gray')
ax3.set_title('5*5, 1')

ax4.axis('off')
ax4.imshow(ersion_3, cmap='gray')
ax3.set_title('5*5, 2')

plt.show()


"""
3. 对读取的图片进行膨胀操作
膨胀: 跟在侵蚀操作后去噪点, 把两个分开的部分连接起来
"""
# 3*3, 1
kernel = np.ones((3, 3), dtype=np.int8)
dilattion_1 = cv2.dilate(gray.copy(), kernel, iterations=1)

# 5*5, 1
kernel = np.ones((5, 5), dtype=np.int8)
dilattion_2 = cv2.dilate(gray.copy(), kernel, iterations=1)

# 5*5, 2
kernel = np.ones((5, 5), dtype=np.int8)
dilattion_3 = cv2.dilate(gray.copy(), kernel, iterations=2)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 8), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(gray.copy(), cmap='gray')
ax1.set_title('orginal image')

ax2.axis('off')
ax2.imshow(dilattion_1, cmap='gray')
ax2.set_title('3*3, 1')

ax3.axis('off')
ax3.imshow(dilattion_2, cmap='gray')
ax3.set_title('5*5, 1')

ax4.axis('off')
ax4.imshow(dilattion_3, cmap='gray')
ax4.set_title('5*5, 2')

plt.show()


"""
4. 对图片进行 opening 操作
opening, 张开
侵蚀 + 膨胀
主要用于清楚噪点
"""
img = cv2.imread('./test_imgs/cv.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(gray, cmap='gray')
plt.show()

# 张开操作
kernel = np.ones((10, 10), dtype=np.int8)
opening_1 = cv2.morphologyEx(gray.copy(), cv2.MORPH_OPEN, kernel)

kernel = np.ones((12, 12), dtype=np.int8)
opening_2 = cv2.morphologyEx(gray.copy(), cv2.MORPH_OPEN, kernel)

kernel = np.ones((15, 15), dtype=np.int8)
opening_3 = cv2.morphologyEx(gray.copy(), cv2.MORPH_OPEN, kernel)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 8), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(gray.copy(), cmap='gray')
ax1.set_title('orginal image')

ax2.axis('off')
ax2.imshow(opening_1, cmap='gray')
ax2.set_title('10*10')

ax3.axis('off')
ax3.imshow(opening_2, cmap='gray')
ax3.set_title('12*12')

ax4.axis('off')
ax4.imshow(opening_3, cmap='gray')
ax4.set_title('15*15')

plt.show()


"""
5. 对图片进行 closing 操作
closing, 闭合
先膨胀 + 侵蚀
主要用于闭合主体内的小洞, 或者一些黑色的点
"""
kernel = np.ones((15, 15), dtype=np.int8)
closing_1 = cv2.morphologyEx(gray.copy(), cv2.MORPH_CLOSE, kernel)


































