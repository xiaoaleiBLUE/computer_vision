# The Computer_vision
---


@[TOC](文章目录)

---

# 前言
这个github项目是记录自己在学习计算机视觉项目方面的编写代码的基础，包括目标检测、人脸识别、关键点检测、文字识别、Gan网络等，其中的收获也很多, 代码编写能力也得到了一些提升，自己的算法能力也在以后继续加强，希望自己每天能及时在代码方面更新一点知识,同时自己的CSDN博客也会更新相关的文章。

---
# 一、常用的python基本语法

>- 在写python 程序之前，搭建环境是第一步，自己在这之前搭建好多环境，有时候自己也是把环境搭建的特别乱，就是这段经历辛苦的经历，遇到问题及时解决才是做好的途径，在后面自己搭建环境很少在出现基本问题错误，但是新的问题也会不断出现，只有自己不断寻找解决问题的办法，难题就会结局，遇到环境搭建错误不要气馁，这些都是需要经过的一部分，寻找解决问题的办法才是关键。
>- 主要介绍python 一些基础语法知识和代码，同时自己在这些基础语法和代码方面之外，回顾python的字符串、列表、字典、循环、分支、面向对象编程等一些基础知识，常用的python基本语法还包含opencv读取图片和保存图片的代码。
![image](https://user-images.githubusercontent.com/107736675/228480911-f6ee3df0-fb33-46cd-942e-3ac3aa624235.png)


# 二、opencv读取视频流
>- opencv 读取摄像头视频流，并在窗口进行实时显示
>- opencv 读取摄像头视频流，设置相应的保存格式，把读取的摄像头视频，保存为本地文件（文件格式:设置相应的保存格式）
>- opencv 读取本地已有的视频流文件（某文件夹下的视频文件），并在窗口进行显示
![image](https://user-images.githubusercontent.com/107736675/228481103-435afcd4-c90f-46e5-bb72-4481800de705.png)

# 三， 四、手掌关键点检测
>- 1. opencv 读取摄像头视频流，并在窗口进行实时显示
>- 2. 在读取的画面上绘制一个方块
>- 3. 通过 mediapipe 包获取手指关键点坐标，将这些关键点进行绘制
>- 4. 判断手指上的某一个关键点是否在方块上
>- 5. 如果该关键点在方块上, 方块跟着手指移动（手上的关键点动，方块随之跟着移动）
![image](https://user-images.githubusercontent.com/107736675/228481010-1d9a4189-af5a-49bf-9369-f6e54a7492d7.png)

# 五、口罩佩戴检测
>- 1. 视频流中检测口罩是否佩戴正确

# 九、Gan网络生成新的图片
## 1.Gan 网路
>- GAN 应用: 数据生成, 图像翻译, 超分辨率(更高清), 图像补全。
>- Adversarial: 对抗对手的意思, 两个模型: Generator(生成模型)，Discriminator(判别模型, 分类模型)
>- 创作者(G)目标: 赝品骗过鉴别者 鉴别者(D)目标: 火眼金睛不被骗
## 2.Gan 网路架构
>- Gan架构： Generator(生成模型)， Discriminator(判别模型, 分类模型), 以下简称 G网络，D网络。
![image](https://user-images.githubusercontent.com/107736675/234520531-97c51219-8ba2-405d-b859-335ff31de73f.png)

## 3.Gan 网路实现基本思路
![image](https://user-images.githubusercontent.com/107736675/234520703-784971b6-5e6e-4aff-aadf-ff53d81bbd23.png)

## 4.本项目中使用Gan网络架构
![image](https://user-images.githubusercontent.com/107736675/234520947-26220dc5-d964-4d49-b577-b35712c156c8.png)

## 5.生成结果
![image](https://user-images.githubusercontent.com/107736675/234521153-cb9e52b1-9e88-424b-af11-9ad9617c1874.png)

# 十一、实现RESNET 18网络
## 1.架构分析
![image](https://user-images.githubusercontent.com/107736675/234800309-e0d860f3-b150-4a27-aa5c-d2d768ad7ff8.png)
![image](https://user-images.githubusercontent.com/107736675/234800214-6a839b17-6d4a-45f7-92ee-a352a9eeb8b3.png)
## 2.基本单元
![image](https://user-images.githubusercontent.com/107736675/234800483-7e9b939f-62b4-4bfa-bc78-dc116d950363.png)

# 十二、实现RESNET 50网络
## 1.网络架构
![image](https://user-images.githubusercontent.com/107736675/234800683-65cc6227-5611-4570-9d25-3ad2418ffa94.png)
![image](https://user-images.githubusercontent.com/107736675/234800685-df6e05dc-2ab9-4124-ba7b-0010590635a9.png)

## 2.基本单元
![image](https://user-images.githubusercontent.com/107736675/234800866-7a921e97-47cb-4498-8869-b1ed7c6a67b6.png)
但是每层基本单元的通道数，卷积核步长(s)不一致,可以定义一个基本的残差单元,然后根据实际情况去修改通道数和卷积核步长(s)。

# 继续更新中

