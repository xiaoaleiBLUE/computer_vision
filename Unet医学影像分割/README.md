# 一、自定义 Unet 网络结构

原Uent的部分  encoder-decoder 中的Concat 连接部分是左侧通道进行裁剪后与右侧通道进行Concat连接。如下图所示，
原Unet网络 encoder-decoder中间的连接部分 256\*136\*136 大小裁剪成 256\*104\*104 和右侧的 256\*104\*104进行拼接成 512\*104\*104
为了省去裁剪，对网络进行直接对encoder-decoder进行拼接, 进行拼接是只需保持特征图的宽高一致就行, 在通道维度进行连接。
修改后的Unet网络结构图直接对 encoder-decoder进行通道连接, 即下图中中间的较长的箭头。

![image](https://github.com/xiaoaleiBLUE/computer_vision/assets/107736675/306a4252-8162-4d7c-acd1-13494f69c538)

对Unet网络进行观察, 可发现有个卷积的基本模块, 都是通过2次基本的卷积，也就是下图中的红色方框（后面两个特征图的通道大小一致）
，所以只需要定义基本的卷积模块(只是输入输出的通道数目不一致)。在左侧不同基本卷积块（即下图红色方框内）之间经过一个下采样（池化）操作, 
在左侧不同基本卷积块（即下图红色方框内）之间经过一个上采样（转置卷积）操作。


# 二、最小卷积模块

![image](https://github.com/xiaoaleiBLUE/computer_vision/assets/107736675/c5d56d22-93f0-4292-a87a-fdbe942daf5b)

自定义模型网络结构：部分结构如下所示
![image](https://github.com/xiaoaleiBLUE/computer_vision/assets/107736675/440d48a3-a328-4cb1-b2d4-8822e75cc37a)


# 三、网络结构代码实现
>- Unet_net.py是自定义的Unet网络模型实现的代码, 相关通道参数也在 forward 中进行备注
## 1.3D_display.py
对于医学影像图片进行3D查看
3D 查看 MRI 图像
![image](https://github.com/xiaoaleiBLUE/computer_vision/assets/107736675/5ad7a17e-dc7c-4a89-b066-b2444cd7c175)


## 2.img_preprocess.py
具体步骤详细看一下代码，代码都有注释
图片预处理
预处理步骤:
>- 1. 读取 NIFIT 格式文件, 加载其中的图片和 mask
>- 2. 显示一层图片(包含 mask)
>- 3. 动态显示整个扫描(显示多层信息)
>- 4. 构造归一化, 标准化函数
>- 5. 预处理所有文件, 保存为 npz 文件, 存在磁盘
>- 6. 检查一下存储的 npz 文件是否合格
处理后的文件夹如下图所示
![image](https://github.com/xiaoaleiBLUE/computer_vision/assets/107736675/15229534-128d-432e-8bb8-373e23c56b38)

## 3.custom_dataset.py
自定义数据集(dataset)代码构建
>- 构建datase基本类需要三个基本类方法：_ _ init _ _(), _ _ getitem _ _(), _ _ len _ _()方法
>- dataset 类主要传入数据集图片路径, 标签, 和数据集变换形式, 先看看基本类型, 采用 类来实现

主要步骤
>- 1. 构造 torch 自定义的 dataset
>- 2. 数据增强
>- 3. 测试一下 dataLoader 加载
>- 4. 测试显示一些图片


## 4.model_train.py
训练模型,
>- 1. 自定义函数数据集Dataset类, 默认为 train 文件夹下, 读取 train 文件下的数据, 如果为 test, 则读取 test 文件夹下的数据, 进行数据增强处理
>- 2. 定义好的Dataset类, 使用 dataloader 加载数据, 并定义 train_dataloader, test_dataloader
>- 3. 导入定义好的 Unet 网络模型
>- 4. 自定义 dice loss 函数
>- 5. 开始训练， 记录测试集和训练的 loss, 并保存最好的模型
最好的模型结构如下
![image](https://github.com/xiaoaleiBLUE/computer_vision/assets/107736675/b7809a09-aaa4-4a34-af11-5f1185d8cf14)




