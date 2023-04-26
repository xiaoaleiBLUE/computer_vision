"""
GAN: train, 网络进行训练
GAN 网络 pytorch 实现
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as tdst
from torch.utils.data import DataLoader
import time

# 预处理
import torchvision.transforms as transforms
import torch.optim as optim

# 模型架构可视化
from torchsummary import summary


# 显示一张图片
img = plt.imread('./no.10_gan/data/img_align_celeba/000008.jpg')
print(img.shape)

plt.imshow(img)
plt.show()

# 导入数据
img_size = 64

img_preprocess = transforms.Compose([
    # 缩放
    transforms.Resize(img_size),
    # 中心裁剪 64*64 的正方形
    transforms.CenterCrop(img_size),
    # PIL图像转为tensor  H*W*C ---> C*H*W
    transforms.ToTensor(),
    # 归一化到[-1,1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 从文件夹读取图片
dataset = tdst.ImageFolder(root='./no.10_gan/data/', transform=img_preprocess)

# 加载成 dataloader
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# 查看批次数量  12324 / 128 = 96.28, 向上取整
print(len(dataloader))

# 显示图片, 其中 x[0]:对应的图片,有shape,  x[1]:对应类别标签,  x[0].shape:torch.Size([128, 3, 64, 64]) 128对应批次大小
for x in dataloader:
    # 设置画布大小
    fig = plt.figure(figsize=(8, 8))
    for i in range(16):

        plt.subplot(4, 4, i+1)

        # 转为 numpy, x[0][i].shape: torch.Size([3, 64, 64])
        img = x[0][i].numpy()

        # 通道顺序变换, 调整通道顺序为 PIL 格式
        img = np.transpose(img, (1, 2, 0))

        # 先转到[0,1]，再乘以255
        img = (img + 1) / 2 * 255

        # 取整
        img = img.astype('int')

        # 显示
        plt.imshow(img)
        plt.axis('off')

    plt.show()
    break


# G 生成网络, 生成者网络
class G_model(nn.Module):

    def __init__(self):
        super(G_model, self).__init__()

        self.main = nn.Sequential(

            # 100*1*1(张量)  ----> 4*4*512
            # out = (input-1)*stride - 2*padding + kernel_size + out_padding
            # out = (1-1)*1 - 2*0 + 4 + 0 = 4
            nn.ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=4, stride=1, padding=0, bias=False),
            # BN
            nn.BatchNorm2d(512),
            # Relu
            nn.ReLU(inplace=True),

            # 4*4*512  ----> 8*8*256
            # out = (4-1)*2 - 2*1 + 4 + 0 = 8
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            # BN
            nn.BatchNorm2d(256),
            # Relu
            nn.ReLU(inplace=True),

            # 8*8*256  ----> 16*16*128
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            # BN
            nn.BatchNorm2d(128),
            # Relu
            nn.ReLU(inplace=True),

            # 16*16*128  ----> 32*32*64
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            # BN
            nn.BatchNorm2d(64),
            # Relu
            nn.ReLU(inplace=True),

            # 32*32*64  ----> 64*64*3
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            # tanh()
            nn.Tanh()

        )

    def forward(self, x):

        return self.main(x)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
G_model = G_model().to(device)

# 输入 (100, 1, 1), 查看架构
print(summary(G_model, (100, 1, 1)))

# 构建一个大小为 100*1*1 的随机张量,  1, 100, 1, 1: 代表 1个 100*1*1 的张量
fixed_noise = torch.randn(1, 100, 1, 1, device=device)

# G 生成网络生成的图片, 伪造的图片
fake_imgs = G_model(fixed_noise)

# 从 gpu 放到 cpu 中, 转为 numpy
fake_imgs_np = fake_imgs.detach().cpu().numpy()

print(fake_imgs_np.shape)                                 # (1, 3, 64, 64)

# fake_imgs_np[0].shape: (3, 64, 64), 调整通道顺序,绘制
plt.imshow(np.transpose(fake_imgs_np[0], (1, 2, 0)))
plt.show()


# D 判别网络网络
class D_model(nn.Module):

    def __init__(self):
        super(D_model, self).__init__()

        self.main = nn.Sequential(

            # 64*64*3  ----> 32*32*64
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True),
            # Relu
            nn.LeakyReLU(0.2, inplace=True),

            # 32*32*64  ----> 16*16*128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True),
            # BN
            nn.BatchNorm2d(128),
            # Relu
            nn.LeakyReLU(0.2, inplace=True),

            # 16*16*128  ----> 8*8*256
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=True),
            # BN
            nn.BatchNorm2d(256),
            # Relu
            nn.LeakyReLU(0.2, inplace=True),

            # 8*8*256  ----> 4*4*512=8192
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=True),
            # BN
            nn.BatchNorm2d(512),
            # Relu
            nn.LeakyReLU(0.2, inplace=True),

            # flatten
            nn.Flatten(),

            # 全连接
            nn.Linear(8192, 1),

            # sigmoid
            nn.Sigmoid()

        )

    def forward(self, x):

        return self.main(x)


D_model = D_model().to(device)

# 查看网络架构
print(summary(D_model, (3, 64, 64)))

# G 生成网络生成伪造的图片, 输入 D
output = D_model(fake_imgs)
print(output.shape)

# 去除多余的维度
output.view(-1)

# loss 损失函数, 二分类的损失函数
loss_fn = nn.BCELoss()

# 优化器
D_optimizer = optim.Adam(D_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
G_optimizer = optim.Adam(G_model.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 开始训练
Epoch_num = 100

for epoch in range(Epoch_num):

    # 获取批次图像
    start_time = time.time()

    for i, data in enumerate(dataloader):
        # ----------------------训练判别网络D：真实数据标记为1------------------------
        # ！！！每次update前清空梯度
        D_model.zero_grad()
        # 获取数据, data[0]是图片, data[1]是类别
        imgs_batch = data[0].to(device)
        # 动态获取图片的batch_size, 最后不可以用BATCH_SIZE，因为数据集数量可能不能被BATCH_SIZE整除
        b_size = imgs_batch.size(0)
        # 计算输出
        output = D_model(imgs_batch).view(-1)
        # 构建全1向量 label
        ones_label = torch.full((b_size, ), 1, dtype=torch.float, device=device)
        # 计算 loss
        d_loss_real = loss_fn(output, ones_label)
        # 反向传播
        d_loss_real.backward()
        # 梯度更新
        D_optimizer.step()

        # -------------------训练判别网络D：假数据标记为0-------------------------------
        # 清楚梯度
        D_model.zero_grad()
        # 构建随机张量
        noise_tensor = torch.randn(b_size, 100, 1, 1, device=device)
        # 生成假的图片
        generated_imgs = G_model(noise_tensor)
        # 假图片的输出, 此时不需要训练 G, 可以 detach
        output = D_model(generated_imgs.detach()).view(-1)
        # 构建全 0 向量
        zeros_label = torch.full((b_size, ), 0, dtype=torch.float, device=device)
        # 计算 loss
        d_loss_fake = loss_fn(output, zeros_label)
        # 反向传播
        d_loss_fake.backward()
        # 梯度更新
        D_optimizer.step()

        # ----------------------训练生成网络G：假数据标记为1--------------------
        # 清楚梯度
        G_model.zero_grad()
        # 随机张量
        noise_tensor = torch.randn(b_size, 100, 1, 1, device=device)
        # 生成假的图片
        generated_imgs = G_model(noise_tensor)
        # !!!!!!!!!!!!!! 假图片的输出，这里不可以detach，否则学习不到
        output = D_model(generated_imgs).view(-1)
        # 构建全1向量
        ones_label = torch.full((b_size, ), 1, dtype=torch.float, device=device)
        # 计算 loss
        g_loss = loss_fn(output, ones_label)
        # 反向传播
        g_loss.backward()
        # 生成网络梯度更新
        G_optimizer.step()

    # 打印训练时间
    print('第{}个epoch执行时间: {}s'.format(epoch, time.time() - start_time))
    # 每一个 epoch 输出结果
    # 用 no_grad 表示梯度不跟踪
    with torch.no_grad():
        # 生成 16 个随机向量
        fixed_noise = torch.randn(16, 100, 1, 1, device=device)
        # 生成图片
        fake_imgs = G_model(fixed_noise).detach().cpu().numpy()
        # 画布大小
        fig = plt.figure(figsize=(10, 10))

        for i in range(fake_imgs.shape[0]):
            plt.subplot(4, 4, i+1)
            img = np.transpose(fake_imgs[i], (1, 2, 0))
            img = (img + 1) / 2 * 255
            img = img.astype('int')
            plt.imshow(img)
            plt.axis('off')

        plt.show()


# 保存模型, 保存生成网络G模型参数
torch.save(G_model.state_dict(), './g_model_course.pt')



















































