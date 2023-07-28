"""
了解加权采用的用法
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


# 设置批次
BATCH_SIZE = 196

# 测试集占比
validation_split = 0.2

# 数据集大小, 这里进行伪造 1000 个
dataset_size = 1000

# 生成索引 list
indices = list(range(dataset_size))


# 分割线
split = int(np.floor(validation_split * dataset_size))
print(split)

# 打乱索引
np.random.shuffle(indices)

# 训练集和测试集索引
train_indices, val_indices = indices[split:], indices[:split]

# 打印大小
print(len(train_indices), len(val_indices))

#
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

"""
就是告诉 Dataloader 哪些索引是训练集, 哪些索引是测试集
"""

# 加载各自的 dataloader
# train_dataloader = DataLoader(dataset_size, batch_size=BATCH_SIZE, sampler=train_sampler)
# test_dataloader = DataLoader(dataset_size, batch_size=BATCH_SIZE, sampler=valid_sampler)

print(len(train_sampler), len(valid_sampler))

# weights：权重列表，可以理解为样本被采样的概率，实际使用时不必加起来等于1
# num_samples：采样数量
# replacement：True，表示一个样本可以重复采样

# 返回的是一个被采样的索引

# torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples, replacement=True)

# 10的位置在索引1上, 所以采样出现 1 比较多,
weights = torch.Tensor([0, 10, 1, 1])

# 4: 采样次数为 4 次
weight_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, 4, True)

print(list(weight_sampler))

# 可以看到位置[0]的采样概率（权重）为0, 所以每次采样输出索引都没有0
# 位置[1]由于权重较大, 被采样的次数较多, [1, 1, 1, 3]






















