"""
author: xiaolei
支持向量 svm
1. 间距: 离超平面最近的点到超平面的距离
2. 合适的超平面: 最大化超平面两边的 margin
"""
from sklearn import svm


"""
先看二分类
"""
# 训练样本特征
X = [[0, 0], [1, 1]]

# 训练样本类别标签
Y = [0, 1]

clf = svm.SVC()

# 训练
clf.fit(X, Y)

# 测试数据样本
test = [[2, 2]]

# 预测
clf.predict(test)


"""
多分类问题
"""
# 训练样本特征
X = [[0], [1], [2], [3], [4]]

# 训练样本的类别标签
Y = [0, 1, 2, 3, 4]

# 测试数据集
test = [[1]]

# 选择一对一策略
clf = svm.SVC(decision_function_shape='ovo')

# 训练
clf.fit(X, Y)

# 查看投票函数
dec = clf.decision_function(test)

# 查看筛选函数的大小, 可以看到是10, 是因为 ovo 策略会设计 5*4/2 = 10个分类器, 然后找出概率最大的
print(dec.shape)









