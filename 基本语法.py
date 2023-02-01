# python 运行环境: 用 conda 创建虚拟环境
# python 包安装: conda > pip > 编译安装
# 手掌关键点识别 Demo 程序: python3.7, opencv, mediapipe

# 用户目录: conda中 powershell prompt 默认打开就是工作目录, 输入 pwd 即打开工作目录
# python 的 5 个知识
# python 入门知识
# 1. python 代码运行方法
# 2. 书写规则, 数据类型及变量
# 3. 流程: 条件与循环
# 4. 数据结构: 列表, 字典, 元组
# 5. 函数, 模块, 包, 库

# 1. python 代码运行方法
# conda 虚拟环境 + 命令行运行, 需要进入同级目录

# 参数
# 可以在命令行使用 python py_1.py  --width=20  --height=40
# 使用 python py_1.py --help 查看需要输入的参数
from argparse import ArgumentParser
parser = ArgumentParser()                                            # 创建解析器
parser.add_argument('--width', type=int, default=960, help='宽度')    # 添加width参数
parser.add_argument('--height', type=int, default=720, help='高度')   # 添加height参数
args = parser.parse_args()                                           # 属性给与args实例
area = int(args.width * args.height)
print('面积为'+str(area))

# “”“  ”“” 三个双引号之间的内容为多行注释, 声明作者, 时间, 描述整个文件, 类的功能
# 计算11月的开支: 房租水电3500   餐饮1200   服装500
print(3500+1200+500)

# 使用变量计算开支
cost_rent = 3500                                               # 房租水电开支
cost_meal = 1300                                               # 餐饮开支
cost_clothing = 300                                            # 服装开支
cost_other = 2000                                              # 其他开支
print(cost_rent+cost_meal+cost_clothing+cost_other)

# 数据类型及变量
# 整数:3500   浮点数:3.1415    字符串:python    布尔值: True
# 流程: 条件与循环
age_xiaohong = 25
age_xiaoming = 27
if age_xiaohong > age_xiaoming:
    print('小红大')
elif age_xiaohong == age_xiaoming:
    print('小红和小明一样大')
else:
    print('小明大')

# for {迭代变量} in {可迭代对象}:   {代码块}
# 数据结构: 列表, 元组, 字典

# 列表: 使用一个 list 变量存储三个字符串, 可修改
fruit_list = ['apple', 'banana', 'tangerine']
del fruit_list[1]                                              # 删除一个元素
print(fruit_list)                                              # ['apple', 'tangerine']
fruit_list.remove('tangerine')                                 # 删除 tangerine 元素

# 元组, 不可修改
fruit_tuple = ('apple', 'banana', 'tangerine')
print(fruit_tuple)                                             # ('apple', 'banana', 'tangerine')
print(fruit_tuple[1])                                          # banana

# 字典: 可以修改
# 定义一个字典元素, 键值对来表示,  键名:键值
# 获取字典的元素: 通过键名来获取
person_dict = {
    'name': '张三',
    'height': 175,
    'age': 23,
    'graduated': True
}
print(person_dict)                                             # 打印字典
print(person_dict['name'])                                     # 获取字典得元素: 张三
person_dict['weight'] = 60                                     # 增加一个元素
person_dict['height'] = 180                                    # 修改元素
del person_dict['name']                                        # 删除一个元素


# 函数, 创建一个函数使用 def 完成
def my_function():
    # 函数执行部分
    print('This is function')


my_function()                                                  # This is function


# 定义有参数的函数
def say_hello(name, age):
    print('{}说, 我今年{}岁'.format(name, age))
    print(name+'说, 我今年'+str(age)+'岁')


say_hello('Tim', 27)                                            # Tim说, 我今年27岁


# 导入自己写的模块: module
# 范围: 函数 < 模块 < 包
# 调用模块内的函数: 模块.函数名称()
import customModule as cm
cm.say_hello()                                                  # 调用customModule模块的 say_hello()函数


# 图像的本质 --- numpy 和 图像基础
# Jupyter Lab 交互式
# 安装: pip install jupyterlab       运行: jupyter-lab
list_a = [[1, 2], [3, 4], [5, 6]]
print(list_a[1][0])                                             # 3

import numpy as np
list_b = [1, 2, 3, 4]
print(type(list_b))                                             # <class 'list'>

my_array = np.array(list_b)
print(type(my_array))                                            # <class 'numpy.ndarray'>

# array 获取单个元素, 多个元素 采用 []
np_array = np.arange(0, 16).reshape((4, 4))
print(np_array[3, 2])                                            # 获取第3行第2列的元素

# 图像是什么: 每个图片可以看成数组,
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open('./Abyssinian_13.jpg')
print(type(img))                                                 # <class 'PIL.JpegImagePlugin.JpegImageFile'>
plt.imshow(img)
plt.show()

img_array = np.array(img)
print(img_array.shape)                                           # (280, 245, 3)

# 显示numpy格式的图片
plt.imshow(img_array)
plt.show()

# copy 图片的numpy数据, 查看复制的array的shape,并绘制, 绘制不同通道的图片
img_arr_copy = img_array.copy()
print(img_arr_copy.shape)                                        # (280, 245, 3)
plt.imshow(img_arr_copy)
plt.show()
plt.imshow(img_arr_copy[:, :, 0])                                # R 通道对图片进行显示
plt.show()
plt.imshow(img_arr_copy[:, :, 0], cmap='gray')                   # 灰度图进行显示
plt.show()
plt.imshow(img_arr_copy[:, :, 1])                                # G 通道对图片进行显示
plt.show()
plt.imshow(img_arr_copy[:, :, 2])                                # B 通道对图片进行显示
plt.show()


# opencv 简介: 常用 C++, 也提供python接口
# opencv 通道 BGR,  matplotlib 通道 RGB
# cv2.imread() 读入图片   cv2.imwrite() 保存图片
import cv2
img = cv2.imread('./Abyssinian_13.jpg')
print(type(img))                                                 # <class 'numpy.ndarray'>
print(img.shape)                                                 # (280, 245, 3)
plt.imshow(img)
plt.show()
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

img_gray = cv2.imread('./Abyssinian_13.jpg', cv2.IMREAD_GRAYSCALE)    # 灰度图模式显示
print(img_gray.shape)                                                 # (280, 245)
plt.imshow(img_gray, cmap='gray')

print(img.shape)                                                      # (280, 245, 3)
img_resize = cv2.resize(img, (900, 300))
print(img_resize.shape)                                               # (300, 900, 3)
plt.imshow(img_resize)
plt.show()

# flip: 0垂直  1水平  -1都翻转
img_flip = cv2.flip(img, 0)
plt.imshow(img_flip)
plt.show()

# opencv 显示图片
#cv2.imshow('Demo', img)                                        # 图片显示一瞬间

# 采用循环控制显示窗口
while True:
    cv2.imshow('Demo', img)
    # 如果等待至少10ms, 并且用户按了 ESC 键, 才会退出
    if cv2.waitKey(10) & 0xFF == 27:
        break
# 关闭所有的窗口
cv2.destroyAllWindows()

# opencv 绘制文字和几何图形

# 创建纯黑色的子图
black_img = np.zeros(shape=(800, 800, 3), dtype=np.int16)
plt.imshow(black_img)
plt.show()

# 利用 opencv 画一个矩形, 红色框和绿色框
cv2.rectangle(img=black_img, pt1=(100, 100), pt2=(400, 300), color=(255, 0, 0), thickness=10)
cv2.rectangle(img=black_img, pt1=(20, 100), pt2=(120, 200), color=(0, 255, 0), thickness=10)
plt.imshow(black_img)
plt.show()

# 利用 opencv 画一个圆形, thickness=-1 表示填充
black_img = np.zeros(shape=(800, 800, 3), dtype=np.int16)
cv2.circle(img=black_img, center=(400, 400), radius=150, color=(255, 0, 0), thickness=10)
plt.imshow(black_img)
plt.show()

# 画一条直线
black_img = np.zeros(shape=(800, 800, 3), dtype=np.int16)
cv2.line(img=black_img, pt1=(0, 0), pt2=(800, 800), color=(255, 0, 0), thickness=5)
plt.imshow(black_img)
plt.show()

# opencv 添加文字: org左下角的位置
font = cv2.FONT_HERSHEY_PLAIN                                      # 字体形式
black_img = np.zeros(shape=(800, 800, 3), dtype=np.int16)
cv2.putText(img=black_img, text='python', org=(150, 150),
            fontFace=font, color=(255, 0, 0), fontScale=4, thickness=5, lineType=cv2.LINE_AA)
plt.imshow(black_img)
plt.show()

# opencv 画多边形, 顶点的位置使用 二维数组 来定义, 顶点设置dtype=np.int32
black_img = np.zeros(shape=(800, 800, 3), dtype=np.int16)
points = np.array([[400, 100], [200, 300],
                   [400, 700], [600, 300]], dtype=np.int32
                  )
# 然后将数组转换为 3 维
pts = points.reshape((-1, 1, 2))
print(pts.shape)                                                   # (4, 1, 2)

# 创建多边形
cv2.polylines(img=black_img, pts=[pts], isClosed=True,
              color=(255, 0, 0), thickness=5)
plt.imshow(black_img)
plt.show()











