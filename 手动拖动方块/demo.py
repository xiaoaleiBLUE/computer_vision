"""
author: xiaoalei
date: 2013-8-23
1. opencv 获取视频流(BGR)的顺序
2. 在画面上画一个方块
3. 通过 mediapipe 获得
4. 判断手指是否在方块上
5. 如果在方块上, 方块跟着手指移动
"""
import math

import cv2
import numpy as np


# 导入 mediapipe
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

#
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# 获取摄像头视频流
cap = cv2.VideoCapture(0)

# 获取画面宽度和高度
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 方块的相关参数, 正方形的参数
square_x = 100
square_y = 100
square_width = 100

# 初始化跟踪移动算法的 L1, L2
L1 = 0
L2 = 0

# 来判断方块是否被激活
on_squre = False


while True:

    # 读取每一帧
    ret, frame = cap.read()

    # 对图像进行处理
    frame = cv2.flip(frame, 1)

    # mediapipe 处理, 图像颜色进行转换
    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)

    # 然后进行交换回来
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 判断是否出现双手
    if results.multi_hand_landmarks:
        # 解析遍历每一双手, 默认为 2, 可以对 hand_landmarks 打印输出进行查看
        for hand_landmarks in results.multi_hand_landmarks:
            # 绘制 21 个关键点
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # 保存 21 个x, y 的坐标
            x_list = []
            y_list = []

            # 对 hand_landmarks(一只手的数据) 进行遍历, 遍历的是每一个关键点
            for landmark in hand_landmarks.landmark:
                # 添加 x 坐标
                x_list.append(landmark.x)
                # 添加 y 坐标
                y_list.append(landmark.y)

            # 获取食指指尖坐标, 要乘以画面高度和宽度
            index_finger_x = int(x_list[8] * width)
            index_finger_y = int(y_list[8] * height)

            # 获取中指指尖坐标
            middle_finger_x = int(x_list[12] * width)
            middle_finger_y = int(y_list[12] * height)

            # 计算食指指尖和中指指尖的距离
            finger_len = math.hypot((index_finger_x - middle_finger_x), (index_finger_y - middle_finger_y))

            # 画一个圆, 进行验证
            # cv2.circle(frame, (index_finger_x, index_finger_y), 20, (255, 0, 255), -1)

            # 如果距离小于30, 算是激活, 如果大于30, 取消激活
            if finger_len < 30:

                # 判断食指指尖是否在方块上
                if (index_finger_x > square_x) and (index_finger_x < (square_x + square_width)) and\
                        (index_finger_y > square_y) and (index_finger_y < (square_y + square_width)):

                    if on_squre == False:
                        # 如果, 食指指尖在方块上就计算和移动, 为了避免负数, 统一加上绝对值
                        L1 = abs(index_finger_x - square_x)
                        L2 = abs(index_finger_y - square_y)

                        on_squre = True

                else:
                    # print("不在方块上")
                    pass

            else:
                # 取消激活
                on_squre = False

            if on_squre:
                # 刷新坐标
                square_x = index_finger_x - L1
                square_y = index_finger_y - L2

    # 画一个方块, -1: 代表实心的
    # cv2.rectangle(frame, (square_x, square_y), (square_x + square_width, square_y + square_width), (255, 0, 0), -1)

    # 画一个半透明的方块, 图片的拷贝
    overlay = frame.copy()
    cv2.rectangle(frame, (square_x, square_y), (square_x + square_width, square_y + square_width), (255, 0, 0), -1)

    # 增加透明度
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    # 显示
    cv2.imshow('Virtual drag', frame)

    # 退出条件
    if cv2.waitKey(10) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()








