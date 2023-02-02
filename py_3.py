"""
autor: xiaoalei
date: 2023-2-2
1. opencv 获取视频流
2. 在画面上或一个方块
3. 通过 mediapipe 获取手指关键点坐标
4. 判断手指是否在方块上
5. 如果在方块上, 方块跟着手指移动
"""
import math

import cv2
import numpy as np
# mediapipe 相关参数
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# 获取摄像头视频流
cap = cv2.VideoCapture(0)

# 获取画面宽度和高度
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 方块的相关参数
square_x = 100
square_y = 100
square_w = 100

L1, L2 = 0, 0

on_squre = False

while True:
    # 读取每一帧
    ret, frame = cap.read()

    # 对图像进行处理
    frame = cv2.flip(frame, 1)

    # mediapipe 处理
    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)

    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 是否出现双手
    if results.multi_hand_landmarks:
        # 解析遍历每一双手
        for hand_landmarks in results.multi_hand_landmarks:
            # 绘制 21 个关键点
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
                )

            # 查看解析参数 使用 print(hand_landmarks) 也行
            # 保存 21 个关键点坐标
            x_list = []
            y_list = []
            for landmarks in hand_landmarks.landmark:
                # 添加 x 坐标
                x_list.append(landmarks.x)
                # 添加 y 坐标
                y_list.append(landmarks.y)
            # print(len(x_list))                          # 长度为 21, 则关键点坐标全部获取
            # 获取食指之间坐标
            index_x = int(x_list[8] * width)
            index_y = int(y_list[8] * height)
            # 获取中指指尖坐标
            index_m_x = int(x_list[12] * width)
            index_m_y = int(y_list[12] * height)

            # 计算食指指尖与中指指尖坐标距离
            l = math.hypot((index_x-index_m_x), (index_y-index_m_y))
            print(l)

            # print(index_x, index_y)
            # # 画一个圆进行验证
            # cv2.circle(frame, center=(index_x, index_y), radius=20,
            #            color=(255, 0, 255), lineType=-1)
            # 如果距离小于30, 则激活
            if l < 30:
                # 判断食指指尖在不在方块上面
                if ((index_x > square_x) and (index_x < (square_x + square_w))) and (
                        (index_y > square_y) and (index_y < (square_y + square_w))
                ):
                    if on_squre == False:
                        print('在方块上')
                        L1 = abs(index_x - square_x)
                        L2 = abs(index_y - square_y)
                        on_squre = True

                else:
                    print('不在方块上')
            else:
                on_squre = False
                
            # 刷新值
            if on_squre:
                square_x = index_x - L1
                square_y = index_y - L2

    # 画一个方块
    cv2.rectangle(frame, (square_x, square_y), (square_x+square_w, square_y+square_w),
                  color=(0, 0, 255), thickness=-1)

    # 显示画面
    cv2.imshow('demo',frame)

    # 退出条件
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
































