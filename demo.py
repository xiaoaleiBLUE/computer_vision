"""
步骤:
1. opencv 获取视频流
2. 在画面上获取一个方块
3. 通过 mediapipe 获取手指关键点坐标
4. 判断手指是否在方块上
5. 如果在方块上, 方块跟着手指移动
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(model_complexity=0,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# 调用摄像头
cap = cv2.VideoCapture(0)

# 获取画面的高度和宽度
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 方块的相关参数
square_x = 100
square_y = 100
square_width = 100

#
L1 = 0
L2 = 0
on_square = False

while True:

    # 返回 frame
    rec, frame = cap.read()

    # 镜像
    frame = cv2.flip(frame, 1)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 推理
    frame.flags.writeable = False
    results = hands.process(frame)

    # 转为 bgr
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 判断是否出现双手
    if results.multi_hand_landmarks:
        # 解析遍历每一只手
        for hand_landmarks in results.multi_hand_landmarks:
            # 绘制21个关键点, hand_landmarks中的landmark: 包含 x, y, z坐标
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            # 保存 21个 x, y坐标
            x_list = []
            y_list = []
            for landmarks in hand_landmarks.landmark:
                # landmarks包含x, y, z的坐标信息: landmarks.x 可查看x坐标数据
                # 添加 x 坐标
                x_list.append(landmarks.x)

                # 添加 y 坐标
                y_list.append(landmarks.y)

            # 获取食指指尖坐标, 8号位置坐标
            index_finger_x = int(x_list[8] * width)
            index_finger_y = int(y_list[8] * height)
            # print(index_finger_x, index_finger_y)

            # 获取中指指尖坐标
            middle_finger_x = int(x_list[12] * width)
            middle_finger_y = int(x_list[12] * height)

            # 计算食指指尖和中指指尖的距离
            finger_len = math.hypot((index_finger_x - middle_finger_x), (index_finger_y - middle_finger_y))

            #

        # 画圆来验证所取得坐标是否对
        # cv2.circle(frame, (index_finger_x, index_finger_y), 20, (255, 0, 255), -1)

        # 如果距离小于 30, 激活,   大于 30, 则不激活
        if finger_len < 30:
            # 判断食指指尖在不在方块上
            if (index_finger_x > square_x) and (index_finger_x < (square_x + square_width)) \
                    and (index_finger_y > square_y) and (index_finger_y < (square_y + square_width)):

                if on_square == False:
                    L1 = abs(index_finger_x - square_x)
                    L2 = abs(index_finger_y - square_y)
                    on_square = True

            else:
                pass
        else:
            # 因为 on_square = False 不更新x, y坐标, 所以 此时手指可甩掉方块
            on_square = False

        # 如果手指在方块上, 就刷新方块的坐标x, y 的坐标
        if on_square:
            square_x = index_finger_x - L1
            square_y - index_finger_y - L2

    # 需要一个遮罩, frame.copy()
    overlay = frame.copy()

    # 画方框
    cv2.rectangle(frame, (square_x, square_y), (square_x + square_width, square_y + square_width), (255, 0, 0),
                  thickness=-1)

    # 增加半透明
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    # 显示
    cv2.imshow('demo', frame)

    # 退出条件
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()





































