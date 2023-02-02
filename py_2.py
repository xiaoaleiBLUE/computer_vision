import numpy as np
import cv2
import matplotlib.pyplot as plt


# opencv 连接 webcam 或 USB 摄像头
# opencv 视频操作

# opencv 读取摄像头视频流, 并显示
# 保存视频流为 mp4文件
cap = cv2.VideoCapture(0)                               # 0 默认摄像头

# 视频格式: DIVX, x264
# fps: 帧率,
fourcc = cv2.VideoWriter_fourcc(*'X264')
fps = 30
width = int((cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
height = int((cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
writer = cv2.VideoWriter('./demo.mp4', fourcc, fps, (width, height))

while True:
    # 返回 frame
    rec, frame = cap.read()
    # 镜像
    frame = cv2.flip(frame, 1)
    # 灰度显示
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 写入画面到文件
    writer.write(frame)
    # 显示画面
    cv2.imshow('demo', frame)

    if cv2.waitKey(10) & 0xFF==27:
        break

writer.release()
cap.release()
cv2.destroyAllWindows()


# opencv 读取视频流
cap = cv2.VideoCapture('./demo.mp4')

if not cap.isOpened():
    print('文件不存在或编码错误')

while cap.isOpened():
    # 读取每一帧
    ret, frame = cap.read()
    if ret:
        # 读到画面,显示画面
        cv2.imshow('demo', frame)
        # time.sleep(0.005)       延时播放, 相当于慢放
        # 退出条件
        if cv2.waitKey(10) & 0xFF==27:
            break
    else:                                          # 没有读到画面
        break

cap.release()
cv2.destroyAllWindows()


































