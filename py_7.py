"""
视频流检测人脸
date: 2023-2-09
1. 构造 haar 人脸检测器, 其他检测器类似
2. 获取视频流
3. 检测每一帧
4. 画人脸框
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

# 构造人脸检测器
haar_face_datector = cv2.CascadeClassifier('./weight/haarcascade_frontalface_default.xml')


while True:
    rec, frame = cap.read()

    # 镜像
    frame = cv2.flip(frame, 1)

    # 转为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    detections = haar_face_datector.detectMultiScale(gray, minNeighbors=7)

    # 解析人脸框
    for (x, y, w, h) in detections:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 显示画面
    cv2.imshow('demo', frame)

    # 退出条件
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()






