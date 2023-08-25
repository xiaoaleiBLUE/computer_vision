"""
author: xiaoalei
opencv 获取视频流并显示
"""
import cv2


cap = cv2.VideoCapture(0)


while True:

    # 读取每一帧
    ret, frame = cap.read()

    # 对图像进行处理
    frame = cv2.flip(frame, 1)

    # 显示
    cv2.imshow('Virtual drag', frame)

    # 退出条件
    if cv2.waitKey(10) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()











