"""
在 opencv 读取视频流中实时检测人脸:
1. 构造 haar 检测器
2. 获取视频流
3. 检测每一帧画面
4. 画人脸框并显示
程序给出了 hog, haar 检测器的人脸检测方法
"""
import cv2
import matplotlib.pyplot as plt
import dlib

"""
采用 hog 检测器
"""
cap = cv2.VideoCapture(0)

hog_face_detector = dlib.get_frontal_face_detector()


while True:

    rec, frame = cap.read()

    frame = cv2.flip(frame, 1)

    detections = hog_face_detector(frame, 1)

    for face in detections:
        x = face.left()
        y = face.top()
        r = face.right()
        b = face.bottom()
        cv2.rectangle(frame, (x, y), (r, b), (0, 255, 0), 5)

    cv2.imshow('face', frame)

    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


"""
采用 haar 检测器, 坐标直接遍历
"""

cap = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier('./weights/haarcascade_frontalface_default.xml')

while True:

    rec, frame = cap.read()

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 添加 minNeighbors 会降低误检率
    detections = face_detector.detectMultiScale(gray, minNeighbors=7)

    for (x, y, w, h) in detections:

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    cv2.imshow('face', frame)

    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


"""
采用 hog 检测器 + 人脸关键点检测
"""
cap = cv2.VideoCapture(0)

# 构造检测器
hog_face_detector = dlib.get_frontal_face_detector()

# 关键点检测模型
shape_detector = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

while True:

    rec, frame = cap.read()

    frame = cv2.flip(frame, 1)

    detections = hog_face_detector(frame, 1)

    for face in detections:
        x, y, r, b = face.left(), face.top(), face.right(), face.bottom

        # 人脸关键点检测
        points = shape_detector(frame, face)

        # 对关键点进行解析
        for point in points.parts():

            cv2.circle(img=frame, center=(point.x, point.y),
                       radius=2, color=(0, 255, 0), thickness=1)

        cv2.rectangle(frame, (x, y), (r, b), (0, 255, 0), 3)

    cv2.imshow('demo', frame)

    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()









