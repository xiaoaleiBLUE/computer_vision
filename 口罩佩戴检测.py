"""
在视频流中识别口罩
采用类的方法: 
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.special import softmax


class MaskRecognition:

    def __init__(self):

        # 加载模型
        self.model = tf.keras.models.load_model('./data/face_mask_model')
        # 打印模型架构
        print(self.model.summary())
        pass

    def imgBlob(self, img):
        """
        图片转为 Blob
        :return:
        """
        # 转为Blob
        img_blob = cv2.dnn.blobFromImage(img, 1, (100, 100), (104, 177, 123), swapRB=True)

        # img_blob的shape (1, 3, 300, 300)
        img_squeeze = np.squeeze(img_blob)  # img_squeeze的shape: (3, 100, 100)
        img_squeeze = img_squeeze.T  # (100, 100, 3)

        # 此时返回显示的图像是翻转的, 不是一个正立的图像, 进行旋转, 顺时针旋转90度
        img_rotate = cv2.rotate(img_squeeze, cv2.ROTATE_90_CLOCKWISE)

        # 垂直的进行镜像
        img_flip = cv2.flip(img_rotate, 1)

        # 去除负数, 大于0保持不变, 小于0被设置为0, 并归一化, img_blob的shape (100, 100, 3)
        img_blob = np.maximum(img_flip, 0) / img_flip.max()

        return img_blob

    def recognize(self):
        """
        识别用的
        :return:
        """
        cap = cv2.VideoCapture(0)

        # 加载 SSD 模型
        face_detector = cv2.dnn.readNetFromCaffe('./weights/deploy.prototxt.txt',
                                                 './weights/res10_300x300_ssd_iter_140000.caffemodel')

        # 获取原图尺寸
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # labels
        labels = ['yes', 'no', 'nose']

        # 显示框颜色
        colors = [(0, 255, 0), (0, 0, 255), (0, 255, 255)]

        while True:

            rec, frame = cap.read()

            frame = cv2.flip(frame, 1)

            # 转为 Blob
            img_blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 177, 123), swapRB=True)

            # 输入
            face_detector.setInput(img_blob)

            # 推理
            detections = face_detector.forward()

            # 获取人脸
            person_count = detections.shape[2]

            # 对人数的人脸进行遍历
            for face_index in range(person_count):
                # 置信度
                confidence = detections[0, 0, face_index, 2]
                if confidence > 0.5:
                    locations = detections[0, 0, face_index, 3:7] * np.array(
                        [frame_w, frame_h, frame_w, frame_h])

                    # 取整, opencv 画框不能为小数
                    l, t, r, b = locations.astype('int')

                    # 截取人脸
                    face_crop = frame[t:b, l:r]

                    # 转为 Blob
                    img_blob = self.imgBlob(face_crop)

                    # 转为四维
                    img_input = img_blob.reshape(1, 100, 100, 3)

                    # 预测
                    result = self.model.predict(img_input)

                    result = softmax(result[0])

                    # 最大值索引
                    max_index = result.argmax()

                    # 最大值
                    max_value = result[max_index]

                    # 对应标签的文字
                    label = labels[max_value]

                    # 显示文本
                    txt = label + ' ' + str(round(max_value*100)) + '%'

                    # 显示框颜色
                    color = colors[max_index]

                    # 绘制文字
                    cv2.putText(frame, txt, (l, t-10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

                    # 画人脸框
                    cv2.rectangle(frame, (l, t), (r, b), color, 5)

                    # 显示 blob
                    # cv2.imshow('demo', img_blob)

            cv2.imshow('demo', frame)

            if cv2.waitKey(10) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


# 实例化
mask = MaskRecognition()
mask.recognize()















