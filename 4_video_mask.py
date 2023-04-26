"""
为视频流添加口罩
读取视频流并添加口罩
"""
import cv2
import numpy as np
import time
import dlib
import glob
import random


class MaskVideo:
    def __init__(self):
        # 加载人脸检测模型
        self.face_detector = cv2.dnn.readNetFromCaffe('./weights/deploy.prototxt.txt',
                                                      './weights/res10_300x300_ssd_iter_140000.caffemodel')

        # 关键点 检测模型
        self.shape_detector = dlib.face_recognition_model_v1('./weights/dlib_face_recognition_resnet_model_v1.dat')

        # 加载 png 图片
        self.mask_list = self.getPngList()

    def getPngList(self):
        png_list = glob.glob('./images/masks/*.png')
        mask_list = []

        for png_file in png_list:

            mask_img = cv2.imread(png_file)
            mask_list.append(mask_img)

        return mask_list

    def getCropedFace(self, frame, conf_thresh=0.5):

        # 缩放图片
        img_resize = cv2.resize(frame, (300, 300))

        # 图像转为 blob
        img_blob = cv2.dnn.blobFromImage(img_resize, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # 输入
        self.face_detector.setInput(img_blob)

        # 推理
        detections = self.face_detector.forward()

        # 查看人脸检测数量
        num_of_detections = detections.shape[2]

        # 遍历人脸
        for index in range(num_of_detections):
            # 置信度
            detection_confidence = detections[0, 0, index, 2]

            # 挑选置信度,找到一个人返回
            if detection_confidence > conf_thresh:
                # 位置
                locations = detections[0, 0, index, 3:7]

                return locations

        return None

    def main(self):

        cap = cv2.VideoCapture(0)
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        png_index = 0

        start_time = time.time()

        while True:

            rec, frame = cap.read()

            frame = cv2.flip(frame, 1)

            # 人脸检测
            locations = self.getCropedFace(frame)

            if locations is not None:

                locations = locations * np.array([frame_w, frame_h, frame_w, frame_h])

                # 矩形坐标
                l, t, r, b = locations.astype('int')

                # 构造 dlib 类型
                face = dlib.rectangle(l, t, r, b)

                # 获取关键点
                points = self.shape_detector(frame, face)

                # 绘制关键点
                # for point in points.parts():
                #     cv2.circle(frame, (point.x, point.y), 2, (0, 255, 0), 1)

                # 获取 mask
                now_time = time.time()
                if now_time - start_time > 3:
                    # 超过 3 秒, 增加1
                    png_index += 1

                    # 是否超过mask的list长度
                    if (png_index + 1) == len(self.mask_list):
                        png_index = 0

                mask_image = self.mask_list[png_index]

                # 获取画面宽度和高度
                (h_m, w_m) = mask_image.shape[:2]

                # 构造口罩图片参考平面
                ptsA = np.asarray([[0, 0], [w_m // 2, 0], [w_m, 0], [w_m, h_m],
                                   [w_m // 2, h_m], [0, h_m]
                                   ])

                # 选取人脸参考点
                points_list = list(points.parts())

                # 左上角, 上中, 右上角, 右下角, 下中, 左下角,
                point_1 = points_list[1].x, points_list[1].y
                point_2 = points_list[28].x, points_list[28].y
                point_3 = points_list[15].x, points_list[15].y
                point_4 = points_list[15].x, points_list[8].y
                point_5 = points_list[8].x, points_list[8].y
                point_6 = points_list[1].x, points_list[8].y

                ptsB = np.asarray([point_1, point_2, point_3, point_4, point_5, point_6])

                # 单位性矩阵
                (H, status) = cv2.findHomography(ptsA, ptsB)

                # 进行透视透视
                mask_warped = cv2.warpPerspective(mask_image, H, (frame_w, frame_h), None,
                                                  cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

                mask_warped = mask_warped.astype(np.uint8)

                imask = mask_warped > 1
                frame[imask] = mask_warped[imask]

                # 画人脸检测框
                cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)

            cv2.imshow('demo', frame)

            if cv2.waitKey(10) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


mv = MaskVideo()
mv.main()























































































