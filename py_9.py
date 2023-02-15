"""
人脸考勤项目
1. 人脸注册: 将人脸特征存进 feature.csv
2. 人脸识别: 将检测的人脸特征与 feature.csv 中的特征做比较, 如果比中把考勤记录写入 attendance.csv

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import dlib
import time
import csv


# 人脸注册的方法
def faceRegister(label_id, name, count, interval):
    """
    :param label_id: 人脸 ID
    :param name: 人脸姓名
    :param count: 采集数量
    :param interval: 采集间隔时间
    :return:
    """
    # 1. 检测人脸
    # 2. 获取68个关键点
    # 3. 获取特征描述符
    cap = cv2.VideoCapture(0)

    # 获取长宽
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 构造人脸检测器
    hog_face_detector = dlib.get_frontal_face_detector()

    # 获取关键点检测模型
    shape_detector = dlib.shape_predictor('./weight/shape_predictor_68_face_landmarks.dat')

    # 特征描述符
    face_descriptor_extractor = dlib.face_recognition_model_v1('./weight/dlib_face_recognition_resnet_model_v1.dat')

    # 开始时间
    start_time = time.time()

    # 采集次数(执行次数)
    collect_count = 0

    # csv 文件写入器
    f = open('./data/feature.csv', 'a', newline="")
    csv_writer = csv.writer(f)

    while True:

        rec, frame = cap.read()

        # 缩放
        frame = cv2.resize(frame, (width//2, height//2))

        frame = cv2.flip(frame, 1)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        detections = hog_face_detector(frame, 1)

        # 遍历人脸, 进行参数解析
        for face in detections:
            l = face.left()
            t = face.top()
            r = face.right()
            b = face.bottom()

            # 获取人脸 68 个关键点坐标
            points = shape_detector(frame, face)

            # 绘制关键点
            for point in points.parts():
                cv2.circle(frame, (point.x, point.y), 2, (0, 255, 0), -1)

            # 矩形人脸框
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)

            # 采集次数
            if collect_count < count:

                # 获取当前时间
                now_time = time.time()

                # 判断时间间隔
                if now_time - start_time > interval:

                    # 获取特征描述符
                    face_descriptor = face_descriptor_extractor.compute_face_descriptor(frame, points)

                    # 特征描述符转为列表
                    face_descriptor = [f for f in face_descriptor]

                    # 写入 csv
                    line = [label_id, name, face_descriptor]
                    csv_writer.writerow(line)

                    collect_count += 1
                    print("采集次数:{}".format(collect_count))

                    start_time = now_time

                else:
                    pass

            else:
                # 采集完毕
                print('采集完毕')
                return




        # 图片显示
        cv2.imshow('face attendance', frame)

        # 退出条件
        if cv2.waitKey(10) & 0xFF == 27:
            break

    f.close()
    cap.release()
    cv2.destroyAllWindows()

    pass

# 获取并组装 csv 文件中的特征
def getFeatureList():

    # 构造列表
    label_list = []
    name_list = []
    feature_list = None
    with open('./data/feature.csv', 'r') as f:
        csv_reader = csv.reader(f)

        for line in csv_reader:
            label_id = line[0]
            name = line[1]

            label_list.append(label_id)
            name_list.append(name)

            # 特征描述符类型: str, 将特征描述符转为 list
            face_descriptor = eval(line[2])

            #
            face_descriptor = np.array(face_descriptor, dtype=np.float64)
            face_descriptor = np.reshape(face_descriptor, (1, -1))
            print(face_descriptor.shape)

            # 进行连接
            if feature_list is None:
                feature_list = face_descriptor

            else:
                np.concatenate((feature_list, face_descriptor), axis=0)

    return label_list, name_list, feature_list



# 人脸识别
# 1. 实时获取视频流中的人脸特征描述符
# 2. 将它与库的特征做距离判断
# 3. 找到预测的ID, NAME
# 4. 考勤纪律存进csv文件, 第一次识别存入, 或者隔一段时间在存入
def faceRecognizer(threshold=0.5):

    cap = cv2.VideoCapture(0)

    # 获取长宽
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 构造人脸检测器
    hog_face_detector = dlib.get_frontal_face_detector()

    # 获取关键点检测模型
    shape_detector = dlib.shape_predictor('./weight/shape_predictor_68_face_landmarks.dat')

    # 特征描述符
    face_descriptor_extractor = dlib.face_recognition_model_v1('./weight/dlib_face_recognition_resnet_model_v1.dat')

    # 读取特征
    label_list, name_list, feature_list = getFeatureList()

    # 字典记录人脸识别记录
    recog_record = {}

    # 写入 csv
    f = open('./data/attendance.csv', 'a', newline="")
    csv_writer = csv.writer(f)

    # 帧率信息
    fps_time = time.time()

    while True:

        rec, frame = cap.read()

        # 缩放
        frame = cv2.resize(frame, (width // 2, height // 2))

        frame = cv2.flip(frame, 1)

        # 检测人脸
        detections = hog_face_detector(frame, 1)

        # 遍历人脸, 进行参数解析
        for face in detections:
            l = face.left()
            t = face.top()
            r = face.right()
            b = face.bottom()

            # 获取人脸 68 个关键点坐标
            points = shape_detector(frame, face)

            # 矩形人脸框
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)

            # 获取特征描述符
            face_descriptor = face_descriptor_extractor.compute_face_descriptor(frame, points)

            # 特征描述符转为列表
            face_descriptor = [f for f in face_descriptor]

            # 计算与库的距离
            face_descriptor = np.array(face_descriptor, dtype=np.float64)
            distances = np.linalg.norm((face_descriptor - feature_list), axis=1)

            # 获取最短距离的索引
            min_index = np.argmin(distances)

            # 获取最短距离
            min_distance = distances[min_index]

            #
            if min_distance < threshold:

                predict_id = label_list[min_index]
                predict_name = name_list[min_index]

                # 人脸框下面增加文字,形式: 'xiaoalei 0.23'


                cv2.putText(frame, predict_name + str(round(min_distance, 2)), (l, b+30),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 225, 0), 1)

                now = time.time()
                need_insert = False
                # 判断是否识别过
                if predict_name in recog_record:
                    # 存过
                    # 隔一段时间在存
                    if time.time() - recog_record[predict_name] > 3:
                        # 超过阈值时间, 再存一次
                        need_insert = True
                        print(predict_name)
                        recog_record[predict_name] = now

                    else:
                        pass
                        need_insert = False

                else:
                    # 没有存过
                    recog_record[predict_name] = now

                    # 将记录存进 csv 文件
                    need_insert = True
                    print(predict_name)

                if need_insert == True:
                    time_local = time.localtime(recog_record[predict_name])

                    # 转换格式
                    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time_local)
                    line = [predict_id, predict_name, min_distance, time_str]
                    csv_writer.writerow(line)


        # 计算帧率
        now = time.time()
        fps = 1 / (now - fps_time)
        fps_time = now

        cv2.putText(frame, 'fps'+str(round(fps, 2)), (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 225, 0), 1)


        # 图片显示
        cv2.imshow('face attendance', frame)

        # 退出条件
        if cv2.waitKey(10) & 0xFF == 27:
            break

    f.close()
    cap.release()
    cv2.destroyAllWindows()




faceRegister(label_id=1, name='lei', count=3, interval=3)
faceRecognizer(threshold=0.5)




























