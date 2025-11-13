#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse

import cv2 as cv
import numpy as np
import mediapipe as mp
import math
# from mtcnn.mtcnn import MTCNN

# from utils import CvFpsCalc
from collections import deque
# import cv2 as cv


# class CvFpsCalc(object):
#     def __init__(self, buffer_len=1):
#         self._start_tick = cv.getTickCount()
#         self._freq = 1000.0 / cv.getTickFrequency()
#         self._difftimes = deque(maxlen=buffer_len)
#
#     def get(self):
#         current_tick = cv.getTickCount()
#         different_time = (current_tick - self._start_tick) * self._freq
#         self._start_tick = current_tick
#
#         self._difftimes.append(different_time)
#
#         fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
#         fps_rounded = round(fps, 2)
#
#         return fps_rounded
# Footer

'''
# 左目：中心
            cv.circle(image, landmark_point[468], 2, (0, 0, 255), -1)
            # 左目：目頭側
            cv.circle(image, landmark_point[469], 2, (0, 0, 255), -1)
            # 左目：上側
            cv.circle(image, landmark_point[470], 2, (0, 0, 255), -1)
            # 左目：目尻側
            cv.circle(image, landmark_point[471], 2, (0, 0, 255), -1)
            # 左目：下側
            cv.circle(image, landmark_point[472], 2, (0, 0, 255), -1)

            # 右目：中心
            cv.circle(image, landmark_point[473], 2, (0, 0, 255), -1)
            # 右目：目尻側
            cv.circle(image, landmark_point[474], 2, (0, 0, 255), -1)
            # 右目：上側
            cv.circle(image, landmark_point[475], 2, (0, 0, 255), -1)
            # 右目：目頭側
            cv.circle(image, landmark_point[476], 2, (0, 0, 255), -1)
            # 右目：下側
            cv.circle(image, landmark_point[477], 2, (0, 0, 255), -1)
'''


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def calc_biase(landmark_points):
    left_x = round(landmark_points[468][0] - landmark_points[33][0], 2)
    left_y = round(landmark_points[468][1] - landmark_points[33][1], 2)
    right_x = round(landmark_points[473][0] - landmark_points[362][0], 2)
    right_y = round(landmark_points[473][1] - landmark_points[362][1], 2)

    # print('left_x: ', left_x)
    # print('left_y: ', left_y)
    # print('right_x: ', right_x)
    # print('right_y: ', right_y)
    return (left_x, left_y), (right_x, right_y)

def calc_biase_and_ratio(landmark_points):
    left_x = round(landmark_points[468][0] - landmark_points[33][0], 2)

    # left_y = round(landmark_points[468][1] - landmark_points[33][1], 2)
    left_y = round(landmark_points[468][1] - landmark_points[27][1], 2)
    left_y_max = round(landmark_points[23][1] - landmark_points[27][1], 2)
    left_y_ratio = left_y / left_y_max

    right_x = round(landmark_points[473][0] - landmark_points[362][0], 2)

    # right_y = round(landmark_points[473][1] - landmark_points[362][1], 2)
    right_y = round(landmark_points[473][1] - landmark_points[257][1], 2)
    right_y_max = round(landmark_points[253][1] - landmark_points[257][1], 2)
    right_y_ratio = right_y / right_y_max
    # print('left_x: ', left_x)
    # print('left_y_ratio: ', left_y_ratio)
    # print('right_x: ', right_x)
    # print('right_y_ratio: ', right_y_ratio)
    return (left_x, left_y_ratio), (right_x, right_y_ratio)

def calc_biase_only(landmark_points):
    left_x = round(landmark_points[468][0] - landmark_points[173][0], 2)

    # print('468: {}, 398: {}'.format(landmark_points[468], landmark_points[173]))
    # left_y = round(landmark_points[468][1] - landmark_points[33][1], 2)
    left_y = round(landmark_points[468][1] - landmark_points[173][1], 2)     #27, 33
    left_y_max = round(landmark_points[23][1] - landmark_points[27][1], 2)
    left_y_ratio = left_y / left_y_max

    right_x = round(landmark_points[473][0] - landmark_points[398][0], 2)

    # right_y = round(landmark_points[473][1] - landmark_points[362][1], 2)
    right_y = round(landmark_points[473][1] - landmark_points[398][1], 2)
    right_y_max = round(landmark_points[253][1] - landmark_points[257][1], 2)
    right_y_ratio = right_y / right_y_max
    # print('left_x: ', left_x)
    # print('left_y_ratio: ', left_y_ratio)
    # print('right_x: ', right_x)
    # print('right_y_ratio: ', right_y_ratio)
    return (left_x, left_y), (right_x, right_y)

def calc_biase_ratio(landmark_points):
    left_x = round(landmark_points[468][0] - landmark_points[130][0], 2)
    left_x_max = round(landmark_points[190][0] - landmark_points[130][0], 2)
    left_x_ratio = left_x/left_x_max

    left_y = round(landmark_points[468][1] - landmark_points[27][1], 2)
    left_y_max = round(landmark_points[23][1] - landmark_points[27][1], 2)
    left_y_ratio = left_y / left_y_max



    # left_y_max = round(landmark_points[133][0] - landmark_points[33][0], 2)
    right_x = round(landmark_points[473][0] - landmark_points[414][0], 2)
    right_x_max = round(landmark_points[359][0] - landmark_points[414][0], 2)
    right_x_ratio = right_x / right_x_max

    right_y = round(landmark_points[473][1] - landmark_points[257][1], 2)
    right_y_max = round(landmark_points[253][1] - landmark_points[257][1], 2)
    right_y_ratio = right_y / right_y_max

    # print('left_x_ratio: ', left_x_ratio)
    # print('left_y_ratio: ', left_y_ratio)
    # print('right_x_ratio: ', right_x_ratio)
    # print('right_y_ratio: ', right_y_ratio)

    return (left_x_ratio, left_y_ratio), (right_x_ratio, right_y_ratio)

def calc_biase_iris(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min((landmark.x * image_width), image_width - 1)
        landmark_y = min((landmark.y * image_height), image_height - 1)

        landmark_point.append((landmark_x, landmark_y))
    return calc_biase(landmark_point)

def calc_biase_iris_ratio(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min((landmark.x * image_width), image_width - 1)
        landmark_y = min((landmark.y * image_height), image_height - 1)

        landmark_point.append((landmark_x, landmark_y))

    return calc_biase_only(landmark_point)
    # return calc_biase_and_ratio(landmark_point)
    # return calc_biase_ratio(landmark_point)
# 人脸对齐
def face_alignment(faceImg, left_eye, right_eye):
    eye_center = ((left_eye[0]+right_eye[0])/2, (left_eye[1]+right_eye[1])/2)
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    angle = math.atan2(dy, dx)*180/math.pi

    RotateMatrix = cv.getRotationMatrix2D(eye_center, angle, scale=1)
    RotImg = cv.warpAffine(faceImg, RotateMatrix, (faceImg.shape[1], faceImg.shape[0]), flags=cv.INTER_CUBIC)
    return RotImg



# 返回眼球中心和半径
def calc_min_enc_losingCircle(landmark_list):
    # 最小外包圆
    center, radius = cv.minEnclosingCircle(np.array(landmark_list))
    center = (int(center[0]), int(center[1]))
    radius = int(radius)
    return center, radius, landmark_list[0]


import cv2

LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

NOSE = [4]

RIGHT_IRIS = [474, 475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]
LEFT_EYE_CORNER = [133]
RIGHT_EYE_CORNER = [398]

# from dlib_detect_nose import dlib_process

#pupil_detection
class pupil_det:
    def __init__(self):
        mp_face_mesh = mp.solutions.face_mesh
        self.mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.count = 0

        # 记载模型
        # self.face_detetor = MTCNN()

        # 检测人脸



        # self.dlib = dlib_process()

    def compute_biase(self, left_eye, right_eye, nose):
        left_to_nose = (nose[0] - left_eye[0][0], nose[1] - left_eye[0][1])
        right_to_nose = (right_eye[0][0] - nose[0], nose[1] - right_eye[0][1])

        return left_to_nose, right_to_nose


    def compute_iris_ratio_in_eye_rect(self, left_iris_center, right_iris_center, left_eye_rect, right_eye_rect):
        left_x_ratio = (left_iris_center[0] - left_eye_rect[0])/left_eye_rect[2]
        left_y_ratio = (left_iris_center[1] - left_eye_rect[1]) / left_eye_rect[3]
        right_x_ratio = (right_iris_center[0] - right_eye_rect[0]) / right_eye_rect[2]
        right_y_ratio = (right_iris_center[1] - right_eye_rect[1]) / right_eye_rect[3]

        left_x_ratio, left_y_ratio, right_x_ratio, right_y_ratio = round(left_x_ratio,2), round(left_y_ratio,2),\
                                                                   round(right_x_ratio,2), round(right_y_ratio,2)

        return (left_x_ratio, left_y_ratio), (right_x_ratio, right_y_ratio)

    def get_eye_rects(self, mesh_points):
        left_eye_rect = cv2.boundingRect(mesh_points[LEFT_EYE])  # x, y, w, h = cv2.boundingRect(cont)
        right_eye_rect = cv2.boundingRect(mesh_points[RIGHT_EYE])
        # image, (x, y), (x+w, y+h), (0, 0, 255), 10
        return left_eye_rect, right_eye_rect

    def process_img(self, frame):

        # frame = self.clahe.apply(frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = self.mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_points = np.array(
                [np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            # center point [x, y], radius
            left_eye_info = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            right_eye_info = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])


            # 测试代码-2-14left_to_nose
            left_eye_rect, right_eye_rect = self.get_eye_rects(mesh_points)
            left_ratio, right_ratio = self.compute_iris_ratio_in_eye_rect(mesh_points[468], mesh_points[473], left_eye_rect, right_eye_rect)

            print('left x: {}, y:{}'.format(left_ratio[0], left_ratio[1]))
            print('left w: {}, h:{}'.format(left_eye_rect[2], left_eye_rect[3]))
            print('right w: {}, h:{}'.format(right_eye_rect[2], right_eye_rect[3]))
            cv2.rectangle(frame, (left_eye_rect[0], left_eye_rect[1]),
                          (left_eye_rect[0]+left_eye_rect[2], left_eye_rect[1]+left_eye_rect[3]), (0,0,255))

            cv2.rectangle(frame, (right_eye_rect[0], right_eye_rect[1]),
                          (right_eye_rect[0] + right_eye_rect[2], right_eye_rect[1] + right_eye_rect[3]), (0, 0, 255))


            # left_eye = mesh_points[468]
            # right_eye = mesh_points[473]

            left_eye_info = (mesh_points[468], left_eye_info[1])
            right_eye_info = (mesh_points[473], right_eye_info[1])

            nose_info = mesh_points[4]
            # print('nose_info: ', nose_info)
            left_to_nose, right_to_nose = self.compute_biase(left_eye_info, right_eye_info, nose_info)

            # return
            return left_eye_info, right_eye_info, left_to_nose, right_to_nose

        else:
            return False, False, False, False

    def process_img_ratio(self, frame):

        # frame = self.clahe.apply(frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = self.mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_points = np.array(
                [np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            # center point [x, y], radius
            # left_eye_info = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            # right_eye_info = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])


            # 测试代码-2-14left_to_nose
            left_eye_rect, right_eye_rect = self.get_eye_rects(mesh_points)
            left_ratio, right_ratio = self.compute_iris_ratio_in_eye_rect(mesh_points[468], mesh_points[473], left_eye_rect, right_eye_rect)

            # print('left x: {}, y:{}'.format(left_ratio[0], left_ratio[1]))
            # print('left w: {}, h:{}'.format(left_eye_rect[2], left_eye_rect[3]))


            return (mesh_points[468], left_ratio), (mesh_points[473], right_ratio)

            # cv2.rectangle(frame, (left_eye_rect[0], left_eye_rect[1]),
            #               (left_eye_rect[0]+left_eye_rect[2], left_eye_rect[1]+left_eye_rect[3]), (0,0,255))
            #
            # cv2.rectangle(frame, (right_eye_rect[0], right_eye_rect[1]),
            #               (right_eye_rect[0] + right_eye_rect[2], right_eye_rect[1] + right_eye_rect[3]), (0, 0, 255))


            # left_eye = mesh_points[468]
            # right_eye = mesh_points[473]

            # left_eye_info = (mesh_points[468], left_eye_info[1])
            # right_eye_info = (mesh_points[473], right_eye_info[1])

            # nose_info = mesh_points[4]
            # print('nose_info: ', nose_info)
            # left_to_nose, right_to_nose = self.compute_biase(left_eye_info, right_eye_info, nose_info)

            # return
            # return left_eye_info, right_eye_info, left_to_nose, right_to_nose

        else:
            return False, False
    def cal_biase_with_eye_corner(self, left_iris_center, left_coner, right_iris_center, right_coner):
        left_dx = left_coner[0] - left_iris_center[0]
        left_dy = left_coner[1] - left_iris_center[1]

        # right_dx = right_iris_center[0] - right_coner[0]
        # right_dy = right_iris_center[1] - right_coner[1]

        right_dx = right_coner[0] - right_iris_center[0]
        right_dy = right_coner[1] - right_iris_center[1]

        return (left_dx, left_dy), (right_dx, right_dy)

    def process_img_biase_corner(self, frame):

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]

        # 获取XML文件，加载人脸检测器
        # faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # 色彩转换，转换为灰度图像

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 调用函数detectMultiScale
        # faces = faceCascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(5, 5))

        # x, y, w, h = faces[0]
        # 打印输出的测试结果
        # print("发现{0}个人脸！".format(len(faces)))
        # 逐个标注人脸
        # for (x, y, w, h) in faces[0]:
        # detections = self.face_detetor.detect_faces(frame)
        # x, y, w, h = detections[0]['box']
        #
        # nose = (x+w*0.5, y+h*0.5)

        results = self.mesh.process(rgb_frame)

        mesh_points = []
        if results.multi_face_landmarks:
            faceLms = results.multi_face_landmarks[0]

            box = calc_bounding_rect(frame, faceLms)
            for id, lm in enumerate(faceLms.landmark):
                # print(lm)
                # ih, iw, ic = img.shape
                # 关键点坐标
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                # print(id, x, y)
                mesh_points.append((x, y))

            nose = (box[0], box[1])

            left_biase, right_biase = self.cal_biase_with_eye_corner(mesh_points[468], nose, mesh_points[473], nose)
            return (mesh_points[468], left_biase), (mesh_points[473], right_biase)
        else:
            return False, False


    def process_img_biase_corner2(self, frame):

        # frame = self.clahe.apply(frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = self.mesh.process(rgb_frame)

        # nose_p = self.dlib.dlib_process_img(rgb_frame)
        # print('nose_p: ', nose_p.x)
        if results.multi_face_landmarks:
            mesh_points = np.array(
                [np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            # center point [x, y], radius
            # left_eye_info = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            # right_eye_info = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            # 测试代码-2-14left_to_nose
            # left_eye_rect, right_eye_rect = self.get_eye_rects(mesh_points)
            # left_ratio, right_ratio = self.compute_iris_ratio_in_eye_rect(mesh_points[468], mesh_points[473], left_eye_rect, right_eye_rect)

            # print('left x: {}, y:{}'.format(left_ratio[0], left_ratio[1]))
            # print('left w: {}, h:{}'.format(left_eye_rect[2], left_eye_rect[3]))
            # left_biase, right_biase = self.cal_biase_with_eye_corner(mesh_points[468], mesh_points[133], mesh_points[473], mesh_points[362])


            #  (300,200)
            nose_position = (mesh_points[4][0], mesh_points[4][1])

            if self.count == 0:
                self.nose = nose_position
                self.count = 1
            # position = (310,263)
            # import random
            # rd = random.uniform(-5,5)
            # print('rd: ', rd)
            # position = (position[0]+rd, position[1]+rd)

            # position = (nose_p.x, nose_p.y)
            left_biase, right_biase = self.cal_biase_with_eye_corner(mesh_points[468], nose_position,
                                                                     mesh_points[473], nose_position)

            # print('nose position x: {}, y:{}'.format(nose_position[0], nose_position[1]))

            # if self.count == 0:
            #     self.count = 1

            # print('righ biase x: {}, y:{}'.format(right_biase[0], right_biase[1]))

            return (mesh_points[468], left_biase), (mesh_points[473], right_biase)

        else:
            return False, False

    def draw_eyes(self, img, left_eye_info, right_eye_info):

        res_img = img.copy()
        left_center, left_radius = left_eye_info[0], left_eye_info[1]
        right_center, right_radius = right_eye_info[0], right_eye_info[1]

        left_center = int(left_center[0]), int(left_center[1])
        right_center = int(right_center[0]), int(right_center[1])

        left_radius, right_radius = int(left_radius), int(right_radius)
        # print('left_center: {}, left_radius: {}'.format(left_center, left_radius))


        cv2.circle(res_img, left_center, 1, (255,0,0), 2)#left_radius
        cv2.circle(res_img, right_center, 1, (255, 0, 0), 2)#left_radius

        # cv.circle(image, right_eye[0], right_eye[1], (0, 255, 0), 2)

        return res_img


NUM_FACE = 1
# from mtcnnruntime import MTCNN
# mtcnn = MTCNN()
import numpy as np
# from collections import deque

class pupil_detection:
    def __init__(self):
        # self.staticMode = staticMode
        # self.maxFace = maxFace
        # self.minDetectionCon = minDetectionCon
        # self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        mpFaceMesh = mp.solutions.face_mesh
        # self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFace, self.minDetectionCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

        # mp_face_mesh = mp.solutions.face_mesh
        self.faceMesh = mpFaceMesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        # self.nose_list = deque(maxlen=20)
        # self.dlib = dlib_process()

    def findFaceLandmark(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)

        faces = []

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACE_CONNECTIONS, self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    # print(lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    #cv2.putText(img, str(id), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)
                    #print(id, x, y)
                    face.append([x,y])
                faces.append(face)
        return img, faces

    def cal_biase_with_eye_corner(self, left_iris_center, left_coner, right_iris_center, right_coner):
        left_dx = left_coner[0] - left_iris_center[0]
        left_dy = left_coner[1] - left_iris_center[1]

        # right_dx = right_iris_center[0] - right_coner[0]
        # right_dy = right_iris_center[1] - right_coner[1]

        right_dx = right_coner[0] - right_iris_center[0]
        right_dy = right_coner[1] - right_iris_center[1]

        return (left_dx, left_dy), (right_dx, right_dy)

    def process_img_biase_corner(self, img):

        # boxes, landmarks = mtcnn.detect(img)
        # nose_pos = (landmarks[0][4], landmarks[0][5])
        # nose_p = self.dlib.dlib_process_img(img)
        # nose_p = (nose_p.x, nose_p.y)
        img, faces = self.findFaceLandmark(img, False)
        if len(faces) > 0:
            face = faces[0]
            left_pupil, nose, right_pupil = face[468], face[4], face[473]
            # left_pupil, nose, right_pupil = face[469], face[4], face[474]

            # print('nose p: ', nose)
            nose = (100, 100)
            # self.nose_list.append(np.array(nose))
            left_biase, right_biase = self.cal_biase_with_eye_corner(left_pupil, nose, right_pupil, nose)
            # print('left_pupil: x {}, y {}, left_biase: x {}, y: {}'.format(left_pupil[0], left_pupil[1], left_biase[0], left_biase[1]))
            # print('right_pupil: x {}, y {}, right_biase: x {}, y: {}'.format(right_pupil[0], right_pupil[1], right_biase[0],
            #                                                                right_biase[1]))
            return (left_pupil, left_biase), (right_pupil, right_biase)
        else:
            return False, False



    def process_img(self, frame):
        pupil_indices = [468, 473]

        left_landmark_indices = [4, 33, 133]            # 4
        right_landmark_indices = [4, 362, 263]          # 4

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = self.faceMesh.process(rgb_frame)

        # pst = []
        if results.multi_face_landmarks:
            mesh_points = np.array(
                [np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            left_pst = [mesh_points[i] for i in left_landmark_indices]
            right_pst = [mesh_points[i] for i in right_landmark_indices]

            pupil_center = [mesh_points[i] for i in pupil_indices]


            return (left_pst, right_pst, pupil_center)
        else:
            return False, False, False

    def process_img_1(self, frame):
        pupil_indices = [468, 473, 4]       # 4为鼻子

        left_landmark_indices = [33, 173]   # 133
        right_landmark_indices = [398, 263] # 362

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = self.faceMesh.process(rgb_frame)

        # pst = []
        if results.multi_face_landmarks:
            mesh_points = np.array(
                [np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            left_pst = [mesh_points[i] for i in left_landmark_indices]
            right_pst = [mesh_points[i] for i in right_landmark_indices]

            pupil_center = [mesh_points[i] for i in pupil_indices]


            return (left_pst, right_pst, pupil_center)
        else:
            return False, False, False

if __name__ == '__main__':
    # main()
    # run()
    PD = pupil_detection()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while 1:
        _, frame = cap.read()

        frame = cv2.flip(frame, 1)

        left_eye_info, right_eye_info = PD.process_img_biase_corner(frame)
        # rotate_img = face_alignment(frame, left_eye_info[0], right_eye_info[0])
        '''
        left_eye_info, right_eye_info, left_to_nose, right_to_nose = PD.process_img(frame)
        show_img = PD.draw_eyes(frame, left_eye_info, right_eye_info)
        '''
        cv2.imshow('test.jpg', frame)
        # cv2.imshow('rotate_img.jpg', rotate_img)
        cv2.waitKey(20)

    cap.release()