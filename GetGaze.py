from iris_detection import pupil_detection
from mouseControl import Mouse_Control
import pynput.mouse
from saveGazeData import Data_saving
import cv2
import numpy as np
import pyautogui as pg

import os
import pandas as pd

import sys
import threading
import time


from sklearn.metrics import mean_squared_error

from Show_points_for_calibration import ShowCalibrationPoint

from pykalman import KalmanFilter
from saveVideos import saveFaceAndScreen_new
from HeadPoseCalibrate import HeadPositionCalibration

from headPose_new import RayPlaneIntersection, headPoseMediapipe

from blink_detection_1 import BlinkDetector_1
import collections
import random
WINDOW_NAME = 'start_calibration'

# 每个点需要校准的帧数，取n帧的均值
Calib_times_per_point = 8#  86#20  calib_times_per_point

EyeTrackingStatusList = ['static', 'stable', 'moving']



def draw_gaze_point(gaze_point, screen_sz, radius=15, img=None):
    if img is None:
        InputImg = np.zeros((screen_sz[1], screen_sz[0], 3), np.uint8)
        InputImg.fill(200)
        # InputImg=ImageGrab.grab(bbox=(0, 0, 1920, 1080))#screen_sz[0],screen_sz[1]
        # InputImg = cv2.cvtColor(np.asarray(InputImg), cv2.COLOR_RGB2BGR)
        # InputImg = cv2.resize(InputImg, (1536, 864))
    else:
        InputImg = img.copy()
    # print('img.shape: ', InputImg.shape)
    # h,w,c = InputImg.shape
    # print(w,h,c)
    # if is_draw_axis:
    #     draw_scale_parameters(InputImg, 0, 0)
    resimg = cv2.circle(InputImg, gaze_point, radius, (255, 255, 0), thickness=-1)        #255,0,255   (0,255,0)
    return resimg

def draw_gaze_point_1(gaze_point, screen_sz, radius=15, img=None):
    if img is None:
        InputImg = np.zeros((screen_sz[1], screen_sz[0], 3), np.uint8)
        InputImg.fill(200)
    else:
        InputImg = img.copy()
    resimg = cv2.circle(InputImg, gaze_point, radius, (255, 255, 0), thickness=2)        #255,0,255   (0,255,0)
    return resimg

def draw_gaze_point_and_head_point(head_point, gaze_point, screen_sz, radius=15, img=None):
    if img is None:
        InputImg = np.zeros((screen_sz[1], screen_sz[0], 3), np.uint8)
        InputImg.fill(200)
    else:
        InputImg = img.copy()

    resimg = cv2.circle(InputImg, gaze_point, radius, (255,0,255), thickness=-1)        #255,0,255   (0,255,0)
    resimg = cv2.circle(resimg, head_point, radius, (255, 255, 0), thickness=-1)  # 255,0,255   (0,255,0)

    return resimg

def draw_headgaze_point(head_point, screen_sz, radius=15, count=0, img=None):
    if img is None:
        InputImg = np.zeros((screen_sz[1], screen_sz[0], 3), np.uint8)
        InputImg.fill(200)
    else:
        InputImg = img.copy()

    # resimg = cv2.circle(InputImg, gaze_point, radius, (255,0,255), thickness=-1)        #255,0,255   (0,255,0)
    # (255,0,255)
    # colors = [(255,0,255), (255,150,255)]

    colors = [(255, 0, 255), (255, 48, 210)]
    # index = 1 if count % 10 == 0 else 0
    color = colors[count]

    resimg = cv2.circle(InputImg, head_point, int(radius/2), color, thickness=-1)  # 255,0,255   (0,255,0)
    resimg = cv2.circle(resimg, head_point, radius, color, 5)  # 255,0,255   (0,255,0)
    resimg = cv2.line(resimg, (head_point[0]-2*radius, head_point[1]), (head_point[0]+2*radius, head_point[1]), color, 5)

    resimg = cv2.line(resimg, (head_point[0], head_point[1]-2*radius), (head_point[0], head_point[1]+2*radius), color, 5)
    return resimg

# 以n帧的平均值来计算

class EyeTracking():
    def __init__(self, capNumber=0, monitor_pixels=(1920,1080), user_dis=40, re_calib=True, recording_calib=False, counting_time=False,
                 onePointCalibration=False,ID_number=0, fer=False, saving=False, VideoSavePath=''):
        super(EyeTracking, self).__init__()
        self.capID = capNumber  # 相机编号
        self.Calib = False
        self.monitor_pixels = monitor_pixels
        self.re_calib = re_calib

        # self.timer = Timer()
        self.is_count_time = counting_time

        self.Mapx, self.Mapy = 0, 0
        self.startMap = False
        self.stop = False

        # add at 10/20  是否是单点校准
        self.onePointCalibration = onePointCalibration
        # self.RayIntersect = RayPlaneIntersection(monitor_pixels, 25.5, user_camere_distance=400)        #500
        
        #
        self.RayIntersect = RayPlaneIntersection(monitor_pixels, 15.5, user_camere_distance=user_dis*10)  # 500   15.6

        # add at 10/1  视频缓存-面部和屏幕
        # saving false不存数据
        self.Videosave = saveFaceAndScreen_new(saving=saving, savePath=VideoSavePath, ID_number=ID_number)    #saveFaceAndScreen(),   saving=True

        self.cap = self.Videosave.videocapture       # 缓冲视频流
        # self.savePath = VideoSavePath

        self.isRecording = recording_calib  # 用于保存截屏数据

        if self.isRecording:
            self.excelName = os.path.join(VideoSavePath, 'gaze.csv')        #存放gaze数据的文件
            # screen_video = os.path.join(VideoSavePath, 'screen.mp4')
            user_video = os.path.join(VideoSavePath, 'user.mp4')
            # self.screen_writer = cv2.VideoWriter(screen_video, cv2.VideoWriter_fourcc(*"mp4v"), 40, (640,360))#(960, 540)  40, 20
            self.user_writer = cv2.VideoWriter(user_video, cv2.VideoWriter_fourcc(*"mp4v"), 40,
                                                 (int(640), int(480)))
        self.start_saving = False

        #检测眨眼
        self.BD = BlinkDetector_1()
        self.ear_thr = None
        # if self.isRecording:
        #     self.RecordingThread = threading.Thread(target=self.screen_shoot)
        #     self.RecordingThread.start()

        # self.source = WebcamSource()
        # next(self.source)  # start webcam

        self.mapTimes = 0

        self.runback = False
        # 保存预测数据
        self.saveTest = False

        # add 2024-2-15
        self.eye_tracking_status = 'initialize' #初始状态

        # add 2024-3-4
        self.head_pose_stable = {'pitch': 0.0, 'yaw': 0.0, 'roll':0.0}     # 稳定跟踪的头部姿态角度,pitch, yaw, roll

        # add 2024-6-26
        # if fer:
        #     self.FER = Facial_expression_Recognition()

        self.range_detection = False
        self.yaw_range, self.pitch_range = None, None


        # 卡尔曼滤波来平滑眼动跟踪结果-尽在stable时候
        # self.kalman = self.kalman_filter_init_py()
        # self.filtered_state_means0 = np.array([0.0, 0.0, 0.0, 0.0])
        # self.filtered_state_covariances0 = np.eye(4)


    def kalman_filter_init_py(self):
        kf = KalmanFilter(transition_matrices=np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]),
                          observation_matrices=np.array([[1, 0, 0, 0], [0, 1, 0, 0]]),
                          transition_covariance=0.03 * np.eye(4))

        return kf

    def kalman_filter_init_opencv(self):
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.003
        kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 1
        return kalman

    def save_subject_name(self, userName):
        self.saveTest = True
        self.subject = userName
        import os
        path0 = os.path.join(os.getcwd(), 'subjectData')
        if not os.path.exists(path0):
            os.mkdir(path0)

        self.subjectDataPath = os.path.join(path0, userName)
        print('path: ', self.subjectDataPath)

        if not os.path.exists(self.subjectDataPath):
            os.mkdir(self.subjectDataPath)

    def init_head_pose_calibration_for_accuracy_test(self):
        self.stop = False
        HPC = HeadPositionCalibration(self.cap)
        HPC.show_face3(WINDOW_NAME)                 #有些相机慢  需要稍微等待
        print('head position calibration over! ')

    def init_head_pose_calibration(self):
        self.stop = False
        HPC = HeadPositionCalibration(self.cap)
        # HPC.show_face2(WINDOW_NAME)
        self.ear_thr,  _, _ = HPC.show_face_with_blink(WINDOW_NAME)       #添加眨眼检测

        self.BD.set_ear_thr(self.ear_thr)       #设置ear检测系数
        print('head position calibration over! ')

    def init_head_pose_calibration_1(self, init=False, ear_thr=None):
        self.stop = False
        HPC = HeadPositionCalibration(self.cap)
        # HPC.show_face2(WINDOW_NAME)
        # Ear_thr_results = HPC.show_face_with_blink(WINDOW_NAME, init)
        Ear_thr_results = HPC.show_face_with_blink_ch(WINDOW_NAME, init)        #中文版

        if init:
            if Ear_thr_results:
                # print('first time ear_thr: ', Ear_thr_results)
                self.ear_thr = Ear_thr_results[0]
        else:
            # print('next time ear_thr: ', ear_thr)
            self.ear_thr = ear_thr
        # print('ear thr setting: ', self.ear_thr)
        self.BD.set_ear_thr(self.ear_thr)
        print('head position calibration over! ')

    def set_limit_range(self, limit_range):
        self.yaw_range, self.pitch_range = limit_range[0], limit_range[1]
        self.range_detection = True

    def get_ear_thr(self):
        return self.BD.get_ear_thr()  #ear_thr

    def set_ear_thr(self, ear):
        self.ear_thr = ear
        self.BD.set_ear_thr(ear)

    def init_calibration(self, scales=(1.0, 1.0)):
        # self.face_mesh = init_face_mesh()

        self.face_mesh = pupil_detection()
        self.head_model = headPoseMediapipe()  # add 10-20

        # 开始存储
        self.Videosave.start_saving()

        # self.start_one_point_calibration()

        left_scales, right_scales = self.start_calibration()
        print('left scale: {}, right scale: {}'.format(left_scales, right_scales))
        # add 11/2
        self.base_scale = scales  # 一开始设置的基准scale
        print('left scale 0: {}, 1: {}'.format(left_scales[0], left_scales[1]))
        self.set_left_scale(left_scales[0], left_scales[1])
        # self.set_left_scale(100.0, 100.0)
        self.set_right_scale(right_scales[0], right_scales[1])

        # if self.isRecording:
        #     self.isRecording = False
        #     self.RecordingThread.join()  # 等待线程结束

    #prompt.jpg
    def show_notice_if_calibration_failed_1(self):
        cv2.namedWindow('notice.jpg', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('notice.jpg', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # img = np.zeros((self.monitor_pixels[1], self.monitor_pixels[0], 3), np.uint8)

        # notice = 'please do not move your head during calibration'
        # cv2.putText(img, notice, (400, 500), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # cv2.putText('')

        time_start = time.time()
        img = cv2.imread('source2/prompt.jpg')
        img = cv2.resize(img, (1920, 1080))
        while True:
            time_elapse = time.time() - time_start

            # if time_elapse >= 2.0:
            #     break
            cv2.imshow('notice.jpg', img)
            # cv2.waitKey(20)
            if cv2.waitKey(20) & 0xFF == ord(' ') or time_elapse >= 5.0:
                break


        cv2.destroyWindow('notice.jpg')
    def show_notice_if_calibration_failed(self):
        cv2.namedWindow('notice.jpg', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('notice.jpg', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        img = np.zeros((self.monitor_pixels[1], self.monitor_pixels[0], 3), np.uint8)

        notice = 'please do not move your head during calibration'
        cv2.putText(img, notice, (400, 500), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # cv2.putText('')

        time_start = time.time()
        while True:
            time_elapse = time.time() - time_start

            # if time_elapse >= 2.0:
            #     break
            cv2.imshow('notice.jpg', img)
            # cv2.waitKey(20)
            if cv2.waitKey(20) & 0xFF == ord(' ') or time_elapse >= 5.0:
                break


        cv2.destroyWindow('notice.jpg')


    def init_calibration_1(self, scales=(1.0, 1.0), ):
        # self.face_mesh = init_face_mesh()

        self.face_mesh = pupil_detection()
        self.head_model = headPoseMediapipe()  # add 10-20  添加头部模块


        Failed, (left_scales, right_scales) = self.start_calibration_2()

        if Failed:
            return False

        self.base_scale = scales  # 一开始设置的基准scale

        self.set_left_scale(left_scales[0], left_scales[1])
        # self.set_left_scale(100.0, 100.0)
        self.set_right_scale(right_scales[0], right_scales[1])

        return True

    def process_center_one_point_numpy(self, face_mesh, Videocapture, frame_counts=10):

        left_eye = []
        right_eye = []

        left_to_noses = []
        right_to_noses = []
        for i in range(frame_counts):

            frame = Videocapture.get_frame()
            # ret, frame = VideoCap.read()
            if frame is False:
                print('相机获取图像错误')
                break
            frame = cv2.flip(frame, 1)

            # add at 10/1
            # if Videosave is not None:
            #     Videosave.set_face(frame)
            #     Videosave.get_current_face_and_screen()

            # debug_image, left_eyes, left_biases, right_eyes, right_biases = process_img(face_mesh, frame)

            # 加入眼睛与鼻子之间的距离信息
            left_eye_info, right_eye_info, left_to_nose, right_to_nose = face_mesh.process_img(frame)

            if left_eye_info is not False:
                # if (len(left_eyes) > 0):

                left_eye.append(left_eye_info[0])
                right_eye.append(right_eye_info[0])
                left_to_noses.append(left_to_nose)
                right_to_noses.append(right_to_nose)
                # eye = (left_eye_info[0], right_eye_info[0])
                # eyes.append(eye)

        if len(left_eye) == 0:
            print('这个点没有检测到一个人脸')
            return False, False, False, False

        left_pupil = np.mean(np.array(left_eye), 0)
        right_pupil = np.mean(np.array(right_eye), 0)

        leftpupil_to_nose = np.mean(np.array(left_to_noses), 0)
        rightpupil_to_nose = np.mean(np.array(right_to_noses), 0)

        # print('left_pupil: {}, right_pupil: {}'.format(left_pupil, right_pupil))
        return left_pupil, right_pupil, leftpupil_to_nose, rightpupil_to_nose

    def process_ratio_one_point_numpy(self, face_mesh, Videocapture, frame_counts=10, coordinate_or_ratio=True):

        left_eye = []
        right_eye = []
        left_ratio = []
        right_ratio = []
        for i in range(frame_counts):

            frame = Videocapture.get_frame()
            # ret, frame = VideoCap.read()
            if frame is False:
                print('相机获取图像错误')
                break
            frame = cv2.flip(frame, 1)

            # 加入眼睛与鼻子之间的距离信息
            # left_eye_info, right_eye_info, left_to_nose, right_to_nose = face_mesh.process_img(frame)

            # (mesh_points[468], left_ratio), (mesh_points[473], right_ratio) \
            left_info, right_info = face_mesh.process_img_ratio(frame)


            if left_info is not False:
                # if (len(left_eyes) > 0):

                left_eye.append(left_info[0])
                right_eye.append(right_info[0])

                #比例
                left_ratio.append(left_info[1])
                right_ratio.append(right_info[1])

        if len(left_eye) == 0:
            print('这个点没有检测到一个人脸')
            return False, False

        left_pupil = np.mean(np.array(left_eye), 0)
        right_pupil = np.mean(np.array(right_eye), 0)

        left_ratio = np.mean(np.array(left_ratio), 0)
        right_ratio = np.mean(np.array(right_ratio), 0)

        print('left_ratio: {}, right_ratio: {}'.format(left_ratio, right_ratio))

        if coordinate_or_ratio:
            return left_pupil, right_pupil
        else:
            return left_ratio, right_ratio
    def process_biase_one_point_numpy(self, face_mesh, Videocapture, frame_counts=10, coordinate_or_biase=True):

        left_eye = []
        right_eye = []
        left_biase = []
        right_biase = []
        for i in range(frame_counts):

            frame = Videocapture.get_frame()
            # ret, frame = VideoCap.read()
            if frame is False:
                print('相机获取图像错误')
                break
            frame = cv2.flip(frame, 1)

            left_info, right_info = face_mesh.process_img_biase_corner(frame)
            if left_info is not False:
                # if (len(left_eyes) > 0):

                left_eye.append(left_info[0])
                right_eye.append(right_info[0])

                #比例
                left_biase.append(left_info[1])
                right_biase.append(right_info[1])

        if len(left_eye) == 0:
            print('这个点没有检测到一个人脸')
            return False, False

        left_pupil = np.mean(np.array(left_eye), 0)
        right_pupil = np.mean(np.array(right_eye), 0)

        left_ratio = np.mean(np.array(left_biase), 0)
        right_ratio = np.mean(np.array(right_biase), 0)

        print('left_biase: {}, right_biase: {}'.format(left_ratio, right_ratio))

        if coordinate_or_biase:
            return left_pupil, right_pupil
        else:
            return left_ratio, right_ratio

    def process_biase_one_point_numpy_with_head(self, face_mesh, headposeModel, Videocapture, frame_counts=10, coordinate_or_biase=True):

        left_eye = []
        right_eye = []
        nose = []
        left_corner = []
        right_corner = []

        # 存储头部姿态信息
        pitchs = []
        yaws = []
        rolls = []


        for i in range(frame_counts):

            frame = Videocapture.get_frame()
            # ret, frame = VideoCap.read()
            if frame is False:
                print('相机获取图像错误')
                break
            frame = cv2.flip(frame, 1)

            # left_info, right_info = face_mesh.process_img_biase_corner(frame)
            left_info, right_info, pupil = face_mesh.process_img_1(frame)

            if pupil is not False:
                # if (len(left_eyes) > 0):
                left_eye.append(pupil[0])
                right_eye.append(pupil[1])
                #鼻子信息
                nose.append(pupil[2])

                # 左右眼角信息
                left_corner.append(left_info)
                right_corner.append(right_info)


            status, euler_angle = headposeModel.process_img(frame)
            if status:
                pitch, yaw, roll = euler_angle
                pitchs.append(pitch)
                yaws.append(yaw)
                rolls.append(roll)

        if len(left_eye) == 0:
            print('这个点没有检测到一个人脸')
            return False, False

        left_pupil = np.mean(np.array(left_eye), 0)
        right_pupil = np.mean(np.array(right_eye), 0)
        nose_c = np.mean(np.array(nose), 0)

        left_corner = np.array(left_corner)
        left_corner = np.mean(left_corner, axis=0)

        right_corner = np.array(right_corner)
        right_corner = np.mean(right_corner, axis=0)

        # left_ratio = np.mean(np.array(left_biase), 0)
        # right_ratio = np.mean(np.array(right_biase), 0)
        #
        # print('left_biase: {}, right_biase: {}'.format(left_ratio, right_ratio))

        mean_pitch = sum(pitchs) / len(pitchs)
        mean_yaw = sum(yaws) / len(yaws)
        mean_roll = sum(rolls) / len(rolls)

        # if coordinate_or_biase:
        return left_pupil, right_pupil, (left_corner, right_corner), (mean_pitch, mean_yaw, mean_roll), nose_c

    def process_biase_one_point_numpy_with_head_1(self, face_mesh, headposeModel, Videocapture, frame_counts=10, coordinate_or_biase=True):

        left_eye = []
        right_eye = []
        nose = []
        left_corner = []
        right_corner = []

        # 存储头部姿态信息
        pitchs = []
        yaws = []
        rolls = []
        ear_list = []
        BD = BlinkDetector_1()

        for i in range(frame_counts):

            frame = Videocapture.get_frame()
            # ret, frame = VideoCap.read()
            if frame is False:
                print('相机获取图像错误')
                break
            frame = cv2.flip(frame, 1)
            ear_list.append(BD.get_current_ear(frame))

            # left_info, right_info = face_mesh.process_img_biase_corner(frame)
            left_info, right_info, pupil = face_mesh.process_img_1(frame)

            if pupil is not False:
                # if (len(left_eyes) > 0):
                left_eye.append(pupil[0])
                right_eye.append(pupil[1])
                #鼻子信息
                nose.append(pupil[2])

                # 左右眼角信息
                left_corner.append(left_info)
                right_corner.append(right_info)


            status, euler_angle = headposeModel.process_img(frame)
            if status:
                pitch, yaw, roll = euler_angle
                pitchs.append(pitch)
                yaws.append(yaw)
                rolls.append(roll)

        if len(left_eye) == 0:
            print('这个点没有检测到一个人脸')
            return False, False

        left_pupil = np.mean(np.array(left_eye), 0)
        right_pupil = np.mean(np.array(right_eye), 0)
        nose_c = np.mean(np.array(nose), 0)

        left_corner = np.array(left_corner)
        left_corner = np.mean(left_corner, axis=0)

        right_corner = np.array(right_corner)
        right_corner = np.mean(right_corner, axis=0)

        # left_ratio = np.mean(np.array(left_biase), 0)
        # right_ratio = np.mean(np.array(right_biase), 0)
        #
        # print('left_biase: {}, right_biase: {}'.format(left_ratio, right_ratio))

        mean_pitch = sum(pitchs) / len(pitchs)
        mean_yaw = sum(yaws) / len(yaws)
        mean_roll = sum(rolls) / len(rolls)

        ear = np.array(ear_list).mean()
        # if coordinate_or_biase:
        return left_pupil, right_pupil, (left_corner, right_corner), (mean_pitch, mean_yaw, mean_roll), nose_c, ear

    def run_eyetracking_back_1(self):
        # VideoCap, face_mesh, win_sz
        self.runback = True
        # runEyeTrackingWithSmooth_back_head_move
        self.TrackingThread = threading.Thread(target=self.runEyeTrackingWithSmooth_back_head_move_2, args=(self.cap, self.face_mesh, self.head_model, self.monitor_pixels, False))
        self.TrackingThread.start()

    def run_eyetracking_back(self):
        # VideoCap, face_mesh, win_sz
        self.runback = True
        # runEyeTrackingWithSmooth_back_head_move
        self.TrackingThread = threading.Thread(target=self.runEyeTrackingWithSmooth_back_head_move_1, args=(self.cap, self.face_mesh, self.head_model, self.monitor_pixels, False))
        self.TrackingThread.start()


    def stop_eyetracking_back(self):
        self.stopTracking()
        # print('stop all thread.0..')
        self.TrackingThread.join()

        #线程结束，不保存数据了
        if self.isRecording:
            self.user_writer.release()
            # self.screen_writer.release()

        # self.video_thread.join()
        # print('stop all thread.1..')

    def screen_shoot(self):
        vid_writer = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, (int(960), int(540)))
        while self.isRecording:
            img = pg.screenshot()

            open_cv_image = np.array(img)

            # Convert RGB to BGR,opencv read image as BGR,but Pillow is RGB
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
            open_cv_image = cv2.resize(open_cv_image, (960,540))
            vid_writer.write(open_cv_image)
            cv2.waitKey(20)
        print('record over!')
        vid_writer.release()


    def start_eye_tracking_one_point(self):

        timer_s = self.timer if self.is_count_time else None
        # runEyeTrackingWithSmooth(self.cap, self.face_mesh, self.mapModel, self.monitor_pixels, self.re_calib, self.calibModel, timer_s)
        # self.runEyeTrackingWithSmooth_public_one_point(self.cap, self.face_mesh, self.mapModel, self.monitor_pixels,
        #                                      self.re_calib, self.calibModel, timer=timer_s)
        # self.runEyeTrackingWithSmooth_public_one_point(self.cap, self.face_mesh, self.monitor_pixels, timer=timer_s)

        self.runEyeTrackingWithSmooth_public_one_point_modify(self.cap, self.face_mesh, self.monitor_pixels, timer=timer_s)
        self.Videosave.stop_saving()
        self.Videosave.release()

    def set_gaze_point(self, pointx, pointy):
        # self.
        self.Mapx = pointx
        self.Mapy = pointy

    def get_gaze_point(self):
        if self.startMap:
            return self.Mapx, self.Mapy
        return False, False

    def set_center_pupil(self, left_eye, right_eye):
        self.left_center_eye = left_eye
        self.right_center_eye = right_eye

    def save_calibration_pupil(self, left_eye, right_eye):
        self.left_center_eye_original = left_eye
        self.right_center_eye_original = right_eye

    def set_eye_corner(self, corner):

        self.left_eye_corner =corner[0]
        self.right_eye_corner = corner[1]

    # 单点校准
    def start_one_point_calibration(self):

        SCP = ShowCalibrationPoint('test.jpg', (1920, 1080), 150)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # center = show_center_point_on_screen(WINDOW_NAME, self.monitor_pixels)
        center = SCP.show_point_i(0, WINDOW_NAME)

        left_eye, right_eye, corner, headpose_angles, nose = self.process_biase_one_point_numpy_with_head(self.face_mesh, self.head_model, self.cap, frame_counts=20, coordinate_or_biase=True)

        self.RayIntersect.set_center_angle(headpose_angles[0], headpose_angles[1], headpose_angles[2])

        self.head_pose_stable['pitch'], self.head_pose_stable['yaw'] = headpose_angles[0], headpose_angles[1]
        self.set_center_pupil(left_eye, right_eye)

        self.save_calibration_pupil(left_eye, right_eye)

        #存放眼角信息
        self.set_eye_corner(corner)
        self.center_one_point = center

        # print('center left eye: {}, right eye: {}'.format(left_eye, right_eye))

        cv2.destroyWindow(WINDOW_NAME)

        # add 2024-3-5
        # self.last_mes = self.current_mes = center
        # self.last_pre = self.current_pre = center

        # print('center left eye: {}, right eye: {}'.format(left_to_nose, right_to_nose))
    def start_computing_datum_error(self):
        SCP = ShowCalibrationPoint('test.jpg', (1920, 1080), 150, True)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        i = 0
        left_eye_lists = []
        right_eye_lists = []

        left_eye_predict = []
        right_eye_predict = []
        left_eye_predict_1 = []
        right_eye_predict_1 = []
        point_lists = []
        while i < 9:
            point = SCP.show_ablation_point_i(i, WINDOW_NAME)  # show_point

            left_eye, right_eye, corner, headpose_angles, nose = self.process_biase_one_point_numpy_with_head(self.face_mesh,
                                                                                                        self.head_model,
                                                                                                        self.cap,
                                                                                                        frame_counts=20,
                                                                                                        coordinate_or_biase=True)

            left_offset, right_offset = self.compute_eye_corner_offsets(corner[0][0], corner[0][1], corner[1][0], corner[1][1])
            # print('left offset: {}, right offset: {}'.format(left_offset, right_offset))

            pupil_datum = self.compute_new_pupil(left_offset, right_offset)
            left_eye_predict_1.append(pupil_datum[0])
            right_eye_predict_1.append(pupil_datum[1])

            point_lists.append(point)
            left_eye_lists.append(left_eye)
            right_eye_lists.append(right_eye)

            left_eye_predict.append(self.left_center_eye)
            right_eye_predict.append(self.right_center_eye)

            print('predict pupil  left: {}, right: {}'.format(self.left_center_eye, self.right_center_eye))
            print('actual pupil  left: {}, right: {}'.format(left_eye, right_eye))
            print('predict 1 pupil  left: {}, right: {}'.format(pupil_datum[0], pupil_datum[1]))

            i += 1
        print('datum_error computation finished!...')
        cv2.destroyWindow(WINDOW_NAME)


        self.compute_datum_error(left_eye_lists, right_eye_lists, left_eye_predict_1, right_eye_predict_1)
        # self.compute_datum_error(left_eye_lists, right_eye_lists, left_eye_predict, right_eye_predict)

        self.save_results(left_eye_lists, right_eye_lists, left_eye_predict_1, right_eye_predict_1)

    def compute_datum_error(self, left_ground, right_ground, left_predict, right_predict):

        # 根均方误差(RMSE)
        rmse_left = np.sqrt(mean_squared_error(left_ground, left_predict))

        rmse_right = np.sqrt(mean_squared_error(right_ground, right_predict))

        print('rmse left: {}, right: {}'.format(rmse_left, rmse_right))

    def save_results(self, left_ground, right_ground, left_predict, right_predict):
        save_predict_left_ground = pd.DataFrame(left_ground)
        save_predict_right_ground = pd.DataFrame(right_ground)
        save_predict_left_predict = pd.DataFrame(left_predict)
        save_predict_right_predict = pd.DataFrame(right_predict)

        save_predict = pd.concat([save_predict_left_ground, save_predict_right_ground, save_predict_left_predict, save_predict_right_predict], axis=1)
        # print('save :', save_predict)

        # userdata = 'subjectData-2-26/accuracy_result.csv'

        userDataSaveRootPath = 'subjectData-5-17-ablation'#head_fixation  head_motion

        userPath = os.path.join(userDataSaveRootPath, self.subject)
        if not os.path.exists(userPath):
            os.mkdir(userPath)
        userdata = os.path.join(userPath, 'ablation_result.csv')
         # save_predict.to_csv('res_eye_gz.csv', header=None)
        save_predict.to_csv(userdata, header=None)

    def judge_calibration_failed(self, headpose_lists):
        head_base = headpose_lists[0]
        numbers = len(headpose_lists)

        thr = 3.0   #2.0
        for i in range(1,numbers):
            pitch_dis = abs(headpose_lists[i][0]-head_base[0])
            yaw_dis = abs(headpose_lists[i][1] - head_base[1])
            if pitch_dis >= thr or yaw_dis >= thr:
                return True
        return False

    def start_calibration(self):

        SCP = ShowCalibrationPoint('test.jpg', (1920, 1080), 150)
        # SCP.show_point()

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        i = 0

        left_eye_lists = []
        right_eye_lists = []
        point_lists = []
        headpose_lists = []
        while i < 5:
            point = SCP.show_point_i(i, WINDOW_NAME)  # show_point
            left_eye, right_eye, corner, headpose_angles, nose = self.process_biase_one_point_numpy_with_head(self.face_mesh,
                                                                                                        self.head_model,
                                                                                                        self.cap,
                                                                                                        frame_counts=20,
                                                                                                        coordinate_or_biase=True)
            # print('pitch: {}, yaw: {}, roll: {}'.format(headpose_angles[0], headpose_angles[1], headpose_angles[2]))

            if i==0:
                self.RayIntersect.set_center_angle(headpose_angles[0], headpose_angles[1], headpose_angles[2])
                self.head_pose_stable['pitch'], self.head_pose_stable['yaw'] = headpose_angles[0], headpose_angles[1]
                self.set_center_pupil(left_eye, right_eye)
                self.save_calibration_pupil(left_eye, right_eye)
                # 存放眼角信息
                self.set_eye_corner(corner)
                self.center_one_point = point
            else:
                point_lists.append(point)
                left_eye_lists.append(left_eye)
                right_eye_lists.append(right_eye)
            headpose_lists.append(headpose_angles)

            i += 1


        print('calibration finished!...')
        cv2.destroyWindow(WINDOW_NAME)
        Failed_or_not = self.judge_calibration_failed(headpose_lists)

        # 计算比例系数
        return Failed_or_not, self.compute_scales(left_eye_lists, right_eye_lists, point_lists)

    def start_calibration_1(self):
        SCP = ShowCalibrationPoint('test.jpg', (1920, 1080), 150)
        # SCP.show_point()

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        i = 0
        left_eye_lists = []
        right_eye_lists = []
        point_lists = []
        headpose_lists = []
        while i < 6:        # 6
            show_id = i%5
            point = SCP.show_point_i(show_id, WINDOW_NAME)  # show_point
            left_eye, right_eye, corner, headpose_angles, nose = self.process_biase_one_point_numpy_with_head(self.face_mesh,
                                                                                                        self.head_model,
                                                                                                        self.cap,
                                                                                                        frame_counts=20,
                                                                                                        coordinate_or_biase=True)
            # print('pitch: {}, yaw: {}, roll: {}'.format(headpose_angles[0], headpose_angles[1], headpose_angles[2]))
            if i == 0 or i == 5:
                self.RayIntersect.set_center_angle(headpose_angles[0], headpose_angles[1], headpose_angles[2])
                # add 7-23
                self.RayIntersect.set_center_nose(nose)

                self.head_pose_stable['pitch'], self.head_pose_stable['yaw'] = headpose_angles[0], headpose_angles[1]
                # print('head calib pitch: {}, yaw: {}'.format(headpose_angles[0], headpose_angles[1]))
                self.set_center_pupil(left_eye, right_eye)
                self.save_calibration_pupil(left_eye, right_eye)
                # 存放眼角信息
                self.set_eye_corner(corner)
                self.center_one_point = point
            else:
                point_lists.append(point)
                left_eye_lists.append(left_eye)
                right_eye_lists.append(right_eye)
            headpose_lists.append(headpose_angles)

            i += 1
        print('calibration finished!...')
        cv2.destroyWindow(WINDOW_NAME)
        Failed_or_not = self.judge_calibration_failed(headpose_lists)

        # 修改限制范围，加上基准的值
        if self.range_detection:
            # self.head_pose_stable['pitch'], self.head_pose_stable['yaw']

            # print('self.head_pose_stable yaw: ', self.head_pose_stable['yaw'])
            # print('type: ', type(self.head_pose_stable['yaw']))
            self.yaw_range += self.head_pose_stable['yaw']

            self.pitch_range += self.head_pose_stable['pitch']

            # print('yaw range: {}, pitch range: {}'.format(self.yaw_range, self.pitch_range))

        # 计算比例系数
        return Failed_or_not, self.compute_scales(left_eye_lists, right_eye_lists, point_lists)

    # 添加ear测试
    def start_calibration_2(self):
        SCP = ShowCalibrationPoint('test.jpg', (1920, 1080), 150)
        # SCP.show_point()

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        i = 0
        left_eye_lists = []
        right_eye_lists = []
        point_lists = []
        headpose_lists = []
        ear_list = []
        while i < 5:        # 
            show_id = i%5
            point = SCP.show_point_i(show_id, WINDOW_NAME)  # show_point
            # process_biase_one_point_numpy_with_head
            left_eye, right_eye, corner, headpose_angles, nose, ear = self.process_biase_one_point_numpy_with_head_1(self.face_mesh,
                                                                                                        self.head_model,
                                                                                                        self.cap,
                                                                                                        frame_counts=20,
                                                                                                        coordinate_or_biase=True)

            ear_list.append(ear)
            # print('pitch: {}, yaw: {}, roll: {}'.format(headpose_angles[0], headpose_angles[1], headpose_angles[2]))
            if i == 0 or i == 5:
                self.RayIntersect.set_center_angle(headpose_angles[0], headpose_angles[1], headpose_angles[2])
                # add 7-23
                self.RayIntersect.set_center_nose(nose)

                self.head_pose_stable['pitch'], self.head_pose_stable['yaw'] = headpose_angles[0], headpose_angles[1]
                # print('head calib pitch: {}, yaw: {}'.format(headpose_angles[0], headpose_angles[1]))
                self.set_center_pupil(left_eye, right_eye)
                self.save_calibration_pupil(left_eye, right_eye)
                # 存放眼角信息
                self.set_eye_corner(corner)
                self.center_one_point = point
            else:
                point_lists.append(point)
                left_eye_lists.append(left_eye)
                right_eye_lists.append(right_eye)
            headpose_lists.append(headpose_angles)

            i += 1
        print('calibration finished!...')
        cv2.destroyWindow(WINDOW_NAME)
        Failed_or_not = self.judge_calibration_failed(headpose_lists)

        max_ear = np.array(ear_list).max()
        # print('max ear: ', max_ear)
        self.BD.set_ear_thr(max_ear * 1.25)
        # 修改限制范围，加上基准的值

        if self.range_detection:
            # self.head_pose_stable['pitch'], self.head_pose_stable['yaw']

            # print('self.head_pose_stable yaw: ', self.head_pose_stable['yaw'])
            # print('type: ', type(self.head_pose_stable['yaw']))
            self.yaw_range += self.head_pose_stable['yaw']

            self.pitch_range += self.head_pose_stable['pitch']

            # print('yaw range: {}, pitch range: {}'.format(self.yaw_range, self.pitch_range))

        # 计算比例系数
        return Failed_or_not, self.compute_scales(left_eye_lists, right_eye_lists, point_lists)


    def compute_scales(self, left_eye_lists, right_eye_lists, point_lists):
        lsx = (point_lists[-1][0]-point_lists[1][0])/(left_eye_lists[-1][0] - left_eye_lists[1][0])
        lsy = (point_lists[2][1] - point_lists[0][1]) / (left_eye_lists[2][1] - left_eye_lists[0][1])

        rsx = (point_lists[-1][0] - point_lists[1][0]) / (right_eye_lists[-1][0] - right_eye_lists[1][0])
        rsy = (point_lists[2][1] - point_lists[0][1]) / (right_eye_lists[2][1] - right_eye_lists[0][1])

        # print('left eye: ', left_eye_lists)
        # print('right eye: ', right_eye_lists)
        # print('point_lists: ', point_lists)
        return (lsx, lsy), (rsx, rsy)

    def compute_scale_with_two_point(self, center_iris_left, center_iris_right, side_iris_left, side_iris_right, center_point, side_point):

        dx_left_eye = side_iris_left[0] - center_iris_left[0]

        dy_left_eye = side_iris_left[1] - center_iris_left[1]

        dx_right_eye = side_iris_right[0] - center_iris_right[0]

        dy_right_eye = side_iris_right[1] - center_iris_right[1]


        dx_screen = side_point[0] - center_point[0]
        dy_screen = side_point[1] - center_point[1]

        left_scale_x = dx_screen/(dx_left_eye*100)
        left_scale_y = dy_screen / (dy_left_eye * 100)
        right_scale_x = dx_screen / (dx_right_eye * 100)
        right_scale_y = dy_screen / (dy_right_eye * 100)

        print('left scalex:{}, scaley:{}, right scalex:{}, scaley:{}'.format(left_scale_x, left_scale_y, right_scale_x, right_scale_y))
    def compute_scale_x_with_two_point(self, center_iris_left, center_iris_right, side_iris_left, side_iris_right, center_point, side_point):

        dx_left_eye = side_iris_left[0] - center_iris_left[0]

        # dy_left_eye = side_iris_left[1] - center_iris_left[1]

        dx_right_eye = side_iris_right[0] - center_iris_right[0]

        # dy_right_eye = side_iris_right[1] - center_iris_right[1]


        dx_screen = side_point[0] - center_point[0]
        # dy_screen = side_point[1] - center_point[1]

        left_scale_x = dx_screen/(dx_left_eye*100)
        # left_scale_y = dy_screen / (dy_left_eye * 100)
        right_scale_x = dx_screen / (dx_right_eye * 100)
        # right_scale_y = dy_screen / (dy_right_eye * 100)

        print('left scalex:{}, right scalex:{}'.format(left_scale_x, right_scale_x))
    def compute_scale_y_with_two_point(self, center_iris_left, center_iris_right, side_iris_left, side_iris_right, center_point, side_point):

        # dx_left_eye = side_iris_left[0] - center_iris_left[0]

        dy_left_eye = side_iris_left[1] - center_iris_left[1]

        # dx_right_eye = side_iris_right[0] - center_iris_right[0]

        dy_right_eye = side_iris_right[1] - center_iris_right[1]


        # dx_screen = side_point[0] - center_point[0]
        dy_screen = side_point[1] - center_point[1]

        # left_scale_x = dx_screen/(dx_left_eye*100)
        left_scale_y = dy_screen / (dy_left_eye * 100)
        # right_scale_x = dx_screen / (dx_right_eye * 100)
        right_scale_y = dy_screen / (dy_right_eye * 100)

        print('left scaley:{}, right scaley:{}'.format(left_scale_y, right_scale_y))


    def nothing(self, x):
        pass

    def stop_online_calibration(self):
        self.stop_online = True



    def Release(self):
        self.cap.release()

    def stopTracking(self):
        self.stop = True

    def FinalRelease(self):
        '''
        self.stopTracking()
        if self.runback:
            self.stop_eyetracking_back()
        '''

        if self.isRecording:
            self.user_writer.release()
        #     self.screen_writer.release()

        self.Videosave.stop_saving()
        self.Videosave.release()
        # print('user_writer release 2')

    def is_stop(self):
        return self.stop

    def set_left_scale(self, scale_x, scale_y):
        self.left_scale_x = scale_x
        self.left_scale_y = scale_y

    def set_right_scale(self, scale_x, scale_y):
        self.right_scale_x = scale_x
        self.right_scale_y = scale_y

    def get_scale(self):

        return (self.left_scale_x, self.left_scale_y), (self.right_scale_x, self.right_scale_y)

    def set_scale(self, scale_v):
        self.set_left_scale(scale_v, scale_v)
        self.set_right_scale(scale_v, scale_v)

    def compute_map_point(self, left, right):
        l_dx = left[0]-self.left_center_eye[0]
        l_dy = left[1] - self.left_center_eye[1]

        r_dx = right[0] - self.right_center_eye[0]
        r_dy = right[1] - self.right_center_eye[1]
        # print('l_dx: {}, l_dy: {}, r_dx: {}, r_dy: {}'.format(l_dx, l_dy, r_dx, r_dy))
        # l_dx = left[0] / self.left_center_eye[0]
        # l_dy = left[1] / self.left_center_eye[1]
        #
        # r_dx = right[0] / self.right_center_eye[0]
        # r_dy = right[1] / self.right_center_eye[1]

        # print('l_dx: ', l_dx)
        # print('l_dy: ', l_dy)
        # print('r_dx: ', r_dx)
        # print('r_dy: ', r_dy)
        left_dx, left_dy = l_dx*self.left_scale_x*100, l_dy*self.left_scale_y*100

        right_dx, right_dy = r_dx * self.right_scale_x*100, r_dy * self.right_scale_y*100

        left_map_x, left_map_y = self.center_one_point[0]+left_dx, self.center_one_point[1]+left_dy
        right_map_x, right_map_y = self.center_one_point[0]+right_dx, self.center_one_point[1] + right_dy

        # print('left_map_x: {}, left_map_y: {}'.format(left_map_x, left_map_y))
        # print('right_map_x: {}, right_map_y: {}'.format(right_map_x, right_map_y))

        return (left_map_x, left_map_y), (right_map_x, right_map_y)

    def compute_map_point_1(self, left, right):
        l_dx = left[0]-self.left_center_eye[0]
        l_dy = left[1] - self.left_center_eye[1]

        r_dx = right[0] - self.right_center_eye[0]
        r_dy = right[1] - self.right_center_eye[1]
        # print('left_center_eye: {}, right_center_eye: {}'.format(self.left_center_eye, self.right_center_eye))
        # print('l_dx: {}, l_dy: {}, r_dx: {}, r_dy: {}'.format(l_dx, l_dy, r_dx, r_dy))

        left_dx, left_dy = l_dx*self.left_scale_x, l_dy*self.left_scale_y

        right_dx, right_dy = r_dx * self.right_scale_x, r_dy * self.right_scale_y

        # print('center point: ', self.center_one_point)
        left_map_x, left_map_y = self.center_one_point[0]+left_dx, self.center_one_point[1]+left_dy
        right_map_x, right_map_y = self.center_one_point[0]+right_dx, self.center_one_point[1] + right_dy

        return (left_map_x, left_map_y), (right_map_x, right_map_y)

    def compute_map_point_2(self, left, right, half=True):
        l_dx = left[0]-self.left_center_eye[0]
        l_dy = left[1] - self.left_center_eye[1]

        r_dx = right[0] - self.right_center_eye[0]
        r_dy = right[1] - self.right_center_eye[1]
        # print('left_center_eye: {}, right_center_eye: {}'.format(self.left_center_eye, self.right_center_eye))
        # print('l_dx: {}, l_dy: {}, r_dx: {}, r_dy: {}'.format(l_dx, l_dy, r_dx, r_dy))

        scale = 2 if half else 1
        left_dx, left_dy = l_dx*self.left_scale_x/scale, l_dy*self.left_scale_y/scale           # /2

        right_dx, right_dy = r_dx * self.right_scale_x/scale, r_dy * self.right_scale_y/scale

        # print('center point: ', self.center_one_point)
        left_map_x, left_map_y = self.center_one_point[0]+left_dx, self.center_one_point[1]+left_dy
        right_map_x, right_map_y = self.center_one_point[0]+right_dx, self.center_one_point[1] + right_dy

        return (left_map_x, left_map_y), (right_map_x, right_map_y)

    def compute_map_point_3(self, left, right, headTracking_to_eyetracking=False, count=None):
        l_dx = left[0]-self.left_center_eye[0]
        l_dy = left[1] - self.left_center_eye[1]

        r_dx = right[0] - self.right_center_eye[0]
        r_dy = right[1] - self.right_center_eye[1]
        # print('left_center_eye: {}, right_center_eye: {}'.format(self.left_center_eye, self.right_center_eye))
        # print('l_dx: {}, l_dy: {}, r_dx: {}, r_dy: {}'.format(l_dx, l_dy, r_dx, r_dy))

        # scale = 2 if half else 1
        # scale = 1

        if headTracking_to_eyetracking and count:
            scale = 5/(3+count/10)          #之前设置为20    50           scale=2/(count/50+1)         #*0.75
        else:
            scale= 5/4      #1 #5/4                       # scale=1

        # print('scale: ', scale)
        left_dx, left_dy = l_dx*self.left_scale_x/scale, l_dy*self.left_scale_y/scale           # /2

        right_dx, right_dy = r_dx * self.right_scale_x/scale, r_dy * self.right_scale_y/scale

        # print('center point: ', self.center_one_point)
        left_map_x, left_map_y = self.center_one_point[0]+left_dx, self.center_one_point[1]+left_dy
        right_map_x, right_map_y = self.center_one_point[0]+right_dx, self.center_one_point[1] + right_dy

        return (left_map_x, left_map_y), (right_map_x, right_map_y)

    def compute_map_point_biase(self, left, right):

        l_dx = left[0]-self.left_center_eye[0]
        l_dy = left[1]-self.left_center_eye[1]

        r_dx = right[0]-self.right_center_eye[0]
        r_dy = right[1]-self.right_center_eye[1]

        # print('l_dx: {}, l_dy: {}, r_dx: {}, r_dy: {}'.format(l_dx, l_dy, r_dx, r_dy))

        fix_scale = 100
        fiy_scale = 200
        left_dx, left_dy = l_dx * self.left_scale_x * fix_scale, l_dy * self.left_scale_y * fix_scale
        right_dx, right_dy = r_dx * self.right_scale_x * fix_scale, r_dy * self.right_scale_y * fix_scale
        # print('left_dx: {}, left_dy: {}, right_dx: {}, right_dy: {}'.format(left_dx, left_dy, right_dx, right_dy))

        left_map_x, left_map_y = self.center_one_point[0] - left_dx, self.center_one_point[1] - left_dy
        right_map_x, right_map_y = self.center_one_point[0] - right_dx, self.center_one_point[1] - right_dy

        # print('left_map_x: {}, left_map_y: {}'.format(left_map_x, left_map_y))
        # print('right_map_x: {}, right_map_y: {}'.format(right_map_x, right_map_y))
        # return (left_map_x, left_map_y), (left_map_x, left_map_y)
        # return (right_map_x, right_map_y), (right_map_x, right_map_y)
        return (left_map_x, left_map_y), (right_map_x, right_map_y)



    def compute_map_point_ratio(self, left, right):

        l_dx = (left[0]-self.left_center_eye[0])*40
        l_dy = (left[1]-self.left_center_eye[1])*12

        r_dx = (right[0]-self.right_center_eye[0])*40
        r_dy = (right[1]-self.right_center_eye[1])*12

        print('l_dx: {}, l_dy: {}'.format(l_dx, l_dy))

        fix_scale = 100
        left_dx, left_dy = l_dx * self.left_scale_x * fix_scale, l_dy * self.left_scale_y * fix_scale
        right_dx, right_dy = r_dx * self.right_scale_x * fix_scale, r_dy * self.right_scale_y * fix_scale
        print('left_dx: {}, left_dy: {}'.format(left_dx, left_dy))

        left_map_x, left_map_y = self.center_one_point[0] + left_dx, self.center_one_point[1] + left_dy
        right_map_x, right_map_y = self.center_one_point[0] + right_dx, self.center_one_point[1] + right_dy

        print('left_map_x: {}, left_map_y: {}'.format(left_map_x, left_map_y))
        print('right_map_x: {}, right_map_y: {}'.format(right_map_x, right_map_y))
        # return (left_map_x, left_map_y), (left_map_x, left_map_y)
        return (left_map_x, left_map_y), (right_map_x, right_map_y)
        # self.left_center_eye

    def runEyeTrackingWithSmooth_public_one_point(self, VideoCap, face_mesh, win_sz, timer=None):
        left_b = []
        right_b = []
        testimg = cv2.imread('data/test.jpg')
        testimg = cv2.resize(testimg, (1920, 1080))

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # vid_writer = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (int(960), int(540)))
        timecount = 0
        if timer:
            timer.set_start_time(cv2.getTickCount())

        # print('run error 0!')
        while True:
            # ret, frame = VideoCap.read()
            frame = VideoCap.get_frame()
            if frame is False:
                return
            frame = cv2.flip(frame, 1)
            # debug_image, left_eyes, left_biases, right_eyes, right_biases = face_mesh.process_img(frame)
            left_eyes, right_eyes, left_to_nose, right_to_nose = face_mesh.process_img(frame)


            if left_eyes is not False:
            # if (len(left_biases) > 0):
                # left_b.append(left_biases[0])
                # right_b.append(right_biases[0])
                left_b.append(left_eyes[0])
                right_b.append(right_eyes[0])

            # print('run error 1!')
            if len(left_b) >= Calib_times_per_point:
                # calc_mean(left_b, right_b)
                self.startMap = True
                # biase = (left_biases[0], right_biases[0])  # 去第一个人脸的结果

                # print('left b 1  :', left_b)
                left_x, left_y, right_x, right_y = self.calc_mean(left_b, right_b)



                map_left, map_right = self.compute_map_point((left_x, left_y), (right_x, right_y))

                map_center = int((map_left[0]+map_right[0])/2), int((map_left[1]+map_right[1])/2)

                # if recalib and calibModel:
                #     mapx, mapy = calibModel.Predict(mapx, mapy)

                del (left_b[0])
                del (right_b[0])

                # add at 10/1
                mapx, mapy = map_center[0], map_center[1]

                mapx = 0 if mapx < 0 else mapx
                mapx = win_sz[0] if mapx > win_sz[0] else mapx

                mapy = 0 if mapy < 0 else mapy
                mapy = win_sz[1] if mapy > win_sz[1] else mapy

                self.set_gaze_point(mapx, mapy)
                # print('mapx: ', mapx)
                # print('mapy: ', mapy)


                gaze_movement = draw_gaze_point((int(mapx), int(mapy)), win_sz, radius=50, img=testimg)
                cv2.imshow(WINDOW_NAME, gaze_movement)

                # modified at 9/30
                # add timer
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    sys.exit()

                if timer is not None:
                    if timer.count_time(10):
                        cv2.destroyAllWindows()
                        break

                # if timer is not None:
                #     if timer.count_time(30):
                #         cv2.destroyAllWindows()
                #         break
                #         # sys.exit()
            if self.is_stop():
                break
        # VideoCap.release()
        self.stopTracking()

        self.Videosave.stop_saving()
        self.Videosave.release()

    def control_mouse_(self):
        MC = Mouse_Control()
        listener = pynput.mouse.Listener(on_click=MC.on_click)
        listener.start()

        while True:
            # if self.stop_control_mouse:
            #     break
            gazePoint = Data_saving.get_mean_gaze()  # get_mean_gaze()

            if gazePoint:
                pg.moveTo(gazePoint[0], gazePoint[1])

            if MC.get_double_click():
                self.stop = True
                listener.stop()
                break
        self.FinalRelease()

    def do_nothing_when_experiment(self):

        MC = Mouse_Control()
        listener = pynput.mouse.Listener(on_click=MC.on_click)
        listener.start()

        # listener.join()
        # '''
        start_t = time.time()
        while True:
            if MC.get_double_click():
                self.stop = True
                listener.stop()
                break
            # if self.is_stop():
            #     break

        end_t = time.time()
        print('release here!')

        self.FinalRelease()
        task_time = end_t - start_t
        # return
        return task_time

    def show_gaze_results_real_time(self, win_sz, timer=None):
        # left_b = []
        # right_b = []
        testimg = cv2.imread('data/test.jpg')
        testimg = cv2.resize(testimg, (1920, 1080))

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # vid_writer = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (int(960), int(540)))
        timecount = 0
        if timer:
            timer.set_start_time(cv2.getTickCount())

        # print('run error 0!')
        count = 0
        index = 0
        while True:
            
            tracking_status = self.eye_tracking_status
            gaze_movement = testimg
            if tracking_status == 'stable':
                # print('change into stable...!')
                gazePoint = Data_saving.get_mean_gaze()#get_mean_gaze()
                # print('gaze point: ', gazePoint)
                if gazePoint is not False:
                    # print
                    mapx, mapy = gazePoint
                    gaze_movement = draw_gaze_point((int(mapx), int(mapy)), win_sz, radius=50, img=testimg)#testimg
            else:
                head_gazepoint = Data_saving.get_mean_headgaze()
                if head_gazepoint is not False:
                    map_headx, map_heady = head_gazepoint
                    gaze_movement = draw_headgaze_point((int(map_headx), int(map_heady)),win_sz, radius=50, count=index, img=testimg)#testimg
                    count += 1
                    if count >= 20:
                        count = 0
                        index = 1 - index

                    # head_target
            if self.range_detection:
                if self.cross_limitation:

                    point_location = (960,540)
                    if self.cross_status == 'UP_TO_MUCH':
                        point_location = (960,200)
                    elif self.cross_status == 'DOWN_TO_MUCH':
                        point_location = (960, 880)
                    elif self.cross_status == 'LEFT_TO_MUCH':
                        point_location = (200, 540)
                    elif self.cross_status == 'RIGHT_TO_MUCH':
                        point_location = (1620, 540)

                    cv2.putText(gaze_movement, self.cross_status, point_location, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            #获取眨眼状态
            # blink_status = self.BD.get_status()
            #
            # cv2.putText(gaze_movement, blink_status, (200, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            '''
            if gazePoint is not False:
                mapx, mapy = gazePoint
                self.set_gaze_point(mapx, mapy)
                if head_gazepoint is not False:
                    map_headx, map_heady = head_gazepoint
                    gaze_movement = draw_gaze_point_and_head_point((int(map_headx), int(map_heady)), (int(mapx), int(mapy)),
                                                      win_sz, radius=50, img=testimg)
                else:
                    gaze_movement = draw_gaze_point((int(mapx), int(mapy)), win_sz, radius=50, img=testimg)  # testimg
            else:
                gaze_movement = testimg.copy()
            '''
            cv2.imshow(WINDOW_NAME, gaze_movement)

            # modified at 9/30
            # add timer
            if cv2.waitKey(20) & 0xFF == ord(' '):      # q
                print('comer here!')
                cv2.destroyAllWindows()
                break
                # sys.exit()

            if timer is not None:
                if timer.count_time(10):
                    cv2.destroyAllWindows()
                    break

                # if timer is not None:
                #     if timer.count_time(30):
                #         cv2.destroyAllWindows()
                #         break
                #         # sys.exit()
            if self.is_stop():
                break
        # VideoCap.release()
        # self.stopTracking()
        print('release here!')
        self.FinalRelease()
        # self.Videosave.stop_saving()
        # self.Videosave.release()

  
    def show_website(self, image_list, win_sz=(1920,1080)):

        if len(image_list) < 2:
            return False

        background = np.zeros((win_sz[1], win_sz[0], 3), np.uint8)
        background.fill(220)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        website_images = []
        for i in range(len(image_list)):
            website_images.append(cv2.resize(image_list[i], win_sz))

        for i in range(2):
            start_time = time.time()
            website = website_images[i]
            while True:
                
                tracking_status = self.eye_tracking_status

                # gaze_movement = website.copy()
                # current =
                # if current - start_time >= 20.0:
                #     break
                elaspe_time = time.time() - start_time
                if elaspe_time <= 20.0:
                    gaze_movement = website.copy()
                else:
                    if i == 0:
                        if elaspe_time <= 23.0:         #呈现三秒钟灰屏
                            gaze_movement = background.copy()
                        else:
                            break
                    else:
                        break
                # if i == 0:
                #     if elaspe_time <= 20.0:
                #         gaze_movement = website.copy()
                #     elif elaspe_time <= 25.0:
                #         gaze_movement = background.copy()
                #     else:
                #         break
                # else:
                #     if elaspe_time <= 20.0:
                #         gaze_movement = website.copy()
                #     else:
                #         break

                if tracking_status == 'stable':
                    gazePoint = Data_saving.get_mean_gaze()  # get_mean_gaze()
                else:
                    gazePoint = Data_saving.get_mean_headgaze()

                if gazePoint:           #draw_gaze_point
                    gaze_movement = draw_gaze_point_1((int(gazePoint[0]), int(gazePoint[1])), win_sz, radius=30, img=gaze_movement)#testimg

                cv2.imshow(WINDOW_NAME, gaze_movement)

                if cv2.waitKey(20) & 0xFF == ord('q'):
                    print('comer here!')
                    # cv2.destroyAllWindows()
                    break


        cv2.destroyAllWindows()
        print('release here!')
        self.FinalRelease()
    def show_car(self, image, win_sz=(1920,1080)):
        # if len(image_list) < 2:
        #     return False

        # background = np.zeros((win_sz[1], win_sz[0], 3), np.uint8)
        # background.fill(220)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        start_time = time.time()

        while True:
            # 
            tracking_status = self.eye_tracking_status

            elaspe_time = time.time() - start_time
            if elaspe_time <= 20.0:
                gaze_movement = image.copy()
            else:
                break


            if tracking_status == 'stable':
                gazePoint = Data_saving.get_mean_gaze()  # get_mean_gaze()
            else:
                gazePoint = Data_saving.get_mean_headgaze()

            if gazePoint:           #draw_gaze_point
                gaze_movement = draw_gaze_point_1((int(gazePoint[0]), int(gazePoint[1])), win_sz, radius=30, img=gaze_movement)#testimg

            cv2.imshow(WINDOW_NAME, gaze_movement)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                print('comer here!')
                # cv2.destroyAllWindows()
                break
        cv2.destroyAllWindows()
        print('release here!')
        self.FinalRelease()

    def show_car_1(self, image, win_sz=(1920,1080)):

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        start_time = time.time()
        while True:
            # 
            tracking_status = self.eye_tracking_status
            elaspe_time = time.time() - start_time
            if elaspe_time <= 20.0:         #
                gaze_movement = image.copy()
            else:
                break
            if tracking_status == 'stable':
                gazePoint = Data_saving.get_mean_gaze()  # get_mean_gaze()
            else:
                gazePoint = Data_saving.get_mean_headgaze()

            if gazePoint:           #draw_gaze_point
                gaze_movement = draw_gaze_point_1((int(gazePoint[0]), int(gazePoint[1])), win_sz, radius=30, img=gaze_movement)#testimg

            cv2.imshow(WINDOW_NAME, gaze_movement)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                print('comer here!')
                # cv2.destroyAllWindows()
                break
        cv2.destroyAllWindows()
        print('release here!')
        self.FinalRelease()

    def show_phone(self, image, task_mode='ui1', win_sz=(1920,1080)):

        background = np.zeros((win_sz[1], win_sz[0], 3), np.uint8)
        background.fill(220)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        start_time = time.time()

        show_app_name = True

        

        app_name_dict = {'ui1': 'data/phone_app_name/ui1_name.bmp', 'ui2': 'data/phone_app_name/ui2_name.bmp',
                         'ui3': 'data/phone_app_name/ui3_name.bmp',
                         'ui4': 'data/phone_app_name/ui4_name.bmp',
                         'ui5': 'data/phone_app_name/ui5_name.bmp', 'ui6': 'data/phone_app_name/ui6_name.bmp'}

        app_name_pic = cv2.imread(app_name_dict[task_mode])
        h, w, _ = app_name_pic.shape

        start_point_x = int(win_sz[0]/2 - w/2)
        start_point_y = int(win_sz[1] / 2 - h / 2)

        # background[start_point_y:start_point_y+h, start_point_x:start_point_x+w] = app_name_pic.copy()

        h1, w1, _ = image.shape

        new_size = (int(w1/3.2), int(h1/3.2))
        sclae_image = cv2.resize(image, new_size)

        print('size of sclae image: ', new_size)
        start_point_x1 = int(win_sz[0] / 2 - new_size[0] / 2)
        start_point_y1 = int(win_sz[1] / 2 - new_size[1] / 2)

        while True:
            # 
            tracking_status = self.eye_tracking_status
            # elaspe_time = time.time() - start_time
            gaze_movement = background.copy()
            if show_app_name:
                gaze_movement[start_point_y:start_point_y + h, start_point_x:start_point_x + w] = app_name_pic.copy()
            else:
                gaze_movement[start_point_y1:start_point_y1 + new_size[1], start_point_x1:start_point_x1 + new_size[0]] = sclae_image.copy()

            # if elaspe_time <= 10.0:  # 设置为10秒钟
            #     gaze_movement = image.copy()
            # else:
            #     break
            if tracking_status == 'stable':
                gazePoint = Data_saving.get_mean_gaze()  # get_mean_gaze()
            else:
                gazePoint = Data_saving.get_mean_headgaze()

            if gazePoint:  # draw_gaze_point
                gaze_movement = draw_gaze_point_1((int(gazePoint[0]), int(gazePoint[1])), win_sz, radius=30,
                                                  img=gaze_movement)  # testimg

            cv2.imshow(WINDOW_NAME, gaze_movement)
            if cv2.waitKey(20) & 0xFF == ord(' '):
                # print('comer here!')
                # cv2.destroyAllWindows()
                if show_app_name:
                    #第一次点击空格键后，进入ui界面
                    show_app_name = False
                    start_time = time.time()        #进入ui界面，计时开始
                else:
                    print('结束该实验')
                    # 统计耗时
                    elaspe_time = time.time() - start_time
                    break
        cv2.destroyAllWindows()
        print('release here!')
        self.FinalRelease()
        return elaspe_time

    def show_random_point(self, win_sz):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        background = np.zeros((win_sz[1], win_sz[0], 3), np.uint8)
        background.fill(220)

        count = 0
        index = 0
        random_count = 0
        x, y = 0, 0

        start_time = time.time()
        while True:
            # 只有稳定跟踪的时候显示眼球注视点，其余状态显示头部指向点
            tracking_status = self.eye_tracking_status

            frame = background.copy()

            current = time.time()
            if current - start_time >= 2.0:
            # if random_count%10 == 0:
                x, y = random.randint(0, 1919), random.randint(0, 1079)
                start_time = current

            cv2.circle(frame, (x, y), 20, (255, 255, 0), thickness=-1)

            # ret, frame = cap.read()
            gaze_movement = cv2.resize(frame, (1920, 1080))

            if tracking_status == 'stable':
                # print('change into stable...!')
                gazePoint = Data_saving.get_mean_gaze()#get_mean_gaze()
                # print('gaze point: ', gazePoint)
                if gazePoint is not False:
                    # print
                    mapx, mapy = gazePoint
                    gaze_movement = draw_gaze_point((int(mapx), int(mapy)), win_sz, radius=50, img=gaze_movement)#testimg
            else:
                head_gazepoint = Data_saving.get_mean_headgaze()
                if head_gazepoint is not False:
                    map_headx, map_heady = head_gazepoint
                    gaze_movement = draw_headgaze_point((int(map_headx), int(map_heady)),win_sz, radius=50, count=index, img=gaze_movement)#testimg
                    count += 1
                    if count >= 20:
                        count = 0
                        index = 1 - index

            cv2.imshow(WINDOW_NAME, gaze_movement)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                print('comer here!')
                cv2.destroyAllWindows()
                break

            if self.is_stop():
                break
        print('release here!')
        self.FinalRelease()
    def show_gaze_results_real_time_on_video(self, win_sz, video, timer=None):
        # left_b = []
        # right_b = []
        testimg = cv2.imread('data/test.jpg')
        testimg = cv2.resize(testimg, (1920, 1080))

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # vid_writer = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (int(960), int(540)))
        timecount = 0

        cap = cv2.VideoCapture(video)

        if timer:
            timer.set_start_time(cv2.getTickCount())

        # print('run error 0!')
        count = 0
        index = 0
        while True:
           
            tracking_status = self.eye_tracking_status

            ret, frame = cap.read()
            if ret:
                gaze_movement = cv2.resize(frame, (1920, 1080))
                if tracking_status == 'stable':
                    # print('change into stable...!')
                    gazePoint = Data_saving.get_mean_gaze()#get_mean_gaze()
                    # print('gaze point: ', gazePoint)
                    if gazePoint is not False:
                        # print
                        mapx, mapy = gazePoint
                        gaze_movement = draw_gaze_point((int(mapx), int(mapy)), win_sz, radius=50, img=gaze_movement)#testimg
                else:
                    head_gazepoint = Data_saving.get_mean_headgaze()
                    if head_gazepoint is not False:
                        map_headx, map_heady = head_gazepoint
                        gaze_movement = draw_headgaze_point((int(map_headx), int(map_heady)),win_sz, radius=50, count=index, img=testimg)#testimg
                        count += 1
                        if count >= 20:
                            count = 0
                            index = 1 - index
                cv2.imshow(WINDOW_NAME, gaze_movement)

                if cv2.waitKey(20) & 0xFF == ord('q'):
                    print('comer here!')
                    cv2.destroyAllWindows()
                    break
            else:
                break
            if self.is_stop():
                break
        print('release here!')
        self.FinalRelease()

    def calc_mean(self, biase_left, biase_right):
        lx, ly, rx, ry = 0.0, 0.0, 0.0, 0.0
        for i in range(len(biase_left)):
            lx += biase_left[i][0]
            ly += biase_left[i][1]
            rx += biase_right[i][0]
            ry += biase_right[i][1]

            # lx += biase_left[i][0]
            # ly += biase_left[i][1]
            # rx += biase_right[i][0]
            # ry += biase_right[i][1]

        return lx / len(biase_left), ly / len(biase_left), rx / len(biase_right), ry / len(biase_right)

    def set_center_one_point(self, headpoint):
        self.center_one_point = headpoint

    def runEyeTrackingWithSmooth_back(self, VideoCap, face_mesh, headposeModel, win_sz, coordinate_or_biase=True):
        left_b = []
        right_b = []
        pitchs = []
        yaws = []
        rolls = []
        while True:
            # ret, frame = VideoCap.read()
            frame = VideoCap.get_frame()
            if frame is False:
                return
            frame = cv2.flip(frame, 1)

            # left_eyes, right_eyes, left_to_nose, right_to_nose = face_mesh.process_img(frame)

            # left_eyes, right_eyes = face_mesh.process_img_ratio(frame)
            left_eyes, right_eyes = face_mesh.process_img_biase_corner(frame)
            status, euler_angle = headposeModel.process_img(frame)
            if status:
                pitch, yaw, roll = euler_angle
                # pitch,yaw,roll = euler_angle[0, 0], euler_angle[1, 0], euler_angle[2, 0]
                # for i in range(len(euler_angle)):
                pitchs.append(pitch)
                yaws.append(yaw)
                rolls.append(roll)

            if left_eyes is not False:
            # if (len(left_biases) > 0):
                if coordinate_or_biase:
                    left_b.append(left_eyes[0])
                    right_b.append(right_eyes[0])
                else:
                    left_b.append(left_eyes[1])
                    right_b.append(right_eyes[1])


            # print('run error 1!')
            if len(left_b) >= Calib_times_per_point:
                # calc_mean(left_b, right_b)
                self.startMap = True

                mean_pitch = sum(pitchs) / len(pitchs)
                mean_yaw = sum(yaws) / len(yaws)
                # mean_roll = sum(rolls) / len(rolls)

                map_headx, map_heady = self.RayIntersect.computer_intersection_1(mean_pitch, mean_yaw)#computer_intersection
                map_headx = 0 if map_headx < 0 else map_headx
                map_headx = win_sz[0] if map_headx > win_sz[0] else map_headx

                map_heady = 0 if map_heady < 0 else map_heady
                map_heady = win_sz[1] if map_heady > win_sz[1] else map_heady

                self.set_center_one_point((int(map_headx), int(map_heady)))

                # biase = (left_biases[0], right_biases[0])  # 去第一个人脸的结果
                # print('left b 1  :', left_b)
                left_x, left_y, right_x, right_y = self.calc_mean(left_b, right_b)
                # print('left_x: {}, left_y: {}, right_x: {}, right_y: {}'.format(left_x, left_y, right_x, right_y))

                if coordinate_or_biase:
                    map_left, map_right = self.compute_map_point((left_x, left_y), (right_x, right_y))
                else:

                    map_left, map_right = self.compute_map_point_biase((left_x, left_y), (right_x, right_y))
                    # map_left, map_right = self.compute_map_point_ratio((left_x, left_y), (right_x, right_y))

                map_center = int((map_left[0]+map_right[0])/2), int((map_left[1]+map_right[1])/2)

                del (left_b[0])
                del (right_b[0])
                del (pitchs[0])
                del (yaws[0])
                # add at 10/1
                mapx, mapy = map_center[0], map_center[1]

                mapx = 0 if mapx < 0 else mapx
                mapx = win_sz[0] if mapx > win_sz[0] else mapx

                mapy = 0 if mapy < 0 else mapy
                mapy = win_sz[1] if mapy > win_sz[1] else mapy

                # add 10/30
                Data_saving.save_gaze(mapx, mapy)
                Data_saving.save_head_point(map_headx, map_heady)

                self.set_gaze_point(mapx, mapy)

            if self.is_stop():
                break

    # def is_head_moving(self):
    #     Data_saving.get_mean_headgaze()

    def runEyeTrackingWithSmooth_back2(self, VideoCap, face_mesh, headposeModel, win_sz, coordinate_or_biase=True):
        # left_b = []
        # right_b = []
        # pitchs = []
        # yaws = []
        # rolls = []

        datalen = Calib_times_per_point
        left_eye_list = collections.deque(maxlen=datalen)
        right_eye_list = collections.deque(maxlen=datalen)
        head_pose_list = collections.deque(maxlen=datalen)
        # yaw_list = collections.deque(maxlen=Calib_times_per_point)
        # roll_list = collections.deque(maxlen=Calib_times_per_point)

        while True:
            # ret, frame = VideoCap.read()
            frame = VideoCap.get_frame()
            if frame is False:
                return
            frame = cv2.flip(frame, 1)
            left_eyes, right_eyes = face_mesh.process_img_biase_corner(frame)
            status, euler_angle = headposeModel.process_img(frame)


            if status:
                pitch, yaw, roll = euler_angle
                # pitch,yaw,roll = euler_angle[0, 0], euler_angle[1, 0], euler_angle[2, 0]
                # for i in range(len(euler_angle)):
                head_pose_list.append((pitch, yaw, roll))
                # pitch_list.append(pitch)
                # yaw_list.append(yaw)
                # roll_list.append(roll)

            if left_eyes is not False:
            # if (len(left_biases) > 0):
                if coordinate_or_biase:
                    left_eye_list.append((left_eyes[0]))
                    # left_eye_list.append(left_eyes[0])
                    right_eye_list.append(right_eyes[0])
                else:
                    left_eye_list.append(left_eyes[1])
                    right_eye_list.append(right_eyes[1])
            # print('run error 1!')
            if len(left_eye_list) >= Calib_times_per_point:
                # calc_mean(left_b, right_b)
                self.startMap = True

                headpose = np.array(head_pose_list)
                headpose = np.mean(headpose, axis=0)

                mean_pitch, mean_yaw = headpose[0], headpose[1]
                # mean_pitch = sum(pitchs) / len(pitchs)
                # mean_yaw = sum(yaws) / len(yaws)
                # mean_roll = sum(rolls) / len(rolls)

                map_headx, map_heady = self.RayIntersect.computer_intersection(mean_pitch, mean_yaw)
                map_headx = 0 if map_headx < 0 else map_headx
                map_headx = win_sz[0] if map_headx > win_sz[0] else map_headx

                map_heady = 0 if map_heady < 0 else map_heady
                map_heady = win_sz[1] if map_heady > win_sz[1] else map_heady

                self.set_center_one_point((int(map_headx), int(map_heady)))

                # biase = (left_biases[0], right_biases[0])  # 去第一个人脸的结果
                # print('left b 1  :', left_b)
                # left_x, left_y, right_x, right_y = self.calc_mean(left_b, right_b)

                left_eye = np.array(left_eye_list)
                left_eye = np.mean(left_eye, axis=0)
                right_eye = np.array(right_eye_list)
                right_eye = np.mean(right_eye, axis=0)
                left_x, left_y, right_x, right_y = left_eye[0], left_eye[1], right_eye[0], right_eye[1]


                # print('left_x: {}, left_y: {}, right_x: {}, right_y: {}'.format(left_x, left_y, right_x, right_y))

                if coordinate_or_biase:
                    map_left, map_right = self.compute_map_point((left_x, left_y), (right_x, right_y))
                else:

                    map_left, map_right = self.compute_map_point_biase((left_x, left_y), (right_x, right_y))
                    # map_left, map_right = self.compute_map_point_ratio((left_x, left_y), (right_x, right_y))

                map_center = int((map_left[0]+map_right[0])/2), int((map_left[1]+map_right[1])/2)

                mapx, mapy = map_center[0], map_center[1]

                mapx = 0 if mapx < 0 else mapx
                mapx = win_sz[0] if mapx > win_sz[0] else mapx

                mapy = 0 if mapy < 0 else mapy
                mapy = win_sz[1] if mapy > win_sz[1] else mapy

                # add 10/30
                Data_saving.save_gaze(mapx, mapy)
                Data_saving.save_head_point(map_headx, map_heady)
                self.set_gaze_point(mapx, mapy)
            if self.is_stop():
                break

    def revise_corner(self, left_corner, right_corner):
        self.left_eye_corner = left_corner
        self.right_eye_corner = right_corner

    def compute_eye_corner_offsets(self, left_left, left_right, right_left, right_right):

        # print('left eye: {}, right: {}'.format(self.left_eye_corner, self.right_eye_corner))
        left_offset_x = (left_left[0] - self.left_eye_corner[0][0]+left_right[0]-self.left_eye_corner[1][0])/2
        left_offset_y = (left_left[1] - self.left_eye_corner[0][1] + left_right[1] - self.left_eye_corner[1][1]) / 2

        right_offset_x = (right_left[0] - self.right_eye_corner[0][0] + right_right[0] - self.right_eye_corner[1][0]) / 2
        right_offset_y = (right_left[1] - self.right_eye_corner[0][1] + right_right[1] - self.right_eye_corner[1][1]) / 2


        return (left_offset_x, left_offset_y), (right_offset_x, right_offset_y)

    def compute_new_pupil(self, left_offset, right_offset):

        # new_left = (self.left_center_eye[0]+left_offset[0], self.left_center_eye[1]+left_offset[1])
        # new_right = (self.right_center_eye[0] + right_offset[0], self.right_center_eye[1] + right_offset[1])

        # left_offset

        new_left = (self.left_center_eye_original[0] + left_offset[0], self.left_center_eye_original[1] + left_offset[1])
        new_right = (self.right_center_eye_original[0] + right_offset[0], self.right_center_eye_original[1] + right_offset[1])

        return new_left, new_right


    def detect_facial_expression(self, img):
        return self.FER.do_fer_for_single_Img(img)


    def is_head_moving(self, head_pose_list, moving_thr, headTracking_to_eyetracking):

        # if len(head_pose_list) == 0:
        #     return False
        pitch_thr, yaw_thr = moving_thr

        pitch_now, yaw_now, _ = head_pose_list[-1]
        headpose = np.array(head_pose_list)
        headpose = np.mean(headpose, axis=0)

        mean_pitch, mean_yaw = headpose[0], headpose[1]
        
        if abs(mean_pitch - self.head_pose_stable['pitch']) >= pitch_thr or \
                abs(mean_yaw - self.head_pose_stable['yaw']) >= yaw_thr:
            if abs(pitch_now - mean_pitch) >= 2 or abs(yaw_now - mean_yaw) >= 2:
                # print('moving....')
                self.eye_tracking_status = 'moving'
                if headTracking_to_eyetracking:
                    headTracking_to_eyetracking = False
            else:
                # print('moving to 1....')
                if self.eye_tracking_status == 'moving':
                    # print('change to static...')

                    self.eye_tracking_status = 'stable' 
                    # 清空缓存
                    self.left_eye_list.clear()
                    self.right_eye_list.clear()

                    Data_saving.clear_gaze() 
                    self.static_to_stable_count = 0
        return headTracking_to_eyetracking, (mean_pitch, mean_yaw)

    def process_static(self, left_eye_corner_list, right_eye_corner_list):
        left_corner = np.array(left_eye_corner_list)
        left_corner = np.mean(left_corner, axis=0)

        right_corner = np.array(right_eye_corner_list)
        right_corner = np.mean(right_corner, axis=0)

        # 左眼左眼角和右眼角
        left_left = left_corner[0]
        left_right = left_corner[1]

        # 右眼左眼角和右眼角
        right_left = right_corner[0]
        right_right = right_corner[1]

        left_offset, right_offset = self.compute_eye_corner_offsets(left_left, left_right, right_left, right_right)
        # print('left offset: {}, right offset: {}'.format(left_offset, right_offset))

        pupil_datum = self.compute_new_pupil(left_offset, right_offset)

        return pupil_datum


    def save_user_and_screen(self, user_img):
        # print('user img shape: ', user_img.shape)
        self.user_writer.write(user_img)
        # print('save ...')

        # screen = pg.screenshot()
        # open_cv_screen = np.array(screen)
        # open_cv_screen = cv2.cvtColor(open_cv_screen, cv2.COLOR_RGB2BGR)
        # open_cv_screen = cv2.resize(open_cv_screen, (640,360))#(960,540)
        # self.screen_writer.write(open_cv_screen)


    def save_gaze_to_excle(self, df):
        # df.to_csv(self.excelName, mode='a')
        df.to_csv(self.excelName, mode='a', header=False, index=False)
        # pass
        # screen_img =
    #

    # def process_stable(self, left_eye_corner_list, right_eye_corner_list):
    #     pass
    def stop_saving_data(self):
        self.start_saving = False

    def start_saving_data(self):
        self.start_saving = True

    def screen_video_saving_thread(self):
        while True:
            if self.is_stop():
                break
            screen = pg.screenshot()
            open_cv_screen = np.array(screen)
            open_cv_screen = cv2.cvtColor(open_cv_screen, cv2.COLOR_RGB2BGR)
            open_cv_screen = cv2.resize(open_cv_screen, (640, 360))  # (960,540)
            self.screen_writer.write(open_cv_screen)

    #鼠标检测线程，每点击一次，保存一次截图
    def mouseResponseThread(self):
        pass



    def runEyeTrackingWithSmooth_back_head_move_2(self, VideoCap, face_mesh, headposeModel, win_sz, detect_fe=False):

        datalen = Calib_times_per_point
        self.left_eye_list = collections.deque(maxlen=datalen)
        self.right_eye_list = collections.deque(maxlen=datalen)

        # add 5-9
        left_eye_corner_list = collections.deque(maxlen=datalen)
        right_eye_corner_list = collections.deque(maxlen=datalen)

        head_pose_list = collections.deque(maxlen=datalen)

        nose_list = collections.deque(maxlen=datalen)

        pupil_datum_left_list = collections.deque(maxlen=datalen)
        pupil_datum_right_list = collections.deque(maxlen=datalen)

        #修改与7-23
        pitch_thr, yaw_thr = 2.0, 2.0       #2.5, 2.5


        self.eye_tracking_status = 'stable'

        self.static_to_stable_count = 0


        self.cross_limitation = False
        self.cross_status = 'NORMAL'
        # self.pitch_cross_status = 'NORMAL'
        # self.yaw_cross_status = 'NORMAL'

        # self.static_to_stable_left_eye_list = collections.deque(maxlen=30)
        # self.static_to_stable_right_eye_list = collections.deque(maxlen=30)

        headTracking_to_eyetracking = False
        half_count = 0

        mean_pitch, mean_yaw = 0.0, 0.0

        # 开始存储
        self.Videosave.start_saving()

        #线程开启
        # self.video_thread = threading.Thread(target=self.screen_video_saving_thread)
        # self.video_thread.start()
        saving_count = 0
        save_gaze = False
        first_save = True
        init_time = 0.0


        # cv2.setMouseCallback('drawing', self.mouseResponse)

        speed_s = []

        while True:
            t_start = time.time()
            frame = VideoCap.get_frame()
            if frame is False:
                return
            frame = cv2.flip(frame, 1)

            if self.isRecording:# and self.start_saving:      #添加start_saving，由开始实验传进来
                # print('开始储存')
                saving_count+=1

                #这里保存数据
                # t0 = time.time()
                if saving_count == 2:           #5
                    # t0 = time.time()
                    self.save_user_and_screen(frame)
                    save_gaze = True
                    #保存眼动数据
                    saving_count = 0
                    # t1 = time.time() - t0
                    # print('save data cost time: ', t1)


            # 这里添加面部表情处理
            # print('error 0')
            if detect_fe:
                # t0 = time.time()

                emotion_box = self.detect_facial_expression(frame)

                if emotion_box:
                    print('expression: ', emotion_box[0])
                    Data_saving.save_emotion_box(emotion_box)
            # print('error 1')

            # t2 = time.time()
            left_pst, right_pst, pupil_center = face_mesh.process_img_1(frame)

            status, euler_angle = headposeModel.process_img(frame)
            # t3 = time.time() - t2
            # print('cost time 2: ', t3)
            if status:
                pitch, yaw, roll = euler_angle
                head_pose_list.append((pitch, yaw, roll))
                headTracking_to_eyetracking, mean_angle = self.is_head_moving(head_pose_list, (pitch_thr, yaw_thr), headTracking_to_eyetracking)


                mean_pitch, mean_yaw = mean_angle
                # print('mean pitch: {}, mean yaw: {}'.format(mean_pitch, mean_yaw))
                #判断是否越界
                if self.range_detection:
                    cross_limitation = True
                    cross_status = 'NORMAL'
                    if mean_pitch > self.pitch_range[0] and mean_pitch <self.pitch_range[1]:
                        if mean_yaw > self.yaw_range[0] and mean_yaw < self.yaw_range[1]:
                            # self.cross_limitation = False
                            cross_limitation = False

                    if cross_limitation:
                        if mean_pitch < self.pitch_range[0]:
                            cross_status = 'DOWN_TO_MUCH'
                        elif mean_pitch > self.pitch_range[1]:
                            cross_status = 'UP_TO_MUCH'
                        elif mean_yaw < self.yaw_range[0]:
                            cross_status = 'LEFT_TO_MUCH'
                        elif mean_yaw > self.yaw_range[1]:
                            cross_status = 'RIGHT_TO_MUCH'
                        # else:
                        #     cross_status = 'NORMAL'

                        ''' 
                        if mean_pitch < self.pitch_range[0]:
                            pitch_cross_status = 'DOWN_TO_MUCH'
                        elif mean_pitch > self.pitch_range[1]:
                            pitch_cross_status = 'UP_TO_MUCH'
                        else:
                            pitch_cross_status = 'NORMAL'

                        if mean_yaw < self.yaw_range[0]:
                            yaw_cross_status = 'LEFT_TO_MUCH'
                        elif mean_pitch > self.yaw_range[1]:
                            yaw_cross_status = 'RIGHT_TO_MUCH'
                        else:
                            yaw_cross_status = 'NORMAL'
                        '''
                    self.cross_status = cross_status
                    # self.pitch_cross_status = pitch_cross_status
                    # self.yaw_cross_status = yaw_cross_status
                    self.cross_limitation = cross_limitation

                    Data_saving.save_cross(cross_limitation, cross_status)


                # nose_1 = None
                # if pupil_center is not False:
                #     nose_list.append(pupil_center[2])
                #     nose = np.array(nose_list)
                #     nose_1 = np.mean(nose, axis=0)
                # print('nose 1: ', nose_1)
                # map_headx, map_heady = self.RayIntersect.computer_intersection_new(mean_pitch, mean_yaw, None)
                # print('pitch: {}, yaw: {}'.format(mean_pitch, mean_yaw))
                Data_saving.save_head_pose(mean_pitch, mean_yaw)
                map_headx, map_heady = self.RayIntersect.computer_intersection(mean_pitch, mean_yaw)

                map_headx = 0 if map_headx < 0 else map_headx
                map_headx = win_sz[0] if map_headx > win_sz[0] else map_headx

                map_heady = 0 if map_heady < 0 else map_heady
                map_heady = win_sz[1] if map_heady > win_sz[1] else map_heady

                Data_saving.save_head_point(map_headx, map_heady)
                # print('head point: {}, {}'.format(map_headx, map_heady))
            if pupil_center is not False:
                self.left_eye_list.append(pupil_center[0])
                # left_eye_list.append(left_eyes[0])
                self.right_eye_list.append(pupil_center[1])

                left_eye_corner_list.append(left_pst)
                right_eye_corner_list.append(right_pst)

                self.startMap = True

                if self.eye_tracking_status in ['moving', 'static']:
                    # print('head  moving 0....')
                    pupil_datum = self.process_static(left_eye_corner_list, right_eye_corner_list)

                    pupil_datum_left_list.append(pupil_datum[0])
                    pupil_datum_right_list.append(pupil_datum[1])

                    self.static_to_stable_count += 1
                    if self.static_to_stable_count == 1:                   # 
                    #
                        left_pupil = np.array(pupil_datum_left_list)
                        left_pupil = np.mean(left_pupil, axis=0)

                        right_pupil = np.array(pupil_datum_right_list)
                        right_pupil = np.mean(right_pupil, axis=0)

                        # self.set_center_pupil(pupil_datum[0], pupil_datum[1])
                        # print('pupil datum: {}, {}'.format(left_pupil, right_pupil))
                        self.set_center_pupil(left_pupil, right_pupil)

                        # self.revise_corner(left_corner, right_corner)

                        headpoint = Data_saving.get_headgaze()
                        self.center_one_point = (int(headpoint[0]), int(headpoint[1]))

                     
                        self.head_pose_stable['pitch'], self.head_pose_stable['yaw'] = mean_pitch, mean_yaw

                        self.eye_tracking_status = 'stable'
                        self.static_to_stable_count = 0

                        headTracking_to_eyetracking = True
                        half_count = 0

                        pupil_datum_left_list.clear()
                        pupil_datum_right_list.clear()

                    Data_saving.save_static_to_stable_counts(self.static_to_stable_count)


                if self.eye_tracking_status == 'stable':           #
                    # print('eye tracking computation ....')
                    if len(self.left_eye_list) > 0:
                        left_eye = np.array(self.left_eye_list)
                        left_eye = np.mean(left_eye, axis=0)
                        right_eye = np.array(self.right_eye_list)
                        right_eye = np.mean(right_eye, axis=0)

                        # print('left_eye: {}, right_eye: {}'.format(left_eye, right_eye))
                        left_x, left_y, right_x, right_y = left_eye[0], left_eye[1], right_eye[0], right_eye[1]

                        # if coordinate_or_biase:
                        # compute_map_point_2
                        if headTracking_to_eyetracking:
                            half_count += 1
                            # if half_count >= 20:        #50
                            if half_count >= 10:  # 20
                                headTracking_to_eyetracking=False
                                half_count = 0
                        # print('headTracking_to_eyetracking: ', headTracking_to_eyetracking)
                        # half = True if headTracking_to_eyetracking else False
                        # map_left, map_right = self.compute_map_point_2((left_x, left_y), (right_x, right_y), headTracking_to_eyetracking)#compute_map_point_1
                        map_left, map_right = self.compute_map_point_3((left_x, left_y), (right_x, right_y),
                                                                       headTracking_to_eyetracking, half_count)

                        map_center = int((map_left[0]+map_right[0])/2), int((map_left[1]+map_right[1])/2)
                        # print('map_center: ', map_center)
                        mapx, mapy = map_center[0], map_center[1]

                        mapx = 0 if mapx < 0 else mapx
                        mapx = win_sz[0] if mapx > win_sz[0] else mapx

                        mapy = 0 if mapy < 0 else mapy
                        mapy = win_sz[1] if mapy > win_sz[1] else mapy


                        # print('stable map_center: {} {}'.format(mapy, mapy))

                        ''' 
                        blink_status = self.BD.open_or_closed(frame)#open_or_closed_new   open_or_closed
                        self.BD.set_status(blink_status)

                        if blink_status == 'open':  # 闭着的时候不保存
                            # print('open')
                            # print('left_pupil: {}, right_pupil: {}'.format(left_pupil, right_pupil))
                            # print('save gaze: {}, {}'.format(mapx, mapy))
                            self.set_gaze_point(mapx, mapy)
                            Data_saving.save_gaze(mapx, mapy)
                        elif blink_status == 'closed':  # 闭着的时候清空结果
                            pass
                            # print('closed..')
                        elif blink_status == 'half-closed':
                            # print('half closed')
                            pass
                        '''
                        Data_saving.save_gaze(mapx, mapy)

                # Data_saving.save_head_point(map_headx, map_heady)
                Data_saving.save_status(self.eye_tracking_status)


            t_end = time.time()

            # print('cost time: ', t_end-t_start)
            speed_s.append(t_end-t_start)
                # self.set_gaze_point(mapx, mapy)
            if self.is_stop():
                break
        # print('mean cost: ', np.array(speed_s).mean())


    def runEyeTrackingWithSmooth_back_head_move_1(self, VideoCap, face_mesh, headposeModel, win_sz, detect_fe=False):

        datalen = Calib_times_per_point
        self.left_eye_list = collections.deque(maxlen=datalen)
        self.right_eye_list = collections.deque(maxlen=datalen)

        # add 5-9
        left_eye_corner_list = collections.deque(maxlen=datalen)
        right_eye_corner_list = collections.deque(maxlen=datalen)

        head_pose_list = collections.deque(maxlen=datalen)

        nose_list = collections.deque(maxlen=datalen)

        pupil_datum_left_list = collections.deque(maxlen=datalen)
        pupil_datum_right_list = collections.deque(maxlen=datalen)


        pitch_thr, yaw_thr = 2.5, 2.5       #


        self.eye_tracking_status = 'stable'

        self.static_to_stable_count = 0
        # self.static_to_stable_left_eye_list = collections.deque(maxlen=30)
        # self.static_to_stable_right_eye_list = collections.deque(maxlen=30)

        headTracking_to_eyetracking = False
        half_count = 0

        mean_pitch, mean_yaw = 0.0, 0.0


        self.Videosave.start_saving()


        # self.video_thread = threading.Thread(target=self.screen_video_saving_thread)
        # self.video_thread.start()
        saving_count = 0
        save_gaze = False
        first_save = True
        init_time = 0.0
        # 加一个鼠标响应事件

        # cv2.setMouseCallback('drawing', self.mouseResponse)

        while True:

            frame = VideoCap.get_frame()
            if frame is False:
                return
            frame = cv2.flip(frame, 1)

            if self.isRecording:# and self.start_saving:      #添加start_saving，由开始实验传进来
                # print('开始储存')
                saving_count+=1

                #这里保存数据
                # t0 = time.time()
                if saving_count == 2:           #5
                    # t0 = time.time()
                    self.save_user_and_screen(frame)
                    save_gaze = True
                    #保存眼动数据
                    saving_count = 0
                    # t1 = time.time() - t0
                    # print('save data cost time: ', t1)


            # 这里添加面部表情处理
            # print('error 0')
            if detect_fe:
                # t0 = time.time()
                emotion_box = self.detect_facial_expression(frame)
                # t1 = time.time()-t0
                # print('cost time: ', t1)
                if emotion_box:
                    print('expression: ', emotion_box[0])
                    Data_saving.save_emotion_box(emotion_box)
            # print('error 1')

            # t2 = time.time()
            left_pst, right_pst, pupil_center = face_mesh.process_img_1(frame)

            status, euler_angle = headposeModel.process_img(frame)
            # t3 = time.time() - t2
            # print('cost time 2: ', t3)
            if status:
                pitch, yaw, roll = euler_angle
                head_pose_list.append((pitch, yaw, roll))
                headTracking_to_eyetracking, mean_angle = self.is_head_moving(head_pose_list, (pitch_thr, yaw_thr), headTracking_to_eyetracking)

                # print('head tracking to eyetracking: ', headTracking_to_eyetracking)
                mean_pitch, mean_yaw = mean_angle



                # nose_1 = None
                # if pupil_center is not False:
                #     nose_list.append(pupil_center[2])
                #     nose = np.array(nose_list)
                #     nose_1 = np.mean(nose, axis=0)
                # print('nose 1: ', nose_1)
                # map_headx, map_heady = self.RayIntersect.computer_intersection_new(mean_pitch, mean_yaw, None)
                map_headx, map_heady = self.RayIntersect.computer_intersection(mean_pitch, mean_yaw)

                map_headx = 0 if map_headx < 0 else map_headx
                map_headx = win_sz[0] if map_headx > win_sz[0] else map_headx

                map_heady = 0 if map_heady < 0 else map_heady
                map_heady = win_sz[1] if map_heady > win_sz[1] else map_heady

                Data_saving.save_head_point(map_headx, map_heady)

            if pupil_center is not False:
                self.left_eye_list.append(pupil_center[0])
                # left_eye_list.append(left_eyes[0])
                self.right_eye_list.append(pupil_center[1])

                left_eye_corner_list.append(left_pst)
                right_eye_corner_list.append(right_pst)

                self.startMap = True


                if self.eye_tracking_status == 'static':

                    pupil_datum = self.process_static(left_eye_corner_list, right_eye_corner_list)

                    pupil_datum_left_list.append(pupil_datum[0])
                    pupil_datum_right_list.append(pupil_datum[1])

                    self.static_to_stable_count += 1
                    if self.static_to_stable_count == 5:                   # 130, 210帧调整时间, 180, 20

                        left_pupil = np.array(pupil_datum_left_list)
                        left_pupil = np.mean(left_pupil, axis=0)

                        right_pupil = np.array(pupil_datum_right_list)
                        right_pupil = np.mean(right_pupil, axis=0)

                        # self.set_center_pupil(pupil_datum[0], pupil_datum[1])
                        self.set_center_pupil(left_pupil, right_pupil)


                        # self.revise_corner(left_corner, right_corner)

                        headpoint = Data_saving.get_headgaze()
                        self.center_one_point = (int(headpoint[0]), int(headpoint[1]))

                        self.head_pose_stable['pitch'], self.head_pose_stable['yaw'] = mean_pitch, mean_yaw

                        self.eye_tracking_status = 'stable'
                        self.static_to_stable_count = 0

                        headTracking_to_eyetracking = True
                        half_count = 0

                        pupil_datum_left_list.clear()
                        pupil_datum_right_list.clear()

                    Data_saving.save_static_to_stable_counts(self.static_to_stable_count)
                elif self.eye_tracking_status == 'stable':           
                    if len(self.left_eye_list) > 0:
                        left_eye = np.array(self.left_eye_list)
                        left_eye = np.mean(left_eye, axis=0)
                        right_eye = np.array(self.right_eye_list)
                        right_eye = np.mean(right_eye, axis=0)

                        # print('left_eye: {}, right_eye: {}'.format(left_eye, right_eye))
                        left_x, left_y, right_x, right_y = left_eye[0], left_eye[1], right_eye[0], right_eye[1]

                        # if coordinate_or_biase:
                        # compute_map_point_2
                        if headTracking_to_eyetracking:
                            half_count += 1
                            # if half_count >= 20:        #50
                            if half_count >= 10:  # 20
                                headTracking_to_eyetracking=False
                                half_count = 0
                        # print('headTracking_to_eyetracking: ', headTracking_to_eyetracking)
                        # half = True if headTracking_to_eyetracking else False
                        # map_left, map_right = self.compute_map_point_2((left_x, left_y), (right_x, right_y), headTracking_to_eyetracking)#compute_map_point_1
                        map_left, map_right = self.compute_map_point_3((left_x, left_y), (right_x, right_y),
                                                                       headTracking_to_eyetracking, half_count)

                        map_center = int((map_left[0]+map_right[0])/2), int((map_left[1]+map_right[1])/2)
                        # print('map_center: ', map_center)
                        mapx, mapy = map_center[0], map_center[1]

                        mapx = 0 if mapx < 0 else mapx
                        mapx = win_sz[0] if mapx > win_sz[0] else mapx

                        mapy = 0 if mapy < 0 else mapy
                        mapy = win_sz[1] if mapy > win_sz[1] else mapy
                        # add 10/30

                        # print('stable map_center: ', map_center)

                        Data_saving.save_gaze(mapx, mapy)

                # Data_saving.save_head_point(map_headx, map_heady)
                Data_saving.save_status(self.eye_tracking_status)

            if self.is_stop():
                break


    def runEyeTrackingWithSmooth_back_head_move(self, VideoCap, face_mesh, headposeModel, win_sz, coordinate_or_biase=True):
        # left_b = []
        # right_b = []
        # pitchs = []
        # yaws = []
        # rolls = []

        datalen = Calib_times_per_point
        self.left_eye_list = collections.deque(maxlen=datalen)
        self.right_eye_list = collections.deque(maxlen=datalen)

        # add 5-9
        left_eye_corner_list = collections.deque(maxlen=datalen)
        right_eye_corner_list = collections.deque(maxlen=datalen)

        head_pose_list = collections.deque(maxlen=datalen)

        pupil_datum_left_list = collections.deque(maxlen=datalen)
        pupil_datum_right_list = collections.deque(maxlen=datalen)

        pitch_thr, yaw_thr = 2.5, 2.5          #2.5, 2.5

        
        self.eye_tracking_status = 'stable'

        self.static_to_stable_count = 0
        self.static_to_stable_left_eye_list = collections.deque(maxlen=30)
        self.static_to_stable_right_eye_list = collections.deque(maxlen=30)


        headTracking_to_eyetracking = False
        half_count = 0
        while True:
            # ret, frame = VideoCap.read()
            # print('still running')
            frame = VideoCap.get_frame()
            if frame is False:
                return
            frame = cv2.flip(frame, 1)
            # print('still error 2')
            # left_eyes, right_eyes = face_mesh.process_img_biase_corner(frame)

            left_pst, right_pst, pupil_center = face_mesh.process_img_1(frame)

            status, euler_angle = headposeModel.process_img(frame)
            if status:
                pitch, yaw, roll = euler_angle
                head_pose_list.append((pitch, yaw, roll))

                # Data_saving.get_mean_headgaze()
                # if abs(pitch-
            # print('left eyes: ', left_eyes)
            if pupil_center is not False:
                self.left_eye_list.append(pupil_center[0])
                # left_eye_list.append(left_eyes[0])
                self.right_eye_list.append(pupil_center[1])

                left_eye_corner_list.append(left_pst)
                right_eye_corner_list.append(right_pst)

                self.startMap = True


                pitch_now, yaw_now, _ = head_pose_list[-1]
                headpose = np.array(head_pose_list)
                headpose = np.mean(headpose, axis=0)

                mean_pitch, mean_yaw = headpose[0], headpose[1]
                # 
               

                if abs(mean_pitch - self.head_pose_stable['pitch']) >= pitch_thr or \
                        abs(mean_yaw - self.head_pose_stable['yaw']) >= yaw_thr:
                    if abs(pitch_now - mean_pitch) >= 2 or abs(yaw_now - mean_yaw) >= 2:
                        # print('moving....')
                        self.eye_tracking_status = 'moving'
                        headTracking_to_eyetracking = False
                    else:
                        # print('moving to 1....')
                        if self.eye_tracking_status == 'moving':
                            # print('change to static...')
                            self.eye_tracking_status = 'static'  # 
                            # 清空缓存
                            self.left_eye_list.clear()
                            self.right_eye_list.clear()
                            Data_saving.clear_gaze()
                            self.static_to_stable_count = 0


                map_headx, map_heady = self.RayIntersect.computer_intersection(mean_pitch, mean_yaw)
                map_headx = 0 if map_headx < 0 else map_headx
                map_headx = win_sz[0] if map_headx > win_sz[0] else map_headx

                map_heady = 0 if map_heady < 0 else map_heady
                map_heady = win_sz[1] if map_heady > win_sz[1] else map_heady

                #注释掉
                # self.set_center_one_point((int(map_headx), int(map_heady)))


                # print('eye tracking status: ', self.eye_tracking_status)

                if self.eye_tracking_status == 'static':

                    left_corner = np.array(left_eye_corner_list)
                    left_corner = np.mean(left_corner, axis=0)

                    right_corner = np.array(right_eye_corner_list)
                    right_corner = np.mean(right_corner, axis=0)


                    left_left = left_corner[0]
                    left_right = left_corner[1]

                    right_left = right_corner[0]
                    right_right = right_corner[1]

                    left_offset, right_offset = self.compute_eye_corner_offsets(left_left, left_right, right_left, right_right)
                    # print('left offset: {}, right offset: {}'.format(left_offset, right_offset))

                    pupil_datum = self.compute_new_pupil(left_offset, right_offset)
                    # print('new pupil_datum: {}'.format(pupil_datum))

                    pupil_datum_left_list.append(pupil_datum[0])
                    pupil_datum_right_list.append(pupil_datum[1])

                    # self.static_to_stable_left_eye_list.append(pupil_center[0])
                    # self.static_to_stable_right_eye_list.append(pupil_center[1])

                    self.static_to_stable_count += 1
                    if self.static_to_stable_count == 5:                   # 
                        # left_eye_n = np.array(self.static_to_stable_left_eye_list)
                        # left_eye_n = np.mean(left_eye_n, axis=0)
                        # right_eye_n = np.array(self.static_to_stable_right_eye_list)
                        # right_eye_n = np.mean(right_eye_n, axis=0)

                        # print('update pupil center: {}, {}'.format(left_eye_n, right_eye_n))

                        # print('update pupil...')
                        #
                        left_pupil = np.array(pupil_datum_left_list)
                        left_pupil = np.mean(left_pupil, axis=0)

                        right_pupil = np.array(pupil_datum_right_list)
                        right_pupil = np.mean(right_pupil, axis=0)

                        # self.set_center_pupil(pupil_datum[0], pupil_datum[1])
                        self.set_center_pupil(left_pupil, right_pupil)


                        # self.revise_corner(left_corner, right_corner)

                        # print('update center point...')
                        self.center_one_point = (int(map_headx), int(map_heady))

                        self.head_pose_stable['pitch'], self.head_pose_stable['yaw'] = mean_pitch, mean_yaw

                        self.eye_tracking_status = 'stable'
                        self.static_to_stable_count = 0

                        headTracking_to_eyetracking = True
                        half_count = 0
                        # print('change to stable -> first gaze point: ', Data_saving.get_mean_gaze())

                        # print('change to stable.....')
                        # print()

                        pupil_datum_left_list.clear()
                        pupil_datum_right_list.clear()


                        # self.left_eye_list.clear()
                        # self.right_eye_list.clear()
                        # self.static_to_stable_left_eye_list.clear()
                        # self.static_to_stable_right_eye_list.clear()


                    Data_saving.save_static_to_stable_counts(self.static_to_stable_count)


                elif self.eye_tracking_status == 'stable':           #

                    # print('len of left eye list: ', len(self.left_eye_list))
                    # print('compute gaze point')
                    if len(self.left_eye_list) > 0:
                        left_eye = np.array(self.left_eye_list)
                        left_eye = np.mean(left_eye, axis=0)
                        right_eye = np.array(self.right_eye_list)
                        right_eye = np.mean(right_eye, axis=0)

                        # print('left_eye: {}, right_eye: {}'.format(left_eye, right_eye))
                        left_x, left_y, right_x, right_y = left_eye[0], left_eye[1], right_eye[0], right_eye[1]

                        # if coordinate_or_biase:
                        # compute_map_point_2
                        if headTracking_to_eyetracking:
                            half_count += 1
                            if half_count >= 20: #50:
                                headTracking_to_eyetracking=False
                                half_count = 0
                        # print('headTracking_to_eyetracking: ', headTracking_to_eyetracking)
                        # half = True if headTracking_to_eyetracking else False
                        # map_left, map_right = self.compute_map_point_2((left_x, left_y), (right_x, right_y), headTracking_to_eyetracking)#compute_map_point_1
                        map_left, map_right = self.compute_map_point_3((left_x, left_y), (right_x, right_y),
                                                                       headTracking_to_eyetracking, half_count)

                        map_center = int((map_left[0]+map_right[0])/2), int((map_left[1]+map_right[1])/2)
                        # print('map_center: ', map_center)

                        mapx, mapy = map_center[0], map_center[1]

                        mapx = 0 if mapx < 0 else mapx
                        mapx = win_sz[0] if mapx > win_sz[0] else mapx

                        mapy = 0 if mapy < 0 else mapy
                        mapy = win_sz[1] if mapy > win_sz[1] else mapy
                        # add 10/30

                        # print('map_center: ', map_center)

                        Data_saving.save_gaze(mapx, mapy)

                Data_saving.save_head_point(map_headx, map_heady)

                Data_saving.save_status(self.eye_tracking_status)
                self.set_gaze_point(mapx, mapy)
            if self.is_stop():
                break
        print('end of thread....')


    def runEyeTrackingWithSmooth_without_head(self, VideoCap, face_mesh, win_sz, coordinate_or_biase=True):
        left_b = []
        right_b = []
        # pitchs = []
        # yaws = []
        # rolls = []
        while True:
            # ret, frame = VideoCap.read()
            frame = VideoCap.get_frame()
            if frame is False:
                return
            frame = cv2.flip(frame, 1)

            # left_eyes, right_eyes, left_to_nose, right_to_nose = face_mesh.process_img(frame)

            # left_eyes, right_eyes = face_mesh.process_img_ratio(frame)
            left_eyes, right_eyes = face_mesh.process_img_biase_corner(frame)
            # status, euler_angle = headposeModel.process_img(frame)
            # if status:
            #     pitch, yaw, roll = euler_angle
            #     # pitch,yaw,roll = euler_angle[0, 0], euler_angle[1, 0], euler_angle[2, 0]
            #     # for i in range(len(euler_angle)):
            #     pitchs.append(pitch)
            #     yaws.append(yaw)
            #     rolls.append(roll)

            if left_eyes is not False:
            # if (len(left_biases) > 0):
                if coordinate_or_biase:
                    left_b.append(left_eyes[0])
                    right_b.append(right_eyes[0])
                else:
                    left_b.append(left_eyes[1])
                    right_b.append(right_eyes[1])


            # print('run error 1!')
            if len(left_b) >= Calib_times_per_point:
                # calc_mean(left_b, right_b)
                self.startMap = True

                # mean_pitch = sum(pitchs) / len(pitchs)
                # mean_yaw = sum(yaws) / len(yaws)
                # mean_roll = sum(rolls) / len(rolls)

                # map_headx, map_heady = self.RayIntersect.computer_intersection(mean_pitch, mean_yaw)
                # map_headx = 0 if map_headx < 0 else map_headx
                # map_headx = win_sz[0] if map_headx > win_sz[0] else map_headx
                #
                # map_heady = 0 if map_heady < 0 else map_heady
                # map_heady = win_sz[1] if map_heady > win_sz[1] else map_heady
                #
                # self.set_center_one_point((int(map_headx), int(map_heady)))

                # biase = (left_biases[0], right_biases[0])  # 去第一个人脸的结果
                # print('left b 1  :', left_b)
                left_x, left_y, right_x, right_y = self.calc_mean(left_b, right_b)
                # print('left_x: {}, left_y: {}, right_x: {}, right_y: {}'.format(left_x, left_y, right_x, right_y))

                if coordinate_or_biase:
                    map_left, map_right = self.compute_map_point((left_x, left_y), (right_x, right_y))
                else:

                    map_left, map_right = self.compute_map_point_biase((left_x, left_y), (right_x, right_y))
                    # map_left, map_right = self.compute_map_point_ratio((left_x, left_y), (right_x, right_y))

                map_center = int((map_left[0]+map_right[0])/2), int((map_left[1]+map_right[1])/2)

                del (left_b[0])
                del (right_b[0])
                # del (pitchs[0])
                # del (yaws[0])
                # add at 10/1
                mapx, mapy = map_center[0], map_center[1]

                mapx = 0 if mapx < 0 else mapx
                mapx = win_sz[0] if mapx > win_sz[0] else mapx

                mapy = 0 if mapy < 0 else mapy
                mapy = win_sz[1] if mapy > win_sz[1] else mapy

                # add 10/30
                Data_saving.save_gaze(mapx, mapy)
                # Data_saving.save_head_point(map_headx, map_heady)

                self.set_gaze_point(mapx, mapy)

            if self.is_stop():
                break
    def set_eyetracking_static(self)
        self.eye_tracking_status = 'static'
        self.left_eye_list.clear()
        self.right_eye_list.clear()
        self.static_to_stable_count = 0
        self.static_to_stable_left_eye_list.clear()
        self.static_to_stable_right_eye_list.clear()
        Data_saving.clear_gaze() 

    def get_static_to_stable_count(self):
        return self.static_to_stable_count


