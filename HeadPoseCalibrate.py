import cv2
import numpy as np
import sys
import pyautogui as pg
from blink_detection_1 import BlinkDetector_1
import os
import time
class HeadPositionCalibration:
    def __init__(self, cap):
        self.videocap = cap
        self.headcalibrationStatus = False



    def callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.headcalibrationStatus = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            img = pg.screenshot()
            open_cv_image = np.array(img)

            # Convert RGB to BGR,opencv read image as BGR,but Pillow is RGB
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
            # open_cv_image = cv2.resize(open_cv_image, (960, 540))
            cv2.imwrite('subjectData/screenshoot.jpg', open_cv_image)



    def show_face3(self, windowName):
        img = np.zeros((1080, 1920, 3), np.uint8)

        start_point = (640, 300)
        end_point = (1280, 780)

        words = 'Please put your head in the ellipse area!'

        s_point = (start_point[0]-20, start_point[1]-50)
        cv2.putText(img, words, s_point, cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255),2)  # FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_COMPLEX

        center = (960, 540)
        # radius = 100
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            # if self.getHeadcalibrationStatus():
            #     break

            # print('error 0')
            if not self.videocap.is_start_queue():
                # 等到图像进来
                time.sleep(0.1)
                continue
            # print('error 1')
            frame = self.videocap.get_frame()

            # ret, frame = self.videocap.read()
            if frame is False:
                print('相机获取图像错误')
                break
            frame = cv2.flip(frame, 1)
            # _, frame = self.videocap.read()
            img[start_point[1]:end_point[1],start_point[0]:end_point[0]] = frame[0:480, 0:640].copy()
            # cv2.circle(img, center, radius, (255, 0, 255), 2)
            cv2.ellipse(img, center, (100,130), 0,0,360,(255,0,255), 3)

            # if show_hand:
            #     cv2.line(img, start_hand_point, (end_hand_point[0], start_hand_point[1]), (0,0,255), 2)
            #     img[start_hand_point[1]:end_hand_point[1], start_hand_point[0]:end_hand_point[0]] = frame[390:480, 170:470].copy()

            cv2.imshow(windowName, img)
            if cv2.waitKey(20) & 0xFF == ord(' '):
                cv2.destroyAllWindows()
                break
                # sys.exit()
            # cv2.waitKey(20)
        # 结束后删除窗口
        # cv2.destroyAllWindows()


    def getHeadcalibrationStatus(self):
        return self.headcalibrationStatus


if __name__ == '__main__':
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    HC = HeadPositionCalibration(cap)
    WINDOW_NAME = 'CALIBRATION'
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    HC.show_face2(WINDOW_NAME)