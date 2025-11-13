from collections import deque
import pyautogui as pg
import cv2
import numpy as np
import threading
import time
import os
import pandas as pd
from PIL import ImageGrab
from saveGazeData import Data_saving

class videoCapture:
    def __init__(self, ID_number=0, flip_id=2):
        self.cap = cv2.VideoCapture(ID_number, cv2.CAP_DSHOW)

        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)                    #使用购买的摄像头
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

        self.imgstream = deque(maxlen=20)
        self.captureThread = threading.Thread(target=self.VideoCaptureThread, args=(flip_id,))
        self.captureThread.start()
        self.stop = False
        self.startQueue = False

    def is_start_queue(self):
        return self.startQueue

    def VideoCaptureThread(self, flip_id):

        count = 0
        while True:
            # print('capture still...')
            ret, frame = self.cap.read()

            if ret is False:
                print('capture error!')
                break

            if flip_id != 2:
                frame = cv2.flip(frame, flip_id)

            self.imgstream.append(frame)
            if count == 0:
                self.startQueue = True
                count = 1

            if self.stop:
                break

    def get_frame(self):
        if self.startQueue:
            return self.imgstream[-1]
        return False

    def stopCapture(self):
        self.stop = True
        self.captureThread.join()
        self._release()

    def _release(self):
        self.cap.release()

# combination of face and writer
class saveFaceAndScreen_new:

    # saving  -> 是否存储
    def __init__(self, buffer_len=20, saving=True, savePath='', ID_number=0, flip_id=2):
        # pass
        self.FaceImgQueue = deque(maxlen=buffer_len)
        # self.ScreenImgQueue =  deque(maxlen=buffer_len)

        self.saveVides = saving #True

        if saving:
            self._create_videos(savePath)
            self.first_save = True

        self.startSaving = False
        self.videocapture = videoCapture(ID_number, flip_id=flip_id)
        if saving:
            self.saveVideoThread = threading.Thread(target=self.continues_saving)

    def _create_videos(self, savepath):
        screen_video = os.path.join(savepath, 'screen.mp4')
        user_video = os.path.join(savepath, 'user.mp4')

        self.screen_writer = cv2.VideoWriter(screen_video, cv2.VideoWriter_fourcc(*"mp4v"), 40, (640, 360))
        self.face_writer = cv2.VideoWriter(user_video, cv2.VideoWriter_fourcc(*"mp4v"), 40, (640, 480))

        # add 2024-7-23
        self.excelName = os.path.join(savepath, 'gaze.csv')

        # def set_face(self, capImg):
    #     self.FaceImgQueue.append(capImg)
    #     self.startSaving = True

    def start_saving(self):
        if self.saveVides:
            self.saveVideoThread.start()

    def stop_saving(self):
        if self.saveVides:
            self.saveVides = False
            self.saveVideoThread.join()

    def continues_saving(self):
        # count = 0
        if self.saveVides is False:
            return
        while self.saveVides:
            t0 = time.time()
            status = self.get_current_face_and_screen()
            t1 = time.time() - t0
            # print('cost time: ', t1)
            if status is False:
                continue
            # if count == 100:
            #     break
            # count += 1


        # self.release()

    def save_gaze_to_excle(self, df):
        # df.to_csv(self.excelName, mode='a')
        df.to_csv(self.excelName, mode='a', header=False, index=False)

    def save_gaze(self):
        eye_tracking_status = Data_saving.get_status()

        GazeSaving = Data_saving.get_mean_gaze() if eye_tracking_status == 'stable' else Data_saving.get_mean_headgaze()
        if GazeSaving:
            # eye_staus = 0 if self.eye_tracking_status == 'stable' else 1
            # eye_status = self.eye_tracking_status
            if self.first_save:
                self.init_time = time.time()
                elapse_time = 0.0
                self.first_save = False
            else:
                elapse_time = time.time() - self.init_time
            df = [eye_tracking_status, GazeSaving[0], GazeSaving[1], round(elapse_time, 2)]
            # print('df :', df)
            # df = {self.eye_tracking_status, GazeSaving[0], GazeSaving[1], time.time()}

            df = pd.DataFrame(df).T
            # print('df: ', df)
            self.save_gaze_to_excle(df)
            # save_gaze = False

    def get_current_face_and_screen(self):
        screen = pg.screenshot()
        open_cv_screen = np.array(screen)
        # Convert RGB to BGR,opencv read image as BGR,but Pillow is RGB
        open_cv_screen = cv2.cvtColor(open_cv_screen, cv2.COLOR_RGB2BGR)

        # screen = ImageGrab.grab()
        # screen = np.array(screen.getdata(), np.uint8)#.reshape(640, 360, 3)
        # screen = cv2.resize(screen, (640, 360))
        face = self.videocapture.get_frame()
        # face = self.FaceImgQueue[-1]
        if face is False:
            print('video capture error!')
            return False
        else:
            face = cv2.resize(face, (640, 480))
            open_cv_screen = cv2.resize(open_cv_screen, (640, 360))
            self.face_writer.write(face)
            self.screen_writer.write(open_cv_screen)
            self.save_gaze()
            return True

    def release(self):
        if self.saveVides:
            self.screen_writer.release()
            self.face_writer.release()
        self.videocapture.stopCapture()


class saveFaceAndScreen:
    def __init__(self, buffer_len=20):
        # pass
        self.FaceImgQueue = deque(maxlen=buffer_len)
        # self.ScreenImgQueue =  deque(maxlen=buffer_len)
        self._create_videos()
        self.saveVides = True
        self.startSaving = False

    def _create_videos(self):
        self.screen_writer = cv2.VideoWriter("VideoData/screen.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, (int(1920), int(1080)))
        self.face_writer = cv2.VideoWriter("VideoData/face.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, (int(640), int(480)))

    def set_face(self, capImg):
        self.FaceImgQueue.append(capImg)
        self.startSaving = True


    def continues_saving(self):
        while self.saveVides:
            self.get_current_face_and_screen()

    def stop_saving(self):
        self.saveVides = False
        # self.release()

    def get_current_face_and_screen(self):
        screen = pg.screenshot()
        open_cv_screen = np.array(screen)

        # Convert RGB to BGR,opencv read image as BGR,but Pillow is RGB
        open_cv_screen = cv2.cvtColor(open_cv_screen, cv2.COLOR_RGB2BGR)
        face = self.FaceImgQueue[-1]
        if face.any():
            self.screen_writer.write(open_cv_screen)
        else:
            print('video capture error!')

        self.face_writer.write(face)

    def release(self):
        self.screen_writer.release()
        self.face_writer.release()


if __name__ =='__main__':

    # VC = videoCapture()
    #
    # while True:
    #     frame = VC.get_frame()
    #     if frame is not False:
    #         cv2.imshow('test.jpg', frame)
    #         cv2.waitKey(20)
    #         # cv2.Waitkey(20)
    # VC.stopCapture()


    Videosave = saveFaceAndScreen_new()

    Videosave.start_saving()
    # Videosave.continues_saving()



    for i in range(5):
        time.sleep(1)

    Videosave.stop_saving()
    Videosave.release()
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # count = 0
    # while count < 100:
    #     # _, frame = cap.read()
    #     # Videosave.set_face(frame)
    #     # Videosave.get_current_face_and_screen()
    #     # count += 1
    #
    # Videosave.release()







