import math

import cv2
import mediapipe as mp
import numpy as np


class headPoseMediapipe:
    def __init__(self):
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5)
        self.face_coordination_in_real_world = np.array([
            [285, 528, 200],
            [285, 371, 152],
            [197, 574, 128],
            [173, 425, 108],
            [360, 574, 128],
            [391, 425, 108]
        ], dtype=np.float64)

    def rotation_matrix_to_angles(self, rotation_matrix):
        """
        Calculate Euler angles from rotation matrix.
        :param rotation_matrix: A 3*3 matrix with the following structure
        [Cosz*Cosy  Cosz*Siny*Sinx - Sinz*Cosx  Cosz*Siny*Cosx + Sinz*Sinx]
        [Sinz*Cosy  Sinz*Siny*Sinx + Sinz*Cosx  Sinz*Siny*Cosx - Cosz*Sinx]
        [  -Siny             CosySinx                   Cosy*Cosx         ]
        :return: Angles in degrees for each axis
        """
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0] ** 2 +
                                                         rotation_matrix[1, 0] ** 2))
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        return np.array([x, y, z]) * 180. / math.pi

    def process_img(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image)
        h, w, _ = image.shape
        face_coordination_in_image = []

        angle_results = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [1, 9, 57, 130, 287, 359]:
                        x, y = int(lm.x * w), int(lm.y * h)
                        face_coordination_in_image.append([x, y])

                face_coordination_in_image = np.array(face_coordination_in_image,
                                                      dtype=np.float64)

                # The camera matrix
                focal_length = 1 * w
                cam_matrix = np.array([[focal_length, 0, w / 2],
                                       [0, focal_length, h / 2],
                                       [0, 0, 1]])

                # The Distance Matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Use solvePnP function to get rotation vector
                success, rotation_vec, transition_vec = cv2.solvePnP(
                    self.face_coordination_in_real_world, face_coordination_in_image,
                    cam_matrix, dist_matrix)

                # Use Rodrigues function to convert rotation vector to matrix
                rotation_matrix, jacobian = cv2.Rodrigues(rotation_vec)
                # pitch, yaw, roll
                result = self.rotation_matrix_to_angles(rotation_matrix)
                angle_results.append(result)
                # print('result: ', result)

                '''
                for i, info in enumerate(zip(('pitch', 'yaw', 'roll'), result)):
                    k, v = info
                    text = f'{k}: {int(v)}'
                    cv2.putText(image, text, (20, i * 30 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
                '''
            return True, angle_results[0]
        else:
            return False, False


MM_TO_IN = 0.0393700787  # 1/25.4   1mm等于多少英寸
# MM_TO_IN = 0.0641026  # 1/15.6   1mm等于多少英寸
import math


class RayPlaneIntersection:
    def __init__(self, win_sz, win_inch, user_camere_distance=500):  # mm
        self.win_sz = win_sz
        self.win_inch = win_inch
        self.distance = user_camere_distance
        self.pixel_per_mm = self.get_mm_pixel_ratio(win_inch)
        self.center = (int(win_sz[0] / 2), int(win_sz[1] / 2))

    def get_mm_pixel_ratio(self, screen_size_inch):
        # from tkinter import Tk
        # root = Tk()
        # width = root.winfo_screenwidth()
        # height = root.winfo_screenheight()

        diagonal_pixel = np.sqrt(np.square(self.win_sz[0]) + np.square(self.win_sz[1]))
        diagonal_mm = screen_size_inch / MM_TO_IN  # 对角线的长度，单位毫米
        pixel_per_mm = diagonal_pixel / diagonal_mm  # 1mm对应多少个像素
        return pixel_per_mm

    def set_center_nose(self, nose):
        self.nose = nose

    def set_center_angle(self, pitch, yaw, roll):
        self.center_p = pitch
        self.center_y = yaw
        self.center_r = roll

    def computer_intersection_new(self, pitch, yaw, nose):
        biase_pitch = pitch - self.center_p
        biase_yaw = yaw - self.center_y

        delta_x = self.distance * math.tan(biase_yaw / 180 * math.pi)
        delta_y = self.distance * math.tan(biase_pitch / 180 * math.pi)

        # print('delta_X: ', delta_x)
        # print('delta_y: ', delta_y)
        #
        if nose is not None:
            delta_nose = nose[0] - self.nose[0], nose[1] - self.nose[1]
        else:
            delta_nose = 0.0, 0.0

        # print('delta nose: ', delta_nose)
        new_x = self.center[0] + delta_nose[0] + delta_x * self.pixel_per_mm
        new_y = self.center[1] + delta_nose[1] - delta_y * self.pixel_per_mm

        # new_x = self.center[0] + delta_x * self.pixel_per_mm
        # new_y = self.center[1] - delta_y * self.pixel_per_mm

        return (int(new_x), int(new_y))

    def computer_intersection(self, pitch, yaw):
        biase_pitch = pitch - self.center_p
        biase_yaw = yaw - self.center_y

        delta_x = self.distance * math.tan(biase_yaw / 180 * math.pi)
        delta_y = self.distance * math.tan(biase_pitch / 180 * math.pi)

        # print('delta_X: ', delta_x)
        # print('delta_y: ', delta_y)
        #
        new_x = self.center[0] + delta_x * self.pixel_per_mm
        new_y = self.center[1] - delta_y * self.pixel_per_mm

        return (int(new_x), int(new_y))

    def computer_intersection_1(self, pitch, yaw):
        biase_pitch = pitch - self.center_p
        biase_yaw = yaw - self.center_y

        # delta_x = self.distance * math.tan(biase_yaw / 180 * math.pi)
        # delta_y = self.distance * math.tan(biase_pitch / 180 * math.pi)

        delta_x = self.distance * (math.tan(yaw / 180 * math.pi) - math.tan(biase_yaw / 180 * math.pi))
        delta_y = self.distance * (math.tan(pitch / 180 * math.pi) - math.tan(biase_pitch / 180 * math.pi))

        # print('delta_X: ', delta_x)
        # print('delta_y: ', delta_y)
        #
        new_x = self.center[0] + delta_x * self.pixel_per_mm
        new_y = self.center[1] - delta_y * self.pixel_per_mm

        return (int(new_x), int(new_y))


import collections

if __name__ == '__main__':
    HeadPose = headPoseMediapipe()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    count = 0
    t_list = 0.0
    datalen = 10
    # left_eye_list = collections.deque(maxlen=datalen)
    # right_eye_list = collections.deque(maxlen=datalen)
    head_pose_list = collections.deque(maxlen=datalen)

    angle_thr = 2.0
    while cap.isOpened():
        success, image = cap.read()
        image = cv2.flip(image, 1)

        t1 = cv2.getTickCount()
        status, result = HeadPose.process_img(image)
        if status:
            print('result: ', result)

            head_pose_list.append(result)

        if count >= datalen:
            pitch_now, yaw_now, _ = head_pose_list[-1]
            headpose = np.array(head_pose_list)
            headpose = np.mean(headpose, axis=0)

            mean_pitch, mean_yaw = headpose[0], headpose[1]

            if abs(pitch_now - mean_pitch) >= angle_thr or abs(yaw_now - mean_yaw) >= angle_thr:
                print('pitch_now - mean_pitch: ', pitch_now - mean_pitch)
                print('yaw_now - mean_yaw: ', yaw_now - mean_yaw)
                print('head is moving!!!!!!!!!!')

        t2 = (cv2.getTickCount() - t1) / cv2.getTickFrequency() * 1000
        t_list += t2

        # if count == 100:
        #     break
        count += 1
        cv2.imshow('test.jpg', image)
        cv2.waitKey(20)

    cap.release()

    print('mean cost: ', t_list / count)