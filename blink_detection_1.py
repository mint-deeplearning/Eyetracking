import cv2 as cv
import mediapipe as mp
# import time
import math
# import numpy as np

# variables
frame_counter = 0
CEF_COUNTER = 0
TOTAL_BLINKS = 0
# constants
CLOSED_EYES_FRAME = 3
FONTS = cv.FONT_HERSHEY_COMPLEX

# face bounder indices
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
             149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# lips indices for Landmarks
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39,
        37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
# Left eyes indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]

# right eyes indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

map_face_mesh = mp.solutions.face_mesh
# camera object

# landmark detection function
def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                  results.multi_face_landmarks[0].landmark]
    if draw:
        [cv.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks
    return mesh_coord


# Euclaidean distance
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return distance


# Blinking Ratio
def blinkRatio(landmarks, right_indices, left_indices):
    # Right eyes
    # horizontal line
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # draw lines on right eyes
    # cv.line(img, rh_right, rh_left, utils_blink.GREEN, 1)
    # cv.line(img, rv_top, rv_bottom, utils_blink.WHITE, 1)

    # LEFT_EYE
    # horizontal line
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rhDistance / (rvDistance+0.01)
    leRatio = lhDistance / (lvDistance+0.01)

    ratio = (reRatio + leRatio) / 2
    return ratio



class BlinkDetector:
    def __init__(self, ear_thr=3.5):
        self.face_mesh = map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.ear_thr = ear_thr

    def open_or_closed(self, frame):
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results = self.face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, False)
            ratio = blinkRatio(mesh_coords, RIGHT_EYE, LEFT_EYE)
            # cv.putText(frame, f'ratio {ratio}', (100, 100), FONTS, 1.0, utils.GREEN, 2)
            return 'closed' if ratio > self.ear_thr else 'open'
        else:
            return False



class BlinkDetector_1:
    #包含校准确认系数的过程
    def __init__(self):
        self.face_mesh = map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.ear_thr = None
        self.blink_status = 'open'

    def set_status(self, status):
        self.blink_status = status

    def get_status(self):
        return self.blink_status

    def get_current_ear(self, frame):
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results = self.face_mesh.process(rgb_frame)
        ratio = False
        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, False)
            ratio = blinkRatio(mesh_coords, RIGHT_EYE, LEFT_EYE)
        return ratio

    def set_ear_thr(self, ear_thr):
        # print('ear thr: ', ear_thr)
        self.ear_thr = ear_thr

    def get_ear_thr(self):
        # print('BD ear THR: ', self.ear_thr)
        return self.ear_thr

    def open_or_closed(self, frame):
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results = self.face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, False)
            ratio = blinkRatio(mesh_coords, RIGHT_EYE, LEFT_EYE)
            # cv.putText(frame, f'ratio {ratio}', (100, 100), FONTS, 1.0, utils.GREEN, 2)
            return 'closed' if ratio > self.ear_thr else 'open'
        else:
            return False

    def open_or_closed_new(self, frame):
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results = self.face_mesh.process(rgb_frame)
        status = None
        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, False)
            ratio = blinkRatio(mesh_coords, RIGHT_EYE, LEFT_EYE)
            # print('ratio" ', ratio)
            # cv.putText(frame, f'ratio {ratio}', (100, 100), FONTS, 1.0, utils.GREEN, 2)
            if ratio > self.ear_thr*1.2:
                status = 'closed'
            elif ratio > self.ear_thr*1.1:         #1.03 1.08   1.05
                status = 'half-closed'
            else:
                status = 'open'

            # return 'closed' if ratio > self.ear_thr else 'open'
        # else:
        #     return False

        return status
