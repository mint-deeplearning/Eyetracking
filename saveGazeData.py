import collections
import numpy as np

max_len=10

class DataSaving:
    def __init__(self):
        self.gaze = collections.deque(maxlen=max_len)
        self.head_gaze = collections.deque(maxlen=max_len)
        self.datum_leftpupil = collections.deque(maxlen=max_len)
        self.datum_rightpupil = collections.deque(maxlen=max_len)

        self.pose = collections.deque(maxlen=max_len)
        self.left_eye = collections.deque(maxlen=max_len)
        self.right_eye = collections.deque(maxlen=max_len)

        self.cross_status = collections.deque(maxlen=max_len)
        self.cross_limitation = collections.deque(maxlen=max_len)

        self.emotion_box_list = collections.deque(maxlen=max_len)
        # self.yaw = collections.deque(maxlen=15)
        self.saving = False
        self.gaze_saving = False
        self.pose_saving = False
        self.emotion_save = False
        self.cross_save = False

        self.tracking_status = ''
        self.static_to_stable_counts = 0
        self.datum_saving = False

        self.eye_corner_saving = False

        self.left = (0,0)
        self.right = (0,0)

    def clear_all(self):
        self.saving = False
        self.gaze_saving = False
        self.pose_saving = False
        self.emotion_save = False
        self.cross_save = False

        self.tracking_status = ''
        self.static_to_stable_counts = 0
        self.datum_saving = False
        self.gaze.clear()
        self.head_gaze.clear()
        self.datum_leftpupil.clear()
        self.datum_rightpupil.clear()

        self.cross_status.clear()
        self.cross_limitation.clear()
        self.pose.clear()
        self.left_eye.clear()
        self.right_eye.clear()
        self.emotion_box_list.clear()

        self.GameScore = None
        self.clip = False

    def clear_gaze(self):

        # print('come here 0?...............')
        self.saving = False
        self.gaze.clear()

    def save_status(self, status):
        self.tracking_status = status

    def save_static_to_stable_counts(self, count):
        self.static_to_stable_counts = count

    def get_static_to_stable_counts(self):
        return self.static_to_stable_counts
    
    def get_status(self):
        return self.tracking_status

    def save_gaze(self, mapx, mapy):
        self.is_save()
        self.gaze.append((mapx,mapy))
    def save_score(self, score):
        self.GameScore = score

    def save_cross(self, cross_limitation, cross_status):
        self.cross_save = True
        self.cross_status.append(cross_status)      #向上、下、左、右转动太多
        self.cross_limitation.append(cross_limitation)  #是否越界


    def save_head_point(self, headx, heady):
        self.gaze_saving = True
        self.head_gaze.append((headx, heady))

    def save_datum_pupil(self, left, right):
        if not self.datum_saving:
            self.datum_saving = True

        self.datum_leftpupil.append(left)
        self.datum_rightpupil.append(right)

    def save_head_pose(self, pitch, yaw):
        if not self.pose_saving:
            self.pose_saving = True
        self.pose.append((pitch, yaw))
        # self.yaw.append(yaw)

    def save_eye_corner(self, left, right):
        if not self.eye_corner_saving:
            self.eye_corner_saving = True

        self.left_eye.append(np.array(left).ravel())
        self.right_eye.append(np.array(right).ravel())

    def get_score(self):
        return self.GameScore

    def get_cross(self):
        if self.cross_save is True:
            return self.cross_limitation[-1], self.cross_status[-1]
        else:
            return False


    def get_mean_eye_corner(self):
        if self.eye_corner_saving is True:
            lefteye = np.array(self.left_eye)

            # print('left eye: ', lefteye)
            # print('left eye shape: ', lefteye.shape)
            righteye = np.array(self.right_eye)
            # print('left eye shape: ', lefteye.shape)
            lefteye = np.mean(lefteye, axis=0)
            righteye = np.mean(righteye, axis=0)

            # print('left eye: ', lefteye)
            # print('right eye: ', righteye)
            # print('left pupil center: ', leftpupil)
            # print('right pupil center: ', rightpupil)
            # gaze_point = (int(gaze_point[0]), int(gaze_point[1]))

            return (lefteye, righteye)
        else:
            return False

    def get_mean_datum_pupil(self):
        if self.datum_saving is True:
            leftpupil = np.array(self.datum_leftpupil)
            rightpupil = np.array(self.datum_rightpupil)

            leftpupil = np.mean(leftpupil, axis=0)
            rightpupil = np.mean(rightpupil, axis=0)
            # print('left pupil center: ', leftpupil)
            # print('right pupil center: ', rightpupil)
            # gaze_point = (int(gaze_point[0]), int(gaze_point[1]))

            return (leftpupil, rightpupil)
        else:
            return False

    def save_original_pupil(self, left, right):
        self.left = left
        self.right = right

    def get_original_pupil(self):
        return self.left, self.right

    def is_save(self):
        self.saving = True

    def get_gaze(self):
        if self.saving is True:
            return self.gaze[-1]
        else:
            return False

    def get_mean_gaze(self):

        if self.saving is True:
            gaze = np.array(self.gaze)

            gaze_point = np.mean(gaze, axis=0)

            gaze_point = (int(gaze_point[0]), int(gaze_point[1]))
            return gaze_point
        else:
            return False

    def clear_gaze_(self):
        print('come here 1?...............')
        lastgaze = self.gaze[-1]
        self.gaze.clear()
        self.gaze.append(lastgaze)

    def get_headgaze(self):
        if self.gaze_saving is True:
            return self.head_gaze[-1]
        else:
            return False

    def get_mean_headgaze(self):

        if self.gaze_saving is True:
            head_gaze = np.array(self.head_gaze)
            # print('head_gaze: ', head_gaze)
            gaze_point = np.mean(head_gaze, axis=0)

            gaze_point = (int(gaze_point[0]), int(gaze_point[1]))
            return gaze_point
        else:
            return False

    def get_mean_headpose(self):

        if self.pose_saving is True:
            head_pose = np.array(self.pose)
            # print('head_gaze: ', head_gaze)
            pose = np.mean(head_pose, axis=0)

            pose = float(pose[0]), float(pose[1])
            return pose
        else:
            return False
    # def set_eye_tracking_static(self):

    def save_emotion_box(self, emotion_box):
        self.emotion_save = True
        self.emotion_box_list.append(emotion_box)

    def get_current_emotion_box(self):
        if self.emotion_save:
            return self.emotion_box_list[-1]
        else:
            return False

    def set_clip(self, staus):
        self.clip = staus

    def get_clip(self):
        return self.clip

Data_saving = DataSaving()

# dq = collections.deque(maxlen=20)
#
# dq.append((1,2))
#
# dq.append((2,3))
#
# dq.append((4,3))
#
# print(len(dq))
#
#
# import numpy as np
#
# dq = np.array(dq)
# print(dq)

# print(np.mean(dq, axis=0))
