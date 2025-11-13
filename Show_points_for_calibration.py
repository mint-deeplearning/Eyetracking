import numpy as np
import cv2
import time
import pyautogui

from saveGazeData import Data_saving

class ShowCalibrationPoint():
    def __init__(self, window_name, monitor_pixels, dis_to_bound=100, ablation=False):
        self.sz = monitor_pixels

        # self.eye_tracker = eye_tracker
        self.point_list = self.generate_calibration_points(monitor_pixels[0], monitor_pixels[1], dis_to_bound)

        if ablation:
            self.ablation_list = self.generate_abaltion_points(monitor_pixels[0], monitor_pixels[1])

        self.wd_name = window_name
        # print('point_list: ', self.point_list2)
        self.t_start = 0.0


    def generate_calibration_points(self, width, height, dis_to_boundary):

        point0 = (width/2, height/2)
        point1 = (width/2, dis_to_boundary)
        point2 = (dis_to_boundary, height / 2)
        point3 = (width/2, height-dis_to_boundary)
        point4 = (width-dis_to_boundary, height / 2)

        point_list = [point0, point1, point2, point3, point4]

        for i in range(len(point_list)):
            p = point_list[i]
            point_list[i] = (int(p[0]), int(p[1]))

        return point_list

    def generate_abaltion_points(self, width, height):

        point_list = []
        for i in range(9):
            row, col = i//3, i%3

            point = (int(width/6+width/3*col), int(height/6+height/3*row))
            point_list.append(point)

        return point_list

    def draw_headgaze_point(self, head_point, img):

        colors = [(255, 0, 255), (255, 48, 210)]
        cv2.circle(img, head_point, 30, colors[0], thickness=-1)  # 255,0,255   (0,255,0)

        # return resimg
    def show_ablation_point_i(self, number, wd_name):
        point = self.ablation_list[number]
        img = np.zeros((self.sz[1], self.sz[0], 3), np.uint8)
        # (255,0,0) 蓝色
        color = (0, 0, 255) #(0, 250, 0) #if is_center else (0, 0, 255)

        radius = 40

        t_start = time.time()
        while True:
            img_show = img.copy()
            t_elapse = time.time() - t_start

            head_gz = Data_saving.get_mean_headgaze()
            if head_gz:
                self.draw_headgaze_point(head_gz, img_show)

            if t_elapse > 2.5:
                t_elapse = 2.5
            r = self.compute_radius(radius, t_elapse)
            cv2.circle(img_show, point, r, color, -1)

            cv2.imshow(wd_name, img_show)
            # cv2.waitKey(20)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
                # cv2.destroyAllWindows()
                # sys.exit()
        return point

    def show_ablation_point(self):

        cv2.namedWindow(self.wd_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.wd_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        point_amount = len(self.ablation_list)

        i = 0
        while i < point_amount:
            # print('first i: ', i)
            # self.t_start = cv2.getTickCount()
            self.show_ablation_point_i(i, self.wd_name)  # show_point
            i += 1
        print('calibration finished!...')
        cv2.destroyAllWindows()

    def show_point(self):

        cv2.namedWindow(self.wd_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.wd_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        point_amount = len(self.point_list)

        i = 0
        while i < point_amount:
            # print('first i: ', i)
            # self.t_start = cv2.getTickCount()
            self.show_point_i(i)  # show_point
            i += 1
        print('calibration finished!...')
        cv2.destroyAllWindows()

    def compute_radius(self, r_start, elapse_t):
        return int(r_start-elapse_t*10)

    def show_point_i(self, number, wd_name):
        point = self.point_list[number]
        img = np.zeros((self.sz[1], self.sz[0], 3), np.uint8)
        # (255,0,0) 蓝色
        color = (0, 0, 255) #(0, 250, 0) #if is_center else (0, 0, 255)


        radius = 40

        t_start = time.time()
        while True:
            img_show = img.copy()
            t_elapse = time.time() - t_start

            if t_elapse >= 1.8: #2.5:
                break
            r = self.compute_radius(radius, t_elapse)
            cv2.circle(img_show, point, r, color, -1)
            cv2.imshow(wd_name, img_show)
            cv2.waitKey(20)

        return point

if __name__ == '__main__':
    SCP = ShowCalibrationPoint('test.jpg', (1920,1080), ablation=True)
    # SCP.show_point()
    SCP.show_ablation_point()


