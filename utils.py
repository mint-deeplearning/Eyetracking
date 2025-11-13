import random
import sys
import time
from datetime import datetime
from enum import Enum
# import pyautogui as pg
import cv2
import numpy as np
from typing import Tuple, Union
import math


# from webcam import WebcamSource
# from run_eye_gaze_one_point import get_gaze

def get_monitor_dimensions() -> Union[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[None, None]]:
    """
    Get monitor dimensions from Gdk.
    from on https://github.com/NVlabs/few_shot_gaze/blob/master/demo/monitor.py
    :return: tuple of monitor width and height in mm and pixels or None
    """
    try:
        import pgi

        pgi.install_as_gi()
        import gi.repository

        gi.require_version('Gdk', '3.0')
        from gi.repository import Gdk

        display = Gdk.Display.get_default()
        screen = display.get_default_screen()
        default_screen = screen.get_default()
        num = default_screen.get_number()

        h_mm = default_screen.get_monitor_height_mm(num)
        w_mm = default_screen.get_monitor_width_mm(num)

        h_pixels = default_screen.get_height()
        w_pixels = default_screen.get_width()

        return (w_mm, h_mm), (w_pixels, h_pixels)

    except ModuleNotFoundError:
        return None, None


FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.8  # 0.5
TEXT_THICKNESS = 2


class TargetOrientation(Enum):
    UP = 38  # 82
    DOWN = 40  # 84
    LEFT = 37  # 81
    RIGHT = 39  # 83


def create_image(monitor_pixels: Tuple[int, int], center=(0, 0), circle_scale=1., orientation=TargetOrientation.RIGHT,
                 target='E') -> Tuple[np.ndarray, float, bool]:
    """
    Create image to display on screen.

    :param monitor_pixels: monitor dimensions in pixels
    :param center: center of the circle and the text
    :param circle_scale: scale of the circle
    :param orientation: orientation of the target
    :param target: char to write on image
    :return: created image, new smaller circle_scale and bool that indicated if it is th last frame in the animation
    """
    width, height = monitor_pixels

    # img = cv2.imread('test.jpg')
    if orientation == TargetOrientation.LEFT or orientation == TargetOrientation.RIGHT:
        img = np.zeros((height, width, 3), np.float32)
        #
        if orientation == TargetOrientation.LEFT:
            center = (width - center[0], center[1])

        end_animation_loop = write_text_on_image(center, circle_scale, img, target)

        if orientation == TargetOrientation.LEFT:
            img = cv2.flip(img, 1)
    else:  # TargetOrientation.UP or TargetOrientation.DOWN
        img = np.zeros((width, height, 3), np.float32)
        center = (center[1], center[0])

        if orientation == TargetOrientation.UP:
            center = (height - center[0], center[1])

        end_animation_loop = write_text_on_image(center, circle_scale, img, target)

        if orientation == TargetOrientation.UP:
            img = cv2.flip(img, 1)

        img = img.transpose((1, 0, 2))

    return img / 255, circle_scale * 0.78, end_animation_loop


def create_image_spider(monitor_pixels: Tuple[int, int], center=(0, 0), circle_scale=1.,
                        orientation=TargetOrientation.RIGHT, target='E') -> Tuple[np.ndarray, float, bool]:
    """
    Create image to display on screen.

    :param monitor_pixels: monitor dimensions in pixels
    :param center: center of the circle and the text
    :param circle_scale: scale of the circle
    :param orientation: orientation of the target
    :param target: char to write on image
    :return: created image, new smaller circle_scale and bool that indicated if it is th last frame in the animation
    """
    width, height = monitor_pixels

    # img = np.zeros((height, width, 3), np.float32)

    img = cv2.imread('data/test.jpg')
    img = cv2.resize(img, (width, height))

    if orientation == TargetOrientation.LEFT:
        center = (width - center[0], center[1])

    end_animation_loop = draw_spider_on_image(center, circle_scale, img, target)

    # if orientation == TargetOrientation.LEFT:
    #     img = cv2.flip(img, 1)

    # return img / 255, circle_scale * 0.78, end_animation_loop
    return img, circle_scale * 0.78, end_animation_loop


# draw spider
def draw_spider_on_image(center: Tuple[int, int], circle_scale: float, img: np.ndarray, target: str):
    """
    Write target on image and check if last frame of the animation.

    :param center: center of the circle and the text
    :param circle_scale: scale of the circle
    :param img: image to write data on
    :param target: char to write
    :return: True if last frame of the animation
    """
    text_size, _ = cv2.getTextSize(target, FONT, TEXT_SCALE, TEXT_THICKNESS)

    # print('radius: ', int(text_size[0] * 5 * circle_scale))
    # print('')
    cv2.circle(img, center, int(text_size[0] * 3.0 * circle_scale), (133, 21, 199),
               -1)  # 5, (112, 25, 25) (32, 32, 32), (100, 100, 100)
    # text_origin = (center[0] - text_size[0] // 2, center[1] + text_size[1] // 2)

    radius = 30
    horizontal_startx, horizontal_endx = center[0] - radius, center[0] + radius
    horizontal_starty, horizontal_endy = center[1], center[1]

    vertical_startx, vertical_endx = center[0], center[0]
    vertical_starty, vertical_endy = center[1] - radius, center[1] + radius
    cv2.line(img, (horizontal_startx, horizontal_starty), (horizontal_endx, horizontal_endy), (255, 255, 255),
             TEXT_THICKNESS)
    cv2.line(img, (vertical_startx, vertical_starty), (vertical_endx, vertical_endy), (255, 255, 255),
             TEXT_THICKNESS)

    end_animation_loop = circle_scale < 0.38  # random.uniform(0.1, 0.5)

    if not end_animation_loop:
        cv2.circle(img, center, 12, (238, 130, 238), -1)  # (17, 112, 170), (255, 0, 0)
    else:
        cv2.circle(img, center, 12, (255, 0, 255), -1)  # (252, 125, 11),

    # cv2.putText(img, target, text_origin, FONT, TEXT_SCALE, (17, 112, 170), TEXT_THICKNESS, cv2.LINE_AA)

    '''
    if not end_animation_loop:
        cv2.putText(img, target, text_origin, FONT, TEXT_SCALE, (17, 112, 170), TEXT_THICKNESS, cv2.LINE_AA)
    else:
        cv2.putText(img, target, text_origin, FONT, TEXT_SCALE, (252, 125, 11), TEXT_THICKNESS, cv2.LINE_AA)
    '''
    return end_animation_loop


def write_text_on_image(center: Tuple[int, int], circle_scale: float, img: np.ndarray, target: str):
    """
    Write target on image and check if last frame of the animation.

    :param center: center of the circle and the text
    :param circle_scale: scale of the circle
    :param img: image to write data on
    :param target: char to write
    :return: True if last frame of the animation
    """
    text_size, _ = cv2.getTextSize(target, FONT, TEXT_SCALE, TEXT_THICKNESS)
    # print('radius: ', int(text_size[0] * 5 * circle_scale))
    # print('')
    cv2.circle(img, center, int(text_size[0] * 3.0 * circle_scale), (100, 100, 100), -1)  # 5, (32, 32, 32)
    text_origin = (center[0] - text_size[0] // 2, center[1] + text_size[1] // 2)

    end_animation_loop = circle_scale < 0.38  # random.uniform(0.1, 0.5)
    if not end_animation_loop:
        cv2.putText(img, target, text_origin, FONT, TEXT_SCALE, (17, 112, 170), TEXT_THICKNESS, cv2.LINE_AA)
    else:
        cv2.putText(img, target, text_origin, FONT, TEXT_SCALE, (252, 125, 11), TEXT_THICKNESS, cv2.LINE_AA)

    return end_animation_loop


def get_random_position_on_screen(monitor_pixels: Tuple[int, int]) -> Tuple[int, int]:
    """
    Get random valid position on monitor.

    :param monitor_pixels: monitor dimensions in pixels
    :return: tuple of random valid x and y coordinated on monitor
    """
    return int(random.uniform(0, 1) * monitor_pixels[0]), int(random.uniform(0, 1) * monitor_pixels[1])


def get_fix_position_on_screen(monitor_pixels, distance_boundary_pixels, times):
    """
    Get fix valid position on monitor.

    :param monitor_pixels: monitor dimensions in pixels
    :param distance_boundary_pixels: distance_boundary: 50
    :param point_radius: raidus
    :param times: call back times from 0->n
    :return: tuple of fix valid x and y coordinated on monitor
    """
    W, H = monitor_pixels[0], monitor_pixels[1]
    m = distance_boundary_pixels
    w0, h0 = (W - 2 * m) / 3, (H - 2 * m) / 3

    rows, cols = times // 4, times % 4

    point_x, point_y = m + cols * w0, m + rows * h0

    return int(point_x), int(point_y)

    # return int(random.uniform(0, 1) * monitor_pixels[0]), int(random.uniform(0, 1) * monitor_pixels[1])


'''
def show_point_on_screen(window_name: str, base_path: str, monitor_pixels: Tuple[int, int], source: WebcamSource) -> Tuple[str, Tuple[int, int], float]:
    """
    Show one target on screen, full animation cycle. Return collected data if data is valid

    :param window_name: name of the window to draw on
    :param base_path: path where to save the image to
    :param monitor_pixels: monitor dimensions in pixels
    :param source: webcam source
    :return: collected data otherwise None
    """
    circle_scale = 1.
    # distance_boundary_pixels = 50  # 像素点距离边界的距离
    # point_radius = 30
    center = get_random_position_on_screen(monitor_pixels)
    # center = get_fix_position_on_screen(monitor_pixels, distance_boundary_pixels, times)
    end_animation_loop = False
    orientation = random.choice(list(TargetOrientation))

    file_name = None
    time_till_capture = None
    # print('1')
    while not end_animation_loop:
        image, circle_scale, end_animation_loop = create_image(monitor_pixels, center, circle_scale, orientation)
        cv2.imshow(window_name, image)

        for _ in range(10):  # workaround to not speed up the animation when buttons are pressed
            if cv2.waitKey(50) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                sys.exit()
    # print('2')
    if end_animation_loop:
        file_name = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        # print('ss: ', file_name)
        start_time_color_change = time.time()

        while time.time() - start_time_color_change < 0.5:
            # print('here 0!')
            # print('value: ', orientation.value)
            # c = cv2.waitKey(42)
            # print('c: ', c)
            # if c == orientation.value:
            if cv2.waitKey(42)& 0xFF == ord('z'):#orientation.value:
                # print('here 1!')
                source.clear_frame_buffer()
                cv2.imwrite(f'{base_path}/{file_name}.jpg', next(source))
                time_till_capture = time.time() - start_time_color_change
                break
    # print('3')
    cv2.imshow(window_name, np.zeros((monitor_pixels[1], monitor_pixels[0], 3), np.float32))
    cv2.waitKey(500)

    return f'{file_name}.jpg', center, time_till_capture
'''


# one point calibration
def show_center_point_on_screen(window_name, monitor_pixels):
    circle_scale = 1.
    distance_boundary_pixels = 50  # 像素点距离边界的距离
    # point_radius = 30
    # center = get_random_position_on_screen(monitor_pixels)
    # center = get_fix_position_on_screen(monitor_pixels, distance_boundary_pixels, times)
    center = (960, 540)
    end_animation_loop = False
    orientation = random.choice(list(TargetOrientation))

    file_name = None
    time_till_capture = None
    # print('1')
    while not end_animation_loop:
        # print('end_animation_loop 0: ', end_animation_loop)
        # print('circle_scale: ', circle_scale)
        image, circle_scale, end_animation_loop = create_image(monitor_pixels, center, circle_scale, orientation)
        cv2.imshow(window_name, image)
        # print('end_animation_loop 1: ', end_animation_loop)
        for _ in range(9):  # workaround to not speed up the animation when buttons are pressed
            if cv2.waitKey(50) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                sys.exit()
    return center


def show_Arbitrary_point_on_screen(window_name, monitor_pixels, point):
    circle_scale = 1.

    end_animation_loop = False
    orientation = random.choice(list(TargetOrientation))

    while not end_animation_loop:

        image, circle_scale, end_animation_loop = create_image(monitor_pixels, point, circle_scale, orientation)
        cv2.imshow(window_name, image)
        for _ in range(9):  # workaround to not speed up the animation when buttons are pressed
            if cv2.waitKey(50) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                sys.exit()
    return point


def show_fixpoint_on_screen(window_name, monitor_pixels, VideoCap, times, base_path=None):
    """
    Show one target on screen, full animation cycle. Return collected data if data is valid

    :param window_name: name of the window to draw on
    :param base_path: path where to save the image to
    :param monitor_pixels: monitor dimensions in pixels
    :param source: webcam source
    :return: collected data otherwise None
    """
    circle_scale = 1.
    distance_boundary_pixels = 50  # 50  # 像素点距离边界的距离
    # point_radius = 30
    # center = get_random_position_on_screen(monitor_pixels)
    center = get_fix_position_on_screen(monitor_pixels, distance_boundary_pixels, times)
    end_animation_loop = False
    orientation = random.choice(list(TargetOrientation))

    file_name = None
    time_till_capture = None
    # print('1')
    while not end_animation_loop:
        image, circle_scale, end_animation_loop = create_image(monitor_pixels, center, circle_scale, orientation,
                                                               target='+')
        cv2.imshow(window_name, image)

        for _ in range(9):  # workaround to not speed up the animation when buttons are pressed
            if cv2.waitKey(50) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                sys.exit()
    # print('2')
    '''
    if end_animation_loop:
        file_name = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        # print('ss: ', file_name)
        start_time_color_change = time.time()

        while time.time() - start_time_color_change < 0.5:
            # 这里可以计算眼动了
            # print('here 0!')
            # print('value: ', orientation.value)
            # c = cv2.waitKey(42)
            # print('c: ', c)
            # if c == orientation.value:
            if cv2.waitKey(42)& 0xFF == ord('z'):#orientation.value:
                # print('here 1!')
                source.clear_frame_buffer()
                # cv2.imwrite(f'{base_path}/{file_name}.jpg', next(source))
                time_till_capture = time.time() - start_time_color_change
                break
    # print('3')
    cv2.imshow(window_name, np.zeros((monitor_pixels[1], monitor_pixels[0], 3), np.float32))
    cv2.waitKey(500)
    '''
    return center


def get_fix_position_on_screen_for_calibration(monitor_pixels, radius):
    # radius  圆半径
    Points = []
    angle = 45 / 180 * math.pi
    w0, h0 = monitor_pixels[0] / 2, monitor_pixels[1] / 2
    p0 = (w0, h0)
    p1 = (w0, h0 - radius)
    # print('cos angle: ')
    p2 = (w0 + radius * math.cos(angle), h0 - radius * math.sin(angle))
    p3 = (w0 + radius, h0)
    p4 = (w0 + radius * math.cos(angle), h0 + radius * math.sin(angle))
    p5 = (w0, h0 + radius)
    p6 = (w0 - radius * math.cos(angle), h0 + radius * math.sin(angle))
    p7 = (w0 - radius, h0)
    p8 = (w0 - radius * math.cos(angle), h0 - radius * math.sin(angle))
    Points.append(p0)
    Points.append(p1)
    Points.append(p2)
    Points.append(p3)
    Points.append(p4)
    Points.append(p5)
    Points.append(p6)
    Points.append(p7)
    Points.append(p8)

    for i in range(len(Points)):
        Points[i] = (int(Points[i][0]), int(Points[i][1]))
    return Points


def get_fix_position_on_screen_for_calibration_2(monitor_pixels, distance_boundary_pixels, times):
    # radius  圆半径
    # Points = []
    # angle = 60
    W, H = monitor_pixels[0], monitor_pixels[1]
    m = distance_boundary_pixels
    w0, h0 = (W - 2 * m) / 2, (H - 2 * m) / 2

    rows, cols = times // 3, times % 3

    point_x, point_y = m + cols * w0, m + rows * h0

    return int(point_x), int(point_y)


def get_classification_point_position_on_screen(monitor_pixels, times):
    # radius  圆半径
    # Points = []
    # angle = 60
    W, H = monitor_pixels[0], monitor_pixels[1]
    # m = distance_boundary_pixels
    # w0, h0 = (W - 2 * m) / 2, (H - 2 * m) / 2

    w0, h0 = W / 4, H / 4

    start_point = W / 8, H / 8

    rows, cols = times // 4, times % 4

    point_x, point_y = start_point[0] + cols * w0, start_point[1] + rows * h0

    return int(point_x), int(point_y)


def get_fix_position_on_screen_for_head_eye_gaze(monitor_pixels, distance_boundary_pixels, times):
    # radius  圆半径
    # Points = []
    # angle = 60
    W, H = monitor_pixels[0], monitor_pixels[1]
    m = distance_boundary_pixels

    a = W / 4

    w0, h0 = (W * 0.5 - 2 * m) / 2, (H - 2 * m) / 3

    rows, cols = times // 3, times % 3

    point_x, point_y = a + m + cols * w0, m + rows * h0

    return int(point_x), int(point_y)


def show_fixpoint_on_screen_head_eye_gaze(window_name, monitor_pixels, times):
    """
    Show one target on screen, full animation cycle. Return collected data if data is valid

    :param window_name: name of the window to draw on
    :param base_path: path where to save the image to
    :param monitor_pixels: monitor dimensions in pixels
    :param source: webcam source
    :return: collected data otherwise None
    """
    circle_scale = 1.
    distance_boundary_pixels = 50  # 50  # 像素点距离边界的距离
    # point_radius = 30
    # center = get_random_position_on_screen(monitor_pixels)
    # center = get_fix_position_on_screen(monitor_pixels, distance_boundary_pixels, times)
    center = get_fix_position_on_screen_for_head_eye_gaze(monitor_pixels, distance_boundary_pixels, times)
    end_animation_loop = False
    orientation = random.choice(list(TargetOrientation))

    file_name = None
    time_till_capture = None
    # print('1')
    while not end_animation_loop:
        image, circle_scale, end_animation_loop = create_image(monitor_pixels, center, circle_scale, orientation)
        cv2.imshow(window_name, image)

        for _ in range(9):  # workaround to not speed up the animation when buttons are pressed
            if cv2.waitKey(50) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                sys.exit()
    return center


# 4 个点
def get_fix_position_on_screen_for_calibration_3(monitor_pixels, distance_boundary_pixels, times):
    # radius  圆半径
    # Points = []
    # angle = 60
    W, H = monitor_pixels[0], monitor_pixels[1]
    m = distance_boundary_pixels
    w0, h0 = (W - 2 * m) / 2, (H - 2 * m) / 2

    rows, cols = times // 3, times % 3

    point_x, point_y = m + cols * w0, m + rows * h0

    return int(point_x), int(point_y)
    # for i in range(len(Points)):
    #     Points[i] = (int(Points[i][0]), int(Points[i][1]))
    # return Points


def show_fixpoint_on_screen_for_calibration(window_name, monitor_pixels, times, base_path=None):
    """
    Show one target on screen, full animation cycle. Return collected data if data is valid

    :param window_name: name of the window to draw on
    :param base_path: path where to save the image to
    :param monitor_pixels: monitor dimensions in pixels
    :param source: webcam source
    :return: collected data otherwise None
    """
    circle_scale = 1.
    distance_boundary_pixels = 200  # 像素点距离边界的距离  200
    # point_radius = 30
    # center = get_random_position_on_screen(monitor_pixels)
    center = get_fix_position_on_screen_for_calibration_2(monitor_pixels, distance_boundary_pixels, times)
    # center = get_fix_position_on_screen(monitor_pixels, distance_boundary_pixels, times)
    end_animation_loop = False
    orientation = random.choice(list(TargetOrientation))

    file_name = None
    time_till_capture = None
    # print('1')
    while not end_animation_loop:
        image, circle_scale, end_animation_loop = create_image(monitor_pixels, center, circle_scale, orientation)
        cv2.imshow(window_name, image)

        for _ in range(9):  # workaround to not speed up the animation when buttons are pressed
            if cv2.waitKey(50) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                sys.exit()
    return center

    # return center


# from run_various_scale_one_point_eye_tracking import tracker

from saveGazeData import Data_saving


def show_fixpoint_on_screen_for_calibration_with_gaze(window_name, monitor_pixels, times):
    """
    Show one target on screen, full animation cycle. Return collected data if data is valid

    :param window_name: name of the window to draw on
    :param base_path: path where to save the image to
    :param monitor_pixels: monitor dimensions in pixels
    :param source: webcam source
    :return: collected data otherwise None
    """

    circle_scale = 1.
    distance_boundary_pixels = 250  # 200  # 像素点距离边界的距离  200
    # point_radius = 30
    # center = get_random_position_on_screen(monitor_pixels)
    center = get_fix_position_on_screen_for_calibration_2(monitor_pixels, distance_boundary_pixels, times)
    # center = get_fix_position_on_screen(monitor_pixels, distance_boundary_pixels, times)
    end_animation_loop = False
    orientation = random.choice(list(TargetOrientation))

    # file_name = None
    # time_till_capture = None
    # print('1')
    while not end_animation_loop:
        image, circle_scale, end_animation_loop = create_image(monitor_pixels, center, circle_scale, orientation)

        Map_location = Data_saving.get_mean_gaze()  # get_gaze()

        img = image.copy()
        if Map_location is not False:
            # print('start mapping!!!')
            cv2.circle(img, Map_location, 30, (255, 0, 255), thickness=-1)

        cv2.imshow(window_name, img)

        for _ in range(4):  # workaround to not speed up the animation when buttons are pressed
            if cv2.waitKey(20) & 0xFF == ord('q'):  # 50
                cv2.destroyAllWindows()
                sys.exit()
    return center
    # return center


def show_fixpoint_on_screen_for_calibration_with_gaze_new(window_name, monitor_pixels, times):
    """
    Show one target on screen, full animation cycle. Return collected data if data is valid

    :param window_name: name of the window to draw on
    :param base_path: path where to save the image to
    :param monitor_pixels: monitor dimensions in pixels
    :param source: webcam source
    :return: collected data otherwise None
    """

    circle_scale = 1.
    distance_boundary_pixels = 250  # 200  # 像素点距离边界的距离  200
    # point_radius = 30
    # center = get_random_position_on_screen(monitor_pixels)
    center = get_fix_position_on_screen_for_calibration_2(monitor_pixels, distance_boundary_pixels, times)
    # center = get_fix_position_on_screen(monitor_pixels, distance_boundary_pixels, times)
    end_animation_loop = False
    orientation = random.choice(list(TargetOrientation))

    # file_name = None
    # time_till_capture = None
    # print('1')
    t1 = cv2.getTickCount()
    while True:
        # while not end_animation_loop:
        t = (cv2.getTickCount() - t1) / cv2.getTickFrequency()

        if t >= 3:
            break
        image, circle_scale, end_animation_loop = create_image(monitor_pixels, center, circle_scale, orientation)
        # image, circle_scale, end_animation_loop = create_image_spider(monitor_pixels, center, circle_scale, orientation)

        Map_location = Data_saving.get_mean_gaze()  # get_gaze()

        img = image.copy()
        if Map_location is not False:
            # print('start mapping!!!')
            cv2.circle(img, Map_location, 30, (255, 0, 255), thickness=-1)

        cv2.imshow(window_name, img)

        for _ in range(4):  # workaround to not speed up the animation when buttons are pressed
            if cv2.waitKey(20) & 0xFF == ord('q'):  # 50
                cv2.destroyAllWindows()
                sys.exit()
    return center


def show_fixpoint_on_screen_for_calibration_beta2(window_name, monitor_pixels, times, base_path=None):
    """
    Show one target on screen, full animation cycle. Return collected data if data is valid

    :param window_name: name of the window to draw on
    :param base_path: path where to save the image to
    :param monitor_pixels: monitor dimensions in pixels
    :param source: webcam source
    :return: collected data otherwise None
    """
    circle_scale = 1.
    # distance_boundary_pixels = 135  # 50  # 像素点距离边界的距离
    # point_radius = 30
    # center = get_random_position_on_screen(monitor_pixels)

    # center = get_fix_position_on_screen(monitor_pixels, distance_boundary_pixels, times)
    center = get_classification_point_position_on_screen(monitor_pixels, times)
    print('center point: ', center)
    end_animation_loop = False
    # orientation = random.choice(list(TargetOrientation))
    orientation = TargetOrientation.RIGHT
    file_name = None
    time_till_capture = None
    # print('1')
    t1 = cv2.getTickCount()
    # while not end_animation_loop:
    while True:
        t = (cv2.getTickCount() - t1) / cv2.getTickFrequency()

        if t >= 3:
            break
        # image, circle_scale, end_animation_loop = create_image(monitor_pixels, center, circle_scale, orientation, target='+')
        image, circle_scale, end_animation_loop = create_image_spider(monitor_pixels, center, circle_scale, orientation)

        Map_location = Data_saving.get_mean_gaze()  # get_gaze()
        img = image.copy()

        start_point = int(center[0] - monitor_pixels[0] / 8), int(center[1] - monitor_pixels[1] / 8)
        end_point = int(center[0] + monitor_pixels[0] / 8), int(center[1] + monitor_pixels[1] / 8)
        cv2.rectangle(img, start_point, end_point, (0, 0, 255), 2)  # (255,0,0)蓝色

        if Map_location is not False:
            # print('start mapping!!!')
            cv2.circle(img, Map_location, 20, (255, 0, 255), thickness=-1)

        cv2.imshow(window_name, img)

        for _ in range(5):  # workaround to not speed up the animation when buttons are pressed
            if cv2.waitKey(50) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                sys.exit()
    return center


def show_fixpoint_on_screen_for_calibration_beta2_new(window_name, monitor_pixels, times, base_path=None):
    """
    Show one target on screen, full animation cycle. Return collected data if data is valid

    :param window_name: name of the window to draw on
    :param base_path: path where to save the image to
    :param monitor_pixels: monitor dimensions in pixels
    :param source: webcam source
    :return: collected data otherwise None
    """
    circle_scale = 1.
    # distance_boundary_pixels = 135  # 50  # 像素点距离边界的距离
    # point_radius = 30
    # center = get_random_position_on_screen(monitor_pixels)

    # center = get_fix_position_on_screen(monitor_pixels, distance_boundary_pixels, times)
    center = get_classification_point_position_on_screen(monitor_pixels, times)
    print('center point: ', center)
    end_animation_loop = False
    # orientation = random.choice(list(TargetOrientation))
    orientation = TargetOrientation.RIGHT
    file_name = None
    time_till_capture = None
    # print('1')

    img = cv2.imread('data/test.jpg')
    img = cv2.resize(img, (1920, 1080))
    t1 = cv2.getTickCount()
    while True:
        t = (cv2.getTickCount() - t1) / cv2.getTickFrequency()

        if t >= 3:
            break

        show_img = img.copy()
        Map_location = Data_saving.get_mean_gaze()  # get_gaze()

        # while not end_animation_loop:
        # image, circle_scale, end_animation_loop = create_image(monitor_pixels, center, circle_scale, orientation, target='+')
        # image, circle_scale, end_animation_loop = create_image_spider(monitor_pixels, center, circle_scale, orientation)

        # Map_location = Data_saving.get_mean_gaze()  # get_gaze()
        # img = image.copy()

        start_point = int(center[0] - monitor_pixels[0] / 8), int(center[1] - monitor_pixels[1] / 8)
        end_point = int(center[0] + monitor_pixels[0] / 8), int(center[1] + monitor_pixels[1] / 8)

        cv2.rectangle(show_img, start_point, end_point, (0, 0, 255), 3)  # (255,0,0)蓝色

        # cv2.circle(show_img, center, 30, (255, 0, 0), 3)  # (255,0,0)蓝色

        if Map_location is not False:
            # print('start mapping!!!')
            cv2.circle(show_img, Map_location, 50, (255, 0, 255), thickness=-1)

        cv2.imshow(window_name, show_img)

        for _ in range(5):  # workaround to not speed up the animation when buttons are pressed
            if cv2.waitKey(20) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                sys.exit()
    return center


def show_fixpoint_on_screen_for_calibration_beta2_new_2(window_name, monitor_pixels, times, base_path=None):
    circle_scale = 1.
    center = get_classification_point_position_on_screen(monitor_pixels, times)
    print('center point: ', center)
    end_animation_loop = False
    # orientation = random.choice(list(TargetOrientation))
    orientation = TargetOrientation.RIGHT
    file_name = None
    time_till_capture = None
    # print('1')

    # img = cv2.imread('data/test.jpg')
    # img = cv2.resize(img, (1920, 1080))
    t1 = cv2.getTickCount()
    while True:
        t = (cv2.getTickCount() - t1) / cv2.getTickFrequency()

        if t >= 3:
            break

        # show_img = img.copy()
        Map_location = Data_saving.get_mean_gaze()  # get_gaze()

        # image, circle_scale, end_animation_loop = create_image(monitor_pixels, center, circle_scale, orientation, target='+')
        image, circle_scale, end_animation_loop = create_image_spider(monitor_pixels, center, circle_scale, orientation)

        # Map_location = Data_saving.get_mean_gaze()  # get_gaze()
        # img = image.copy()

        start_point = int(center[0] - monitor_pixels[0] / 8), int(center[1] - monitor_pixels[1] / 8)
        end_point = int(center[0] + monitor_pixels[0] / 8), int(center[1] + monitor_pixels[1] / 8)

        show_img = image.copy()
        cv2.rectangle(show_img, start_point, end_point, (0, 0, 255), 3)  # (255,0,0)蓝色

        # cv2.circle(show_img, center, 30, (255, 0, 0), 3)  # (255,0,0)蓝色

        if Map_location is not False:
            # print('start mapping!!!')
            cv2.circle(show_img, Map_location, 20, (0, 0, 255), thickness=3)  # (255, 0, 255)

        cv2.imshow(window_name, show_img)

        for _ in range(5):  # workaround to not speed up the animation when buttons are pressed
            if cv2.waitKey(20) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                sys.exit()
    return center


def show_fixpoint_on_screen_for_calibration_beta3(window_name, monitor_pixels, times, base_path=None):
    """
    Show one target on screen, full animation cycle. Return collected data if data is valid

    :param window_name: name of the window to draw on
    :param base_path: path where to save the image to
    :param monitor_pixels: monitor dimensions in pixels
    :param source: webcam source
    :return: collected data otherwise None
    """
    circle_scale = 1.
    distance_boundary_pixels = 135  # 50  # 像素点距离边界的距离
    # point_radius = 30
    # center = get_random_position_on_screen(monitor_pixels)
    # center = get_fix_position_on_screen(monitor_pixels, distance_boundary_pixels, times)

    radius = 400
    centers = get_fix_position_on_screen_for_calibration(monitor_pixels, radius)

    end_animation_loop = False
    orientation = random.choice(list(TargetOrientation))

    file_name = None
    time_till_capture = None
    # print('1')
    while not end_animation_loop:
        image, circle_scale, end_animation_loop = create_image(monitor_pixels, centers[times], circle_scale,
                                                               orientation)
        cv2.imshow(window_name, image)

        for _ in range(9):  # workaround to not speed up the animation when buttons are pressed
            if cv2.waitKey(50) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                sys.exit()
    return centers[times]


def generate_calibration_points(width, height):
    # point_list = []
    p_1 = width / 4, height / 4
    p_2 = 3 * width / 4, height / 4
    p_3 = width / 4, 3 * height / 4
    p_4 = 3 * width / 4, 3 * height / 4
    point_1 = [p_1, p_2, p_3, p_4]

    w2, h2 = width / 8, height / 8

    point_list = []
    for i in range(4):
        p_center = point_1[i]
        p_center_1 = p_center[0] - w2, p_center[1] - h2
        p_center_2 = p_center[0] + w2, p_center[1] - h2
        p_center_3 = p_center[0] - w2, p_center[1] + h2
        p_center_4 = p_center[0] + w2, p_center[1] + h2
        point_2 = [p_center, p_center_1, p_center_2, p_center_3, p_center_4]
        point_list.append(point_2)

    return point_list


def get_show_point(point_list, count):
    m = count // 5
    n = count % 5
    if n == 0:
        return True, point_list[m][n]  # 需要等头部指向这里，并进入稳定跟踪状态
    else:
        return False, point_list[m][n]

