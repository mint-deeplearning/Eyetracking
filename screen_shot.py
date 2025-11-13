# from mss import mss
import numpy as np
import cv2
# def get_screen_shot():
#     with mss() as sct:
#         monitor = sct.monitors[1]
#         sct_img = sct.grab(monitor)
#         return sct_img


def post_screen_shot(im):
    frame = np.array(im, dtype=np.uint8)
    return np.flip(frame[:, :, :3], 2)


def get_game_screen_shot():
    # grab = post_screen_shot(get_screen_shot())
    grab = np.zeros((1080,1920), dtype=np.int8)
    grab = cv2.cvtColor(grab, cv2.COLOR_BGR2RGB)
    # print('grab shape: ', grab.shape)
    grab = grab[220:860, 320:1600]
    grab = cv2.pyrDown(grab)
    # grab = cv2.resize(grab, (640, 320))
    # print('grab shape: ', grab.shape)
    return grab

if __name__ == '__main__':
    grabImg = get_game_screen_shot()
    cv2.imshow('test.jpg', grabImg)
    cv2.waitKey(0)
