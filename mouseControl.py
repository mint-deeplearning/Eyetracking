# import pynput.mouse as mouse
import time
# import queue
import collections

# double_click_interval = 0.25

class Mouse_Control:
    def __init__(self, double_click_interval=0.25):
        self.interval = double_click_interval
        self.click_time = collections.deque(maxlen=10)
        self.double_c = False
    def on_click(self, x, y, button, pressed):
        if pressed:
            # print('鼠标点击在 ({0}, {1})'.format(x, y))
            self.click_time.append(time.time())

            if len(self.click_time) > 1:  # 保存的单机次数大于2
                interval = self.click_time[-1] - self.click_time[-2]
                # print('interval: ', interval)
                if interval < self.interval:
                    print('双击一次!!!!!')
                    self.double_c = True

    def get_double_click(self):
        return self.double_c

# listener = mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
# MC = Mouse_Control()
# listener = mouse.Listener(on_click=MC.on_click)
# # 启动监听器
# listener.start()
# # 阻塞主线程，使监听器持续监听
# listener.join()