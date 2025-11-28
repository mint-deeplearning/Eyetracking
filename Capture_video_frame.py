import PySimpleGUI as sg
import cv2
import threading

layout = [
    [sg.Text('照相机: ')],
    [sg.Image(key="-IMGSRC-",size=(640, 480))],
    [sg.Button('拍照')],
    [sg.Button('录像'), sg.Text('', key='-TIME-'), sg.Button('停止录像', disabled=True)]
]

window = sg.Window('Python GUI', layout, keep_on_top=True)
mutex = threading.Lock()

def compute_time(start_time, end_time):
    return (end_time-start_time)/cv2.getTickFrequency()

def set_window_img(imgbytes):
    if imgbytes is not None:
        state = False
        for i in range(5):
            if window.is_close():
                state = True
                break
        if not state:
            window['-IMGSRC-'].update(data=imgbytes)

class CaptureThread(threading.Thread):
    def __init__(self, number=0):
        super().__init__()
        self.camera_id = number
        self.isCapture = True

    def set_iscapture(self, True_or_False):
        mutex.acquire()
        self.isCapture = True_or_False
        mutex.release()

    def get_iscapture(self):
        return self.isCapture

    def run(self):
        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        start_time = cv2.getTickCount()
        while self.isCapture:
            ret, img = cap.read()
            if ret:
                img = cv2.resize(img, (640, 480))
                imgbytes = cv2.imencode('.png', img)[1].tobytes()
                if self.get_iscapture():
                    set_window_img(imgbytes)
                    end_time = cv2.getTickCount()
                    elapse = compute_time(start_time, end_time)
                    string = str(int(elapse))+ ' 秒'
                    # window['-TIME-'].update(datetime.now().strftime("%H:%M:%S"))
                    window['-TIME-'].update(string)
        cap.release()
        print('thread capture stopped!')

while True:
    event, values = window.read()
    if event==None:
        break
    if event=='拍照':
        print('take a photo')
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        ret, img = cap.read()
        img = cv2.resize(img, window['-IMGSRC-'].get_size())
        imgbytes = cv2.imencode('.png', img)[1].tobytes()
        window['-IMGSRC-'].update(data=imgbytes)
        cap.release()
    if event=='录像':
        print('record')
        CaptureThr = CaptureThread(0)
        CaptureThr.start()
        # window['-TIME-'].update(datetime.now().strftime("%H:%M:%S"))
        window['停止录像'].update(disabled=False)
    if event=='停止录像':
        CaptureThr.set_iscapture(False)
        window['停止录像'].update(disabled=True)
window.close()
