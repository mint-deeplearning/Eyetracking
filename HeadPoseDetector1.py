# Import SixDRepNet
from sixdrepnet import SixDRepNet
import cv2


class HeadPoseDetector_1:
    def __init__(self):
        self.model = SixDRepNet()#gpu_id=-1

    def process_img(self, image):
        pitch, yaw, roll = self.model.predict(image)

        return True, (pitch, -yaw, roll)


# Create model
# Weights are automatically downloaded
# model = SixDRepNet()
''' 
HP = HeadPoseDetector_1()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # pitch, yaw, roll = model.predict(frame)
    status, res = HP.process_img(frame)
    print('res: ', res)
    # model.draw_axis(frame, yaw, pitch, roll)
    cv2.imshow("test_window", frame)
    cv2.waitKey(20)
'''