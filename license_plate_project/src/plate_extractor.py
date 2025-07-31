import cv2
import numpy as np
import datetime

IMG_PATH = '.../img/car_01.jpg'
SAVE_PATH = '../extracted_plates'

img = cv2.imread(IMG_PATH)
draw = img.copy()

pts = np.zeros((4, 2), dtype=np.float32)
pts_cnt = 0