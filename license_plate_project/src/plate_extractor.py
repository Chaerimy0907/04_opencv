import cv2
import numpy as np
import datetime

SAVE_PATH = '../extracted_plates'

img = cv2.imread('../img/car_01.jpg')
draw = img.copy()

pts = np.zeros((4, 2), dtype=np.float32)
pts_cnt = 0

def onMouse(event, x, y, flags, param):     # 마우스 이벤트 콜백 함수 구현
    global pts_cnt                          # 마우스로 찍은 좌표의 개수 저장

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(draw, (x, y), 5, (0, 0, 255), -1)       # 좌표에 빨간색 동그라미 표시
        pts[pts_cnt] = [x, y]       # 마우스 좌표 저장
        pts_cnt += 1

cv2.namedWindow('License Plate Extractor')
cv2.setMouseCallback('License Plate Extractor', onMouse)