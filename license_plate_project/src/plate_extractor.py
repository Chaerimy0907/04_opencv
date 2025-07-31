import cv2
import numpy as np
import datetime

# 이미지 폴더 및 저장 폴더 경로 설정
IMG_PATH = '../img'
SAVE_PATH = '../extracted_plates'

# 이미지 불러오기
img = cv2.imread('../img/car_01.jpg')
draw = img.copy()
#img_files = [f"car_{i:02d}.jpg" for i in range(1,6)]

# 클린한 점의 좌표를 저장할 배열과 개수 카운터
pts = np.zeros((4, 2), dtype=np.float32)
pts_cnt = 0

# 마우스 이벤트 콜백 함수
def onMouse(event, x, y, flags, param):
    global pts_cnt, draw, pts
    
    # 마우스 왼쪽 버튼 클릭 시 동작
    if event == cv2.EVENT_LBUTTONDOWN:
        # 클릭한 좌표에 빨간 점 표시
        cv2.circle(draw, (x, y), 10, (0, 0, 255), -1)
        
        # 클릭한 좌표를 저장
        pts[pts_cnt] = [x, y]
        pts_cnt += 1

        # 4개의 점을 모두 클릭했을 때 실행
        if pts_cnt == 4:
            # 좌표 정렬 알고리즘
            sm = pts.sum(axis = 1)          # 각 점의 x+y 값 계산
            diff = np.diff(pts, axis = 1)   # 각 점의 x-y 값 계산

            topLeft = pts[np.argmin(sm)]        # x+y 최소값 -> 좌상단
            bottomRight = pts[np.argmax(sm)]    # x+y 최대값 -> 우하단
            topRight = pts[np.argmin(diff)]     # x-y 최소값 -> 우상단
            bottomLeft = pts[np.argmax(diff)]   # x-y 최대값 -> 좌하단
            
            # 정렬된 좌표 배열 생성
            rect = np.array([topLeft, topRight, bottomRight, bottomLeft], dtype=np.float32)

            if rect.shape != (4, 2):
                pts_cnt = 0
                draw = img.copy()
                return
            
            width, height = 300, 150
            dst_pts = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)

            # 원근 변환 행렬 계산
            mtrx = cv2.getPerspectiveTransform(rect, dst_pts)

            # 원근 변환 적용
            result = cv2.warpPerspective(img, mtrx, (width, height))

            # 파일 이름을 타임스탬프로 생성
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{SAVE_PATH}/plate_{timestamp}.png"

            # 추출된 번호판 이미지를 파일로 저장
            cv2.imwrite(filename, result)
            print(f"저장 완료 : {filename}")

            # 추출된 번호판 이미지를 새 창에 표시
            cv2.imshow("Extracted Plate", result)

            # 다음 클릭을 위해 점 개수와 이미지 초기화
            pts_cnt = 0
            draw = img.copy()
    
# 윈도우 생성 및 마우스 콜백 등록
cv2.namedWindow('License Plate Extractor')
cv2.setMouseCallback('License Plate Extractor', onMouse)

# 메인 루프
while True:
    cv2.imshow("License Plate Extractor", draw)     # 점이 찍힌 이미지를 표시
    key = cv2.waitKey(1) & 0xFF

    # q를 누르면 종료
    if key == ord('q'):
        break

cv2.destroyAllWindows()