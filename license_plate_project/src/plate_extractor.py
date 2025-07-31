import cv2
import numpy as np
import datetime
import os

# 이미지 폴더 및 저장 폴더 경로 설정
IMG_PATH = '../img'
SAVE_PATH = '../extracted_plates'

# 이미지 불러오기
#img = cv2.imread('../img/car_05.jpg')
#draw = img.copy()

# car_01.jpg ~ car_05.jpg 파일 리스트 생성
img_files = [f"car_{i:02d}.jpg" for i in range(1,6)]

# 클린한 점의 좌표를 저장할 배열과 개수 카운터
pts = np.zeros((4, 2), dtype=np.float32)
pts_cnt = 0
current_idx = 0     # 현재 작업 중인 이미지 번호

# 마우스 이벤트 콜백 함수
def onMouse(event, x, y, flags, param):
    global pts_cnt, draw, pts, img, current_idx
    
    # 마우스 왼쪽 버튼 클릭 시 동작
    if event == cv2.EVENT_LBUTTONDOWN:
        # 클릭한 좌표에 빨간 점 표시
        cv2.circle(draw, (x, y), 5, (0, 0, 255), -1)
        
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

            # 번호판 이미지를 그레이스케일로 변환
            gray_plate = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

            # 바이레터럴 블러 적용
            blur = cv2.bilateralFilter(gray_plate, 9, 75, 75)

            # 캐니 엣지 적용
            edges = cv2.Canny(blur, 100, 200)

            # 경계 강조 (이진화)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            ## 파일 이름을 타임스탬프로 생성
            #timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            #filename = f"{SAVE_PATH}/plate_{timestamp}.png"

            ## 추출된 번호판 이미지를 파일로 저장
            #cv2.imwrite(filename, result)
            #print(f"저장 완료 : {filename}")

            # 필터링을 거친 최종 추출 번호판 이미지 저장
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{SAVE_PATH}/plate_{timestamp}.png"
            cv2.imwrite(filename, thresh)
            print(f"저장 완료 : {filename}")

            # 추출된 번호판 이미지를 새 창에 표시
            cv2.imshow("Extracted Plate", thresh)

            # 다음 이미지로 이동
            current_idx += 1
            if current_idx < len(img_files):
                load_next_image()
            else:
                print("모든 이미지 작업 완료")
                cv2.destroyAllWindows()

def load_next_image():
    # 다음 이미지를 로드하고 초기화
    global img, draw, pts_cnt, pts
    img_path = os.path.join(IMG_PATH, img_files[current_idx])
    img = cv2.imread(img_path)
    draw = img.copy()
    pts_cnt = 0
    pts = np.zeros((4, 2), dtype=np.float32)
    cv2.imshow("License Plate Extractor", draw)

# 첫 번째 이미지 로드
load_next_image()

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