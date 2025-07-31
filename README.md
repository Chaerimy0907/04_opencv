# 번호판 이미지 처리 + 숫자 윤곽선 검출

## 개요
- 자동차 이미지에서 번호판을 **마우스로 4점 클릭해 추출**
- 클릭한 영역을 **원근 변환(Perspective Transform)** 으로 보정
- 이진화 및 형태학적 연산을 통해 **글자 강조**
- 번호판의 **숫자/문자에 윤곽선 표시** 및 결과 시각화

---

## 사용 기술
- OpenCV
- Matplotlib
- Numpy

---

## 실행 흐름
1. 코드 실행 -> car_01.jpg 표시
2. 번호판 4개의 점 클릭 -> 번호판 추출 및 저장
3. 윤곽선 결과 시각화
4. 자동으로 다음 이미지 로드
5. 모든 이미지 처리 완료 후 종료

---

## 주요 흐름

### 1. 이미지 로드 및 마우스 이벤트
- 번호판 이미지를 한 장씩 자동으로 로드
```python
def load_next_image():
    # 다음 이미지를 로드하고 초기화
    global img, draw, pts_cnt, pts
    img_path = os.path.join(IMG_PATH, img_files[current_idx])
    img = cv2.imread(img_path)
    draw = img.copy()
    pts_cnt = 0
    pts = np.zeros((4, 2), dtype=np.float32)
    cv2.imshow("License Plate Extractor", draw)
```
- 마우스로 번호판의 4개 모서리 클릭
```python
# 마우스 이벤트 콜백 함수
def onMouse(event, x, y, flags, param):

cv2.setMouseCallback('License Plate Extractor', onMouse)
```

## 2. 번호판 추출 (원근 변환)
- 추출된 번호판 이미지는 extracted_plates(원본)
```python
# 원근 변환 행렬 계산
mtrx = cv2.getPerspectiveTransform(rect, dst_pts)

# 원근 변환 적용
result = cv2.warpPerspective(img, mtrx, (width, height))
```

## 3. 전처리 (이진화 + 형태학적 연산)
- 이미지를 그레이스케일로 변환 후 바이레터럴 블러 적용
- 바이레터럴 블러 적용한 결과에 이진화 적용
- 필터링된 이미지는 processed_plates(필터링 후) 폴더에 저장됨
```python
# 번호판 이미지를 그레이스케일로 변환
gray_plate = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

# 바이레터럴 블러 적용
blur = cv2.bilateralFilter(gray_plate, 9, 75, 75)

# 경계 강조 (이진화)
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

## 4. 윤곽선 검출 및 시각화 (검출 안 됨.. 수정 필요)
- 글자 크기와 비율 조건을 완화해 숫자와 문자의 윤곽선을 검출
- Matplotlib으로 이진화 결과 + 윤곽선 검출 결과를 동시에 출력
```python
# 윤곽선 검출
contours, _ = cv2.findContours(thresh_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
  area = cv2.contourArea(contour)
  x, y, w, h = cv2.boundingRect(contour)

  if area > 20 and h > 10:
    filtered_contours.append(contour)
```

---

## 실행 결과 예시
<img width="990" height="461" alt="result" src="https://github.com/user-attachments/assets/142df306-0020-473c-a7a6-43850d55371d" />
