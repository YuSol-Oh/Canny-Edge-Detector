# Canny-Edge-Detector
컴퓨터비전 - Edge detector 구현해보기

(1) Gray Scale 변환 : 컬러 이미지를 그레이 스케일로 변환

- 입력 이미지를 gray scale로 변환하는 것.
- gray scale 이미지 : 컬러 정보가 없는 흑백 이미지로, edge detect에 적합한 형태로 전환.
```python
# (1) 그레이스케일 변환 ------------------------------------------------------
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
```

(2) 가우시안 블러링 : edge 검출 전, 이미지를 부드럽게 만들기 위해 가우시안 블러 적용
- gray scale 이미지에 가우시안 블러를 적용하여 이미지의 노이즈를 제거
- 가우시안 블러 : 이미지의 부드러운 형태를 유지하면서 edge detect를 더 정확하게 만들어줌.
```python
# (2) 가우시안 블러 -----------------------------------------------------------
# 1. 가우시안 커널 : 가우시안 필터를 반환
def gaussian_kernel(size, sigma): # (필터 크기-홀수, 가우시안 분포 표준편차)

    kernel = np.fromfunction( # 주어진 크기 -> 2D 배열 생성
        lambda x, y: (1/ (2 * np.pi * sigma ** 2)) * np.exp(- ((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)), # 각 요소 (x,y)에 대한 가우시안 값을 계산
        (size, size)
    )
    return kernel / np.sum(kernel) # 정규화 (총 합 1)
# 2. 이미지에 커널을 적용 : 필터(커널)가 적용된 이미지 반환
def convolve(image, kernel): # (입력 이미지, 가우시안 커널)

    result = np.zeros_like(image, dtype=float) # 결과를 저장할 이미지와 동일한 크기의 0으로 초기화된 배열
    image_padded = np.pad(image, pad_width=(kernel.shape[0]//2, kernel.shape[1]//2), mode='constant', constant_values=0) # 입력 이미지 주변을 0으로 패딩한 이미지 생성

    # 각 픽셀에 커널 적용
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            # 현재 픽셀 위치에서 커널 크기만큼의 부분 이미지 추출 -> 각 요소에 커널을 곱한 값들의 합 계산
            result[i, j] = np.sum(image_padded[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)

    return result.astype(np.uint8) # 결과를 정수형으로 변환하여 반환
# 3. 가우시안 블러 적용 : 가우시안 블러가 적용된 이미지 반환
def gaussian_blur(image, kernel_size, sigma): # (입력 이미지, 필터 크기-홀수, 가우시안 분포 표준편차)

    kernel = gaussian_kernel(kernel_size, sigma)
    return convolve(image, kernel)
```

(3) Gradient 및 edge 강도 계산 : 이미지에서 강한 edge를 찾기 위해 gradient를 계산
- 가우시안 블러가 적용된 이미지에서 Sobel 필터를 사용하여 각 픽셀에서의 gradient와 edge 강도를 계산.
- sobel 필터 : 수평 및 수직 방향의 변화를 측정하여 이미지의 edge를 찾음
```python
# (3) Gradient 및 Edge 강도 계산 -------------------------------------------
def calculate_gradients(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3) # 수평 방향의 edge magnitude 계산
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3) # 수직 방향의 edge magnitude 계산
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2) # edge 강도의 크기 계산 (이미지의 각 픽셀에서의 gradient 크기)
    angle = np.arctan2(sobel_y, sobel_x) # edge 강도의 방향 계산 (이미지의 각 픽셀에서의 gradient 방향)
    return magnitude, angle
```

(4) Non-Maximum Suppression (NMS, 비최대 억제) : edge가 아닌 픽셀을 제거하여 억제
- gradient 방향을 기반으로 edge 픽셀을 선택하는 방법.
- 일반적으로, edge 픽셀은 gradient 크기가 상대적으로 큰 지점이며, 그 방향은 edge의 방향을 나타냄. 하지만 실제 이미지에서는 gradient의 크기가 큰 지점 주변에도 여러 픽셀이 존재할 수 있다. 이때, 비최대 억제는 각 픽셀의 gradient 방향을 확인하고, 그 방향으로 gradient 크기가 가장 큰 픽셀만을 선택하고 나머지는 억제하는 방식으로 작동.

① 각 픽셀에서의 gradient 크기와 방향을 계산

② 픽셀 주변의 두 방향을 선택하여 현재 픽셀의 gradient 크기와 비교

③ 만약 현재 픽셀의 gradient 크기가 주변의 두 픽셀보다 크다면 해당 픽셀은 edge로 판단하고 값을 유지. 그렇지 않으면 현재 픽셀의 값을 0으로 설정하여 억제.

⇒ 이미지에서 중복된 edge가 줄어들고, 보다 명확한 edge 정보를 얻을 수 있음. 주로 Canny edge 검출과 같은 알고리즘에서 비최대 억제 단계가 수행됨.
```python
# (4) Non-Maximum Suppression (비 최대 억제) --------------------------------------------
def non_maximum_suppression(magnitude, angle):
    suppressed_magnitude = np.zeros_like(magnitude) # 결과 이미지 초기화 (입력 이미지와 동일 크기)
    angle = np.rad2deg(angle) % 180 # 각 픽셀의 gradient 방향을 0도에서 180도로 변환
    # 이미지 내부의 픽셀에 대해 반복
    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            q = 255
            r = 255
            # gradient 방향에 따라 주변의 두 픽셀을 선택
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            elif 22.5 <= angle[i, j] < 67.5:
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            elif 67.5 <= angle[i, j] < 112.5:
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]
            # 현재 픽셀이 주변의 픽셀보다 크면, 그 값을 결과 이미지에 저장
            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                suppressed_magnitude[i, j] = magnitude[i, j]
    return suppressed_magnitude
```

(5) 이중 임계값 적용 : 픽셀을 강한 edge, 약한 edge, 또는 edge가 아닌 픽셀로 분류
- edge 픽셀을 두 가지 강도로 나누는 기법.
- edge 픽셀을 강한 edge와 약한 edge로 분류하여 후속 처리에 활용.

① 이미지의 모든 픽셀에 대해 두 개의 임계값을 설정 → 이 임계값을 기준으로 픽셀이 강한 edge인지, 약한 edge인지를 구분.

② 임계값보다 큰 gradient 크기를 가지는 픽셀은 강한 edge로 설정. 낮은 임계값과 높은 임계값 사이에 있는 gradient 크기를 가지는 픽셀은 약한 edge로 설정.

③ 강한 edge와 약한 edge로 이루어진 이진 edge 맵을 생성.

⇒ edge를 구분하여 높은 신뢰도의 강한 edge와 상대적으로 낮은 신뢰도의 약한 edge로 나누어 줌으로써, 특히 노이즈가 많은 이미지에서 정확한 edge를 추출하는 데 도움을 줌.
```python
# (5) 이중 임계값 적용 ------------------------------------------
def double_threshold(image, low_threshold, high_threshold):
    strong_edges = (image > high_threshold) # 강한 edge -> 임계값을 초과하는 픽셀
    weak_edges = (image >= low_threshold) & (image <= high_threshold) # 약한 edge -> 낮은 임계값과 높은 임계값 사이의 픽셀로 설정
    return strong_edges, weak_edges
```

(6) Edge Tracking by Hysteresis (연결된 edge 추적) : 약한 edge를 강한 edge에 연결
- 이중 임계값을 적용한 후, 강한 edge와 연결된 약한 edge를 찾기 위해 연결된 edge 추적을 수행.
- Depth-First Search(DFS)를 사용하여 강한 edge와 연결된 약한 edge를 찾아 최종적인 edge 맵을 생성
```python
# (6) 연결된 엣지 추적 --------------------------------------------
def edge_tracking_by_hysteresis(strong_edges, weak_edges):
    visited = np.zeros_like(strong_edges, dtype=bool) # 방문 여부를 나타내는 배열 초기화
    stack = [] # DFS에 사용할 스택 초기화
    # 이미지 내부의 픽셀에 대해 반복
    for i in range(1, strong_edges.shape[0] - 1):
        for j in range(1, strong_edges.shape[1] - 1):
            # 강한 edge 픽셀을 찾고, 방문하지 않았다면 스택에 추가
            if strong_edges[i, j] and not visited[i, j]:
                stack.append((i, j))
                # 스택이 빌 때까지 반복
                while stack:
                    # 스택에서 픽셀을 꺼내어 방문 표시
                    current_i, current_j = stack.pop()
                    visited[current_i, current_j] = True
                    # 현재 픽셀 주변의 픽셀에 대해 반복
                    for x in range(current_i-1, current_i+2):
                        for y in range(current_j-1, current_j+2):
                            # 방문하지 않은 약한 edge 픽셀이면 스택에 추가
                            if not visited[x, y] and weak_edges[x, y]:
                                stack.append((x, y))

    return visited
```
```python
# Canny 엣지 검출기 ------------------------------
def canny_edge_detector(image, low_threshold, high_threshold, kernel_size=5):
    gray = grayscale(image) # 입력 이미지 -> gray scale
    blurred = gaussian_blur(gray, kernel_size=5, sigma=1.0) # 가우시안 블러 적용 -> 노이즈 감소
    magnitude, angle = calculate_gradients(blurred) # gradient 계산
    suppressed_magnitude = non_maximum_suppression(magnitude, angle) # 비최대 억제를 적용 -> edge 픽셀 강조
    strong_edges, weak_edges = double_threshold(suppressed_magnitude, low_threshold, high_threshold) # 이중 임계값을 적용하여 강한 edge와 약한 edge 추적
    edges = edge_tracking_by_hysteresis(strong_edges, weak_edges) # 강한 edge와 연결된 약한 edge 추적
    return edges.astype(np.uint8) * 255 # 최종적인 edge map을 반환하고 0 또는 255 값으로 픽셀 강도 조절
```

▶ 결과
```python
# 결과
import os
import matplotlib.pyplot as plt

image_folder = './컴퓨터비전/이미지'
image_files = [f'image_{i}.jpg' for i in range(1, 14)]

for image_file in image_files:
  # 이미지 불러오기
  image_path = os.path.join(image_folder, image_file)
  image = cv2.imread(image_path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # Canny 엣지 검출 수행
  canny_edges = canny_edge_detector(image, low_threshold=50, high_threshold=150, kernel_size = 5)

  # 시각화
  plt.figure(figsize=(12, 6))

  plt.subplot(1, 2, 1)
  plt.imshow(image)
  plt.title('Original Image')
  plt.axis('off')

  plt.subplot(1, 2, 2)
  plt.imshow(canny_edges, cmap='gray')
  plt.title('Canny Edge Detection')
  plt.axis('off')

  plt.show()
```
① 직선 edge를 가진 객체 이미지와, 객체와 배경의 색감 대비가 명확한 이미지에 대해서는 edge detect가 작은 디테일들 까지도 선명하게 잘 되었다. 

![image](https://github.com/YuSol-Oh/Canny-Edge-Detector/assets/77186075/7b15470e-9c0f-4c3a-bdfd-e4e52be4c206)
![image](https://github.com/YuSol-Oh/Canny-Edge-Detector/assets/77186075/7b0733db-f7c5-4b92-a626-35e1188692ae)
![image](https://github.com/YuSol-Oh/Canny-Edge-Detector/assets/77186075/9d2dec03-109c-460e-b7bc-7f7035ea0f73)
![image](https://github.com/YuSol-Oh/Canny-Edge-Detector/assets/77186075/ec77da8b-8347-4073-ac01-8f21b532073e)
![image](https://github.com/YuSol-Oh/Canny-Edge-Detector/assets/77186075/a1ca83af-b6bf-4db2-95f3-32997cc9571e)
![image](https://github.com/YuSol-Oh/Canny-Edge-Detector/assets/77186075/e67ca67f-6816-444b-8528-8769e80b3478)
![image](https://github.com/YuSol-Oh/Canny-Edge-Detector/assets/77186075/819697b5-a1f2-4938-b265-7adcc8012411)

② 하지만, 곡선 edge를 가지는 객체 이미지이거나, 객체가 배경과 비슷한 색을 가진 경우에는 edge detect가 불분명하게 되는 것을 확인할 수 있었다.

![image](https://github.com/YuSol-Oh/Canny-Edge-Detector/assets/77186075/13181a51-aca6-4cce-966b-e5e5bfca2f79)
![image](https://github.com/YuSol-Oh/Canny-Edge-Detector/assets/77186075/5fe36fdb-c515-4c9d-bd3d-ee99ff00ad8d)
