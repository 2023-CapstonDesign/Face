from keras_facenet import FaceNet
from mtcnn import MTCNN
import PIL
import numpy as np

# FaceNet 모델 초기화
embedder = FaceNet()

# MTCNN 초기화
detector = MTCNN()

# 입력 이미지 로드
image1 = PIL.Image.open('pss.jpg').convert('RGB')
image2 = PIL.Image.open('dong.jpg').convert('RGB')

# 이미지를 numpy 배열로 변환
pixel1 = np.asarray(image1)
pixel2 = np.asarray(image2)

# 얼굴 감지 및 얼굴 영역 추출
result1 = detector.detect_faces(pixel1)
result2 = detector.detect_faces(pixel2)

# 첫 번째 이미지에서 얼굴추출
x1, y1, w1, h1 = result1[0]['box']
face1 = pixel1[y1:y1+h1, x1:x1+w1]

# 두 번째 이미지에서 얼굴추출
x2, y2, w2, h2 = result2[0]['box']
face2 = pixel2[y2:y2+h2, x2:x2+w2]

# 얼굴을 FaceNet 모델에 전달하여 feature 벡터 생성
embedding1 = embedder.embeddings([face1])[0]
embedding2 = embedder.embeddings([face2])[0]

# 얼굴 특징 벡터 간의 거리 계산
distance = np.linalg.norm(embedding1 - embedding2)

# 거리를 유사도로 변환(일치율 계산에 사용)
similarity = 1 / (1 + distance)

# 인물 일치율 계산
match_rate = similarity * 100

match_rate = round(match_rate,2)

# 인물 일치율 출력
print("인물 일치율:", match_rate, "%")

# 거리를 기준으로 동일 인물 여부 판단
threshold = 0.8  #임계값
if distance < threshold:
    print("동일한 인물로 판단됩니다.")
else:
    print("다른 인물로 판단됩니다.")