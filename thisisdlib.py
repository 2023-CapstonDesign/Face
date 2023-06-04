import dlib
import numpy as np

def get_face_embeddings(image_path):
    # 얼굴 인식기 생성
    face_detector = dlib.get_frontal_face_detector()
    
    #사전에 다운로드한 파일
    shape_predictor = dlib.shape_predictor("C:/shape_predictor_5_face_landmarks.dat/shape_predictor_5_face_landmarks.dat")
    face_recognizer = dlib.face_recognition_model_v1("C:/dlib_face_recognition_resnet_model_v1.dat/dlib_face_recognition_resnet_model_v1.dat")

    # 이미지 읽기
    img = dlib.load_rgb_image(image_path)

    # 얼굴 인식
    detected_faces = face_detector(img, 1)

    # 얼굴이 하나 이상 감지된 경우
    if len(detected_faces) > 0:
        face = detected_faces[0]
        shape = shape_predictor(img, face)
        face_embedding = np.array(face_recognizer.compute_face_descriptor(img, shape))
        return face_embedding
    else:
        return None

def compare_embeddings(emb1, emb2):
    # 얼굴 임베딩 간의 유클리드 거리 계산
    distance = np.linalg.norm(emb1 - emb2)
    
    # 임계값 설정하여 유사도 판단
    threshold = 0.6
    if distance < threshold:
        return True
    else:
        return False

# 이미지 파일 경로
image1_path = "pss.jpg"
image2_path = "sin.jpg"

# 얼굴 임베딩 추출
embedding1 = get_face_embeddings(image1_path)
embedding2 = get_face_embeddings(image2_path)

# 얼굴 임베딩 비교
if embedding1 is not None and embedding2 is not None:
    result = compare_embeddings(embedding1, embedding2)
    print("Is the same person:", result)
else:
    print("Failed to detect faces image.")
    
    
if compare_embeddings(embedding1, embedding2) == True:
    print("동일한 인물입니다.")
else:
    print("다른 인물입니다.")
