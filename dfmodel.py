import os
from deepface import DeepFace
import pandas as pd
# 이미지가 들어있는 폴더 경로
folder_path = './celeb'

# 지원하는 이미지 확장자
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# 결과를 저장할 리스트
results = []

for filename in os.listdir(folder_path):
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        file_path = os.path.join(folder_path, filename)
        try:
            # 임베딩 추출
            embedding = DeepFace.represent(
                img_path=file_path,
                detector_backend='retinaface',  # 원하는 백엔드로 변경 가능
                model_name='ArcFace'            # 원하는 모델로 변경 가능
            )
            # 여러 얼굴이 검출될 수 있으니 첫 번째 얼굴만 사용
            if isinstance(embedding, list) and len(embedding) > 0:
                embedding_vector = embedding[0]['embedding']
                results.append({'filename': filename, 'embedding': embedding_vector})
            else:
                results.append({'filename': filename, 'embedding': None, 'error': 'No face detected'})
        except Exception as e:
            results.append({'filename': filename, 'embedding': None, 'error': str(e)})

# 결과 미리보기
df = pd.DataFrame(results)
df.to_csv('embeddings.csv', index=False)