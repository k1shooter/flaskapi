from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from deepface import DeepFace
import io
import os
import tempfile
app=Flask(__name__)
app.debug=True

@app.route("/")
def index():
    return "hello"

model = load_model("./keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    image = Image.open(file.stream).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = float(prediction[0][index])

    return jsonify({
        "class": class_name[2:],
        "confidence": confidence_score
    })

@app.route("/embedding", methods=["POST"])
def get_embedding():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # 임시 파일에 이미지를 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            file.save(tmp)
            tmp_path = tmp.name

        # DeepFace로 임베딩 추출
        embedding = DeepFace.represent(
            img_path=tmp_path,
            detector_backend='retinaface',
            model_name='ArcFace'
        )
        os.remove(tmp_path)  # 임시 파일 삭제

        if isinstance(embedding, list) and len(embedding) > 0:
            embedding_vector = embedding[0]['embedding']
            facial_area =embedding[0]['facial_area']
            facial_confidence=embedding[0]['face_confidence']
        else:
            return jsonify({"error": "No face detected"}), 400

        return jsonify({"embedding": embedding_vector,
                        "facial_area":facial_area,
                         "facial_confidence":facial_confidence })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)