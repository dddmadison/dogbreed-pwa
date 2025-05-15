from flask import Flask, request, render_template, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import pandas as pd
import zipfile

app = Flask(__name__, static_folder="static", template_folder="templates")

# 절대 경로 기준
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 모델 zip 경로 및 대상 모델 파일 경로
model_zip_path = os.path.join(BASE_DIR, 'models', 'dog_breed_light.keras.zip')
model_dir = os.path.join(BASE_DIR, 'models')
model_path = os.path.join(model_dir, 'dog_breed_light.keras')

# 🔓 Render 서버에서 압축 해제 (한 번만 실행됨)
if not os.path.exists(model_path) and os.path.exists(model_zip_path):
    with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
        zip_ref.extractall(model_dir)

# 라벨 로딩
labels_path = os.path.join(BASE_DIR, 'dogbreed_dataset', 'labels.csv')
labels_df = pd.read_csv(labels_path)
DOGBREED_CLASSES = labels_df["breed"].unique()
DOGBREED_CLASSES.sort()

@app.route('/')
def home():
    return render_template("index_pwa.html")

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided.'}), 400
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file provided.'}), 400

        # ✅ 요청 시점에 모델 로딩
        model = tf.keras.models.load_model(model_path)

        # ✅ 이미지 파일 저장 없이 stream으로 직접 처리
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape((1, 224, 224, 3))

        # 예측 수행
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions, axis=1)[0]
        predicted_probability = predictions[0][predicted_index]
        predicted_breed = DOGBREED_CLASSES[predicted_index]

        return render_template(
            'index_pwa.html',
            predicted_breed=predicted_breed,
            predicted_probability="{:.1%}".format(predicted_probability)
        )

    except Exception as e:
        app.logger.error(f'[예측 오류] {e}')
        return jsonify({'error': '서버 오류 발생', 'detail': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
