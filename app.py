from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd

# Flask 설정
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 라벨 목록 (예: label.csv에서 추출된 120개의 견종 이름 리스트)
labels = [...]  # ["beagle", "poodle", ...]

# 모델 로드
model = load_model("../models/dog_breed_model.keras")
labels_df = pd.read_csv("../../dogbreed_dataset/labels.csv")
labels = labels_df["breed"].unique().tolist()


# 이미지 전처리 함수
def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# 라우팅
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files or request.files["file"].filename == "":
            return render_template("index.html", error="No file selected.")
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # 예측
            img_array = preprocess_img(filepath)
            preds = model.predict(img_array)[0]
            pred_idx = np.argmax(preds)
            pred_label = labels[pred_idx]
            confidence = preds[pred_idx] * 100

            return render_template("result.html",
                                   user_image=filepath,
                                   dogcat_class=f"{pred_label} ({confidence:.1f}%)")
    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render가 자동 지정한 포트를 읽음
    app.run(host="0.0.0.0", port=port)
