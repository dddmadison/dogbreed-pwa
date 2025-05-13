from flask import Flask, render_template, request
import os, gc, time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ─── 기본 설정 ──────────────────────────────────────────
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# 라벨 한 번만 메모리에
labels = pd.read_csv("static/labels.csv")["breed"].unique().tolist()

# ─── 전처리 함수 ────────────────────────────────────────
def preprocess_img(path):
    img = image.load_img(path, target_size=(224, 224))
    arr = image.img_to_array(img)
    return np.expand_dims(arr, 0) / 255.0

# ─── 라우팅 ─────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return render_template("index.html", error="No file selected.")

        # 저장
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(save_path)

        # ✔ 요청 시점에만 모델 로드
        start = time.time()
        model = load_model("models/dog_breed_model.keras")

        # 예측
        preds = model.predict(preprocess_img(save_path))[0]
        pred_idx = int(np.argmax(preds))
        pred_label = labels[pred_idx]
        confidence = preds[pred_idx] * 100

        # ✔ 메모리 해제
        tf.keras.backend.clear_session()
        del model
        gc.collect()
        print("⏱ prediction elapsed:", round(time.time() - start, 2), "sec")

        return render_template(
            "result.html",
            user_image=save_path,
            dogcat_class=f"{pred_label} ({confidence:.1f}%)"
        )

    return render_template("index.html")

# ─── Render용 포트 바인딩 ───────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
