from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import cv2
import os

app = Flask(__name__)

# =========================
# CORS (Production safe)
# =========================
CORS(app)

IMG_SIZE = 128


# =========================
# MODELS PATH
# =========================
MODEL_STAGE2 = os.getenv("MODEL_STAGE2", "fingerprint_robust_model.keras")
MODEL_STAGE3 = os.getenv("MODEL_STAGE3", "fingerprint_forensic_model.keras")


# =========================
# CUSTOM LAYER
# =========================
@tf.keras.utils.register_keras_serializable()
class L2Normalize(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)


# =========================
# SAFE MODEL LOADING
# =========================
print("Loading models...")

try:
    model2 = tf.keras.models.load_model(
        MODEL_STAGE2,
        compile=False,
        custom_objects={"L2Normalize": L2Normalize}
    )

    model3 = tf.keras.models.load_model(
        MODEL_STAGE3,
        compile=False,
        custom_objects={"L2Normalize": L2Normalize}
    )

    print("Models loaded ✔")

except Exception as e:
    print("❌ Model loading failed:", e)
    model2 = None
    model3 = None


# =========================
# PREPROCESS IMAGE
# =========================
def preprocess(file):
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Invalid image file")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) * 2.0

    return img.reshape(1, IMG_SIZE, IMG_SIZE, 1), img


# =========================
# VALIDATION
# =========================
def is_fingerprint(img_raw):
    img_uint8 = ((img_raw + 1) / 2 * 255).astype(np.uint8)

    edges = cv2.Canny(img_uint8, 100, 200)
    edge_ratio = np.sum(edges > 0) / (IMG_SIZE * IMG_SIZE)

    variance = np.std(img_raw)

    return edge_ratio >= 0.015 and variance >= 0.03


# =========================
# ANALYSIS
# =========================
def analyze_fingerprint(img_raw):

    img_uint8 = ((img_raw + 1) / 2 * 255).astype(np.uint8)

    quality = float(np.std(img_raw))

    edges = cv2.Canny(img_uint8, 100, 200)
    minutiae = int(np.sum(edges > 0))

    density_ratio = minutiae / (IMG_SIZE * IMG_SIZE)

    ridge_score = (
        0.9 if density_ratio > 0.08 else
        0.6 if density_ratio > 0.04 else
        0.3
    )

    if quality > 0.6:
        fp_type = "Whorl"
    elif quality > 0.4:
        fp_type = "Loop"
    else:
        fp_type = "Arch"

    return {
        "type": fp_type,
        "quality": round(quality, 3),
        "minutiae": minutiae,
        "ridge_density": round(ridge_score, 3),
        "clarity": round(quality * 100, 2)
    }


# =========================
# HEALTH CHECK
# =========================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "AI Fingerprint Service Running ✔",
        "models_loaded": model2 is not None and model3 is not None
    })


# =========================
# COMPARE API
# =========================
@app.route("/compare", methods=["POST"])
def compare():

    try:
        if model2 is None or model3 is None:
            return jsonify({"error": "Models not loaded"}), 500

        file1 = request.files.get("img1")
        file2 = request.files.get("img2")

        if not file1 or not file2:
            return jsonify({"error": "Missing images"}), 400

        img1_tensor, img1_raw = preprocess(file1)
        img2_tensor, img2_raw = preprocess(file2)

        # INVALID CHECK
        if not is_fingerprint(img1_raw) or not is_fingerprint(img2_raw):
            return jsonify({
                "fingerA": analyze_fingerprint(img1_raw),
                "fingerB": analyze_fingerprint(img2_raw),
                "similarity": {
                    "stage2": 0,
                    "stage3": 0,
                    "final_score": 0,
                    "result": "INVALID_INPUT",
                    "confidence": 0,
                    "match_score": 0
                }
            })

        fingerA = analyze_fingerprint(img1_raw)
        fingerB = analyze_fingerprint(img2_raw)

        # EMBEDDINGS
        e2_1 = np.squeeze(model2.predict(img1_tensor, verbose=0))
        e2_2 = np.squeeze(model2.predict(img2_tensor, verbose=0))

        e3_1 = np.squeeze(model3.predict(img1_tensor, verbose=0))
        e3_2 = np.squeeze(model3.predict(img2_tensor, verbose=0))

        # DISTANCE
        d2_raw = float(np.linalg.norm(e2_1 - e2_2))
        d3_raw = float(np.linalg.norm(e3_1 - e3_2))

        d2 = max(0.0, min(1.0, 1 - (d2_raw / 10.0)))
        d3 = max(0.0, min(1.0, 1 - (d3_raw / 10.0)))

        final = (0.4 * d2) + (0.6 * d3)
        final = max(0.0, min(final, 1.0))

        if final >= 0.88:
            result = "MATCH"
        elif final >= 0.65:
            result = "SIMILAR"
        else:
            result = "DIFFERENT"

        confidence = (0.7 * final) + (0.3 * d3)

        return jsonify({
            "fingerA": fingerA,
            "fingerB": fingerB,
            "similarity": {
                "stage2": float(d2),
                "stage3": float(d3),
                "final_score": float(final),
                "result": result,
                "confidence": round(confidence, 3),
                "match_score": round(final, 3)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# RUN
# =========================
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)