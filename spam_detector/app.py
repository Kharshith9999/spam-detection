"""
Flask API backend for the Spam/Ham Email Classifier
"""

import os
import json
import joblib
from flask import Flask, request, jsonify, send_from_directory

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "spam_model.pkl")
STATS_PATH = os.path.join(BASE_DIR, "model_stats.json")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")

# ─── Load Model ───────────────────────────────────────────────────────────────
print("🔄  Loading model...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}.\n"
        "Please run: python spam_detector/train_model.py"
    )

model = joblib.load(MODEL_PATH)
print("✅  Model loaded!")

with open(STATS_PATH) as f:
    model_stats = json.load(f)

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400

    # Predict
    prediction   = model.predict([text])[0]           # 0=ham, 1=spam
    probabilities = model.predict_proba([text])[0]     # [ham_prob, spam_prob]

    label        = "spam" if prediction == 1 else "ham"
    spam_prob    = round(float(probabilities[1]) * 100, 2)
    ham_prob     = round(float(probabilities[0]) * 100, 2)
    confidence   = spam_prob if label == "spam" else ham_prob

    return jsonify({
        "label":      label,
        "is_spam":    bool(prediction == 1),
        "spam_prob":  spam_prob,
        "ham_prob":   ham_prob,
        "confidence": confidence,
    })

@app.route("/api/stats", methods=["GET"])
def stats():
    return jsonify(model_stats)

# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
