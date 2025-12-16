from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os
import json

app = Flask(__name__)

# ==================================================
# Paths (Render / Local safe)
# ==================================================
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

# ==================================================
# Load model, scaler, metadata
# ==================================================
try:
    model = joblib.load(os.path.join(MODEL_DIR, "best_model_Logistic_Regression.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

    with open(os.path.join(MODEL_DIR, "metadata.json"), "r") as f:
        METADATA = json.load(f)

    MODEL_LOADED = True
    print("✅ Model, scaler, and metadata loaded successfully!")
    print("📌 Model:", METADATA.get("best_model"))
    print("📌 Features expected:", len(scaler.feature_names_in_))

except Exception as e:
    MODEL_LOADED = False
    METADATA = {}
    print(f"❌ Error during loading: {e}")

# ==================================================
# Feature information for UI sliders
# ==================================================
FEATURE_INFO = [
    {"name": "mean radius", "display_name": "Mean Radius", "min": 6.0, "max": 30.0, "default": 14.0, "step": 0.1},
    {"name": "mean texture", "display_name": "Mean Texture", "min": 9.0, "max": 40.0, "default": 19.0, "step": 0.1},
    {"name": "mean perimeter", "display_name": "Mean Perimeter", "min": 43.0, "max": 190.0, "default": 92.0, "step": 0.1},
    {"name": "mean area", "display_name": "Mean Area", "min": 143.0, "max": 2500.0, "default": 654.0, "step": 1.0},
    {"name": "mean smoothness", "display_name": "Mean Smoothness", "min": 0.05, "max": 0.20, "default": 0.10, "step": 0.001}
]

# ==================================================
# Routes
# ==================================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/feature_ranges', methods=['GET'])
def feature_ranges():
    return jsonify({
        "features": FEATURE_INFO,
        "model_loaded": MODEL_LOADED,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    if not MODEL_LOADED:
        return jsonify({
            "error": "Model not loaded",
            "prediction": None,
            "prediction_label": None,
            "probabilities": {"malignant": 0.0, "benign": 0.0},
            "confidence": 0.0
        }), 503

    try:
        data = request.json or {}

        # 🔥 CRITICAL FIX: use training mean for missing features
        input_df = pd.DataFrame(
            [scaler.mean_],
            columns=scaler.feature_names_in_
        )

        # Update with provided values
        for feature, value in data.items():
            if feature in input_df.columns:
                input_df.at[0, feature] = float(value)

        # Scale & predict
        input_scaled = scaler.transform(input_df)
        prediction = int(model.predict(input_scaled)[0])
        probabilities = model.predict_proba(input_scaled)[0]

        return jsonify({
            "prediction": prediction,
            "prediction_label": "malignant" if prediction == 0 else "benign",
            "probabilities": {
                "malignant": float(probabilities[0]),
                "benign": float(probabilities[1])
            },
            "confidence": float(max(probabilities)) * 100,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "prediction": None,
            "prediction_label": None,
            "probabilities": {"malignant": 0.0, "benign": 0.0},
            "confidence": 0.0
        }), 400

@app.route('/api/model_info', methods=['GET'])
def model_info():
    return jsonify({
        "model_loaded": MODEL_LOADED,
        "model_type": METADATA.get("best_model", "Logistic Regression"),
        "features_count": len(scaler.feature_names_in_) if MODEL_LOADED else 0,
        "accuracy": METADATA.get("best_accuracy", 0.0),
        "recall": METADATA.get("best_recall", 0.0),
        "precision": METADATA.get("best_precision", 0.0),
        "f1_score": METADATA.get("best_f1", 0.0),
        "dataset": METADATA.get("dataset", "Breast Cancer Wisconsin (Diagnostic)"),
        "samples": METADATA.get("training_summary", {}).get("total_samples", 0),
        "classes": METADATA.get("training_summary", {}).get("class_distribution", {})
    })

# ==================================================
# Local run (ignored by Gunicorn / Render)
# ==================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
