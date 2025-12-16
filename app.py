from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os
import json

app = Flask(__name__)

# ==================================================
# Paths (Notebook + Deployment safe)
# ==================================================
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Running inside Jupyter Notebook
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
    {"name": "mean smoothness", "display_name": "Mean Smoothness", "min": 0.05, "max": 0.17, "default": 0.10, "step": 0.001},
    {"name": "mean compactness", "display_name": "Mean Compactness", "min": 0.02, "max": 0.35, "default": 0.09, "step": 0.001},
    {"name": "mean concavity", "display_name": "Mean Concavity", "min": 0.0, "max": 0.43, "default": 0.09, "step": 0.001},
    {"name": "mean concave points", "display_name": "Mean Concave Points", "min": 0.0, "max": 0.20, "default": 0.05, "step": 0.001},
    {"name": "mean symmetry", "display_name": "Mean Symmetry", "min": 0.11, "max": 0.30, "default": 0.18, "step": 0.001},
    {"name": "mean fractal dimension", "display_name": "Mean Fractal Dimension", "min": 0.05, "max": 0.10, "default": 0.06, "step": 0.001}
]

# ==================================================
# Routes
# ==================================================
@app.route('/')
def home():
    return render_template('index.html')

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
        return jsonify({"error": "Model not loaded"}), 503

    try:
        data = request.json

        # Create input dataframe with correct feature order
        input_df = pd.DataFrame(
            np.zeros((1, len(scaler.feature_names_in_))),
            columns=scaler.feature_names_in_
        )

        for feature, value in data.items():
            if feature in input_df.columns:
                input_df.at[0, feature] = float(value)

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
        return jsonify({"error": str(e)}), 400

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
# Local run only (ignored by Render / Gunicorn)
# ==================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
