from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load the trained model and scaler
try:
    model = joblib.load('saved_models/best_model_Random_Forest.pkl')
    scaler = joblib.load('saved_models/scaler.pkl')
    MODEL_LOADED = True
    print("✅ Model and scaler loaded successfully!")
except Exception as e:
    MODEL_LOADED = False
    print(f"❌ Error loading model: {e}")

# Feature information for the UI
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

@app.route('/')
def home():
    """Render the main HTML page"""
    return render_template('index.html')

@app.route('/api/feature_ranges', methods=['GET'])
def get_feature_ranges():
    """API endpoint to get feature ranges for UI sliders"""
    return jsonify({
        "features": FEATURE_INFO,
        "model_loaded": MODEL_LOADED,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint to make predictions"""
    if not MODEL_LOADED:
        return jsonify({
            "error": "Model not loaded",
            "prediction": None,
            "probabilities": None,
            "confidence": 0
        }), 503
    
    try:
        # Get data from request
        data = request.json
        
        # Create a DataFrame with all expected features
        input_df = pd.DataFrame(columns=scaler.feature_names_in_)
        
        # Fill with zeros initially
        for feature in scaler.feature_names_in_:
            input_df[feature] = [0.0]
        
        # Update with values from request
        for feature, value in data.items():
            if feature in input_df.columns:
                input_df[feature] = [float(value)]
        
        # Ensure correct order
        input_df = input_df[scaler.feature_names_in_]
        
        # Scale the features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Prepare response
        response = {
            "prediction": int(prediction),
            "prediction_label": "malignant" if prediction == 0 else "benign",
            "probabilities": {
                "malignant": float(prediction_proba[0]),
                "benign": float(prediction_proba[1])
            },
            "confidence": float(max(prediction_proba)) * 100,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "prediction": None,
            "probabilities": None,
            "confidence": 0
        }), 400

@app.route('/api/model_info', methods=['GET'])
def model_info():
    """API endpoint to get model information"""
    info = {
        "model_loaded": MODEL_LOADED,
        "model_type": "Random Forest Classifier" if MODEL_LOADED else "Unknown",
        "features_count": len(scaler.feature_names_in_) if MODEL_LOADED else 0,
        "accuracy": 0.9649,
        "recall": 0.9825,
        "precision": 0.9655,
        "f1_score": 0.974,
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "samples": 569,
        "classes": {"malignant": 212, "benign": 357}
    }
    return jsonify(info)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
