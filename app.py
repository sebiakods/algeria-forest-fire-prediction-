import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Get the project root directory (current directory since app.py is in root)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Load model and scaler
model = joblib.load(os.path.join(PROJECT_ROOT, "models", "random_forest_model.pkl"))
scaler = joblib.load(os.path.join(PROJECT_ROOT, "models", "scaler.pkl"))

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Algeria Forest Fire Prediction API",
        "endpoints": {
            "/predict": "POST - Send features to get fire risk prediction",
            "/health": "GET - Check API health"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Expected features: day, month, year, Temperature, RH, Ws, Rain, 
        # FFMC, DMC, DC, ISI, BUI, FWI
        features = data.get("features")
        
        if not features:
            return jsonify({"error": "No features provided"}), 400
        
        if len(features) != 13:
            return jsonify({"error": f"Expected 13 features, got {len(features)}"}), 400
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        return jsonify({
            "fire_risk": int(prediction),
            "risk_level": "High" if prediction == 1 else "Low",
            "confidence": float(max(probability)),
            "probabilities": {
                "no_fire": float(probability[0]),
                "fire": float(probability[1])
            }
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)