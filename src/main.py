import sys
import io
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import folium
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify
import os
import joblib

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load dataset
df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "algeria_forest_fires.csv"))

print(df.head())

# Keep "Region" as text
if "Region" in df.columns:
    df["Region"] = df["Region"].str.lower().str.strip()
else:
    print("Warning: 'Region' column not found in dataset.")

# Convert target variable "Classes" (fire/not fire) to numerical
df["Classes"] = df["Classes"].map({"fire": 1, "not fire": 0})

# Define features - exclude 'Classes' target and 'Region' text column
X = df.drop(columns=["Classes", "Region"])
y = df["Classes"]

print(f"\nFeatures used: {X.columns.tolist()}")
print(f"Target distribution:\n{y.value_counts()}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
joblib.dump(model, os.path.join(PROJECT_ROOT, "models", "random_forest_model.pkl"))
joblib.dump(scaler, os.path.join(PROJECT_ROOT, "models", "scaler.pkl"))
print("Model and scaler saved to models/ directory")

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Add predictions to dataframe
df["Prediction"] = model.predict(X_scaled)

# Count fires by region
fire_by_region = df[df["Prediction"] == 1]["Region"].value_counts()
print("\nHigh Risk Regions:")
print(fire_by_region)

# Create a bar chart
plt.figure(figsize=(12, 6))
fire_by_region.plot(kind='bar', color='red', alpha=0.7)
plt.title("Fire Risk by Region (Model Predictions)")
plt.xlabel("Region")
plt.ylabel("Number of High Risk Predictions")
plt.xticks(rotation=45)
plt.tight_layout()
os.makedirs(os.path.join(PROJECT_ROOT, "maps"), exist_ok=True)
plt.savefig(os.path.join(PROJECT_ROOT, "maps", "fire_risk_barchart.png"))
print(f"\nBar chart saved to: {os.path.join(PROJECT_ROOT, 'maps', 'fire_risk_barchart.png')}")
plt.show()

# Create a simple HTML map with markers
print("\nCreating interactive map...")
fire_map = folium.Map(location=[28, 2], zoom_start=6)

# Add markers for high-risk areas (approximate coordinates)
region_coords = {
    "chlef": [36.165, 1.331],
    "guelma": [36.462, 7.426],
    "tipaza": [36.590, 2.447],
    "ain defla": [36.264, 1.968],
    "bejaia": [36.756, 5.084],
    "tizi ouzou": [36.712, 4.046],
    "skikda": [36.876, 6.909],
    "bouira": [36.375, 3.902],
    "el tarf": [36.767, 8.313],
    "jijel": [36.820, 5.766],
}

# Add markers for high risk regions
high_risk_regions = df[df["Prediction"] == 1]["Region"].unique()
print(f"High risk regions found: {len(high_risk_regions)}")

for region in high_risk_regions:
    if region in region_coords:
        folium.Marker(
            location=region_coords[region],
            popup=f" HIGH FIRE RISK: {region.title()}",
            icon=folium.Icon(color="red", icon="fire", prefix="fa"),
        ).add_to(fire_map)
        print(f"  Added marker for {region}")

# Add markers for low risk regions
low_risk_regions = df[df["Prediction"] == 0]["Region"].unique()
for region in low_risk_regions:
    if region in region_coords:
        folium.Marker(
            location=region_coords[region],
            popup=f" Low Fire Risk: {region.title()}",
            icon=folium.Icon(color="green", icon="leaf", prefix="fa"),
        ).add_to(fire_map)

# Save the map
map_path = os.path.join(PROJECT_ROOT, "maps", "fire_risk_map.html")
fire_map.save(map_path)
print(f"\n Fire risk map saved as {map_path}")
print(f"    Open this file in your browser to see the interactive map!")

# Create a summary report
summary_report = f"""
========================================
ALGERIA FOREST FIRE PREDICTION REPORT
========================================

Dataset Information:
- Total samples: {len(df)}
- Fire cases: {y.sum()}
- Non-fire cases: {len(df) - y.sum()}

Model Performance:
- Accuracy: {accuracy_score(y_test, y_pred):.2%}
- Model used: Random Forest Classifier

High Risk Regions:
{fire_by_region.to_string()}

Map Location:
{map_path}

========================================
"""

# Save summary report
report_path = os.path.join(PROJECT_ROOT, "maps", "prediction_report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(summary_report)
print(f"\n    Summary report saved to: {report_path}")

# Flask API
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Algeria Forest Fire Prediction API",
        "status": "running",
        "model_accuracy": f"{accuracy_score(y_test, y_pred):.2%}"
    })

@app.route("/predict", methods=["POST"])
def predict_fire():
    try:
        data = request.get_json()
        
        if not data or "features" not in data:
            return jsonify({"error": "Please provide 'features' in request body"}), 400
        
        features = data["features"]
        
        # Check if we have 13 features
        if len(features) != 13:
            return jsonify({"error": f"Expected 13 features, got {len(features)}"}), 400
        
        # Scale features
        input_data = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        return jsonify({
            "fire_risk": int(prediction),
            "risk_level": "HIGH" if prediction == 1 else "LOW",
            "confidence": float(max(probability)),
            "probabilities": {
                "no_fire": float(probability[0]),
                "fire": float(probability[1])
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": True})

if __name__ == "__main__":
    print("\n" + "="*50)
    print("   Starting Flask API Server")
    print("="*50)
    print(f" API URL: http://127.0.0.1:5000")
    print(f" Model Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(f"      Map available at: {map_path}")
    print("\nAvailable endpoints:")
    print("  GET  /        - API information")
    print("  GET  /health  - Health check")
    print("  POST /predict - Make predictions")
    print("\nExample POST request:")
    print('  {"features": [9,8,2023,27,41,15,3.8,51.7,3.5,143.9,10.6,48.1,10.1]}')
    print("="*50)
    print("\n API is ready! Press Ctrl+C to stop\n")
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)