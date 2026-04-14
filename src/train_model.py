import sys
import io
import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load and prepare data
df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "algeria_forest_fires.csv"))
df["Classes"] = df["Classes"].map({"fire": 1, "not fire": 0})

# Prepare features
X = df.drop(columns=["Classes", "Region"])
y = df["Classes"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Save model and scaler
os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
joblib.dump(model, os.path.join(PROJECT_ROOT, "models", "random_forest_model.pkl"))
joblib.dump(scaler, os.path.join(PROJECT_ROOT, "models", "scaler.pkl"))

print(" Model and scaler saved successfully!")
print(f"Model saved to: {os.path.join(PROJECT_ROOT, 'models', 'random_forest_model.pkl')}")
print(f"Scaler saved to: {os.path.join(PROJECT_ROOT, 'models', 'scaler.pkl')}")