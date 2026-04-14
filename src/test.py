import os
import sys
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load dataset
dataset = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "algeria_forest_fires.csv"))

# Display basic information
print("Dataset shape:", dataset.shape)
print(dataset.head())

# Rename target column for consistency
if "Classes" in dataset.columns:
    dataset.rename(columns={"Classes": "fire"}, inplace=True)

# Ensure correct mapping of fire labels
dataset["fire"] = dataset["fire"].str.strip().map({"fire": 1, "not fire": 0})

# Drop any NaN values in "fire"
dataset.dropna(subset=["fire"], inplace=True)

# Convert fire column to integer
dataset["fire"] = dataset["fire"].astype(int)

# Check for missing values
print("Missing values per column:\n", dataset.isnull().sum())

# ===== FIX: Exclude non-numeric columns =====
# Define features (X) - exclude 'fire' target and 'Region' text column
X = dataset.drop(columns=["fire", "Region"])  # Remove Region column
y = dataset["fire"]

print(f"\nFeatures shape: {X.shape}")
print(f"Features columns: {X.columns.tolist()}")
print(f"Target distribution:\n{y.value_counts()}")

# Visualizations
# Histogram of all numerical features
dataset.select_dtypes(include=[np.number]).hist(figsize=(12, 8), bins=30)
plt.tight_layout()
plt.show()

# Boxplot to detect outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=dataset.select_dtypes(include=[np.number]))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Data visualization - Count of fire occurrences
plt.figure(figsize=(6, 4))
sns.countplot(x="fire", data=dataset, palette="coolwarm", hue="fire", legend=False)
plt.title("Fire Occurrences")
plt.xlabel("Fire (1 = fire, 0 = no fire)")
plt.ylabel("Count")
plt.show()

# Standardize features (better for Logistic Regression, SVM, and Neural Networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Print dataset shapes
print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# Train Multiple Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000),
}

print("\n" + "="*60)
print("MODEL COMPARISON RESULTS")
print("="*60)

# Train and evaluate each model
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Classification Report:")
    print(classification_report(y_test, y_pred))
    print("-" * 50)

# Find best model
best_model_name = max(results, key=results.get)
print(f"\n{'='*60}")
print(f"BEST MODEL: {best_model_name} with accuracy: {results[best_model_name]:.4f}")
print(f"{'='*60}")

# Perform cross-validation for Random Forest
print("\nCross-Validation for Random Forest:")
rf = RandomForestClassifier(n_estimators=100)
cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
print(f"  Random Forest Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Hyperparameter Tuning for Random Forest
print("\nHyperparameter Tuning for Random Forest...")
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print Best Parameters
print(f"\nBest Parameters for Random Forest: {grid_search.best_params_}")

# Evaluate the best Random Forest model
best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test)
best_accuracy = accuracy_score(y_test, y_pred_best)
print(f"\nOptimized Random Forest Accuracy: {best_accuracy:.4f}")
print(classification_report(y_test, y_pred_best))

# Save the best model
from joblib import dump
dump(best_rf, os.path.join(PROJECT_ROOT, "models", "random_forest_model.pkl"))
dump(scaler, os.path.join(PROJECT_ROOT, "models", "scaler.pkl"))
print(f"\n[SUCCESS] Best model saved to {os.path.join(PROJECT_ROOT, 'models', 'random_forest_model.pkl')}")

# Confusion Matrix for best model
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()