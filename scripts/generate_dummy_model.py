import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# Paths
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "fraud_model_rf.joblib"
ENCODER_PATH = MODELS_DIR / "encoders.joblib"

print("🛠️ Generating dummy fraud model and encoders...")

# 1. Create dummy encoders
encoders = {}
for col in ["category", "gender", "state", "merchant"]:
    le = LabelEncoder()
    le.fit(["dummy_val", "other_val"])
    encoders[col] = le

joblib.dump(encoders, ENCODER_PATH)
print(f"✅ Saved encoders to {ENCODER_PATH}")

# 2. Create dummy RandomForest model
# Features: category, amt, gender, state, merchant, hour, day_of_week
X = np.zeros((10, 7))
y = np.random.randint(0, 2, 10)

rf = RandomForestClassifier(n_estimators=10)
rf.fit(X, y)

joblib.dump(rf, MODEL_PATH)
print(f"✅ Saved dummy model to {MODEL_PATH}")
