import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
import sys

# Ensure we can import from the project root if run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import settings

def fix_data():
    print("🚀 Veriscan GuardAgent Data Synchronization...")
    
    if not settings.FRAUD_TRAIN_PATH.exists():
        print(f"❌ Cannot find {settings.FRAUD_TRAIN_PATH}. Please run prepare_fraud_data.py first.")
        return

    # Using joblib for models; ensuring models dir exists if we were to train
    if not settings.MODEL_RF_PATH.exists():
        print(f"❌ Cannot find {settings.MODEL_RF_PATH}. Please run train_fraud_model.py first.")
        return

    # 1. Load data and model
    print("📂 Loading data and model...")
    df = pd.read_csv(settings.FRAUD_TRAIN_PATH)
    model = joblib.load(settings.MODEL_RF_PATH)
    encoders = joblib.load(settings.ENCODERS_PATH)

    # ... (rest of the code using df)
    # 3. Predict Scores
    print("🔮 Generating fraud scores...")
    features = ["category", "amt", "gender", "state", "merchant", "hour", "day_of_week"]
    X = df[features].copy()
    for col in ["category", "gender", "state", "merchant"]:
        X[col] = encoders[col].transform(X[col].astype(str))
    
    probs = model.predict_proba(X)[:, 1]
    
    # 4. Create Fraud Scores Output
    print(f"💾 Saving {settings.FRAUD_SCORES_PATH}...")
    scores_df = pd.DataFrame({
        "TRANSACTION_ID": [f"TXN_{i:06d}" for i in range(len(df))],
        "USER_ID": df["USER_ID"],
        "CATEGORY": df["category"],
        "MERCHANT": df["merchant"],
        "COMBINED_RISK_SCORE": probs * 100,
        "RISK_LEVEL": ["CRITICAL" if p > 0.8 else "HIGH" if p > 0.5 else "MEDIUM" if p > 0.2 else "LOW" for p in probs],
        "ZSCORE_FLAG": np.random.uniform(0, 5, len(df)), # Synthetic for analyst flavor
        "VELOCITY_FLAG": np.random.uniform(0, 5, len(df)),
        "GEOGRAPHIC_RISK_SCORE": np.random.uniform(0, 10, len(df))
    })
    scores_df.to_csv(settings.FRAUD_SCORES_PATH, index=False)

    # 5. Create Auth Profiles Output
    print(f"💾 Saving {settings.AUTH_PROFILES_PATH}...")
    profiles = scores_df.groupby("USER_ID").agg(
        AVG_RISK=("COMBINED_RISK_SCORE", "mean"),
        HIGH_RISK_COUNT=("RISK_LEVEL", lambda x: (x == "HIGH").sum() + (x == "CRITICAL").sum()),
        TXN_COUNT=("USER_ID", "count")
    ).reset_index()
    profiles["RECOMMENDED_SECURITY_LEVEL"] = profiles["AVG_RISK"].apply(
        lambda x: "MFA_REQUIRED" if x > 40 else "Standard"
    )
    profiles.to_csv(settings.AUTH_PROFILES_PATH, index=False)

    # 6. Ensure transactions_3000.csv exists (satisfies HabitModel)
    print(f"💾 Saving {settings.TXN_3000_PATH}...")
    # Add dummy location if missing
    if "location" not in df.columns:
        df["location"] = df["state"].apply(lambda s: f"Sample City, {s}")
    
    # Standardize columns for HabitModel
    df_mini = df.copy()
    df_mini.columns = [c.upper() for c in df_mini.columns]
    df_mini.to_csv(settings.TXN_3000_PATH, index=False)

    print("\n✅ Sync complete. GuardAgent tools are now ready to use.")

if __name__ == "__main__":
    fix_data()
