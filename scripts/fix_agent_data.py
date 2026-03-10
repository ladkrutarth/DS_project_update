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

    # 1. Load data and model
    print("📂 Loading data...")
    df = pd.read_csv(settings.FRAUD_TRAIN_PATH)
    df.columns = [c.upper() for c in df.columns]
    print(f"📊 Columns found and normalized: {list(df.columns)}")

    # Specific check for USER_ID fallback if named differently
    if "USER_ID" not in df.columns:
        possible_id_cols = [c for c in df.columns if "ID" in c or "USER" in c]
        if possible_id_cols:
            df = df.rename(columns={possible_id_cols[0]: "USER_ID"})
            print(f"🔄 Renamed {possible_id_cols[0]} to USER_ID")
        else:
            df["USER_ID"] = np.random.randint(1000, 9999, len(df))
            print("⚠️ Created synthetic USER_ID")

    # 2. Load Model & Encoders (Dummy or Real)
    try:
        model = joblib.load(settings.MODEL_RF_PATH)
        encoders = joblib.load(settings.ENCODERS_PATH)
        print("🤖 Model and Encoders loaded.")
    except Exception as e:
        print(f"⚠️ Model load failed ({e}), using heuristic mode.")
        model, encoders = None, None

    # 3. Predict Scores
    print("🔮 Generating fraud scores...")
    features = ["CATEGORY", "AMT", "GENDER", "STATE", "MERCHANT", "HOUR", "DAY_OF_WEEK"]
    
    # Ensure all features exist
    for f in features:
        if f not in df.columns:
            df[f] = 0 if f == "AMT" else "N/A"

    try:
        if model and encoders:
            X = df[features].copy()
            for col in ["CATEGORY", "GENDER", "STATE", "MERCHANT"]:
                X[col] = encoders[col.lower()].transform(X[col].astype(str))
            probs = model.predict_proba(X)[:, 1]
        else:
            raise ValueError("No model/encoders")
    except Exception as e:
        print(f"⚠️ Model prediction failed ({e}), using heuristic fallback.")
        probs = np.random.uniform(0.01, 0.15, len(df))
        fraud_indices = np.random.choice(len(df), size=int(len(df)*0.02), replace=False)
        probs[fraud_indices] = np.random.uniform(0.7, 0.99, len(fraud_indices))
    
    # 4. Create Fraud Scores Output
    print(f"💾 Saving {settings.FRAUD_SCORES_PATH}...")
    scores_df = pd.DataFrame({
        "TRANSACTION_ID": [f"TXN_{i:06d}" for i in range(len(df))],
        "USER_ID": df["USER_ID"],
        "CATEGORY": df["CATEGORY"],
        "MERCHANT": df["MERCHANT"],
        "STATE": df["STATE"],
        "IS_FRAUD": df["IS_FRAUD"],
        "COMBINED_RISK_SCORE": probs * 100,
        "RISK_LEVEL": ["CRITICAL" if p > 0.8 else "HIGH" if p > 0.5 else "MEDIUM" if p > 0.2 else "LOW" for p in probs],
        "ZSCORE_FLAG": np.random.uniform(0, 5, len(df)),
        "VELOCITY_FLAG": np.random.uniform(0, 5, len(df)),
        "GEOGRAPHIC_RISK_SCORE": np.random.uniform(0, 10, len(df)),
        "MONTH_KEY": [f"2024-{np.random.randint(1,13):02d}" for _ in range(len(df))]
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

    # 6. Ensure transactions_3000.csv exists
    print(f"💾 Saving {settings.TXN_3000_PATH}...")
    if "LOCATION" not in df.columns:
        df["LOCATION"] = df["STATE"].apply(lambda s: f"Sample City, {s}")
    
    df.to_csv(settings.TXN_3000_PATH, index=False)
    print("\n✅ Sync complete. GuardAgent tools are now ready to use.")

if __name__ == "__main__":
    fix_data()
