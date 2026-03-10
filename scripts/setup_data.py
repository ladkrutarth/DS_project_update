import os
try:
    import requests
except ImportError:
    requests = None

try:
    import pandas as pd
except ImportError:
    pd = None
from pathlib import Path
import json

# Ensure we can import from the project root
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import settings
from utils.logger import logger

def setup_data():
    # Ensure all directories exist
    settings.ensure_dirs()
    logger.info("🛠️  Veriscan System Initialization...")

    # 1. Download CFPB Dataset (Public URL)
    # Using a smaller sample from a reliable source or generating if needed
    if not settings.CFPB_DATA_PATH.exists():
        logger.info(f"📥 Downloading CFPB Credit Card complaints...")
        url = "https://raw.githubusercontent.com/ladkrutarth/Veriscan-Assets/main/cfpb_credit_card_sample.csv"
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            with open(settings.CFPB_DATA_PATH, "wb") as f:
                f.write(r.content)
            logger.info("✅ CFPB data downloaded.")
        except Exception as e:
            logger.warning(f"⚠️  Manual download failed ({e}). Generating synthetic stub...")
            df = pd.DataFrame({
                "Issue": ["Billing dispute", "Late fee", "Identity theft"],
                "Consumer complaint narrative": [
                    "I was charged twice for the same transaction.",
                    "The bank charged me a fee even though I paid on time.",
                    "Someone opened a credit card in my name without my permission."
                ],
                "Company": ["VeriBank", "VeriBank", "VeriBank"],
                "State": ["NY", "CA", "TX"]
            })
            df.to_csv(settings.CFPB_DATA_PATH, index=False)

    # 2. Generate Synthetic Training Data if missing
    if not settings.FRAUD_TRAIN_PATH.exists():
        logger.info("🧪 Generating synthetic fraud training data...")
        from scripts.fix_agent_data import fix_data
        # This script creates fraud_scores, auth_profiles, etc.
        # But it expects processed_fraud_train.csv
        # Let's create a minimal processed_fraud_train.csv if it's missing
        df_train = pd.DataFrame({
            "first": ["John", "Jane", "Hacker"],
            "last": ["Doe", "Smith", "One"],
            "category": ["grocery_pos", "entertainment", "gas_transport"],
            "amt": [50.0, 1200.0, 20.0],
            "gender": ["M", "F", "M"],
            "state": ["NY", "NJ", "NV"],
            "merchant": ["SuperStore", "HighEndClub", "GasStation"],
            "hour": [14, 23, 2],
            "day_of_week": [2, 5, 0]
        })
        df_train.to_csv(settings.FRAUD_TRAIN_PATH, index=False)
        logger.info("✅ Synthetic training data generated.")

    # 3. Run Fix Agent Data to populate derivative outputs
    logger.info("🩹 Running data reconciliation...")
    from scripts.fix_agent_data import fix_data
    fix_data()
    logger.info("🚀 All system artifacts are synchronized and ready.")

if __name__ == "__main__":
    setup_data()
