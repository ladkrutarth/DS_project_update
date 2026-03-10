import requests
import pandas as pd
import numpy as np
import json
from pathlib import Path

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

    # 1. Download/Generate CFPB Dataset
    if not settings.CFPB_DATA_PATH.exists() or settings.CFPB_DATA_PATH.stat().st_size < 1000:
        logger.info(f"📥 Preparing CFPB Credit Card complaints...")
        # Generate 500 synthetic rows for a rich tab experience
        issues = ["Billing dispute", "Late fee", "Identity theft", "Credit reporting error", "Unauthorized charge"]
        narratives = [
            "I was charged twice for the same transaction at a merchant.",
            "The bank charged me a fee even though I paid on time according to my records.",
            "Someone opened a credit card in my name without my permission and spent money.",
            "My credit report shows a late payment that I actually paid on time.",
            "I see a transaction for $500 that I never authorized."
        ]
        companies = ["VeriBank", "Citibank", "Chase", "Wells Fargo", "Capital One"]
        states = ["NY", "CA", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]
        
        df_cfpb = pd.DataFrame({
            "Issue": [np.random.choice(issues) for _ in range(500)],
            "Consumer complaint narrative": [np.random.choice(narratives) for _ in range(500)],
            "Company": [np.random.choice(companies) for _ in range(500)],
            "State": [np.random.choice(states) for _ in range(500)]
        })
        df_cfpb.to_csv(settings.CFPB_DATA_PATH, index=False)
        logger.info("✅ CFPB dataset enriched.")

    # 2. Generate Synthetic Training Data
    if not settings.FRAUD_TRAIN_PATH.exists() or settings.FRAUD_TRAIN_PATH.stat().st_size < 1000:
        logger.info("🧪 Generating enriched synthetic fraud training data...")
        # Generate 2000 rows for rich charts
        num_rows = 2000
        first_names = ["John", "Jane", "Alice", "Bob", "Charlie", "Diana", "Edward", "Fiona", "George", "Hannah"]
        last_names = ["Smith", "Doe", "Johnson", "Brown", "Taylor", "Miller", "Wilson", "Moore", "Anderson", "Thomas"]
        categories = ["grocery_pos", "entertainment", "gas_transport", "shopping_net", "food_out", "health_fitness"]
        genders = ["M", "F"]
        states = ["NY", "CA", "TX", "FL", "IL", "NJ", "NV", "AZ", "WA", "MA"]
        merchants = ["SuperStore", "HighEndClub", "GasStation", "Amazon", "Starbucks", "Nike", "Apple", "Walmart"]
        
        df_train = pd.DataFrame({
            "first": [np.random.choice(first_names) for _ in range(num_rows)],
            "last": [np.random.choice(last_names) for _ in range(num_rows)],
            "category": [np.random.choice(categories) for _ in range(num_rows)],
            "amt": np.random.uniform(5, 2000, num_rows),
            "gender": [np.random.choice(genders) for _ in range(num_rows)],
            "state": [np.random.choice(states) for _ in range(num_rows)],
            "merchant": [np.random.choice(merchants) for _ in range(num_rows)],
            "hour": np.random.randint(0, 24, num_rows),
            "day_of_week": np.random.randint(0, 7, num_rows),
            "is_fraud": np.random.choice([0, 1], num_rows, p=[0.97, 0.03])
        })
        df_train.to_csv(settings.FRAUD_TRAIN_PATH, index=False)
        logger.info("✅ Enriched fraud training data generated.")

    # 4. Generate Synthetic Spending DNA Data
    if not settings.DNA_DATA_PATH.exists() or settings.DNA_DATA_PATH.stat().st_size < 5000:
        logger.info("🧬 Generating enriched synthetic Spending DNA data...")
        users = [f"user_{i}" for i in range(101, 115)] + ["demo_user", "premium_user", "test_subject"]
        dna_records = []
        for user in users:
            # 100 sessions per user for a stable baseline
            for _ in range(100):
                dna_records.append({
                    "user_id": user,
                    "avg_txn_amount": np.random.uniform(10, 800),
                    "location_entropy": np.random.uniform(0.05, 0.95),
                    "weekend_ratio": np.random.uniform(0, 0.6),
                    "category_diversity": np.random.uniform(0.1, 0.9),
                    "time_of_day_pref": np.random.randint(0, 4),
                    "risk_appetite_score": np.random.uniform(0, 1),
                    "spending_velocity": np.random.uniform(0.5, 15),
                    "merchant_loyalty_score": np.random.uniform(0.2, 0.95),
                    "trust_score": np.random.uniform(0.5, 0.99),
                    "dna_deviation_score": np.random.uniform(0, 0.4),
                    "is_anomalous_session": np.random.choice([0, 1], p=[0.95, 0.05])
                })
        df_dna = pd.DataFrame(dna_records)
        df_dna.to_csv(settings.DNA_DATA_PATH, index=False)
        logger.info("✅ Enriched Spending DNA data generated.")

    # 5. Generate Financial Advisor Dataset
    if not settings.ADVISOR_DATA_PATH.exists() or settings.ADVISOR_DATA_PATH.stat().st_size < 1000:
        logger.info("💰 Generating enriched financial advisor dataset...")
        num_records = 1000
        users = [f"user_{i}" for i in range(101, 115)] + ["demo_user", "premium_user", "test_subject"]
        categories = ["grocery_pos", "entertainment", "gas_transport", "shopping_net", "food_out", "health_fitness", "subscriptions", "utilities"]
        merchants = ["SuperStore", "HighEndClub", "GasStation", "Amazon", "Starbucks", "Nike", "Apple", "Walmart", "Netflix", "Spotify", "Hulu"]
        archetypes = ["Frugal", "Impulsive", "Strategic", "Conservative"]
        
        advisor_records = []
        for i in range(num_records):
            user = np.random.choice(users)
            dt = pd.Timestamp("2024-01-01") + pd.Timedelta(days=np.random.randint(0, 270))
            advisor_records.append({
                "user_id": user,
                "transaction_date": dt,
                "month_key": dt.strftime("%Y-%m"),
                "amount": np.random.uniform(5, 500),
                "category": np.random.choice(categories),
                "merchant": np.random.choice(merchants),
                "is_subscription": np.random.choice([True, False], p=[0.1, 0.9]),
                "credit_score_impact_category": np.random.choice(["positive", "negative", "neutral"], p=[0.4, 0.1, 0.5]),
                "archetype": np.random.choice(archetypes),
                "spending_velocity_7d": np.random.uniform(100, 2000),
                "risk_score": np.random.uniform(0, 0.3),
                "is_fraud_flag": 0
            })
        df_advisor = pd.DataFrame(advisor_records)
        df_advisor.to_csv(settings.ADVISOR_DATA_PATH, index=False)
        logger.info("✅ Financial Advisor dataset enriched.")

    logger.info("🚀 All system artifacts are synchronized and ready.")

if __name__ == "__main__":
    setup_data()
