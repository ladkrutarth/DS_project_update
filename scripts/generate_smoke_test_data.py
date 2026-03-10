#!/usr/bin/env python3
import sys
import csv
from pathlib import Path

# Ensure we can import from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# We'll try to get settings, but if config.py fails due to pydantic import, we'll use fallbacks.
try:
    from config.config import settings
    DATA_DIR = settings.DATA_DIR
    CFPB_DATA_PATH = settings.CFPB_DATA_PATH
    FRAUD_TRAIN_PATH = settings.FRAUD_TRAIN_PATH
    ADVISOR_DATA_PATH = settings.ADVISOR_DATA_PATH
    DNA_DATA_PATH = settings.DNA_DATA_PATH
    TRANSACTIONS_DATA_PATH = settings.TRANSACTIONS_DATA_PATH
except Exception:
    print("⚠️  Could not load settings. Using default paths.")
    DATA_DIR = PROJECT_ROOT / "dataset" / "csv_data"
    CFPB_DATA_PATH = DATA_DIR / "cfpb_credit_card.csv"
    FRAUD_TRAIN_PATH = DATA_DIR / "processed_fraud_train.csv"
    ADVISOR_DATA_PATH = DATA_DIR / "financial_advisor_dataset.csv"
    DNA_DATA_PATH = DATA_DIR / "spending_dna_dataset.csv"
    TRANSACTIONS_DATA_PATH = DATA_DIR / "transactions_3000.csv"

def write_csv(path, headers, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"✅ Created {path}")

def generate_stubs():
    print("🧪 Generating Resilient Smoke Test Stubs...")
    
    # 1. CFPB Stub
    write_csv(CFPB_DATA_PATH, 
              ["Issue", "Consumer complaint narrative", "Company", "State", "Product"],
              [["Billing dispute", "Stub narrative", "VeriBank", "NY", "Credit card"],
               ["Late fee", "Stub narrative", "VeriBank", "CA", "Credit card"]])

    # 2. Fraud Training Stub
    write_csv(FRAUD_TRAIN_PATH,
              ["first", "last", "category", "amt", "gender", "state", "merchant", "hour", "day_of_week", "is_fraud"],
              [["Alice", "Test", "shopping_net", 50.0, "F", "NY", "Merchant A", 12, 1, 0],
               ["Bob", "Test", "travel", 1500.0, "M", "CA", "Merchant B", 1, 5, 1]])

    # 3. Advisor Dataset Stub
    write_csv(ADVISOR_DATA_PATH,
              ["user_id", "transaction_date", "amount", "category", "description"],
              [["user_123", "2026-03-01", 45.99, "Food", "Lunch"],
               ["user_123", "2026-03-02", -20.00, "Subscription", "Netflix"]])

    # 4. Spending DNA Stub
    write_csv(DNA_DATA_PATH,
              ["user_id", "hour_of_day", "amount", "category_encoded"],
              [["user_123", 14.5, 65.0, 3]])

    # 5. Transactions Stub
    write_csv(TRANSACTIONS_DATA_PATH,
              ["user_id", "transaction_date", "amount", "category", "merchant", "hour", "day_of_week"],
              [["user_123", "2026-03-01", 50.0, "grocery_pos", "SuperStore", 14, 2],
               ["user_123", "2026-03-02", 120.0, "entertainment", "Cinema", 20, 3]])

    print("🚀 Resilient smoke test data initialization complete.")

if __name__ == "__main__":
    generate_stubs()
