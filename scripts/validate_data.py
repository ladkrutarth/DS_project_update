#!/usr/bin/env python3
import sys
import csv
from pathlib import Path

# Ensure we can import from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try to get settings
try:
    from config.config import settings
    PATHS = {
        "CFPB Data": settings.CFPB_DATA_PATH,
        "Fraud Training": settings.FRAUD_TRAIN_PATH,
        "Financial Advisor": settings.ADVISOR_DATA_PATH,
        "Spending DNA": settings.DNA_DATA_PATH,
        "Transactions": settings.TRANSACTIONS_DATA_PATH
    }
    DATA_DIR = settings.DATA_DIR
    MODELS_DIR = settings.MODELS_DIR
except Exception:
    print("⚠️  Could not load settings. Using defaults.")
    DATA_DIR = PROJECT_ROOT / "dataset" / "csv_data"
    MODELS_DIR = PROJECT_ROOT / "models"
    PATHS = {
        "CFPB Data": DATA_DIR / "cfpb_credit_card.csv",
        "Fraud Training": DATA_DIR / "processed_fraud_train.csv",
        "Financial Advisor": DATA_DIR / "financial_advisor_dataset.csv",
        "Spending DNA": DATA_DIR / "spending_dna_dataset.csv",
        "Transactions": DATA_DIR / "transactions_3000.csv"
    }

def validate_csv(name, path):
    print(f"🔍 Validating {name}...")
    if not path.exists():
        print(f"❌ FAIL: {path} does not exist.")
        return False
    
    try:
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            headers = next(reader)
            if not headers:
                print(f"❌ FAIL: {path} has no headers.")
                return False
            
            # Check for at least one data row
            try:
                first_row = next(reader)
                if not first_row:
                    print(f"❌ FAIL: {path} has no data.")
                    return False
            except StopIteration:
                print(f"❌ FAIL: {path} has no data.")
                return False
                
        print(f"✅ PASS: Valid CSV at {path} ({len(headers)} columns)")
        return True
    except Exception as e:
        print(f"❌ FAIL: Error reading {path}: {e}")
        return False

def main():
    print("🛡️  Veriscan Resilient Validation Suite\n" + "="*40)
    
    overall_pass = True
    
    for name, path in PATHS.items():
        if not validate_csv(name, path):
            overall_pass = False
            
    print("\n📂 Checking Directories...")
    for dname, dpath in [("DATA_DIR", DATA_DIR), ("MODELS_DIR", MODELS_DIR)]:
        if dpath.exists():
            print(f"✅ {dname}: {dpath}")
        else:
            print(f"❌ {dname} MISSING: {dpath}")
            overall_pass = False
            
    print("\n" + "="*40)
    if overall_pass:
        print("🎉 SUMMARY: Data Integrity Audit PASSED.")
    else:
        print("⚠️  SUMMARY: Data Integrity Audit FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main()
