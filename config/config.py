import os
from pathlib import Path
from typing import Optional
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    # Minimal fallback for environments without pydantic_settings
    class BaseSettings: pass
    class SettingsConfigDict: pass

class VeriscanSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # --- Project Metadata ---
    PROJECT_NAME: str = "Veriscan"
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

    # --- Environment & Platform ---
    ENV: str = "development"
    DEBUG: bool = True
    # Fast mode uses a smaller model (1B instead of 8B) for quicker local demos
    VERISCAN_FAST_MODE: bool = os.environ.get("VERISCAN_FAST_MODE", "false").lower() == "true"

    # --- API Configuration ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    VERISCAN_API_URL: str = "http://localhost:8000"

    # --- LLM Configuration ---
    # Default is Llama-3-8B; in fast mode we prefer Llama-3.2-1B
    DEFAULT_LLM_MODEL: str = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
    FAST_LLM_MODEL: str = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    
    @property
    def LLM_MODEL_ID(self) -> str:
        return self.FAST_LLM_MODEL if self.VERISCAN_FAST_MODE else self.DEFAULT_LLM_MODEL

    LLM_MAX_TOKENS: int = 500
    LLM_TEMPERATURE: float = 0.2

    # --- Data Paths ---
    DATA_DIR: Path = PROJECT_ROOT / "dataset" / "csv_data"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    
    # Specific Files
    CFPB_DATA_PATH: Path = DATA_DIR / "cfpb_credit_card.csv"
    FRAUD_TRAIN_PATH: Path = DATA_DIR / "processed_fraud_train.csv"
    FRAUD_SCORES_PATH: Path = DATA_DIR / "fraud_scores_output.csv"
    AUTH_PROFILES_PATH: Path = DATA_DIR / "auth_profiles_output.csv"
    ADVISOR_DATA_PATH: Path = DATA_DIR / "financial_advisor_dataset.csv"
    DNA_DATA_PATH: Path = DATA_DIR / "spending_dna_dataset.csv"
    TRANSACTIONS_DATA_PATH: Path = DATA_DIR / "transactions_3000.csv"
    TXN_3000_PATH: Path = DATA_DIR / "transactions_3000.csv"
    
    # Model Files
    MODEL_RF_PATH: Path = MODELS_DIR / "fraud_model_rf.joblib"
    ENCODERS_PATH: Path = MODELS_DIR / "encoders.joblib"
    
    RAG_DB_PATH: Path = PROJECT_ROOT / ".chroma_db_local"

    # --- Aesthetics (Streamlit) ---
    CHART_TEXT_COLOR: str = "#0f172a"
    PRIMARY_COLOR: str = "#6366f1"
    SECONDARY_COLOR: str = "#a855f7"
    TEXT_COLOR_MAIN: str = "#1e293b"
    TEXT_COLOR_DARK: str = "#0f172a"
    BG_COLOR_LIGHT: str = "#f8fafc"

    # --- RAG Hyperparameters ---
    RAG_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    RAG_COLLECTION_NAME: str = "veriscan_intel_local"
    RAG_CFPB_INDEX_ROWS: int = 1000
    RAG_BATCH_SIZE: int = 100
    RAG_CONFIDENCE_THRESHOLD: float = 0.4
    RAG_DISTANCE_SCALE: float = 1.5
    RAG_CONTEXT_COUNT: int = 3

    # --- Financial Advisor Parameters ---
    ADVISOR_SAVINGS_TARGET: float = 100.0
    ADVISOR_TREND_THRESHOLD_PCT: float = 5.0
    
    # --- Credit Scoring Model ---
    CREDIT_BASE_SCORE: int = 680
    CREDIT_SCORE_MIN: int = 580
    CREDIT_SCORE_MAX: int = 850
    CREDIT_BASE_LIMIT_MIN: float = 5000.0
    CREDIT_LIMIT_MULTIPLIER: int = 3
    CREDIT_UTILIZATION_THRESHOLD: float = 0.3
    CREDIT_POS_WEIGHT: int = 40
    CREDIT_NEG_WEIGHT: int = 60

    # --- Spending DNA & Biometrics ---
    DNA_TRUST_THRESHOLD: float = 0.75
    DNA_MODERATE_THRESHOLD: float = 0.55
    DNA_DEVIATION_SCALE: float = 1.8

    # --- Transaction Analyst Parameters ---
    ANALYST_FRAUD_THRESHOLD: float = 0.7
    ANALYST_DEFAULT_LIMIT: int = 20
    ANALYST_DEFAULT_WINDOW: int = 30
    
    # --- Hardware & Resource Stability ---
    # Allowed: "auto", "cpu", "mps", "cuda"
    DEVICE: str = os.environ.get("VERISCAN_DEVICE", "auto")
    # Minimum free RAM in GB required to attempt loading the 8B model
    MEMORY_THRESHOLD_GB: float = 8.0
    # Whether to force CPU for embeddings even if GPU is available
    FORCE_CPU_EMBEDDINGS: bool = os.environ.get("VERISCAN_FORCE_CPU", "false").lower() == "true"

    def get_device(self) -> str:
        if self.DEVICE != "auto":
            return self.DEVICE
        
        # Simple auto-detection
        try:
            import torch
            if torch.cuda.is_available(): return "cuda"
            if torch.backends.mps.is_available(): return "mps"
        except (ImportError, AttributeError):
            pass
        return "cpu"

    # --- Logging & Artifact Management ---
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
    EXPERIMENTS_DIR: Path = ARTIFACTS_DIR / "experiments"
    LOG_LEVEL: str = os.environ.get("VERISCAN_LOG_LEVEL", "INFO")

    def get_experiment_path(self, name: str) -> Path:
        """Create a timestamped experiment folder."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.EXPERIMENTS_DIR / f"{name}_{timestamp}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_dirs(self):
        """Ensure all required system directories exist."""
        for d in [self.DATA_DIR, self.MODELS_DIR, self.LOGS_DIR, self.ARTIFACTS_DIR, self.EXPERIMENTS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

def get_settings():
    return VeriscanSettings()

settings = get_settings()
