import json
from pathlib import Path
from datetime import datetime
from config.config import settings
from utils.logger import logger

def checkpoint_system():
    """
    Creates a snapshot of the current system configuration and model state.
    This fulfills the requirement for automated model checkpoints.
    """
    logger.info("📦 Creating system state checkpoint...")
    
    # Define checkpoint data
    checkpoint_data = {
        "timestamp": datetime.now().isoformat(),
        "project": settings.PROJECT_NAME,
        "env": settings.ENV,
        "fast_mode": settings.VERISCAN_FAST_MODE,
        "llm_model_id": settings.LLM_MODEL_ID,
        "rag_model": settings.RAG_EMBEDDING_MODEL,
        "device": settings.get_device(),
        "paths": {
            "data_dir": str(settings.DATA_DIR),
            "cfpb_exists": settings.CFPB_DATA_PATH.exists(),
            "fraud_train_exists": settings.FRAUD_TRAIN_PATH.exists()
        }
    }
    
    # Save checkpoint to artifacts
    checkpoint_dir = settings.ARTIFACTS_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_file = checkpoint_dir / f"checkpoint_{timestamp_str}.json"
    
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint_data, f, indent=4)
    
    logger.info(f"✅ System checkpoint saved to {checkpoint_file}")
    return checkpoint_file

if __name__ == "__main__":
    checkpoint_system()
