import json
import time
from pathlib import Path
from datetime import datetime
from config.config import settings
from utils.logger import logger

class MetricTracker:
    """
    Tracks and saves experiment metrics to timestamped folders.
    """
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.experiment_dir = settings.get_experiment_path(experiment_name)
        self.metrics = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "results": {}
        }
        logger.info(f"🚀 Experiment '{experiment_name}' started. Artifacts: {self.experiment_dir}")

    def log_metric(self, key: str, value: any):
        """Log a single metric value."""
        self.metrics["results"][key] = value
        logger.info(f"📊 Metric: {key} = {value}")

    def save(self):
        """Save all tracked metrics to a JSON file in the experiment directory."""
        self.metrics["end_time"] = datetime.now().isoformat()
        metrics_file = self.experiment_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=4)
        logger.info(f"✅ Experiment metrics saved to {metrics_file}")
        return metrics_file

    def save_artifact(self, name: str, data: any, format: str = "json"):
        """Save a data artifact (e.g., model results) to the experiment folder."""
        path = self.experiment_dir / name
        if format == "json":
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
        else:
            with open(path, "w") as f:
                f.write(str(data))
        logger.info(f"💾 Artifact saved: {path}")
        return path

if __name__ == "__main__":
    # Test execution
    tracker = MetricTracker("smoke_test")
    tracker.log_metric("accuracy", 0.95)
    tracker.log_metric("latency_ms", 120)
    tracker.save()
