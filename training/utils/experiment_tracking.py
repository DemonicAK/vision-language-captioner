"""Experiment tracking and reproducibility utilities.

This module provides comprehensive experiment tracking including:
- Run summaries with config snapshots
- Git commit tracking
- Metrics logging (CSV and JSON)
- Artifact management
- Seed management for reproducibility
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import random
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # TensorFlow seed
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
    except ImportError:
        pass
    
    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    logger.info(f"Set global random seed to {seed}")


def get_git_info() -> Dict[str, Any]:
    """Get current git repository information.
    
    Returns:
        Dictionary with git commit, branch, and status.
    """
    info: Dict[str, Any] = {
        "commit": None,
        "branch": None,
        "dirty": None,
        "remote_url": None,
    }
    
    try:
        # Get current commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["commit"] = result.stdout.strip()
        
        # Get current branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()
        
        # Check if repo is dirty
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["dirty"] = len(result.stdout.strip()) > 0
        
        # Get remote URL
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["remote_url"] = result.stdout.strip()
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.warning("Could not retrieve git information")
    
    return info


@dataclass
class RunMetadata:
    """Metadata for a training run.
    
    Attributes:
        run_id: Unique identifier for this run.
        run_name: Human-readable name for the run.
        start_time: Timestamp when run started.
        end_time: Timestamp when run ended.
        status: Run status (running, completed, failed).
        git_info: Git repository information.
        environment: Environment variables and system info.
    """
    run_id: str
    run_name: str
    start_time: str
    end_time: Optional[str] = None
    status: str = "running"
    git_info: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(cls, run_name: Optional[str] = None) -> "RunMetadata":
        """Create new run metadata with auto-generated ID."""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        if run_name is None:
            run_name = f"run_{run_id}"
        
        return cls(
            run_id=run_id,
            run_name=run_name,
            start_time=datetime.now().isoformat(),
            git_info=get_git_info(),
            environment=cls._get_environment_info(),
        )
    
    @staticmethod
    def _get_environment_info() -> Dict[str, Any]:
        """Collect environment information."""
        import platform
        
        info: Dict[str, Any] = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "hostname": platform.node(),
        }
        
        # TensorFlow info
        try:
            import tensorflow as tf
            info["tensorflow_version"] = tf.__version__
            info["gpu_available"] = len(tf.config.list_physical_devices("GPU")) > 0
            info["gpu_devices"] = [
                d.name for d in tf.config.list_physical_devices("GPU")
            ]
        except ImportError:
            pass
        
        return info
    
    def complete(self) -> None:
        """Mark run as completed."""
        self.end_time = datetime.now().isoformat()
        self.status = "completed"
    
    def fail(self, error: str) -> None:
        """Mark run as failed."""
        self.end_time = datetime.now().isoformat()
        self.status = "failed"
        self.environment["error"] = error


@dataclass
class MetricsRecord:
    """Single metrics record for an epoch.
    
    Attributes:
        epoch: Epoch number.
        train_loss: Training loss.
        val_loss: Validation loss (optional).
        bleu1: BLEU-1 score (optional).
        bleu2: BLEU-2 score (optional).
        bleu3: BLEU-3 score (optional).
        bleu4: BLEU-4 score (optional).
        learning_rate: Current learning rate.
        epoch_time_seconds: Time taken for epoch.
        additional: Additional custom metrics.
    """
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    bleu1: Optional[float] = None
    bleu2: Optional[float] = None
    bleu3: Optional[float] = None
    bleu4: Optional[float] = None
    learning_rate: Optional[float] = None
    epoch_time_seconds: Optional[float] = None
    additional: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = asdict(self)
        # Flatten additional metrics
        additional = result.pop("additional", {})
        result.update(additional)
        return {k: v for k, v in result.items() if v is not None}


class ExperimentTracker:
    """Comprehensive experiment tracking and artifact management.
    
    Tracks training runs including:
    - Configuration snapshots
    - Epoch-wise metrics
    - BLEU scores and evaluation results
    - Model checkpoints and artifacts
    - TensorBoard logs
    
    Example:
        >>> tracker = ExperimentTracker(output_dir="experiments/run_001")
        >>> tracker.log_config(config_dict)
        >>> tracker.log_metrics(epoch=1, train_loss=2.5, val_loss=2.8)
        >>> tracker.log_bleu_scores(bleu1=0.65, bleu2=0.45, bleu3=0.35, bleu4=0.25)
        >>> tracker.finalize()
    """
    
    def __init__(
        self,
        output_dir: str | Path,
        run_name: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize experiment tracker.
        
        Args:
            output_dir: Base directory for experiment outputs.
            run_name: Human-readable name for the run.
            seed: Random seed for reproducibility.
        """
        self._output_dir = Path(output_dir)
        self._metadata = RunMetadata.create(run_name)
        self._seed = seed
        
        # Create directory structure
        self._setup_directories()
        
        # Set seed if provided
        if seed is not None:
            set_global_seed(seed)
            self._metadata.environment["seed"] = seed
        
        # Initialize metrics storage
        self._metrics: List[MetricsRecord] = []
        self._bleu_scores: Dict[str, float] = {}
        self._config: Dict[str, Any] = {}
        
        logger.info(f"Initialized experiment tracker: {self._metadata.run_id}")
    
    def _setup_directories(self) -> None:
        """Create output directory structure."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        (self._output_dir / "logs").mkdir(exist_ok=True)
        (self._output_dir / "checkpoints").mkdir(exist_ok=True)
        (self._output_dir / "artifacts").mkdir(exist_ok=True)
    
    @property
    def run_id(self) -> str:
        """Get run ID."""
        return self._metadata.run_id
    
    @property
    def output_dir(self) -> Path:
        """Get output directory."""
        return self._output_dir
    
    @property
    def logs_dir(self) -> Path:
        """Get TensorBoard logs directory."""
        return self._output_dir / "logs"
    
    @property
    def checkpoints_dir(self) -> Path:
        """Get checkpoints directory."""
        return self._output_dir / "checkpoints"
    
    @property
    def artifacts_dir(self) -> Path:
        """Get artifacts directory."""
        return self._output_dir / "artifacts"
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration snapshot.
        
        Args:
            config: Configuration dictionary.
        """
        self._config = config.copy()
        
        # Add config hash for change detection
        config_str = json.dumps(config, sort_keys=True)
        self._config["_config_hash"] = hashlib.md5(
            config_str.encode()
        ).hexdigest()[:8]
        
        # Save immediately
        config_path = self._output_dir / "config_snapshot.json"
        with open(config_path, "w") as f:
            json.dump(self._config, f, indent=2, default=str)
        
        logger.info(f"Logged config snapshot to {config_path}")
    
    def log_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        epoch_time_seconds: Optional[float] = None,
        **additional: Any,
    ) -> None:
        """Log metrics for an epoch.
        
        Args:
            epoch: Epoch number.
            train_loss: Training loss.
            val_loss: Validation loss.
            learning_rate: Current learning rate.
            epoch_time_seconds: Time taken for epoch.
            **additional: Additional custom metrics.
        """
        record = MetricsRecord(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=learning_rate,
            epoch_time_seconds=epoch_time_seconds,
            additional=additional,
        )
        self._metrics.append(record)
        
        # Append to CSV immediately
        self._append_metrics_csv(record)
        
        logger.debug(f"Logged metrics for epoch {epoch}")
    
    def _append_metrics_csv(self, record: MetricsRecord) -> None:
        """Append metrics record to CSV file."""
        csv_path = self._output_dir / "metrics.csv"
        
        record_dict = record.to_dict()
        file_exists = csv_path.exists()
        
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=record_dict.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(record_dict)
    
    def log_bleu_scores(
        self,
        bleu1: float,
        bleu2: float,
        bleu3: float,
        bleu4: float,
        split: str = "test",
    ) -> None:
        """Log BLEU evaluation scores.
        
        Args:
            bleu1: BLEU-1 score.
            bleu2: BLEU-2 score.
            bleu3: BLEU-3 score.
            bleu4: BLEU-4 score.
            split: Data split (train/val/test).
        """
        self._bleu_scores[f"{split}_bleu1"] = bleu1
        self._bleu_scores[f"{split}_bleu2"] = bleu2
        self._bleu_scores[f"{split}_bleu3"] = bleu3
        self._bleu_scores[f"{split}_bleu4"] = bleu4
        
        logger.info(
            f"BLEU scores ({split}): "
            f"BLEU-1={bleu1:.4f}, BLEU-2={bleu2:.4f}, "
            f"BLEU-3={bleu3:.4f}, BLEU-4={bleu4:.4f}"
        )
    
    def log_artifact(
        self,
        name: str,
        data: Any,
        artifact_type: str = "json",
    ) -> Path:
        """Log an artifact.
        
        Args:
            name: Artifact name.
            data: Artifact data.
            artifact_type: Type of artifact (json, numpy, pickle).
            
        Returns:
            Path to saved artifact.
        """
        artifact_path = self.artifacts_dir / name
        
        if artifact_type == "json":
            if not name.endswith(".json"):
                artifact_path = artifact_path.with_suffix(".json")
            with open(artifact_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        elif artifact_type == "numpy":
            if not name.endswith(".npy"):
                artifact_path = artifact_path.with_suffix(".npy")
            np.save(artifact_path, data)
        else:
            raise ValueError(f"Unknown artifact type: {artifact_type}")
        
        logger.info(f"Saved artifact: {artifact_path}")
        return artifact_path
    
    def save_model_info(
        self,
        model_path: str,
        vocab_size: int,
        max_length: int,
        feature_dim: int,
    ) -> None:
        """Save model metadata for deployment.
        
        Args:
            model_path: Path to saved model.
            vocab_size: Vocabulary size.
            max_length: Maximum sequence length.
            feature_dim: Feature dimension.
        """
        model_info = {
            "model_path": str(model_path),
            "vocab_size": vocab_size,
            "max_length": max_length,
            "feature_dim": feature_dim,
            "created_at": datetime.now().isoformat(),
            "run_id": self._metadata.run_id,
        }
        
        self.log_artifact("model_info", model_info)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get complete run summary.
        
        Returns:
            Dictionary with all run information.
        """
        # Compute summary statistics from metrics
        if self._metrics:
            final_metrics = self._metrics[-1]
            best_val_loss_idx = min(
                range(len(self._metrics)),
                key=lambda i: self._metrics[i].val_loss or float("inf"),
            )
            best_metrics = self._metrics[best_val_loss_idx]
        else:
            final_metrics = None
            best_metrics = None
        
        summary = {
            "metadata": asdict(self._metadata),
            "config": self._config,
            "final_metrics": final_metrics.to_dict() if final_metrics else None,
            "best_metrics": best_metrics.to_dict() if best_metrics else None,
            "bleu_scores": self._bleu_scores,
            "total_epochs": len(self._metrics),
            "artifacts": {
                "metrics_csv": str(self._output_dir / "metrics.csv"),
                "config_snapshot": str(self._output_dir / "config_snapshot.json"),
                "tensorboard_logs": str(self.logs_dir),
            },
        }
        
        return summary
    
    def finalize(self, error: Optional[str] = None) -> Path:
        """Finalize the experiment and save run summary.
        
        Args:
            error: Error message if run failed.
            
        Returns:
            Path to run summary file.
        """
        if error:
            self._metadata.fail(error)
        else:
            self._metadata.complete()
        
        summary = self.get_summary()
        
        # Save run summary
        summary_path = self._output_dir / "run_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Finalized experiment. Summary saved to {summary_path}")
        return summary_path


class MetricsLogger:
    """Lightweight metrics logger for training callbacks.
    
    Integrates with Keras callbacks to log metrics during training.
    
    Example:
        >>> logger = MetricsLogger(tracker)
        >>> callback = logger.get_keras_callback()
        >>> model.fit(..., callbacks=[callback])
    """
    
    def __init__(self, tracker: ExperimentTracker) -> None:
        """Initialize metrics logger.
        
        Args:
            tracker: Experiment tracker instance.
        """
        self._tracker = tracker
        self._epoch_start_time: Optional[float] = None
    
    def get_keras_callback(self) -> "MetricsLoggingCallback":
        """Get Keras callback for automatic metrics logging.
        
        Returns:
            Keras callback instance.
        """
        return MetricsLoggingCallback(self._tracker)


class MetricsLoggingCallback:
    """Keras callback for logging metrics to ExperimentTracker.
    
    Automatically logs training and validation metrics at each epoch.
    """
    
    def __init__(self, tracker: ExperimentTracker) -> None:
        """Initialize callback.
        
        Args:
            tracker: Experiment tracker instance.
        """
        self._tracker = tracker
        self._epoch_start_time: float = 0
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Called at the start of an epoch."""
        import time
        self._epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Called at the end of an epoch."""
        import time
        
        logs = logs or {}
        epoch_time = time.time() - self._epoch_start_time
        
        self._tracker.log_metrics(
            epoch=epoch + 1,  # 1-indexed
            train_loss=logs.get("loss", 0.0),
            val_loss=logs.get("val_loss"),
            learning_rate=logs.get("lr"),
            epoch_time_seconds=epoch_time,
            accuracy=logs.get("accuracy"),
        )


# TensorFlow Keras callback wrapper
try:
    import tensorflow as tf
    
    class TFMetricsLoggingCallback(tf.keras.callbacks.Callback):
        """TensorFlow Keras callback for experiment tracking."""
        
        def __init__(self, tracker: ExperimentTracker) -> None:
            super().__init__()
            self._tracker = tracker
            self._epoch_start_time: float = 0
        
        def on_epoch_begin(
            self, epoch: int, logs: Optional[Dict] = None
        ) -> None:
            import time
            self._epoch_start_time = time.time()
        
        def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
            import time
            
            logs = logs or {}
            epoch_time = time.time() - self._epoch_start_time
            
            # Get learning rate from optimizer
            lr = None
            if hasattr(self.model, "optimizer") and self.model.optimizer:
                try:
                    lr = float(self.model.optimizer.learning_rate)
                except Exception:
                    pass
            
            self._tracker.log_metrics(
                epoch=epoch + 1,
                train_loss=logs.get("loss", 0.0),
                val_loss=logs.get("val_loss"),
                learning_rate=lr,
                epoch_time_seconds=epoch_time,
                accuracy=logs.get("accuracy"),
            )

except ImportError:
    TFMetricsLoggingCallback = None  # type: ignore
