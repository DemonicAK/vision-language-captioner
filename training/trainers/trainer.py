"""Model trainer abstraction.

This module provides a high-level training interface
for image captioning models with experiment tracking.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import tensorflow as tf

from training.configs import Config

if TYPE_CHECKING:
    from training.utils.experiment_tracking import ExperimentTracker

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Container for training results.
    
    Attributes:
        history: Training history dictionary.
        best_epoch: Epoch with best validation loss.
        final_train_loss: Final training loss.
        final_val_loss: Final validation loss.
        model_path: Path to saved model.
        run_summary_path: Path to experiment run summary.
    """
    history: Dict[str, List[float]]
    best_epoch: int
    final_train_loss: float
    final_val_loss: Optional[float] = None
    model_path: Optional[str] = None
    run_summary_path: Optional[str] = None
    
    @property
    def epochs_trained(self) -> int:
        """Number of epochs trained."""
        return len(self.history.get("loss", []))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "history": self.history,
            "best_epoch": self.best_epoch,
            "final_train_loss": self.final_train_loss,
            "final_val_loss": self.final_val_loss,
            "model_path": self.model_path,
            "run_summary_path": self.run_summary_path,
            "epochs_trained": self.epochs_trained,
        }


class CallbackFactory:
    """Factory for creating training callbacks.
    
    Creates commonly used Keras callbacks with
    sensible defaults.
    """
    
    @staticmethod
    def model_checkpoint(
        filepath: str,
        monitor: str = "loss",
        save_best_only: bool = True,
    ) -> tf.keras.callbacks.ModelCheckpoint:
        """Create model checkpoint callback."""
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor=monitor,
            save_best_only=save_best_only,
            verbose=1,
        )
    
    @staticmethod
    def early_stopping(
        patience: int = 5,
        monitor: str = "val_loss",
        restore_best_weights: bool = True,
    ) -> tf.keras.callbacks.EarlyStopping:
        """Create early stopping callback."""
        return tf.keras.callbacks.EarlyStopping(
            patience=patience,
            monitor=monitor,
            restore_best_weights=restore_best_weights,
            verbose=1,
        )
    
    @staticmethod
    def reduce_lr_on_plateau(
        factor: float = 0.5,
        patience: int = 3,
        min_lr: float = 1e-6,
        monitor: str = "val_loss",
    ) -> tf.keras.callbacks.ReduceLROnPlateau:
        """Create learning rate reduction callback."""
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=1,
        )
    
    @staticmethod
    def tensorboard(
        log_dir: str,
        histogram_freq: int = 1,
    ) -> tf.keras.callbacks.TensorBoard:
        """Create TensorBoard callback."""
        return tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=histogram_freq,
        )


class Trainer:
    """High-level trainer for caption models.
    
    Encapsulates the training loop with callbacks,
    logging, experiment tracking, and model saving.
    
    Attributes:
        model: Keras model to train.
        config: Training configuration.
        artifacts_dir: Directory for saving artifacts.
        tracker: Experiment tracker for logging.
    
    Example:
        >>> trainer = Trainer(model, config, run_name="experiment_001")
        >>> result = trainer.train(train_ds, val_ds, steps_per_epoch=100)
        >>> trainer.save_model("final_model.keras")
    """
    
    def __init__(
        self,
        model: tf.keras.Model,
        config: Config,
        artifacts_dir: Optional[str] = None,
        run_name: Optional[str] = None,
        tracker: Optional["ExperimentTracker"] = None,
    ) -> None:
        """Initialize trainer.
        
        Args:
            model: Keras model to train.
            config: Training configuration.
            artifacts_dir: Directory for saving artifacts.
            run_name: Name for the training run (for experiment tracking).
            tracker: Optional pre-configured experiment tracker.
        """
        self._model = model
        self._config = config
        self._artifacts_dir = Path(
            artifacts_dir or config.training.artifacts_dir
        )
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        self._history: Optional[tf.keras.callbacks.History] = None
        
        # Initialize experiment tracker
        if tracker is not None:
            self._tracker = tracker
        else:
            from training.utils.experiment_tracking import ExperimentTracker
            self._tracker = ExperimentTracker(
                output_dir=self._artifacts_dir,
                run_name=run_name,
                seed=config.data.random_seed,
            )
        
        # Log config
        self._log_config()
    
    @property
    def model(self) -> tf.keras.Model:
        """Get the model being trained."""
        return self._model
    
    @property
    def history(self) -> Optional[Dict[str, List[float]]]:
        """Get training history."""
        if self._history is None:
            return None
        return self._history.history
    
    @property
    def tracker(self) -> "ExperimentTracker":
        """Get the experiment tracker."""
        return self._tracker
    
    def _log_config(self) -> None:
        """Log configuration to experiment tracker."""
        config_dict = {
            "data": asdict(self._config.data),
            "model": asdict(self._config.model),
            "training": asdict(self._config.training),
        }
        self._tracker.log_config(config_dict)
    
    def _create_callbacks(
        self,
        extra_callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    ) -> List[tf.keras.callbacks.Callback]:
        """Create training callbacks.
        
        Args:
            extra_callbacks: Additional callbacks to include.
            
        Returns:
            List of callbacks.
        """
        cfg = self._config.training
        
        checkpoint_path = str(self._artifacts_dir / "checkpoint.keras")
        
        callbacks = [
            CallbackFactory.model_checkpoint(checkpoint_path),
            CallbackFactory.early_stopping(
                patience=cfg.early_stopping_patience,
            ),
            CallbackFactory.reduce_lr_on_plateau(
                factor=cfg.lr_decay_factor,
                patience=cfg.lr_patience,
                min_lr=cfg.min_lr,
            ),
        ]
        
        # TensorBoard logging (use tracker's logs directory)
        log_dir = str(self._tracker.logs_dir)
        callbacks.append(CallbackFactory.tensorboard(log_dir))
        
        # Add experiment tracking callback
        from training.utils.experiment_tracking import TFMetricsLoggingCallback
        if TFMetricsLoggingCallback is not None:
            callbacks.append(TFMetricsLoggingCallback(self._tracker))
        
        if extra_callbacks:
            callbacks.extend(extra_callbacks)
        
        return callbacks
    
    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: Optional[tf.data.Dataset] = None,
        steps_per_epoch: int = 100,
        validation_steps: Optional[int] = None,
        epochs: Optional[int] = None,
        extra_callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    ) -> TrainingResult:
        """Train the model.
        
        Args:
            train_dataset: Training dataset.
            val_dataset: Validation dataset (optional).
            steps_per_epoch: Number of steps per epoch.
            validation_steps: Number of validation steps.
            epochs: Number of epochs (uses config if not provided).
            extra_callbacks: Additional callbacks.
            
        Returns:
            TrainingResult with training metrics.
        """
        epochs = epochs or self._config.training.epochs
        callbacks = self._create_callbacks(extra_callbacks)
        
        logger.info(
            f"Starting training for {epochs} epochs, "
            f"{steps_per_epoch} steps/epoch"
        )
        
        # Train
        self._history = self._model.fit(
            train_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1,
        )
        
        # Find best epoch
        history = self._history.history
        val_losses = history.get("val_loss", history.get("loss", []))
        best_epoch = int(tf.argmin(val_losses).numpy()) + 1
        
        result = TrainingResult(
            history=history,
            best_epoch=best_epoch,
            final_train_loss=history["loss"][-1],
            final_val_loss=history.get("val_loss", [None])[-1],
        )
        
        logger.info(
            f"Training complete. Best epoch: {best_epoch}, "
            f"Final loss: {result.final_train_loss:.4f}"
        )
        
        return result
    
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
        self._tracker.log_bleu_scores(bleu1, bleu2, bleu3, bleu4, split)
    
    def finalize(self, error: Optional[str] = None) -> str:
        """Finalize training run and save summary.
        
        Args:
            error: Error message if run failed.
            
        Returns:
            Path to run summary file.
        """
        summary_path = self._tracker.finalize(error)
        return str(summary_path)
    
    def save_model(
        self,
        filename: str = "image_caption_model_final.keras",
    ) -> str:
        """Save the trained model.
        
        Args:
            filename: Output filename.
            
        Returns:
            Path to saved model.
        """
        output_path = self._artifacts_dir / filename
        self._model.save(str(output_path))
        logger.info(f"Saved model to {output_path}")
        return str(output_path)
    
    def load_weights(self, weights_path: str) -> None:
        """Load model weights.
        
        Args:
            weights_path: Path to weights file.
        """
        self._model.load_weights(weights_path)
        logger.info(f"Loaded weights from {weights_path}")


class DistributedTrainer(Trainer):
    """Trainer with distributed training support.
    
    Extends Trainer with multi-GPU training capabilities
    using TensorFlow distribution strategies.
    """
    
    def __init__(
        self,
        model: tf.keras.Model,
        config: Config,
        strategy: Optional[tf.distribute.Strategy] = None,
        artifacts_dir: Optional[str] = None,
    ) -> None:
        """Initialize distributed trainer.
        
        Args:
            model: Keras model to train.
            config: Training configuration.
            strategy: Distribution strategy (auto-detected if None).
            artifacts_dir: Directory for saving artifacts.
        """
        super().__init__(model, config, artifacts_dir)
        
        if strategy is None:
            strategy = self._detect_strategy()
        
        self._strategy = strategy
        logger.info(f"Using distribution strategy: {type(strategy).__name__}")
    
    @staticmethod
    def _detect_strategy() -> tf.distribute.Strategy:
        """Auto-detect the best distribution strategy."""
        gpus = tf.config.list_physical_devices("GPU")
        
        if len(gpus) > 1:
            return tf.distribute.MirroredStrategy()
        elif len(gpus) == 1:
            return tf.distribute.OneDeviceStrategy("/gpu:0")
        else:
            return tf.distribute.OneDeviceStrategy("/cpu:0")
    
    @property
    def strategy(self) -> tf.distribute.Strategy:
        """Get the distribution strategy."""
        return self._strategy
