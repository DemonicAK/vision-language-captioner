"""Custom training callbacks.

This module provides custom Keras callbacks for
monitoring training progress and computing metrics.
"""

from __future__ import annotations

import gc
import logging
import time
from typing import Any, Dict, List, Optional

import tensorflow as tf

logger = logging.getLogger(__name__)


class MetricsCallback(tf.keras.callbacks.Callback):
    """Callback for computing and logging custom metrics.
    
    Tracks training metrics and provides periodic logging
    with customizable frequency.
    
    Attributes:
        log_frequency: How often to log metrics (in epochs).
    
    Example:
        >>> callback = MetricsCallback(log_frequency=5)
        >>> model.fit(data, callbacks=[callback])
    """
    
    def __init__(
        self,
        log_frequency: int = 1,
        log_to_file: Optional[str] = None,
    ) -> None:
        """Initialize metrics callback.
        
        Args:
            log_frequency: How often to log metrics (epochs).
            log_to_file: Path to log file (optional).
        """
        super().__init__()
        self._log_frequency = log_frequency
        self._log_to_file = log_to_file
        self._epoch_times: List[float] = []
        self._epoch_start: float = 0
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of training."""
        logger.info("Training started")
        self._train_start = time.time()
    
    def on_epoch_begin(
        self,
        epoch: int,
        logs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Called at the beginning of each epoch."""
        gc.collect()
        self._epoch_start = time.time()
    
    def on_epoch_end(
        self,
        epoch: int,
        logs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Called at the end of each epoch."""
        logs = logs or {}
        epoch_time = time.time() - self._epoch_start
        self._epoch_times.append(epoch_time)
        
        if (epoch + 1) % self._log_frequency == 0:
            loss = logs.get("loss", 0)
            val_loss = logs.get("val_loss")
            acc = logs.get("accuracy", logs.get("acc"))
            
            msg = (
                f"Epoch {epoch + 1}: "
                f"loss={loss:.4f}"
            )
            
            if val_loss is not None:
                msg += f", val_loss={val_loss:.4f}"
            if acc is not None:
                msg += f", acc={acc:.4f}"
            
            msg += f", time={epoch_time:.1f}s"
            
            logger.info(msg)
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of training."""
        total_time = time.time() - self._train_start
        avg_epoch_time = sum(self._epoch_times) / max(len(self._epoch_times), 1)
        
        logger.info(
            f"Training complete. "
            f"Total time: {total_time:.1f}s, "
            f"Avg epoch time: {avg_epoch_time:.1f}s"
        )


class ProgressCallback(tf.keras.callbacks.Callback):
    """Progress callback with ETA estimation.
    
    Provides detailed progress updates with
    estimated time remaining.
    """
    
    def __init__(self, total_epochs: int) -> None:
        """Initialize progress callback.
        
        Args:
            total_epochs: Total number of training epochs.
        """
        super().__init__()
        self._total_epochs = total_epochs
        self._epoch_times: List[float] = []
    
    def on_epoch_end(
        self,
        epoch: int,
        logs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Called at the end of each epoch."""
        if hasattr(self, "_epoch_start"):
            epoch_time = time.time() - self._epoch_start
            self._epoch_times.append(epoch_time)
            
            # Estimate remaining time
            avg_time = sum(self._epoch_times) / len(self._epoch_times)
            remaining = (self._total_epochs - epoch - 1) * avg_time
            
            hours, remainder = divmod(remaining, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            logger.info(
                f"Progress: {epoch + 1}/{self._total_epochs} "
                f"({100 * (epoch + 1) / self._total_epochs:.1f}%) - "
                f"ETA: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
            )
    
    def on_epoch_begin(
        self,
        epoch: int,
        logs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Called at the beginning of each epoch."""
        self._epoch_start = time.time()
