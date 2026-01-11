"""Model loading and management for inference.

This module provides a production-safe model loader that avoids
global state and supports lazy loading, versioning, and concurrency.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for model loading.
    
    Attributes:
        model_path: Path to the Keras model file.
        vocab_path: Path to vocabulary directory (containing wordtoix.json, ixtoword.json).
        max_length: Maximum sequence length.
        feature_extractor_name: Name of the feature extractor backbone.
        image_size: Input image size as (height, width).
    """
    model_path: str
    vocab_path: str
    max_length: int = 38
    feature_extractor_name: str = "EfficientNetB3"
    image_size: Tuple[int, int] = (300, 300)
    
    @classmethod
    def from_artifacts_dir(
        cls,
        artifacts_dir: str,
        model_filename: str = "image_caption_model_final.keras",
    ) -> "ModelConfig":
        """Create config from artifacts directory.
        
        Args:
            artifacts_dir: Path to artifacts directory.
            model_filename: Name of the model file.
            
        Returns:
            ModelConfig instance.
        """
        artifacts_path = Path(artifacts_dir)
        return cls(
            model_path=str(artifacts_path / model_filename),
            vocab_path=str(artifacts_path),
        )


class ModelRegistry:
    """Thread-safe registry for managing loaded models.
    
    Implements singleton pattern with lazy loading to avoid
    loading models at import time.
    
    Example:
        >>> registry = ModelRegistry()
        >>> model_bundle = registry.get_model("default")
        >>> caption = model_bundle.predict(image_features)
    """
    
    _instance: Optional["ModelRegistry"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "ModelRegistry":
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize registry (only once due to singleton)."""
        if self._initialized:
            return
        
        self._models: Dict[str, "ModelBundle"] = {}
        self._configs: Dict[str, ModelConfig] = {}
        self._model_lock = threading.Lock()
        self._initialized = True
    
    def register(self, name: str, config: ModelConfig) -> None:
        """Register a model configuration.
        
        Args:
            name: Model identifier.
            config: Model configuration.
        """
        self._configs[name] = config
        logger.info(f"Registered model config: {name}")
    
    def get_model(self, name: str = "default") -> "ModelBundle":
        """Get or load a model by name.
        
        Thread-safe lazy loading of models.
        
        Args:
            name: Model identifier.
            
        Returns:
            Loaded ModelBundle.
            
        Raises:
            KeyError: If model name not registered.
        """
        if name not in self._configs:
            raise KeyError(f"Model '{name}' not registered")
        
        if name not in self._models:
            with self._model_lock:
                # Double-check after acquiring lock
                if name not in self._models:
                    config = self._configs[name]
                    self._models[name] = ModelBundle.load(config)
                    logger.info(f"Loaded model: {name}")
        
        return self._models[name]
    
    def is_loaded(self, name: str) -> bool:
        """Check if a model is loaded.
        
        Args:
            name: Model identifier.
            
        Returns:
            True if model is loaded.
        """
        return name in self._models
    
    def unload(self, name: str) -> None:
        """Unload a model to free memory.
        
        Args:
            name: Model identifier.
        """
        with self._model_lock:
            if name in self._models:
                del self._models[name]
                logger.info(f"Unloaded model: {name}")
    
    def clear(self) -> None:
        """Unload all models."""
        with self._model_lock:
            self._models.clear()
            logger.info("Cleared all models")


class ModelBundle:
    """Bundle containing model, vocabulary, and feature extractor.
    
    Encapsulates all components needed for inference.
    
    Attributes:
        model: Loaded Keras model.
        word_to_index: Word to index mapping.
        index_to_word: Index to word mapping.
        feature_extractor: Image feature extractor.
        config: Model configuration.
    """
    
    def __init__(
        self,
        model: Any,  # tf.keras.Model
        word_to_index: Dict[str, int],
        index_to_word: Dict[int, str],
        feature_extractor: Any,  # tf.keras.Model
        config: ModelConfig,
    ) -> None:
        """Initialize model bundle.
        
        Args:
            model: Keras caption model.
            word_to_index: Vocabulary mapping.
            index_to_word: Reverse vocabulary mapping.
            feature_extractor: Feature extraction model.
            config: Model configuration.
        """
        self._model = model
        self._word_to_index = word_to_index
        self._index_to_word = index_to_word
        self._feature_extractor = feature_extractor
        self._config = config
    
    @property
    def model(self) -> Any:
        """Get caption model."""
        return self._model
    
    @property
    def word_to_index(self) -> Dict[str, int]:
        """Get word to index mapping."""
        return self._word_to_index
    
    @property
    def index_to_word(self) -> Dict[int, str]:
        """Get index to word mapping."""
        return self._index_to_word
    
    @property
    def feature_extractor(self) -> Any:
        """Get feature extractor."""
        return self._feature_extractor
    
    @property
    def config(self) -> ModelConfig:
        """Get model config."""
        return self._config
    
    @property
    def max_length(self) -> int:
        """Get maximum sequence length."""
        return self._config.max_length
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self._word_to_index) + 1
    
    @classmethod
    def load(cls, config: ModelConfig) -> "ModelBundle":
        """Load model bundle from config.
        
        Args:
            config: Model configuration.
            
        Returns:
            Loaded ModelBundle.
        """
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        
        logger.info(f"Loading model from {config.model_path}")
        
        # Load caption model
        model = load_model(config.model_path, compile=False)
        
        # Load vocabulary
        vocab_path = Path(config.vocab_path)
        
        with open(vocab_path / "wordtoix.json", "r") as f:
            word_to_index = json.load(f)
        
        with open(vocab_path / "ixtoword.json", "r") as f:
            index_to_word_str = json.load(f)
            index_to_word = {int(k): v for k, v in index_to_word_str.items()}
        
        logger.info(f"Loaded vocabulary: {len(word_to_index)} words")
        
        # Load feature extractor
        feature_extractor = cls._create_feature_extractor(config)
        
        return cls(
            model=model,
            word_to_index=word_to_index,
            index_to_word=index_to_word,
            feature_extractor=feature_extractor,
            config=config,
        )
    
    @staticmethod
    def _create_feature_extractor(config: ModelConfig) -> Any:
        """Create feature extractor model.
        
        Args:
            config: Model configuration.
            
        Returns:
            Feature extractor model.
        """
        from tensorflow.keras.applications import EfficientNetB3
        
        extractor_map = {
            "EfficientNetB3": EfficientNetB3,
        }
        
        extractor_class = extractor_map.get(config.feature_extractor_name)
        if extractor_class is None:
            raise ValueError(
                f"Unknown feature extractor: {config.feature_extractor_name}"
            )
        
        extractor = extractor_class(
            include_top=False,
            weights="imagenet",
            pooling="avg",
        )
        extractor.trainable = False
        
        logger.info(f"Created feature extractor: {config.feature_extractor_name}")
        return extractor


def get_default_model_bundle(artifacts_dir: str = "/app/artifacts") -> ModelBundle:
    """Get the default model bundle, loading if necessary.
    
    Convenience function for simple use cases.
    
    Args:
        artifacts_dir: Path to artifacts directory.
        
    Returns:
        Loaded ModelBundle.
    """
    registry = ModelRegistry()
    
    if "default" not in registry._configs:
        config = ModelConfig.from_artifacts_dir(artifacts_dir)
        registry.register("default", config)
    
    return registry.get_model("default")


# Lazy loading wrapper for backward compatibility
class _LazyModelLoader:
    """Lazy loader that provides backward compatibility.
    
    Loads model only when attributes are accessed.
    """
    
    _bundle: Optional[ModelBundle] = None
    _artifacts_dir: str = "/app/artifacts"
    
    @classmethod
    def _ensure_loaded(cls) -> ModelBundle:
        """Ensure model is loaded."""
        if cls._bundle is None:
            cls._bundle = get_default_model_bundle(cls._artifacts_dir)
        return cls._bundle
    
    @classmethod
    def get_model(cls) -> Any:
        """Get caption model."""
        return cls._ensure_loaded().model
    
    @classmethod
    def get_word_to_index(cls) -> Dict[str, int]:
        """Get word to index mapping."""
        return cls._ensure_loaded().word_to_index
    
    @classmethod
    def get_index_to_word(cls) -> Dict[int, str]:
        """Get index to word mapping."""
        return cls._ensure_loaded().index_to_word
    
    @classmethod
    def get_feature_extractor(cls) -> Any:
        """Get feature extractor."""
        return cls._ensure_loaded().feature_extractor
    
    @classmethod
    def get_max_length(cls) -> int:
        """Get max sequence length."""
        return cls._ensure_loaded().max_length
