"""Abstract base class for feature extractors.

This module defines the interface that all feature extractors
must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np


class BaseFeatureExtractor(ABC):
    """Abstract base class for image feature extractors.
    
    Defines the interface for extracting feature vectors from images
    using pre-trained CNN backbones.
    
    Subclasses must implement:
        - build_model(): Create the feature extraction model.
        - preprocess(): Preprocess images for the model.
        - feature_dim: Property returning feature dimension.
    
    Example:
        >>> extractor = SomeExtractor(image_size=(300, 300))
        >>> features = extractor.extract_features(image_keys, images_path)
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (299, 299),
        batch_size: int = 32,
        weights: str = "imagenet",
    ) -> None:
        """Initialize the feature extractor.
        
        Args:
            image_size: Input image size as (height, width).
            batch_size: Batch size for feature extraction.
            weights: Pre-trained weights to use.
        """
        self._image_size = image_size
        self._batch_size = batch_size
        self._weights = weights
        self._model: Optional[object] = None
    
    @property
    def image_size(self) -> Tuple[int, int]:
        """Input image size."""
        return self._image_size
    
    @property
    def batch_size(self) -> int:
        """Batch size for extraction."""
        return self._batch_size
    
    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Dimension of extracted features."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the feature extractor."""
        pass
    
    @abstractmethod
    def build_model(self) -> None:
        """Build the feature extraction model.
        
        Must be called before extracting features.
        """
        pass
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for the model.
        
        Args:
            image: Input image array.
            
        Returns:
            Preprocessed image array.
        """
        pass
    
    @abstractmethod
    def extract_features(
        self,
        image_keys: List[str],
        images_path: str,
        verbose: int = 1,
    ) -> Dict[str, np.ndarray]:
        """Extract features for a batch of images.
        
        Args:
            image_keys: List of image identifiers (without extension).
            images_path: Path to directory containing images.
            verbose: Verbosity level (0=silent, 1=progress).
            
        Returns:
            Dictionary mapping image_key to feature vector.
        """
        pass
    
    def save_features(
        self,
        features: Dict[str, np.ndarray],
        output_path: str | Path,
        use_float16: bool = True,
    ) -> None:
        """Save extracted features to file.
        
        Args:
            features: Dictionary of features to save.
            output_path: Path to output file.
            use_float16: Whether to save as float16 for 50% memory reduction.
                        EfficientNet features don't need float32 precision.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if use_float16:
            features = {k: v.astype(np.float16) for k, v in features.items()}
            
        np.save(output_path, features)
    
    @staticmethod
    def load_features(
        input_path: str | Path,
        mmap_mode: Optional[Literal["r", "r+", "w+", "c"]] = None,
    ) -> Dict[str, np.ndarray]:
        """Load features from file.
        
        Args:
            input_path: Path to features file.
            mmap_mode: Memory-mapping mode. Use "r" for read-only mmap
                      to avoid loading entire file into RAM. Options:
                      - None: Load all into RAM (default, original behavior)
                      - "r": Read-only memory-mapped (recommended for large files)
                      - "r+": Read-write memory-mapped
                      - "c": Copy-on-write memory-mapped
            
        Returns:
            Dictionary of loaded features.
        """
        return np.load(input_path, allow_pickle=True, mmap_mode=mmap_mode).item()
