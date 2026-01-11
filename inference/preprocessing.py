"""Image preprocessing for inference.

This module provides image preprocessing utilities that work
with the model registry system.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple, TYPE_CHECKING

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

if TYPE_CHECKING:
    from PIL import Image as PILImage

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Preprocessor for converting images to feature vectors.
    
    Handles image resizing, normalization, and feature extraction.
    
    Example:
        >>> preprocessor = ImagePreprocessor(feature_extractor, (300, 300))
        >>> features = preprocessor.process(pil_image)
    """
    
    def __init__(
        self,
        feature_extractor: Any,
        image_size: Tuple[int, int] = (300, 300),
    ) -> None:
        """Initialize preprocessor.
        
        Args:
            feature_extractor: Feature extraction model.
            image_size: Target image size (height, width).
        """
        self._feature_extractor = feature_extractor
        self._image_size = image_size
    
    @property
    def image_size(self) -> Tuple[int, int]:
        """Get target image size."""
        return self._image_size
    
    def _load_and_preprocess(self, image: "PILImage.Image") -> tf.Tensor:
        """Preprocess PIL image for feature extraction.
        
        Args:
            image: PIL Image object.
            
        Returns:
            Preprocessed tensor ready for model.
        """
        # Resize image
        image_resized = image.resize(self._image_size)
        
        # Convert to numpy array
        image_array = np.array(image_resized, dtype=np.float32)
        
        # Apply EfficientNet preprocessing
        image_array = preprocess_input(image_array)
        
        # Add batch dimension
        image_batch = tf.expand_dims(image_array, axis=0)
        
        return image_batch
    
    def extract_features(self, image: "PILImage.Image") -> tf.Tensor:
        """Extract feature vector from image.
        
        Args:
            image: PIL Image object.
            
        Returns:
            Feature tensor of shape (1, feature_dim).
        """
        preprocessed = self._load_and_preprocess(image)
        features = self._feature_extractor(preprocessed, training=False)
        return features
    
    def process(self, image: "PILImage.Image") -> tf.Tensor:
        """Process image and return features.
        
        Alias for extract_features for convenience.
        
        Args:
            image: PIL Image object.
            
        Returns:
            Feature tensor.
        """
        return self.extract_features(image)


def create_preprocessor_from_bundle(bundle: Any) -> ImagePreprocessor:
    """Create preprocessor from model bundle.
    
    Args:
        bundle: ModelBundle instance.
        
    Returns:
        Configured ImagePreprocessor.
    """
    return ImagePreprocessor(
        feature_extractor=bundle.feature_extractor,
        image_size=bundle.config.image_size,
    )


# Legacy compatibility function
def image_preprocessor(image: "PILImage.Image") -> tf.Tensor:
    """Legacy interface for image preprocessing.
    
    Uses lazy loading to avoid import-time model loading.
    
    Args:
        image: PIL Image object.
        
    Returns:
        Feature tensor.
    """
    from inference.model_registry import _LazyModelLoader
    
    feature_extractor = _LazyModelLoader.get_feature_extractor()
    preprocessor = ImagePreprocessor(feature_extractor)
    return preprocessor.extract_features(image)
