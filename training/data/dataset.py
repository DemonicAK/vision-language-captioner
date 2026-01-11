"""TensorFlow dataset builders for training.

This module provides dataset construction utilities for creating
efficient tf.data pipelines for caption model training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
from requests import options
import tensorflow as tf

from training.data.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


@dataclass
class TrainingSample:
    """Single training sample for caption model.
    
    Attributes:
        image_key: Image identifier.
        input_sequence: Partial caption sequence (input).
        target_word: Next word to predict (target).
    """
    image_key: str
    input_sequence: List[int]
    target_word: int


class DatasetBuilder:
    """Builder for TensorFlow training datasets.
    
    Creates efficient tf.data pipelines for training caption models
    with image features and text sequences.
    
    Attributes:
        max_length: Maximum sequence length.
        feature_dim: Dimension of image features.
        batch_size: Number of samples per batch.
    
    Example:
        >>> builder = DatasetBuilder(max_length=38, feature_dim=1536, batch_size=64)
        >>> samples = builder.build_samples(descriptions, features, tokenizer)
        >>> dataset = builder.create_dataset(samples, features)
    """
    
    def __init__(
        self,
        max_length: int,
        feature_dim: int,
        batch_size: int = 64,
        shuffle_buffer: int = 10000,
    ) -> None:
        """Initialize dataset builder.
        
        Args:
            max_length: Maximum sequence length for padding.
            feature_dim: Dimension of image features.
            batch_size: Number of samples per batch.
            shuffle_buffer: Size of shuffle buffer.
        """
        self._max_length = max_length
        self._feature_dim = feature_dim
        self._batch_size = batch_size
        self._shuffle_buffer = shuffle_buffer
    
    @property
    def max_length(self) -> int:
        """Maximum sequence length."""
        return self._max_length
    
    @property
    def feature_dim(self) -> int:
        """Dimension of image features."""
        return self._feature_dim
    
    @property
    def batch_size(self) -> int:
        """Batch size."""
        return self._batch_size
    
    def build_samples(
        self,
        descriptions: Dict[str, List[str]],
        features: Dict[str, np.ndarray],
        tokenizer: Tokenizer,
    ) -> List[TrainingSample]:
        """Build training samples from descriptions and features.
        
        Creates (image, partial_sequence, next_word) tuples for
        teacher forcing training.
        
        Args:
            descriptions: Dictionary of image_id -> captions.
            features: Dictionary of image_id -> feature vectors.
            tokenizer: Fitted tokenizer instance.
            
        Returns:
            List of TrainingSample objects.
        """
        samples: List[TrainingSample] = []
        
        for image_key, captions in descriptions.items():
            if image_key not in features:
                continue
            
            for caption in captions:
                sequence = tokenizer.encode(caption)
                
                # Create input-output pairs for each position
                for i in range(1, len(sequence)):
                    samples.append(TrainingSample(
                        image_key=image_key,
                        input_sequence=sequence[:i],
                        target_word=sequence[i],
                    ))
        
        logger.info(f"Built {len(samples)} training samples")
        return samples
    
    def create_dataset(
        self,
        samples: List[TrainingSample],
        features: Dict[str, np.ndarray],
        shuffle: bool = True,
        repeat: bool = True,
    ) -> tf.data.Dataset:
        """Create TensorFlow dataset from samples.
        
        Args:
            samples: List of TrainingSample objects.
            features: Dictionary of image_id -> feature vectors.
            shuffle: Whether to shuffle the dataset.
            repeat: Whether to repeat the dataset infinitely.
            
        Returns:
            tf.data.Dataset yielding ((image_features, text_input), target).
        """
        def generator() -> Iterator[Tuple[str, np.ndarray, int]]:
            for sample in samples:
                yield (
                    sample.image_key,
                    np.array(sample.input_sequence, dtype=np.int32),
                    np.int32(sample.target_word),
                )
        
        output_signature = (
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature,
        )
        
        def map_fn(
            key: tf.Tensor,
            input_seq: tf.Tensor,
            target: tf.Tensor,
        ) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
            # Look up image features
            image_features = tf.numpy_function(
                lambda k: features[k.decode("utf-8")].astype(np.float32),
                [key],
                tf.float32,
            )
            image_features.set_shape([self._feature_dim])
            
            # Pad input sequence
            padded_seq = tf.pad(
                input_seq,
                [[0, self._max_length - tf.shape(input_seq)[0]]],
            )
            padded_seq = padded_seq[:self._max_length]
            
            return (image_features, padded_seq), target
        
        if shuffle:
            dataset = dataset.shuffle(self._shuffle_buffer)
        
        dataset = (
            dataset
            .map(map_fn, num_parallel_calls=4)
            .batch(self._batch_size, drop_remainder=True)
        )
        
        if repeat:
            dataset = dataset.repeat()
        
        dataset = dataset.prefetch(1)
        options = tf.data.Options()
        options.experimental_deterministic = False
        dataset = dataset.with_options(options)
        
        return dataset
    
    def compute_steps_per_epoch(self, num_samples: int) -> int:
        """Compute number of steps per epoch.
        
        Args:
            num_samples: Total number of training samples.
            
        Returns:
            Number of batches per epoch.
        """
        return max(1, num_samples // self._batch_size)
