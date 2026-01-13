"""TensorFlow dataset builders for training.

This module provides dataset construction utilities for creating
efficient tf.data pipelines for caption model training.

Uses pure TensorFlow operations (from_tensor_slices) for optimal
GPU performance without Python callbacks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from training.data.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


@dataclass
class PreparedData:
    """Pre-computed tensors for efficient tf.data pipeline.
    
    Attributes:
        image_features: Array of shape (num_samples, feature_dim).
        input_sequences: Padded sequences of shape (num_samples, max_length).
        targets: Target word indices of shape (num_samples,).
    """
    image_features: np.ndarray
    input_sequences: np.ndarray
    targets: np.ndarray


class DatasetBuilder:
    """Builder for TensorFlow training datasets.
    
    Creates efficient tf.data pipelines for training caption models
    with image features and text sequences. Uses from_tensor_slices
    for pure TensorFlow operations without Python callbacks.
    
    Attributes:
        max_length: Maximum sequence length.
        feature_dim: Dimension of image features.
        batch_size: Number of samples per batch.
    
    Example:
        >>> builder = DatasetBuilder(max_length=38, feature_dim=1536, batch_size=64)
        >>> prepared = builder.prepare_data(descriptions, features, tokenizer)
        >>> dataset = builder.create_dataset(prepared)
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
    
    def prepare_data(
        self,
        descriptions: Dict[str, List[str]],
        features: Dict[str, np.ndarray],
        tokenizer: Tokenizer,
    ) -> PreparedData:
        """Prepare all training data as numpy arrays.
        
        Pre-computes all samples with padding for efficient
        tf.data.from_tensor_slices pipeline.
        
        Args:
            descriptions: Dictionary of image_id -> captions.
            features: Dictionary of image_id -> feature vectors.
            tokenizer: Fitted tokenizer instance.
            
        Returns:
            PreparedData with pre-computed arrays.
        """
        image_features_list: List[np.ndarray] = []
        input_sequences_list: List[np.ndarray] = []
        targets_list: List[int] = []
        
        for image_key, captions in descriptions.items():
            if image_key not in features:
                continue
            
            img_feat = features[image_key]
            
            for caption in captions:
                sequence = tokenizer.encode(caption)
                
                # Create input-output pairs for each position
                for i in range(1, len(sequence)):
                    # Pad input sequence to max_length
                    input_seq = sequence[:i]
                    padded = np.zeros(self._max_length, dtype=np.int32)
                    seq_len = min(len(input_seq), self._max_length)
                    padded[:seq_len] = input_seq[:seq_len]
                    
                    image_features_list.append(img_feat)
                    input_sequences_list.append(padded)
                    targets_list.append(sequence[i])
        
        prepared = PreparedData(
            image_features=np.array(image_features_list, dtype=np.float32),
            input_sequences=np.array(input_sequences_list, dtype=np.int32),
            targets=np.array(targets_list, dtype=np.int32),
        )
        
        logger.info(f"Prepared {len(targets_list)} training samples as tensors")
        return prepared
    
    def create_dataset(
        self,
        prepared: PreparedData,
        shuffle: bool = True,
        repeat: bool = True,
    ) -> tf.data.Dataset:
        """Create TensorFlow dataset from prepared data.
        
        Uses from_tensor_slices for pure TensorFlow operations,
        enabling better GPU utilization without Python callbacks.
        
        Args:
            prepared: PreparedData with pre-computed arrays.
            shuffle: Whether to shuffle the dataset.
            repeat: Whether to repeat the dataset infinitely.
            
        Returns:
            tf.data.Dataset yielding ((image_features, text_input), target).
        """
        dataset = tf.data.Dataset.from_tensor_slices((
            prepared.image_features,
            prepared.input_sequences,
            prepared.targets,
        ))
        
        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=self._shuffle_buffer,
                reshuffle_each_iteration=True,
            )
        
        def format_sample(
            image_features: tf.Tensor,
            input_seq: tf.Tensor,
            target: tf.Tensor,
        ) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
            """Format sample as ((image, sequence), target)."""
            return (image_features, input_seq), target
        
        dataset = (
            dataset
            .map(format_sample, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self._batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )
        
        if repeat:
            dataset = dataset.repeat()
        
        # Performance optimizations
        options = tf.data.Options()
        options.experimental_deterministic = False
        options.threading.private_threadpool_size = 8
        dataset = dataset.with_options(options)
        
        return dataset
    
    def compute_steps_per_epoch(self, prepared: PreparedData) -> int:
        """Compute number of steps per epoch.
        
        Args:
            prepared: PreparedData with pre-computed arrays.
            
        Returns:
            Number of batches per epoch.
        """
        num_samples = len(prepared.targets)
        return max(1, num_samples // self._batch_size)
