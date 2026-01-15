"""TensorFlow dataset builders for training.

This module provides dataset construction utilities for creating
efficient tf.data pipelines for caption model training.

Supports two modes:
1. **Streaming mode (default)**: Memory-efficient for large datasets.
   - Loads features from memory-mapped .npy files
   - Generates samples lazily per batch
   - Never materializes all samples in RAM

2. **Eager mode**: Pre-loads all samples for small datasets.
   - Faster iteration but high memory usage
   - Suitable for datasets that fit in RAM

Uses pure TensorFlow operations where possible for optimal
GPU performance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator

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


@dataclass
class StreamingDataSpec:
    """Specification for streaming dataset without materializing all data.
    
    Instead of storing all samples in memory, this stores only the
    metadata needed to generate samples on-the-fly.
    
    Attributes:
        feature_keys: Ordered list of image keys for indexing into mmap.
        feature_array: Memory-mapped feature array (num_images, feature_dim).
        sample_indices: List of (image_idx, caption_idx, position) tuples.
        tokenizer: Tokenizer for encoding captions.
        descriptions: Dict mapping image_key -> list of captions.
        max_length: Maximum sequence length.
        num_samples: Total number of training samples.
    """
    feature_keys: List[str]
    feature_array: np.ndarray  # Memory-mapped
    sample_indices: List[Tuple[int, int, int]]  # (img_idx, cap_idx, position)
    tokenizer: Tokenizer
    descriptions: Dict[str, List[str]]
    max_length: int
    num_samples: int = field(init=False)
    
    def __post_init__(self):
        self.num_samples = len(self.sample_indices)


class DatasetBuilder:
    """Builder for TensorFlow training datasets.
    
    Creates efficient tf.data pipelines for training caption models
    with image features and text sequences.
    
    Supports two modes:
    - **Streaming mode**: Memory-efficient, loads samples lazily from mmap.
    - **Eager mode**: Pre-loads all samples (high memory usage).
    
    Attributes:
        max_length: Maximum sequence length.
        feature_dim: Dimension of image features.
        batch_size: Number of samples per batch.
        streaming: Whether to use memory-efficient streaming mode.
    
    Example (Streaming - recommended for Kaggle):
        >>> builder = DatasetBuilder(max_length=38, feature_dim=1536, streaming=True)
        >>> spec = builder.prepare_streaming_data(
        ...     descriptions, features_path, tokenizer, feature_keys
        ... )
        >>> dataset = builder.create_streaming_dataset(spec)
    
    Example (Eager - only for small datasets):
        >>> builder = DatasetBuilder(max_length=38, feature_dim=1536, streaming=False)
        >>> prepared = builder.prepare_data(descriptions, features, tokenizer)
        >>> dataset = builder.create_dataset(prepared)
    """
    
    def __init__(
        self,
        max_length: int,
        feature_dim: int,
        batch_size: int = 64,
        shuffle_buffer: int = 10000,
        streaming: bool = True,
        use_float16: bool = True,
    ) -> None:
        """Initialize dataset builder.
        
        Args:
            max_length: Maximum sequence length for padding.
            feature_dim: Dimension of image features.
            batch_size: Number of samples per batch.
            shuffle_buffer: Size of shuffle buffer.
            streaming: Whether to use memory-efficient streaming mode.
                      Highly recommended for large datasets (>10k samples).
            use_float16: Whether to cast features to float16 for memory savings.
                        Automatically converts to float32 for model input.
        """
        self._max_length = max_length
        self._feature_dim = feature_dim
        self._batch_size = batch_size
        self._shuffle_buffer = shuffle_buffer
        self._streaming = streaming
        self._use_float16 = use_float16
    
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
    
    @property
    def streaming(self) -> bool:
        """Whether streaming mode is enabled."""
        return self._streaming
    
    # =========================================================================
    # Streaming Mode Methods (Memory Efficient - Recommended for Kaggle)
    # =========================================================================
    
    def prepare_streaming_data(
        self,
        descriptions: Dict[str, List[str]],
        features_path: str | Path,
        tokenizer: Tokenizer,
        feature_keys: Optional[List[str]] = None,
    ) -> StreamingDataSpec:
        """Prepare streaming dataset specification without loading all data.
        
        This method only builds an index of samples to generate. The actual
        feature loading happens lazily during iteration via memory-mapping.
        
        Memory usage: O(num_samples * 3 ints) instead of O(num_samples * feature_dim).
        For 1.8M samples: ~22MB vs ~11GB.
        
        Args:
            descriptions: Dictionary of image_id -> captions.
            features_path: Path to saved features .npy file.
            tokenizer: Fitted tokenizer instance.
            feature_keys: Optional ordered list of feature keys. If None,
                         will be inferred from features file.
        
        Returns:
            StreamingDataSpec with metadata for lazy sample generation.
        """
        features_path = Path(features_path)
        
        # Load features with memory-mapping (zero-copy, doesn't load into RAM)
        logger.info(f"Memory-mapping features from {features_path}")
        features_data = np.load(features_path, allow_pickle=True, mmap_mode="r")
        
        # Handle dict-style features (key -> vector) vs array-style
        if isinstance(features_data, np.ndarray) and features_data.dtype == object:
            # Features saved as dict
            features_dict = features_data.item()
            if feature_keys is None:
                feature_keys = list(features_dict.keys())
            
            # Build stacked feature array for mmap-like access
            # This is a one-time cost but much smaller than full expansion
            logger.info(f"Building feature index for {len(feature_keys)} images")
            feature_array = np.stack(
                [features_dict[k] for k in feature_keys],
                axis=0,
            )
            if self._use_float16:
                feature_array = feature_array.astype(np.float16)
        else:
            # Features already in array format
            feature_array = features_data
            if feature_keys is None:
                raise ValueError(
                    "feature_keys must be provided when features are in array format"
                )
        
        # Build key-to-index mapping
        key_to_idx = {k: i for i, k in enumerate(feature_keys)}
        
        # Build sample index: (image_idx, caption_idx, position)
        # This is O(num_samples) but only stores 3 ints per sample
        sample_indices: List[Tuple[int, int, int]] = []
        
        for image_key, captions in descriptions.items():
            if image_key not in key_to_idx:
                continue
            
            img_idx = key_to_idx[image_key]
            
            for cap_idx, caption in enumerate(captions):
                sequence = tokenizer.encode(caption)
                seq_len = len(sequence)
                
                # Each position in the caption creates one training sample
                for pos in range(1, seq_len):
                    sample_indices.append((img_idx, cap_idx, pos))
        
        logger.info(
            f"Built streaming spec: {len(sample_indices):,} samples "
            f"from {len(feature_keys):,} images (memory-efficient)"
        )
        
        return StreamingDataSpec(
            feature_keys=feature_keys,
            feature_array=feature_array,
            sample_indices=sample_indices,
            tokenizer=tokenizer,
            descriptions=descriptions,
            max_length=self._max_length,
        )
    
    def _sample_generator(
        self,
        spec: StreamingDataSpec,
        indices: np.ndarray,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, int]]:
        """Generate samples one at a time from streaming spec.
        
        Args:
            spec: StreamingDataSpec with metadata.
            indices: Shuffled array of sample indices.
            
        Yields:
            Tuple of (image_features, padded_sequence, target).
        """
        for idx in indices:
            img_idx, cap_idx, pos = spec.sample_indices[idx]
            
            # Get feature (from mmap - only this slice loaded)
            feature = spec.feature_array[img_idx].astype(np.float32)
            
            # Get caption and encode
            image_key = spec.feature_keys[img_idx]
            caption = spec.descriptions[image_key][cap_idx]
            sequence = spec.tokenizer.encode(caption)
            
            # Build input sequence with padding
            input_seq = sequence[:pos]
            padded = np.zeros(spec.max_length, dtype=np.int32)
            seq_len = min(len(input_seq), spec.max_length)
            padded[:seq_len] = input_seq[:seq_len]
            
            # Target is next word
            target = sequence[pos]
            
            yield feature, padded, target
    
    def create_streaming_dataset(
        self,
        spec: StreamingDataSpec,
        shuffle: bool = True,
        repeat: bool = True,
    ) -> tf.data.Dataset:
        """Create TensorFlow dataset from streaming spec.
        
        Loads samples lazily from memory-mapped features, never
        materializing all data in RAM at once.
        
        Args:
            spec: StreamingDataSpec with metadata.
            shuffle: Whether to shuffle sample order each epoch.
            repeat: Whether to repeat infinitely.
            
        Returns:
            tf.data.Dataset yielding ((image_features, text_input), target).
        """
        num_samples = spec.num_samples
        
        def make_generator():
            """Create generator with fresh shuffle each call."""
            indices = np.arange(num_samples)
            if shuffle:
                np.random.shuffle(indices)
            return self._sample_generator(spec, indices)
        
        # Create dataset from generator
        dataset = tf.data.Dataset.from_generator(
            make_generator,
            output_signature=(
                tf.TensorSpec(shape=(self._feature_dim,), dtype=tf.float32),
                tf.TensorSpec(shape=(self._max_length,), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
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
        dataset = dataset.with_options(options)
        
        return dataset
    
    def compute_streaming_steps(self, spec: StreamingDataSpec) -> int:
        """Compute steps per epoch for streaming dataset.
        
        Args:
            spec: StreamingDataSpec with metadata.
            
        Returns:
            Number of batches per epoch.
        """
        return max(1, spec.num_samples // self._batch_size)
    
    # =========================================================================
    # Eager Mode Methods (Original - High Memory Usage)
    # =========================================================================
    
    def prepare_data(
        self,
        descriptions: Dict[str, List[str]],
        features: Dict[str, np.ndarray],
        tokenizer: Tokenizer,
    ) -> PreparedData:
        """Prepare all training data as numpy arrays.
        
        ⚠️ WARNING: This method loads ALL samples into memory at once.
        For large datasets (>100k samples), this can cause OOM errors,
        especially on Kaggle (13-16GB RAM limit).
        
        For memory-efficient training, use prepare_streaming_data() instead.
        
        Pre-computes all samples with padding for efficient
        tf.data.from_tensor_slices pipeline.
        
        Args:
            descriptions: Dictionary of image_id -> captions.
            features: Dictionary of image_id -> feature vectors.
            tokenizer: Fitted tokenizer instance.
            
        Returns:
            PreparedData with pre-computed arrays.
        """
        import warnings
        
        # Estimate memory usage
        total_samples = sum(
            len(tokenizer.encode(cap)) - 1
            for caps in descriptions.values()
            for cap in caps
            if any(k in features for k in descriptions.keys())
        )
        feature_dim = next(iter(features.values())).shape[0] if features else 0
        estimated_gb = (total_samples * feature_dim * 4) / (1024**3)
        
        if estimated_gb > 4.0:
            warnings.warn(
                f"⚠️ Estimated memory usage: {estimated_gb:.1f}GB for {total_samples:,} samples. "
                f"This may cause OOM on memory-constrained systems like Kaggle. "
                f"Consider using prepare_streaming_data() instead for memory-efficient training.",
                ResourceWarning,
                stacklevel=2,
            )
        
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
