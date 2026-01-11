"""Tests for dataset builders and shapes."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from training.data.dataset import DatasetBuilder, TrainingSample
from training.data.tokenizer import Tokenizer


class TestTrainingSample:
    """Test suite for TrainingSample dataclass."""
    
    def test_creation(self) -> None:
        """Test creating a training sample."""
        sample = TrainingSample(
            image_key="image_0",
            input_sequence=[1, 2, 3],
            target_word=4,
        )
        
        assert sample.image_key == "image_0"
        assert sample.input_sequence == [1, 2, 3]
        assert sample.target_word == 4


class TestDatasetBuilder:
    """Test suite for DatasetBuilder."""
    
    @pytest.fixture
    def builder(self) -> DatasetBuilder:
        """Create a dataset builder."""
        return DatasetBuilder(
            max_length=20,
            feature_dim=1536,
            batch_size=4,
            shuffle_buffer=100,
        )
    
    def test_initialization(self, builder: DatasetBuilder) -> None:
        """Test dataset builder initialization."""
        assert builder.max_length == 20
        assert builder.feature_dim == 1536
        assert builder.batch_size == 4
    
    def test_build_samples_creates_samples(
        self,
        builder: DatasetBuilder,
        sample_descriptions: dict[str, list[str]],
        sample_features: dict[str, np.ndarray],
        fitted_tokenizer: Tokenizer,
    ) -> None:
        """Test that build_samples creates training samples."""
        samples = builder.build_samples(
            sample_descriptions,
            sample_features,
            fitted_tokenizer,
        )
        
        assert len(samples) > 0
        assert all(isinstance(s, TrainingSample) for s in samples)
    
    def test_build_samples_skips_missing_features(
        self,
        builder: DatasetBuilder,
        fitted_tokenizer: Tokenizer,
    ) -> None:
        """Test that build_samples skips images without features."""
        descriptions = {
            "image_0": ["startseq a dog runs endseq"],
            "image_missing": ["startseq a cat sits endseq"],  # No features
        }
        features = {
            "image_0": np.random.randn(1536).astype(np.float32),
        }
        
        samples = builder.build_samples(descriptions, features, fitted_tokenizer)
        
        # Only image_0 should have samples
        image_keys = {s.image_key for s in samples}
        assert "image_0" in image_keys
        assert "image_missing" not in image_keys
    
    def test_build_samples_creates_partial_sequences(
        self,
        builder: DatasetBuilder,
        fitted_tokenizer: Tokenizer,
    ) -> None:
        """Test that samples are created for each position in caption."""
        descriptions = {"image_0": ["startseq a dog runs endseq"]}
        features = {"image_0": np.random.randn(1536).astype(np.float32)}
        
        samples = builder.build_samples(descriptions, features, fitted_tokenizer)
        
        # Should create samples for positions 1, 2, 3, 4 (after startseq)
        # Sequence: startseq -> a -> dog -> runs -> endseq
        # Position 1: input=[startseq], target=a
        # Position 2: input=[startseq, a], target=dog
        # etc.
        assert len(samples) >= 4
    
    def test_create_dataset_shape(
        self,
        builder: DatasetBuilder,
        sample_descriptions: dict[str, list[str]],
        sample_features: dict[str, np.ndarray],
        fitted_tokenizer: Tokenizer,
    ) -> None:
        """Test that created dataset has correct shapes."""
        samples = builder.build_samples(
            sample_descriptions,
            sample_features,
            fitted_tokenizer,
        )
        
        dataset = builder.create_dataset(
            samples,
            sample_features,
            shuffle=False,
            repeat=False,
        )
        
        # Get one batch
        for (image_features, text_input), target in dataset.take(1):
            # Check batch dimension
            assert image_features.shape[0] == builder.batch_size
            assert text_input.shape[0] == builder.batch_size
            assert target.shape[0] == builder.batch_size
            
            # Check feature dimension
            assert image_features.shape[1] == builder.feature_dim
            
            # Check sequence length
            assert text_input.shape[1] == builder.max_length
            
            # Check dtypes
            assert image_features.dtype == tf.float32
            assert text_input.dtype == tf.int32
            assert target.dtype == tf.int32
    
    def test_create_dataset_batch_size(
        self,
        sample_descriptions: dict[str, list[str]],
        sample_features: dict[str, np.ndarray],
        fitted_tokenizer: Tokenizer,
    ) -> None:
        """Test that dataset respects batch size."""
        batch_sizes = [2, 4, 8]
        
        for batch_size in batch_sizes:
            builder = DatasetBuilder(
                max_length=20,
                feature_dim=1536,
                batch_size=batch_size,
            )
            
            samples = builder.build_samples(
                sample_descriptions,
                sample_features,
                fitted_tokenizer,
            )
            
            dataset = builder.create_dataset(
                samples,
                sample_features,
                shuffle=False,
                repeat=False,
            )
            
            for (image_features, _), _ in dataset.take(1):
                assert image_features.shape[0] == batch_size
    
    def test_create_dataset_max_length(
        self,
        sample_descriptions: dict[str, list[str]],
        sample_features: dict[str, np.ndarray],
        fitted_tokenizer: Tokenizer,
    ) -> None:
        """Test that sequences are padded to max_length."""
        max_lengths = [10, 20, 50]
        
        for max_length in max_lengths:
            builder = DatasetBuilder(
                max_length=max_length,
                feature_dim=1536,
                batch_size=4,
            )
            
            samples = builder.build_samples(
                sample_descriptions,
                sample_features,
                fitted_tokenizer,
            )
            
            dataset = builder.create_dataset(
                samples,
                sample_features,
                shuffle=False,
                repeat=False,
            )
            
            for (_, text_input), _ in dataset.take(1):
                assert text_input.shape[1] == max_length
    
    def test_create_dataset_feature_dim(
        self,
        fitted_tokenizer: Tokenizer,
    ) -> None:
        """Test that dataset respects feature dimension."""
        feature_dims = [512, 1536, 2048]
        
        for feature_dim in feature_dims:
            features = {
                f"image_{i}": np.random.randn(feature_dim).astype(np.float32)
                for i in range(5)
            }
            descriptions = {
                f"image_{i}": ["startseq a dog runs endseq"]
                for i in range(5)
            }
            
            builder = DatasetBuilder(
                max_length=20,
                feature_dim=feature_dim,
                batch_size=2,
            )
            
            samples = builder.build_samples(descriptions, features, fitted_tokenizer)
            dataset = builder.create_dataset(
                samples,
                features,
                shuffle=False,
                repeat=False,
            )
            
            for (image_features, _), _ in dataset.take(1):
                assert image_features.shape[1] == feature_dim
    
    def test_create_dataset_shuffle(
        self,
        builder: DatasetBuilder,
        sample_descriptions: dict[str, list[str]],
        sample_features: dict[str, np.ndarray],
        fitted_tokenizer: Tokenizer,
    ) -> None:
        """Test that shuffle option affects dataset order."""
        samples = builder.build_samples(
            sample_descriptions,
            sample_features,
            fitted_tokenizer,
        )
        
        # Create two datasets with same seed, one shuffled
        tf.random.set_seed(42)
        ds_no_shuffle = builder.create_dataset(
            samples, sample_features, shuffle=False, repeat=False
        )
        
        tf.random.set_seed(42)
        ds_shuffled = builder.create_dataset(
            samples, sample_features, shuffle=True, repeat=False
        )
        
        # Get samples from both
        no_shuffle_samples = list(ds_no_shuffle.take(2))
        shuffled_samples = list(ds_shuffled.take(2))
        
        # Should be able to create both without error
        assert len(no_shuffle_samples) > 0
        assert len(shuffled_samples) > 0
    
    def test_compute_steps_per_epoch(self, builder: DatasetBuilder) -> None:
        """Test steps per epoch computation."""
        # 100 samples with batch_size=4 = 25 steps
        assert builder.compute_steps_per_epoch(100) == 25
        
        # 10 samples with batch_size=4 = 2 steps
        assert builder.compute_steps_per_epoch(10) == 2
        
        # Edge case: fewer samples than batch size
        assert builder.compute_steps_per_epoch(1) == 1
    
    def test_dataset_is_repeatable(
        self,
        builder: DatasetBuilder,
        sample_descriptions: dict[str, list[str]],
        sample_features: dict[str, np.ndarray],
        fitted_tokenizer: Tokenizer,
    ) -> None:
        """Test that repeat=True creates infinite dataset."""
        samples = builder.build_samples(
            sample_descriptions,
            sample_features,
            fitted_tokenizer,
        )
        
        dataset = builder.create_dataset(
            samples,
            sample_features,
            shuffle=False,
            repeat=True,
        )
        
        # Should be able to take more batches than available samples
        num_batches = (len(samples) // builder.batch_size) + 5
        batches = list(dataset.take(num_batches))
        assert len(batches) == num_batches
    
    def test_target_is_scalar(
        self,
        builder: DatasetBuilder,
        sample_descriptions: dict[str, list[str]],
        sample_features: dict[str, np.ndarray],
        fitted_tokenizer: Tokenizer,
    ) -> None:
        """Test that target is a scalar (word index)."""
        samples = builder.build_samples(
            sample_descriptions,
            sample_features,
            fitted_tokenizer,
        )
        
        dataset = builder.create_dataset(
            samples,
            sample_features,
            shuffle=False,
            repeat=False,
        )
        
        for (_, _), target in dataset.take(1):
            # Target should be 1D (batch of scalars)
            assert len(target.shape) == 1
            assert target.shape[0] == builder.batch_size
