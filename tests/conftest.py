"""Pytest fixtures and configuration."""

from __future__ import annotations

import numpy as np
import pytest

from training.data.tokenizer import Tokenizer
from training.configs.training_config import (
    Config,
    DataConfig,
    ModelConfig,
    TrainingConfig,
)


@pytest.fixture
def sample_captions() -> list[str]:
    """Sample captions for testing."""
    return [
        "startseq a dog runs in the park endseq",
        "startseq a cat sits on a mat endseq",
        "startseq a man walks down the street endseq",
        "startseq a woman holds a baby endseq",
        "startseq two dogs play in the park endseq",
        "startseq a bird flies in the sky endseq",
        "startseq a child plays with a ball endseq",
        "startseq a dog runs fast endseq",
        "startseq the cat sits quietly endseq",
        "startseq a person walks a dog endseq",
    ]


@pytest.fixture
def fitted_tokenizer(sample_captions: list[str]) -> Tokenizer:
    """Pre-fitted tokenizer for testing."""
    tokenizer = Tokenizer(min_word_count=1)
    tokenizer.fit(sample_captions)
    return tokenizer


@pytest.fixture
def sample_features() -> dict[str, np.ndarray]:
    """Sample image features for testing."""
    np.random.seed(42)
    return {
        f"image_{i}": np.random.randn(1536).astype(np.float32)
        for i in range(10)
    }


@pytest.fixture
def sample_descriptions() -> dict[str, list[str]]:
    """Sample descriptions for testing."""
    return {
        "image_0": ["startseq a dog runs endseq", "startseq the dog is running endseq"],
        "image_1": ["startseq a cat sits endseq", "startseq the cat is sitting endseq"],
        "image_2": ["startseq a man walks endseq"],
        "image_3": ["startseq a woman stands endseq"],
        "image_4": ["startseq two dogs play endseq"],
    }


@pytest.fixture
def valid_data_config() -> DataConfig:
    """Valid data configuration."""
    return DataConfig(
        images_path="/path/to/images",
        captions_file="/path/to/captions.txt",
        glove_path="/path/to/glove.txt",
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        word_count_threshold=5,
        random_seed=42,
    )


@pytest.fixture
def valid_model_config() -> ModelConfig:
    """Valid model configuration."""
    return ModelConfig(
        feature_extractor="EfficientNetB3",
        feature_dim=1536,
        embedding_dim=200,
        hidden_dim=256,
        num_attention_heads=4,
        dropout_rate=0.3,
        recurrent_dropout=0.2,
        image_size=(300, 300),
    )


@pytest.fixture
def valid_training_config() -> TrainingConfig:
    """Valid training configuration."""
    return TrainingConfig(
        batch_size=64,
        epochs=20,
        learning_rate=1e-4,
        lr_decay_factor=0.5,
        lr_patience=3,
        min_lr=1e-6,
        early_stopping_patience=5,
        use_mixed_precision=True,
        artifacts_dir="../shared/artifacts",
    )


@pytest.fixture
def valid_config(
    valid_data_config: DataConfig,
    valid_model_config: ModelConfig,
    valid_training_config: TrainingConfig,
) -> Config:
    """Complete valid configuration."""
    return Config(
        data=valid_data_config,
        model=valid_model_config,
        training=valid_training_config,
    )
