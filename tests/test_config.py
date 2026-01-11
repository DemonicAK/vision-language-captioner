"""Tests for configuration validation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from training.configs.training_config import (
    Config,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    load_config,
)


class TestDataConfigValidation:
    """Test suite for DataConfig validation."""
    
    def test_valid_config(self, valid_data_config: DataConfig) -> None:
        """Test that valid config passes validation."""
        assert valid_data_config.train_ratio == 0.6
        assert valid_data_config.val_ratio == 0.2
        assert valid_data_config.test_ratio == 0.2
    
    def test_invalid_train_ratio_too_high(self) -> None:
        """Test that train_ratio > 1 raises error."""
        with pytest.raises(ValueError, match="train_ratio must be in"):
            DataConfig(
                images_path="/path",
                captions_file="/path",
                glove_path="/path",
                train_ratio=1.5,
            )
    
    def test_invalid_train_ratio_too_low(self) -> None:
        """Test that train_ratio <= 0 raises error."""
        with pytest.raises(ValueError, match="train_ratio must be in"):
            DataConfig(
                images_path="/path",
                captions_file="/path",
                glove_path="/path",
                train_ratio=0.0,
            )
    
    def test_invalid_val_ratio_negative(self) -> None:
        """Test that negative val_ratio raises error."""
        with pytest.raises(ValueError, match="val_ratio must be in"):
            DataConfig(
                images_path="/path",
                captions_file="/path",
                glove_path="/path",
                val_ratio=-0.1,
            )
    
    def test_invalid_test_ratio_negative(self) -> None:
        """Test that negative test_ratio raises error."""
        with pytest.raises(ValueError, match="test_ratio must be in"):
            DataConfig(
                images_path="/path",
                captions_file="/path",
                glove_path="/path",
                test_ratio=-0.1,
            )
    
    def test_ratios_must_sum_to_one(self) -> None:
        """Test that data split ratios must sum to 1.0."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            DataConfig(
                images_path="/path",
                captions_file="/path",
                glove_path="/path",
                train_ratio=0.5,
                val_ratio=0.2,
                test_ratio=0.1,  # Sum = 0.8
            )
    
    def test_ratios_sum_to_one_with_tolerance(self) -> None:
        """Test that small floating point errors are tolerated."""
        # Should not raise - small floating point differences
        config = DataConfig(
            images_path="/path",
            captions_file="/path",
            glove_path="/path",
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )
        assert abs(
            config.train_ratio + config.val_ratio + config.test_ratio - 1.0
        ) < 1e-6
    
    def test_config_is_frozen(self, valid_data_config: DataConfig) -> None:
        """Test that DataConfig is immutable (frozen)."""
        with pytest.raises(AttributeError):
            valid_data_config.train_ratio = 0.8  # type: ignore


class TestModelConfigValidation:
    """Test suite for ModelConfig validation."""
    
    def test_valid_config(self, valid_model_config: ModelConfig) -> None:
        """Test that valid config passes validation."""
        assert valid_model_config.feature_dim == 1536
        assert valid_model_config.embedding_dim == 200
        assert valid_model_config.dropout_rate == 0.3
    
    def test_invalid_feature_dim_zero(self) -> None:
        """Test that feature_dim <= 0 raises error."""
        with pytest.raises(ValueError, match="feature_dim must be positive"):
            ModelConfig(feature_dim=0)
    
    def test_invalid_feature_dim_negative(self) -> None:
        """Test that negative feature_dim raises error."""
        with pytest.raises(ValueError, match="feature_dim must be positive"):
            ModelConfig(feature_dim=-100)
    
    def test_invalid_embedding_dim_zero(self) -> None:
        """Test that embedding_dim <= 0 raises error."""
        with pytest.raises(ValueError, match="embedding_dim must be positive"):
            ModelConfig(embedding_dim=0)
    
    def test_invalid_dropout_rate_negative(self) -> None:
        """Test that negative dropout_rate raises error."""
        with pytest.raises(ValueError, match="dropout_rate must be in"):
            ModelConfig(dropout_rate=-0.1)
    
    def test_invalid_dropout_rate_too_high(self) -> None:
        """Test that dropout_rate >= 1 raises error."""
        with pytest.raises(ValueError, match="dropout_rate must be in"):
            ModelConfig(dropout_rate=1.0)
    
    def test_config_is_frozen(self, valid_model_config: ModelConfig) -> None:
        """Test that ModelConfig is immutable (frozen)."""
        with pytest.raises(AttributeError):
            valid_model_config.feature_dim = 512  # type: ignore


class TestTrainingConfigValidation:
    """Test suite for TrainingConfig validation."""
    
    def test_valid_config(self, valid_training_config: TrainingConfig) -> None:
        """Test that valid config passes validation."""
        assert valid_training_config.batch_size == 64
        assert valid_training_config.epochs == 20
        assert valid_training_config.learning_rate == 1e-4
    
    def test_invalid_batch_size_zero(self) -> None:
        """Test that batch_size <= 0 raises error."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            TrainingConfig(batch_size=0)
    
    def test_invalid_batch_size_negative(self) -> None:
        """Test that negative batch_size raises error."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            TrainingConfig(batch_size=-1)
    
    def test_invalid_epochs_zero(self) -> None:
        """Test that epochs <= 0 raises error."""
        with pytest.raises(ValueError, match="epochs must be positive"):
            TrainingConfig(epochs=0)
    
    def test_invalid_epochs_negative(self) -> None:
        """Test that negative epochs raises error."""
        with pytest.raises(ValueError, match="epochs must be positive"):
            TrainingConfig(epochs=-10)
    
    def test_invalid_learning_rate_zero(self) -> None:
        """Test that learning_rate <= 0 raises error."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            TrainingConfig(learning_rate=0)
    
    def test_invalid_learning_rate_negative(self) -> None:
        """Test that negative learning_rate raises error."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            TrainingConfig(learning_rate=-1e-4)
    
    def test_config_is_frozen(self, valid_training_config: TrainingConfig) -> None:
        """Test that TrainingConfig is immutable (frozen)."""
        with pytest.raises(AttributeError):
            valid_training_config.batch_size = 128  # type: ignore


class TestConfigFromDict:
    """Test suite for Config.from_dict()."""
    
    def test_from_dict_complete(self) -> None:
        """Test creating config from complete dictionary."""
        config_dict = {
            "images_path": "/data/images",
            "captions_file": "/data/captions.txt",
            "glove_path": "/data/glove.txt",
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "word_count_threshold": 10,
            "random_seed": 123,
            "feature_extractor": "EfficientNetB3",
            "feature_dim": 2048,
            "embedding_dim": 300,
            "hidden_dim": 512,
            "num_attention_heads": 8,
            "dropout_rate": 0.5,
            "recurrent_dropout": 0.3,
            "image_size": [224, 224],
            "batch_size": 32,
            "epochs": 50,
            "learning_rate": 1e-3,
            "lr_decay_factor": 0.1,
            "lr_patience": 5,
            "min_lr": 1e-7,
            "early_stopping_patience": 10,
            "use_mixed_precision": False,
            "artifacts_dir": "/output/artifacts",
        }
        
        config = Config.from_dict(config_dict)
        
        assert config.data.images_path == "/data/images"
        assert config.data.train_ratio == 0.7
        assert config.model.feature_dim == 2048
        assert config.model.image_size == (224, 224)
        assert config.training.batch_size == 32
        assert config.training.epochs == 50
    
    def test_from_dict_with_defaults(self) -> None:
        """Test that from_dict uses defaults for missing values."""
        config_dict = {
            "images_path": "/data/images",
            "captions_file": "/data/captions.txt",
            "glove_path": "/data/glove.txt",
            # Using defaults for everything else
        }
        
        config = Config.from_dict(config_dict)
        
        # Check defaults are applied
        assert config.data.train_ratio == 0.6
        assert config.model.feature_dim == 1536
        assert config.training.batch_size == 64


class TestLoadConfig:
    """Test suite for load_config() function."""
    
    def test_load_config_from_yaml(self) -> None:
        """Test loading config from YAML file."""
        config_dict = {
            "images_path": "/data/images",
            "captions_file": "/data/captions.txt",
            "glove_path": "/data/glove.txt",
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "test_ratio": 0.2,
            "batch_size": 32,
            "epochs": 10,
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config_dict, f)
            
            config = load_config(config_path)
            
            assert config.data.images_path == "/data/images"
            assert config.training.batch_size == 32
    
    def test_load_config_file_not_found(self) -> None:
        """Test that load_config raises error for missing file."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/nonexistent/config.yaml")
    
    def test_load_config_invalid_yaml(self) -> None:
        """Test that load_config raises error for invalid YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "invalid.yaml"
            with open(config_path, "w") as f:
                f.write("invalid: yaml: content: [")
            
            with pytest.raises(yaml.YAMLError):
                load_config(config_path)


class TestConfigIntegration:
    """Integration tests for configuration system."""
    
    def test_full_config_workflow(self) -> None:
        """Test complete config creation and access workflow."""
        # Create individual configs
        data = DataConfig(
            images_path="/data/images",
            captions_file="/data/captions.txt",
            glove_path="/data/glove.txt",
        )
        model = ModelConfig()
        training = TrainingConfig()
        
        # Create root config
        config = Config(data=data, model=model, training=training)
        
        # Access nested values
        assert config.data.random_seed == 42
        assert config.model.feature_extractor == "EfficientNetB3"
        assert config.training.use_mixed_precision is True
    
    def test_config_validation_on_creation(self) -> None:
        """Test that validation happens during config creation."""
        # Should raise during DataConfig creation
        with pytest.raises(ValueError):
            Config(
                data=DataConfig(
                    images_path="/path",
                    captions_file="/path",
                    glove_path="/path",
                    train_ratio=2.0,  # Invalid
                ),
                model=ModelConfig(),
                training=TrainingConfig(),
            )
