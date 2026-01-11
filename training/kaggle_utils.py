"""Kaggle-specific configuration and helpers.

This module provides utilities for running the training pipeline
on Kaggle notebooks.

Usage:
    In Kaggle notebook:
    ```python
    from training.kaggle_utils import setup_kaggle_training
    setup_kaggle_training()
    
    from training.train import TrainingPipeline
    pipeline = TrainingPipeline("training/config.kaggle.yaml")
    pipeline.run()
    ```
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


def is_kaggle_environment() -> bool:
    """Check if running in Kaggle notebook environment.
    
    Returns:
        True if in Kaggle, False otherwise.
    """
    return os.path.exists("/kaggle/input") and os.path.exists("/kaggle/working")


def setup_kaggle_training(
    verbose: bool = True,
) -> dict:
    """Setup training environment for Kaggle.
    
    Configures:
    - Working directory
    - Input datasets mounting
    - GPU availability
    - Memory optimization
    
    Args:
        verbose: Whether to print setup information.
        
    Returns:
        Dictionary with Kaggle environment info.
    """
    if not is_kaggle_environment():
        logger.warning("Not in Kaggle environment")
        return {}
    
    info = {
        "input_path": "/kaggle/input",
        "working_path": "/kaggle/working",
        "is_kaggle": True,
    }
    
    # Check available datasets
    input_path = Path("/kaggle/input")
    available_datasets = []
    if input_path.exists():
        available_datasets = [d.name for d in input_path.iterdir() if d.is_dir()]
    
    info["available_datasets"] = available_datasets
    
    # Check GPU availability
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        info["gpus_available"] = len(gpus)
        if verbose:
            logger.info(f"GPUs available: {len(gpus)}")
    except Exception as e:
        logger.warning(f"Could not detect GPUs: {e}")
        info["gpus_available"] = 0
    
    if verbose:
        logger.info(f"Kaggle environment detected")
        logger.info(f"Available datasets: {', '.join(available_datasets)}")
        logger.info(f"Working directory: {info['working_path']}")
    
    return info


def create_kaggle_config(
    output_path: str | Path = "training/config.kaggle.yaml",
    flickr_dataset: str = "flickr8k",
    glove_dataset: str = "glove6b200d",
) -> None:
    """Create Kaggle-specific configuration file.
    
    Args:
        output_path: Where to save the config file.
        flickr_dataset: Name of Flickr8k dataset on Kaggle.
        glove_dataset: Name of GloVe dataset on Kaggle.
    """
    output_path = Path(output_path)
    
    config = {
        # Data Configuration
        "images_path": f"/kaggle/input/{flickr_dataset}/Images/",
        "captions_file": f"/kaggle/input/{flickr_dataset}/captions.txt",
        "glove_path": f"/kaggle/input/{glove_dataset}/glove.6B.200d.txt",
        
        # Data splits
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "word_count_threshold": 5,
        "random_seed": 42,
        
        # Model Configuration
        "feature_extractor": "EfficientNetB3",
        "feature_dim": 1536,
        "image_size": [300, 300],
        "embedding_dim": 200,
        "hidden_dim": 256,
        "num_attention_heads": 4,
        "dropout_rate": 0.3,
        "recurrent_dropout": 0.2,
        
        # Training Configuration
        "batch_size": 64,
        "epochs": 20,
        "learning_rate": 0.0001,
        "lr_decay_factor": 0.5,
        "lr_patience": 3,
        "min_lr": 0.000001,
        "early_stopping_patience": 5,
        "use_mixed_precision": True,
        
        # Output to Kaggle working directory
        "artifacts_dir": "/kaggle/working",
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Created Kaggle config at {output_path}")


def get_kaggle_example_notebook() -> str:
    """Return example Kaggle notebook code.
    
    Returns:
        Python code for running on Kaggle.
    """
    return '''# ============================================================================
# Image Captioning Training - Kaggle Notebook Example
# ============================================================================
# This code trains an image captioning model on Flickr8k dataset
# Run this in a Kaggle notebook with GPU enabled

# Cell 1: Setup environment
!pip install -q pyyaml tqdm nltk

# Cell 2: Mount datasets
import os
print("Input datasets:", os.listdir("/kaggle/input"))

# Cell 3: Create Kaggle config
from training.kaggle_utils import create_kaggle_config, setup_kaggle_training

# Setup Kaggle environment
env_info = setup_kaggle_training(verbose=True)

# Create config file
create_kaggle_config()

# Cell 4: Run training
from training.train import TrainingPipeline

pipeline = TrainingPipeline("training/config.kaggle.yaml")
pipeline.run()

# Cell 5: Results
import os
output_files = os.listdir("/kaggle/working")
print("\\nGenerated files:")
for f in sorted(output_files):
    size = os.path.getsize(f"/kaggle/working/{f}") / 1e6  # MB
    print(f"  {f}: {size:.1f} MB")
'''


class KaggleNotebookHelper:
    """Helper class for running training in Kaggle notebooks.
    
    Provides utilities for:
    - Checking dataset availability
    - Managing GPU memory
    - Monitoring training progress
    - Downloading outputs
    
    Example:
        >>> helper = KaggleNotebookHelper()
        >>> helper.check_datasets()
        >>> helper.run_training()
    """
    
    def __init__(self) -> None:
        """Initialize Kaggle helper."""
        if not is_kaggle_environment():
            raise RuntimeError("Not in Kaggle environment")
        
        self._working_dir = Path("/kaggle/working")
        self._input_dir = Path("/kaggle/input")
    
    def check_datasets(self) -> dict:
        """Check available datasets.
        
        Returns:
            Dictionary of dataset info.
        """
        datasets = {}
        for dataset_dir in self._input_dir.iterdir():
            if dataset_dir.is_dir():
                files = list(dataset_dir.glob("**/*"))
                datasets[dataset_dir.name] = {
                    "path": str(dataset_dir),
                    "files": len(files),
                }
        
        logger.info(f"Found {len(datasets)} datasets")
        return datasets
    
    def setup_memory(self) -> None:
        """Optimize GPU memory usage."""
        try:
            import tensorflow as tf
            
            gpus = tf.config.list_physical_devices("GPU")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            logger.info(f"Configured memory growth for {len(gpus)} GPU(s)")
        except Exception as e:
            logger.warning(f"Could not configure memory: {e}")
    
    def run_training(
        self,
        config_path: str = "training/config.kaggle.yaml",
    ) -> None:
        """Run training pipeline.
        
        Args:
            config_path: Path to configuration file.
        """
        from training.train import TrainingPipeline
        
        self.setup_memory()
        
        pipeline = TrainingPipeline(config_path)
        pipeline.run()
    
    def list_outputs(self) -> list:
        """List generated output files.
        
        Returns:
            List of output files.
        """
        outputs = sorted(self._working_dir.glob("*"))
        for output in outputs:
            size_mb = output.stat().st_size / 1e6
            logger.info(f"  {output.name}: {size_mb:.1f} MB")
        
        return outputs
