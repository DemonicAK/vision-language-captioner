"""Mixed precision and GPU utilities.

This module provides utilities for setting up
mixed precision training and GPU configuration.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import tensorflow as tf

logger = logging.getLogger(__name__)


def setup_mixed_precision(policy: str = "mixed_float16") -> None:
    """Enable mixed precision training.
    
    Args:
        policy: Mixed precision policy name.
    """
    try:
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info(f"Enabled mixed precision with policy: {policy}")
    except Exception as e:
        logger.warning(f"Failed to set mixed precision: {e}")


def setup_gpu(
    memory_growth: bool = True,
    visible_devices: Optional[List[int]] = None,
) -> List[tf.config.PhysicalDevice]:
    """Configure GPU settings with performance optimizations.
    
    Args:
        memory_growth: Whether to enable memory growth.
        visible_devices: List of GPU indices to make visible.
        
    Returns:
        List of configured GPU devices.
    """
    # Performance environment variables (set before TF initialization ideally)
    os.environ.setdefault("TF_GPU_THREAD_MODE", "gpu_private")
    os.environ.setdefault("TF_GPU_THREAD_COUNT", "2")
    os.environ.setdefault("TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32", "1")
    os.environ.setdefault("TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32", "1")
    
    # Enable XLA globally
    tf.config.optimizer.set_jit(True)
    
    gpus = tf.config.list_physical_devices("GPU")
    
    if not gpus:
        logger.warning("No GPUs detected")
        return []
    
    logger.info(f"Found {len(gpus)} GPU(s)")
    
    # Filter visible devices
    if visible_devices is not None:
        gpus = [gpus[i] for i in visible_devices if i < len(gpus)]
        tf.config.set_visible_devices(gpus, "GPU")
        logger.info(f"Using GPUs: {visible_devices}")
    
    # Enable memory growth
    if memory_growth:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Enabled memory growth for {gpu.name}")
            except RuntimeError as e:
                logger.warning(f"Could not set memory growth: {e}")
    
    logger.info("XLA JIT compilation enabled globally")
    
    return gpus
    
    return gpus


def get_distribution_strategy() -> tf.distribute.Strategy:
    """Get appropriate distribution strategy.
    
    Returns:
        Distribution strategy based on available devices.
    """
    gpus = tf.config.list_physical_devices("GPU")
    
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        logger.info(f"Using MirroredStrategy with {len(gpus)} GPUs")
    elif len(gpus) == 1:
        strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
        logger.info("Using single GPU strategy")
    else:
        strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
        logger.info("Using CPU strategy")
    
    return strategy
