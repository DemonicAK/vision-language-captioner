"""Resume training from checkpoint for Kaggle sessions.

This is a SEPARATE script for resuming interrupted training.
It does NOT modify the main training pipeline.

Usage (Kaggle):
    # In a new Kaggle notebook cell:
    %run training/resume_training.py \
        --checkpoint /kaggle/input/your-checkpoint-dataset/checkpoint.keras \
        --initial-epoch 12 \
        --total-epochs 22 \
        --config training/config.kaggle.yaml

Your main train.py remains UNCHANGED.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.configs import load_config
from training.data import CaptionLoader, Tokenizer, DatasetBuilder
from training.features import get_feature_extractor
from training.trainers import Trainer
from training.evaluation.inference import GreedyDecoder
from training.evaluation.metrics import CaptionEvaluator
from training.utils import (
    setup_logging,
    get_logger,
    setup_mixed_precision,
    setup_gpu,
    ensure_dir,
)

logger = get_logger(__name__)


def resume_training(
    checkpoint_path: str,
    config_path: str,
    initial_epoch: int,
    total_epochs: int,
    learning_rate: float | None = None,
) -> None:
    """Resume training from a checkpoint.
    
    Args:
        checkpoint_path: Path to the saved .keras checkpoint.
        config_path: Path to config YAML.
        initial_epoch: Epoch number to resume from (0-indexed).
        total_epochs: Target total epochs.
        learning_rate: Optional new learning rate (keeps original if None).
    """
    setup_logging()
    logger.info("=" * 60)
    logger.info("RESUMING TRAINING FROM CHECKPOINT")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Resuming from epoch {initial_epoch} → target {total_epochs}")
    logger.info("=" * 60)
    
    # Load config
    config = load_config(config_path)
    
    # Setup
    setup_gpu(memory_growth=True)
    if config.training.use_mixed_precision:
        setup_mixed_precision()
    
    artifacts_dir = Path(config.training.artifacts_dir)
    if not artifacts_dir.is_absolute():
        artifacts_dir = Path(__file__).parent.parent / config.training.artifacts_dir
    ensure_dir(artifacts_dir)
    
    # =========================================
    # LOAD CHECKPOINT
    # =========================================
    logger.info(f"Loading model from {checkpoint_path}")
    model = keras.models.load_model(checkpoint_path)
    logger.info(f"✓ Model loaded. Params: {model.count_params():,}")
    
    # Recompile with (optionally new) learning rate
    lr = learning_rate or config.training.learning_rate
    logger.info(f"Recompiling with learning_rate={lr}")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        jit_compile=True,
    )
    
    # =========================================
    # REBUILD DATASETS (needed for training)
    # =========================================
    logger.info("Rebuilding datasets...")
    
    # Load tokenizer from artifacts
    tokenizer = Tokenizer.load(artifacts_dir)
    logger.info(f"✓ Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    
    # Load captions
    loader = CaptionLoader()
    descriptions = loader.load(config.data.captions_file)
    splits = loader.create_splits(
        descriptions,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        seed=config.data.random_seed,
    )
    
    # Get max length
    all_descriptions = {**splits.train, **splits.val, **splits.test}
    max_length = loader.get_max_caption_length(all_descriptions)
    
    # Load features from disk
    features_path = artifacts_dir / "features.npy"
    logger.info(f"Loading features from {features_path}")
    features = np.load(features_path, allow_pickle=True).item()
    feature_keys = list(features.keys())
    logger.info(f"✓ Features loaded. {len(feature_keys)} images")
    
    # Build streaming datasets
    builder = DatasetBuilder(
        max_length=max_length,
        feature_dim=config.model.feature_dim,
        batch_size=config.training.batch_size,
        streaming=True,
        use_float16=True,
    )
    
    train_spec = builder.prepare_streaming_data(
        descriptions=splits.train,
        features_path=features_path,
        tokenizer=tokenizer,
        feature_keys=feature_keys,
    )
    val_spec = builder.prepare_streaming_data(
        descriptions=splits.val,
        features_path=features_path,
        tokenizer=tokenizer,
        feature_keys=feature_keys,
    )
    
    train_ds = builder.create_streaming_dataset(train_spec, shuffle=True, repeat=True)
    val_ds = builder.create_streaming_dataset(val_spec, shuffle=False, repeat=True)
    
    steps_per_epoch = builder.compute_streaming_steps(train_spec)
    val_steps = builder.compute_streaming_steps(val_spec)
    
    logger.info(f"Train: {train_spec.num_samples:,} samples, {steps_per_epoch} steps/epoch")
    logger.info(f"Val: {val_spec.num_samples:,} samples, {val_steps} steps")
    
    # Clear memory
    gc.collect()
    
    # =========================================
    # CALLBACKS (append-friendly)
    # =========================================
    cfg = config.training
    
    callbacks = [
        # Save best model
        keras.callbacks.ModelCheckpoint(
            str(artifacts_dir / "checkpoint.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        # Save every epoch (for safety)
        keras.callbacks.ModelCheckpoint(
            str(artifacts_dir / "checkpoint_epoch_{epoch:02d}.keras"),
            save_best_only=False,
            verbose=0,
        ),
        # Reduce LR on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg.lr_decay_factor,
            patience=cfg.lr_patience,
            min_lr=cfg.min_lr,
            verbose=1,
        ),
        # CSV logger (APPEND mode!)
        keras.callbacks.CSVLogger(
            str(artifacts_dir / "logs" / "training_log.csv"),
            append=True,  # Critical for resuming!
        ),
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]
    
    # =========================================
    # RESUME TRAINING
    # =========================================
    logger.info(f"Resuming training: epoch {initial_epoch} → {total_epochs}")
    
    history = model.fit(
        train_ds,
        epochs=total_epochs,
        initial_epoch=initial_epoch,  # THE KEY PARAMETER!
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1,
    )
    
    # =========================================
    # SAVE FINAL MODEL
    # =========================================
    final_model_path = artifacts_dir / f"image_caption_model_final_{initial_epoch}.keras"
    model.save(final_model_path)
    logger.info(f"✓ Final model saved to {final_model_path}")
    
    # =========================================
    # QUICK BLEU EVALUATION
    # =========================================
    logger.info("Running quick BLEU evaluation...")
    
    decoder = GreedyDecoder(tokenizer=tokenizer, max_length=max_length)
    evaluator = CaptionEvaluator(smoothing=True)
    
    # Sample 500 images for quick eval
    import random
    test_keys = [k for k in splits.test.keys() if k in features]
    sample_keys = random.sample(test_keys, min(500, len(test_keys)))
    
    predictions, references = [], []
    for key in sample_keys:
        feature = features[key]
        if len(feature.shape) == 1:
            feature = np.expand_dims(feature, 0)
        pred = decoder.decode(model, feature)
        predictions.append(pred)
        references.append(splits.test[key])
    
    scores = evaluator.evaluate(predictions, references)
    
    logger.info("=" * 60)
    logger.info("TRAINING RESUMED SUCCESSFULLY!")
    logger.info(f"Trained epochs: {initial_epoch} → {total_epochs}")
    logger.info(f"Final train loss: {history.history['loss'][-1]:.4f}")
    logger.info(f"Final val loss: {history.history['val_loss'][-1]:.4f}")
    logger.info(f"BLEU-1: {scores.bleu1:.4f}")
    logger.info(f"BLEU-4: {scores.bleu4:.4f}")
    logger.info(f"Model saved: {final_model_path}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Resume training from checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint .keras file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="training/config.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--initial-epoch",
        type=int,
        required=True,
        help="Epoch number to resume from (e.g., 12 if you completed epochs 0-11)",
    )
    parser.add_argument(
        "--total-epochs",
        type=int,
        required=True,
        help="Target total epochs (e.g., 22 for 10 more epochs)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate (optional)",
    )
    
    args = parser.parse_args()
    
    resume_training(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        initial_epoch=args.initial_epoch,
        total_epochs=args.total_epochs,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
