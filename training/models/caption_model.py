"""Image captioning model architecture.

This module provides the main caption model implementation
with modular encoder-decoder architecture.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from training.models.components.attention import MultiHeadCrossAttention
from training.models.components.encoders import (
    ImageProjection,
    LSTMEncoder,
    TextEmbedding,
)

logger = logging.getLogger(__name__)


class ImageEncoder(tf.keras.layers.Layer):
    """Image encoder component.
    
    Projects image features to the shared embedding space.
    
    Attributes:
        projection_dim: Output projection dimension.
        dropout_rate: Dropout rate for regularization.
    """
    
    def __init__(
        self,
        projection_dim: int = 256,
        dropout_rate: float = 0.3,
        **kwargs,
    ) -> None:
        """Initialize image encoder.
        
        Args:
            projection_dim: Dimension of projected features.
            dropout_rate: Dropout rate.
            **kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        self._projection = ImageProjection(
            projection_dim=projection_dim,
            dropout_rate=dropout_rate,
        )
    
    def call(
        self,
        inputs: tf.Tensor,
        training: Optional[bool] = None,
    ) -> tf.Tensor:
        """Encode image features.
        
        Args:
            inputs: Image features of shape (batch, feature_dim).
            training: Whether in training mode.
            
        Returns:
            Projected features of shape (batch, projection_dim).
        """
        return self._projection(inputs, training=training)


class TextEncoder(tf.keras.layers.Layer):
    """Text encoder component.
    
    Embeds and encodes text sequences using LSTM.
    
    Attributes:
        vocab_size: Vocabulary size.
        embedding_dim: Word embedding dimension.
        hidden_dim: LSTM hidden dimension.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 200,
        hidden_dim: int = 256,
        embedding_matrix: Optional[np.ndarray] = None,
        dropout_rate: float = 0.3,
        recurrent_dropout: float = 0.2,
        **kwargs,
    ) -> None:
        """Initialize text encoder.
        
        Args:
            vocab_size: Size of vocabulary.
            embedding_dim: Word embedding dimension.
            hidden_dim: LSTM hidden dimension.
            embedding_matrix: Pre-trained embeddings (optional).
            dropout_rate: Dropout rate.
            recurrent_dropout: LSTM recurrent dropout.
            **kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        
        self._embedding = TextEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            embedding_matrix=embedding_matrix,
            trainable_embeddings=False,
        )
        
        self._lstm = LSTMEncoder(
            units=hidden_dim,
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
        )
    
    def call(
        self,
        inputs: tf.Tensor,
        training: Optional[bool] = None,
    ) -> tf.Tensor:
        """Encode text sequence.
        
        Args:
            inputs: Token indices of shape (batch, seq_len).
            training: Whether in training mode.
            
        Returns:
            Encoded sequence of shape (batch, seq_len, hidden_dim).
        """
        embedded = self._embedding(inputs)
        encoded = self._lstm(embedded, training=training)
        return encoded


class AttentionLayer(tf.keras.layers.Layer):
    """Attention fusion layer.
    
    Combines image and text representations using
    multi-head cross attention.
    
    Attributes:
        num_heads: Number of attention heads.
        key_dim: Dimension of attention keys.
    """
    
    def __init__(
        self,
        num_heads: int = 4,
        key_dim: int = 256,
        **kwargs,
    ) -> None:
        """Initialize attention layer.
        
        Args:
            num_heads: Number of attention heads.
            key_dim: Dimension of attention keys.
            **kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        self._attention = MultiHeadCrossAttention(
            num_heads=num_heads,
            key_dim=key_dim,
        )
    
    def call(
        self,
        image_features: tf.Tensor,
        text_sequence: tf.Tensor,
        training: Optional[bool] = None,
    ) -> tf.Tensor:
        """Apply cross attention.
        
        Args:
            image_features: Image features of shape (batch, dim).
            text_sequence: Text sequence of shape (batch, seq, dim).
            training: Whether in training mode.
            
        Returns:
            Context vector of shape (batch, dim).
        """
        return self._attention(image_features, text_sequence, training=training)


class CaptionModel(tf.keras.Model):
    """Image captioning model.
    
    Encoder-decoder architecture with attention mechanism
    for generating image captions.
    
    Architecture:
        1. Image encoder: Projects CNN features to embedding space.
        2. Text encoder: Embeds and encodes partial captions with LSTM.
        3. Attention: Cross-attention between image and text.
        4. Decoder: Fuses context and predicts next word.
    
    Example:
        >>> model = CaptionModel(vocab_size=5000, embedding_matrix=glove)
        >>> model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        >>> model.fit(train_dataset, epochs=10)
    """
    
    def __init__(
        self,
        vocab_size: int,
        feature_dim: int = 1536,
        embedding_dim: int = 200,
        hidden_dim: int = 256,
        num_attention_heads: int = 4,
        dropout_rate: float = 0.3,
        recurrent_dropout: float = 0.2,
        embedding_matrix: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        """Initialize caption model.
        
        Args:
            vocab_size: Size of vocabulary.
            feature_dim: Dimension of image features (from CNN).
            embedding_dim: Word embedding dimension.
            hidden_dim: Hidden dimension for encoders.
            num_attention_heads: Number of attention heads.
            dropout_rate: Dropout rate for regularization.
            recurrent_dropout: LSTM recurrent dropout rate.
            embedding_matrix: Pre-trained embedding matrix.
            **kwargs: Additional model arguments.
        """
        super().__init__(**kwargs)
        
        self._vocab_size = vocab_size
        self._feature_dim = feature_dim
        self._embedding_dim = embedding_dim
        self._hidden_dim = hidden_dim
        
        # Encoders
        self._image_encoder = ImageEncoder(
            projection_dim=hidden_dim,
            dropout_rate=dropout_rate,
        )
        
        self._text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            embedding_matrix=embedding_matrix,
            dropout_rate=dropout_rate,
            recurrent_dropout=recurrent_dropout,
        )
        
        # Attention
        self._attention = AttentionLayer(
            num_heads=num_attention_heads,
            key_dim=hidden_dim,
        )
        
        # Decoder
        self._concat = tf.keras.layers.Concatenate()
        self._decoder_dense = tf.keras.layers.Dense(
            hidden_dim,
            activation="relu",
            kernel_initializer="he_normal",
        )
        self._decoder_dropout = tf.keras.layers.Dropout(0.5)
        
        # Output (explicit float32 for mixed precision compatibility)
        self._output_layer = tf.keras.layers.Dense(
            vocab_size,
            activation="softmax",
            dtype="float32",
        )
    
    def call(
        self,
        inputs: Tuple[tf.Tensor, tf.Tensor],
        training: Optional[bool] = None,
    ) -> tf.Tensor:
        """Forward pass.
        
        Args:
            inputs: Tuple of (image_features, text_input).
                - image_features: Shape (batch, feature_dim).
                - text_input: Shape (batch, max_length).
            training: Whether in training mode.
            
        Returns:
            Predictions of shape (batch, vocab_size).
        """
        image_features, text_input = inputs
        
        # Encode image
        image_encoded = self._image_encoder(image_features, training=training)
        
        # Encode text
        text_encoded = self._text_encoder(text_input, training=training)
        
        # Apply attention
        context = self._attention(
            image_encoded,
            text_encoded,
            training=training,
        )
        
        # Decode
        merged = self._concat([context, image_encoded])
        decoded = self._decoder_dense(merged)
        decoded = self._decoder_dropout(decoded, training=training)
        
        # Output
        output = self._output_layer(decoded)
        
        return output
    
    def get_config(self) -> dict:
        """Get model configuration."""
        return {
            "vocab_size": self._vocab_size,
            "feature_dim": self._feature_dim,
            "embedding_dim": self._embedding_dim,
            "hidden_dim": self._hidden_dim,
        }


def build_caption_model(
    vocab_size: int,
    embedding_matrix: np.ndarray,
    max_length: int,
    feature_dim: int = 1536,
    embedding_dim: int = 200,
    hidden_dim: int = 256,
    num_attention_heads: int = 4,
    dropout_rate: float = 0.3,
    recurrent_dropout: float = 0.2,
    learning_rate: float = 1e-4,
) -> tf.keras.Model:
    """Build and compile caption model using functional API.
    
    This is an alternative to the CaptionModel class, using
    the Keras functional API for easier serialization.
    
    Args:
        vocab_size: Size of vocabulary.
        embedding_matrix: Pre-trained embedding matrix.
        max_length: Maximum sequence length.
        feature_dim: Dimension of image features.
        embedding_dim: Word embedding dimension.
        hidden_dim: Hidden dimension.
        num_attention_heads: Number of attention heads.
        dropout_rate: Dropout rate.
        recurrent_dropout: LSTM recurrent dropout.
        learning_rate: Learning rate for optimizer.
        
    Returns:
        Compiled Keras model.
    """
    # Inputs
    image_input = tf.keras.Input(shape=(feature_dim,), name="image_features")
    text_input = tf.keras.Input(shape=(max_length,), name="text_input")
    
    # Image branch
    img_proj = tf.keras.layers.Dense(hidden_dim, activation="relu")(image_input)
    img_proj = tf.keras.layers.Dropout(dropout_rate)(img_proj)
    img_proj = tf.keras.layers.LayerNormalization()(img_proj)
    
    # Text branch
    text_emb = tf.keras.layers.Embedding(
        vocab_size,
        embedding_dim,
        mask_zero=True,
        weights=[embedding_matrix],
        trainable=False,
    )(text_input)
    
    lstm_out = tf.keras.layers.LSTM(
        hidden_dim,
        return_sequences=True,
        dropout=dropout_rate,
        recurrent_dropout=recurrent_dropout,
    )(text_emb)
    
    # Attention
    query = tf.keras.layers.Reshape((1, hidden_dim))(img_proj)
    attention = tf.keras.layers.MultiHeadAttention(
        num_heads=num_attention_heads,
        key_dim=hidden_dim,
    )(query, lstm_out)
    context = tf.keras.layers.GlobalAveragePooling1D()(attention)
    
    # Fusion
    merged = tf.keras.layers.Concatenate()([context, img_proj])
    merged = tf.keras.layers.Dense(hidden_dim, activation="relu")(merged)
    merged = tf.keras.layers.Dropout(0.5)(merged)
    
    # Output
    output = tf.keras.layers.Dense(
        vocab_size,
        activation="softmax",
        dtype="float32",
    )(merged)
    
    # Build model
    model = tf.keras.Model(
        inputs=[image_input, text_input],
        outputs=output,
        name="caption_model",
    )
    
    # Compile with JIT compilation for faster training
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        # Use legacy optimizer for better mixed precision support
    )
    
    # Enable XLA JIT compilation for faster execution
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        jit_compile=True,  # Enable XLA JIT compilation
    )
    
    logger.info(f"Built caption model with {model.count_params():,} parameters")
    logger.info("XLA JIT compilation enabled for faster training")
    
    return model
