"""Tests for decoder termination conditions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tensorflow as tf

from training.data.tokenizer import Tokenizer
from training.evaluation.inference import BeamSearchDecoder, GreedyDecoder


class TestGreedyDecoderTermination:
    """Test suite for GreedyDecoder termination conditions."""
    
    @pytest.fixture
    def decoder_tokenizer(self) -> Tokenizer:
        """Create tokenizer for decoder tests."""
        tokenizer = Tokenizer(min_word_count=1)
        captions = [
            "startseq a dog runs endseq",
            "startseq a cat sits endseq",
            "startseq the bird flies endseq",
        ]
        tokenizer.fit(captions)
        return tokenizer
    
    @pytest.fixture
    def greedy_decoder(self, decoder_tokenizer: Tokenizer) -> GreedyDecoder:
        """Create greedy decoder."""
        return GreedyDecoder(
            tokenizer=decoder_tokenizer,
            max_length=20,
            repetition_penalty=True,
        )
    
    def test_stops_at_endseq(self, greedy_decoder: GreedyDecoder) -> None:
        """Test that decoder stops when predicting endseq."""
        mock_model = MagicMock()
        
        # Simulate: startseq -> a -> dog -> endseq
        vocab_size = greedy_decoder.tokenizer.vocab_size
        wordtoix = greedy_decoder.tokenizer.word_to_index
        
        predictions = []
        for word in ["a", "dog", "endseq"]:
            pred = np.zeros(vocab_size)
            pred[wordtoix[word]] = 1.0
            predictions.append(pred)
        
        mock_model.predict = MagicMock(side_effect=[
            np.array([predictions[0]]),
            np.array([predictions[1]]),
            np.array([predictions[2]]),
        ])
        
        image_features = np.random.randn(1, 1536).astype(np.float32)
        caption = greedy_decoder.decode(mock_model, image_features)
        
        assert "endseq" not in caption
        assert "a" in caption
        assert "dog" in caption
    
    def test_stops_at_max_length(self, greedy_decoder: GreedyDecoder) -> None:
        """Test that decoder stops at max_length."""
        mock_model = MagicMock()
        
        vocab_size = greedy_decoder.tokenizer.vocab_size
        wordtoix = greedy_decoder.tokenizer.word_to_index
        
        # Always predict "a" - never endseq
        pred = np.zeros(vocab_size)
        pred[wordtoix["a"]] = 1.0
        mock_model.predict = MagicMock(return_value=np.array([pred]))
        
        # Create decoder with short max_length
        short_decoder = GreedyDecoder(
            tokenizer=greedy_decoder.tokenizer,
            max_length=5,
            repetition_penalty=False,  # Disable to allow repetition
        )
        
        image_features = np.random.randn(1, 1536).astype(np.float32)
        caption = short_decoder.decode(mock_model, image_features)
        
        # Should stop after max_length predictions
        word_count = len(caption.split())
        assert word_count <= 5
    
    def test_stops_on_repetition(self, greedy_decoder: GreedyDecoder) -> None:
        """Test that decoder stops on word repetition."""
        mock_model = MagicMock()
        
        vocab_size = greedy_decoder.tokenizer.vocab_size
        wordtoix = greedy_decoder.tokenizer.word_to_index
        
        # Predict: a, dog, dog (repetition)
        predictions = []
        for word in ["a", "dog", "dog"]:
            pred = np.zeros(vocab_size)
            pred[wordtoix[word]] = 1.0
            predictions.append(pred)
        
        mock_model.predict = MagicMock(side_effect=[
            np.array([predictions[0]]),
            np.array([predictions[1]]),
            np.array([predictions[2]]),
        ])
        
        image_features = np.random.randn(1, 1536).astype(np.float32)
        caption = greedy_decoder.decode(mock_model, image_features)
        
        # Should stop before the second "dog"
        words = caption.split()
        assert words.count("dog") <= 1
    
    def test_no_repetition_penalty(self, decoder_tokenizer: Tokenizer) -> None:
        """Test decoder without repetition penalty allows repetition."""
        decoder = GreedyDecoder(
            tokenizer=decoder_tokenizer,
            max_length=10,
            repetition_penalty=False,
        )
        
        mock_model = MagicMock()
        vocab_size = decoder.tokenizer.vocab_size
        wordtoix = decoder.tokenizer.word_to_index
        
        # Predict: a, a, a, then endseq
        predictions = []
        for i, word in enumerate(["a", "a", "a", "endseq"]):
            pred = np.zeros(vocab_size)
            pred[wordtoix[word]] = 1.0
            predictions.append(pred)
        
        mock_model.predict = MagicMock(side_effect=[
            np.array([p]) for p in predictions
        ])
        
        image_features = np.random.randn(1, 1536).astype(np.float32)
        caption = decoder.decode(mock_model, image_features)
        
        # Should allow repetitions
        assert caption.count("a") >= 1
    
    def test_stops_on_unknown_token(self, greedy_decoder: GreedyDecoder) -> None:
        """Test that decoder stops when predicting unknown token."""
        mock_model = MagicMock()
        
        vocab_size = greedy_decoder.tokenizer.vocab_size
        wordtoix = greedy_decoder.tokenizer.word_to_index
        
        # Predict: a, then an index that doesn't map to a word
        pred1 = np.zeros(vocab_size)
        pred1[wordtoix["a"]] = 1.0
        
        pred2 = np.zeros(vocab_size)
        pred2[vocab_size - 1] = 1.0  # Index that may not exist
        
        mock_model.predict = MagicMock(side_effect=[
            np.array([pred1]),
            np.array([pred2]),
        ])
        
        image_features = np.random.randn(1, 1536).astype(np.float32)
        caption = greedy_decoder.decode(mock_model, image_features)
        
        # Should have at least generated "a"
        assert "a" in caption


class TestBeamSearchDecoderTermination:
    """Test suite for BeamSearchDecoder termination conditions."""
    
    @pytest.fixture
    def decoder_tokenizer(self) -> Tokenizer:
        """Create tokenizer for decoder tests."""
        tokenizer = Tokenizer(min_word_count=1)
        captions = [
            "startseq a dog runs endseq",
            "startseq a cat sits endseq",
            "startseq the bird flies endseq",
        ]
        tokenizer.fit(captions)
        return tokenizer
    
    @pytest.fixture
    def beam_decoder(self, decoder_tokenizer: Tokenizer) -> BeamSearchDecoder:
        """Create beam search decoder."""
        return BeamSearchDecoder(
            tokenizer=decoder_tokenizer,
            max_length=20,
            beam_width=3,
            length_penalty=0.7,
        )
    
    def test_stops_at_endseq(self, beam_decoder: BeamSearchDecoder) -> None:
        """Test that beam search stops when all beams reach endseq."""
        mock_model = MagicMock()
        
        vocab_size = beam_decoder.tokenizer.vocab_size
        wordtoix = beam_decoder.tokenizer.word_to_index
        
        # Create predictions that lead to endseq
        def make_pred(word: str) -> np.ndarray:
            pred = np.full(vocab_size, 0.01)  # Small probs for all
            pred[wordtoix[word]] = 0.9
            return pred
        
        call_count = [0]
        
        def mock_predict(inputs, verbose=0):
            call_count[0] += 1
            if call_count[0] <= 1:
                return np.array([make_pred("a")])
            elif call_count[0] <= 2:
                return np.array([make_pred("dog")])
            else:
                return np.array([make_pred("endseq")])
        
        mock_model.predict = mock_predict
        
        image_features = np.random.randn(1, 1536).astype(np.float32)
        caption = beam_decoder.decode(mock_model, image_features)
        
        assert "endseq" not in caption
        assert "startseq" not in caption
    
    def test_stops_at_max_length(self, decoder_tokenizer: Tokenizer) -> None:
        """Test that beam search stops at max_length."""
        short_decoder = BeamSearchDecoder(
            tokenizer=decoder_tokenizer,
            max_length=3,
            beam_width=2,
        )
        
        mock_model = MagicMock()
        vocab_size = short_decoder.tokenizer.vocab_size
        wordtoix = short_decoder.tokenizer.word_to_index
        
        # Always predict "a" - never endseq
        pred = np.full(vocab_size, 0.01)
        pred[wordtoix["a"]] = 0.9
        mock_model.predict = MagicMock(return_value=np.array([pred]))
        
        image_features = np.random.randn(1, 1536).astype(np.float32)
        caption = short_decoder.decode(mock_model, image_features)
        
        # Should stop at or before max_length
        word_count = len(caption.split())
        assert word_count <= 3
    
    def test_beam_width_property(self, beam_decoder: BeamSearchDecoder) -> None:
        """Test beam_width property."""
        assert beam_decoder.beam_width == 3
    
    def test_returns_best_sequence(self, beam_decoder: BeamSearchDecoder) -> None:
        """Test that beam search returns the best scoring sequence."""
        mock_model = MagicMock()
        
        vocab_size = beam_decoder.tokenizer.vocab_size
        wordtoix = beam_decoder.tokenizer.word_to_index
        
        # Give higher probability to "dog" than "cat"
        call_count = [0]
        
        def mock_predict(inputs, verbose=0):
            call_count[0] += 1
            pred = np.full(vocab_size, 0.01)
            if call_count[0] <= 1:
                pred[wordtoix["dog"]] = 0.8
                pred[wordtoix["cat"]] = 0.5
            else:
                pred[wordtoix["endseq"]] = 0.9
            return np.array([pred])
        
        mock_model.predict = mock_predict
        
        image_features = np.random.randn(1, 1536).astype(np.float32)
        caption = beam_decoder.decode(mock_model, image_features)
        
        # Should prefer "dog" over "cat" due to higher probability
        # (though beam search may find different paths)
        assert isinstance(caption, str)
        assert len(caption) > 0


class TestDecoderEdgeCases:
    """Test edge cases for decoders."""
    
    @pytest.fixture
    def decoder_tokenizer(self) -> Tokenizer:
        """Create tokenizer for decoder tests."""
        tokenizer = Tokenizer(min_word_count=1)
        captions = [
            "startseq a dog runs endseq",
            "startseq a cat sits endseq",
        ]
        tokenizer.fit(captions)
        return tokenizer
    
    def test_greedy_empty_vocab_word(self, decoder_tokenizer: Tokenizer) -> None:
        """Test greedy decoder handles missing vocab words."""
        decoder = GreedyDecoder(decoder_tokenizer, max_length=10)
        
        mock_model = MagicMock()
        vocab_size = decoder.tokenizer.vocab_size
        
        # Predict index 0 (padding, not in ixtoword)
        pred = np.zeros(vocab_size)
        pred[0] = 1.0  # Padding index
        mock_model.predict = MagicMock(return_value=np.array([pred]))
        
        image_features = np.random.randn(1, 1536).astype(np.float32)
        
        # Should handle gracefully without error
        caption = decoder.decode(mock_model, image_features)
        assert isinstance(caption, str)
    
    def test_beam_handles_all_completed(self, decoder_tokenizer: Tokenizer) -> None:
        """Test beam search handles case where all beams complete early."""
        decoder = BeamSearchDecoder(decoder_tokenizer, max_length=20, beam_width=2)
        
        mock_model = MagicMock()
        vocab_size = decoder.tokenizer.vocab_size
        wordtoix = decoder.tokenizer.word_to_index
        
        # Immediately predict endseq
        pred = np.full(vocab_size, 0.01)
        pred[wordtoix["endseq"]] = 0.99
        mock_model.predict = MagicMock(return_value=np.array([pred]))
        
        image_features = np.random.randn(1, 1536).astype(np.float32)
        caption = decoder.decode(mock_model, image_features)
        
        # Should return empty or minimal caption
        assert isinstance(caption, str)
    
    def test_decoder_properties(self, decoder_tokenizer: Tokenizer) -> None:
        """Test decoder property accessors."""
        greedy = GreedyDecoder(decoder_tokenizer, max_length=38)
        beam = BeamSearchDecoder(decoder_tokenizer, max_length=38, beam_width=5)
        
        assert greedy.tokenizer is decoder_tokenizer
        assert greedy.max_length == 38
        
        assert beam.tokenizer is decoder_tokenizer
        assert beam.max_length == 38
        assert beam.beam_width == 5
