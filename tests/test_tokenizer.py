"""Tests for the Tokenizer class."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from training.data.tokenizer import Tokenizer, GloVeEmbeddings


class TestTokenizer:
    """Test suite for Tokenizer."""
    
    def test_initialization(self) -> None:
        """Test tokenizer initialization."""
        tokenizer = Tokenizer(min_word_count=5, max_vocab_size=1000)
        
        assert tokenizer._min_word_count == 5
        assert tokenizer._max_vocab_size == 1000
        assert not tokenizer.is_fitted
        assert tokenizer.vocab_size == 1  # Only padding
    
    def test_fit_builds_vocabulary(self, sample_captions: list[str]) -> None:
        """Test that fit() builds vocabulary correctly."""
        tokenizer = Tokenizer(min_word_count=1)
        tokenizer.fit(sample_captions)
        
        assert tokenizer.is_fitted
        assert tokenizer.vocab_size > 1
        
        # Check common words are in vocabulary
        word_to_idx = tokenizer.word_to_index
        assert "a" in word_to_idx
        assert "startseq" in word_to_idx
        assert "endseq" in word_to_idx
    
    def test_fit_respects_min_word_count(self) -> None:
        """Test that fit() filters by minimum word count."""
        captions = [
            "the dog runs",
            "the cat sits",
            "the bird flies",
            "a rare word",  # 'rare' appears only once
        ]
        
        tokenizer = Tokenizer(min_word_count=2)
        tokenizer.fit(captions)
        
        # 'the' appears 3 times, should be in vocab
        assert "the" in tokenizer.word_to_index
        
        # 'rare' appears once, should NOT be in vocab
        assert "rare" not in tokenizer.word_to_index
    
    def test_fit_respects_max_vocab_size(self) -> None:
        """Test that fit() respects maximum vocabulary size."""
        # Create captions with many unique words
        captions = [f"word{i}" for i in range(100)]
        
        tokenizer = Tokenizer(min_word_count=1, max_vocab_size=10)
        tokenizer.fit(captions)
        
        # vocab_size includes +1 for padding
        assert tokenizer.vocab_size <= 11
    
    def test_fit_returns_self(self, sample_captions: list[str]) -> None:
        """Test that fit() returns self for method chaining."""
        tokenizer = Tokenizer()
        result = tokenizer.fit(sample_captions)
        
        assert result is tokenizer
    
    def test_encode_basic(self, fitted_tokenizer: Tokenizer) -> None:
        """Test basic encoding functionality."""
        text = "a dog runs"
        encoded = fitted_tokenizer.encode(text)
        
        assert isinstance(encoded, list)
        assert all(isinstance(idx, int) for idx in encoded)
        assert len(encoded) == 3
    
    def test_encode_handles_unknown_words(self, fitted_tokenizer: Tokenizer) -> None:
        """Test that encode() handles unknown words gracefully."""
        text = "a xyzunknownword dog"
        encoded = fitted_tokenizer.encode(text)
        
        # Unknown words should be skipped
        assert len(encoded) == 2  # 'a' and 'dog'
    
    def test_encode_requires_fitted(self) -> None:
        """Test that encode() raises error if not fitted."""
        tokenizer = Tokenizer()
        
        with pytest.raises(RuntimeError, match="must be fitted"):
            tokenizer.encode("some text")
    
    def test_decode_basic(self, fitted_tokenizer: Tokenizer) -> None:
        """Test basic decoding functionality."""
        text = "a dog runs"
        encoded = fitted_tokenizer.encode(text)
        decoded = fitted_tokenizer.decode(encoded, skip_special=False)
        
        assert decoded == text
    
    def test_decode_skips_special_tokens(self, fitted_tokenizer: Tokenizer) -> None:
        """Test that decode() can skip special tokens."""
        text = "startseq a dog runs endseq"
        encoded = fitted_tokenizer.encode(text)
        decoded = fitted_tokenizer.decode(encoded, skip_special=True)
        
        assert "startseq" not in decoded
        assert "endseq" not in decoded
    
    def test_decode_handles_padding(self, fitted_tokenizer: Tokenizer) -> None:
        """Test that decode() handles padding (index 0)."""
        encoded_with_padding = [0, 0] + fitted_tokenizer.encode("a dog")
        decoded = fitted_tokenizer.decode(encoded_with_padding)
        
        # Should skip padding
        assert decoded.strip() == "a dog"
    
    def test_decode_stops_at_endseq(self, fitted_tokenizer: Tokenizer) -> None:
        """Test that decode() stops at endseq token."""
        text = "a dog endseq cat"  # cat should be ignored
        encoded = fitted_tokenizer.encode(text)
        decoded = fitted_tokenizer.decode(encoded, skip_special=False)
        
        assert "cat" not in decoded
    
    def test_word_to_index_is_copy(self, fitted_tokenizer: Tokenizer) -> None:
        """Test that word_to_index returns a copy."""
        wti1 = fitted_tokenizer.word_to_index
        wti2 = fitted_tokenizer.word_to_index
        
        wti1["new_word"] = 9999
        assert "new_word" not in wti2
        assert "new_word" not in fitted_tokenizer._word_to_index
    
    def test_index_to_word_is_copy(self, fitted_tokenizer: Tokenizer) -> None:
        """Test that index_to_word returns a copy."""
        itw1 = fitted_tokenizer.index_to_word
        itw2 = fitted_tokenizer.index_to_word
        
        itw1[9999] = "new_word"
        assert 9999 not in itw2
        assert 9999 not in fitted_tokenizer._index_to_word
    
    def test_save_and_load(self, fitted_tokenizer: Tokenizer) -> None:
        """Test saving and loading tokenizer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            fitted_tokenizer.save(tmpdir)
            
            # Check files exist
            assert (Path(tmpdir) / "wordtoix.json").exists()
            assert (Path(tmpdir) / "ixtoword.json").exists()
            
            # Load
            loaded = Tokenizer.load(tmpdir)
            
            # Verify loaded tokenizer
            assert loaded.is_fitted
            assert loaded.vocab_size == fitted_tokenizer.vocab_size
            assert loaded.word_to_index == fitted_tokenizer.word_to_index
    
    def test_save_creates_directory(self, fitted_tokenizer: Tokenizer) -> None:
        """Test that save() creates directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new" / "nested" / "dir"
            fitted_tokenizer.save(new_dir)
            
            assert new_dir.exists()
            assert (new_dir / "wordtoix.json").exists()
    
    def test_roundtrip_encoding(self, fitted_tokenizer: Tokenizer) -> None:
        """Test that encode/decode roundtrips correctly."""
        texts = [
            "a dog runs",
            "the cat sits on a mat",
            "a man walks",
        ]
        
        for text in texts:
            encoded = fitted_tokenizer.encode(text)
            decoded = fitted_tokenizer.decode(encoded, skip_special=False)
            assert decoded == text
    
    def test_encode_empty_string(self, fitted_tokenizer: Tokenizer) -> None:
        """Test encoding empty string."""
        encoded = fitted_tokenizer.encode("")
        assert encoded == []
    
    def test_decode_empty_list(self, fitted_tokenizer: Tokenizer) -> None:
        """Test decoding empty list."""
        decoded = fitted_tokenizer.decode([])
        assert decoded == ""


class TestGloVeEmbeddings:
    """Test suite for GloVeEmbeddings."""
    
    def test_initialization(self) -> None:
        """Test GloVe embeddings initialization."""
        glove = GloVeEmbeddings("/path/to/glove.txt", embedding_dim=200)
        
        assert glove.embedding_dim == 200
        assert not glove.is_loaded
    
    def test_load_creates_embeddings(self, fitted_tokenizer: Tokenizer) -> None:
        """Test loading GloVe file creates embeddings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock GloVe file
            glove_path = Path(tmpdir) / "glove.txt"
            with open(glove_path, "w") as f:
                f.write("dog " + " ".join(["0.1"] * 50) + "\n")
                f.write("cat " + " ".join(["0.2"] * 50) + "\n")
                f.write("runs " + " ".join(["0.3"] * 50) + "\n")
            
            glove = GloVeEmbeddings(glove_path, embedding_dim=50)
            glove.load()
            
            assert glove.is_loaded
    
    def test_load_file_not_found(self) -> None:
        """Test that load() raises error for missing file."""
        glove = GloVeEmbeddings("/nonexistent/path.txt")
        
        with pytest.raises(FileNotFoundError):
            glove.load()
    
    def test_create_embedding_matrix(self, fitted_tokenizer: Tokenizer) -> None:
        """Test creating embedding matrix from GloVe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock GloVe file with words from our vocab
            glove_path = Path(tmpdir) / "glove.txt"
            with open(glove_path, "w") as f:
                for word in ["a", "dog", "cat", "runs", "sits"]:
                    f.write(word + " " + " ".join(["0.1"] * 50) + "\n")
            
            glove = GloVeEmbeddings(glove_path, embedding_dim=50)
            glove.load()
            
            matrix = glove.create_embedding_matrix(fitted_tokenizer)
            
            assert matrix.shape == (fitted_tokenizer.vocab_size, 50)
            assert matrix.dtype == np.float32
    
    def test_create_embedding_matrix_requires_loaded(
        self, fitted_tokenizer: Tokenizer
    ) -> None:
        """Test that create_embedding_matrix requires loaded embeddings."""
        glove = GloVeEmbeddings("/path/to/glove.txt")
        
        with pytest.raises(RuntimeError, match="must be loaded"):
            glove.create_embedding_matrix(fitted_tokenizer)
