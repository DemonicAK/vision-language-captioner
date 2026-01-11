"""Performance and load tests for inference service."""

from __future__ import annotations

import asyncio
import io
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Skip if dependencies not available
pytest.importorskip("PIL")


class TestInferencePerformance:
    """Performance tests for inference components."""
    
    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Create mock model for performance testing."""
        mock = MagicMock()
        # Simulate realistic prediction time
        vocab_size = 5000
        pred = np.random.rand(vocab_size).astype(np.float32)
        pred = pred / pred.sum()  # Normalize
        mock.predict = MagicMock(return_value=np.array([pred]))
        return mock
    
    @pytest.fixture
    def mock_word_to_index(self) -> Dict[str, int]:
        """Create mock vocabulary."""
        words = ["startseq", "endseq", "a", "the", "dog", "cat", "runs", "sits"]
        words.extend([f"word{i}" for i in range(100)])
        return {w: i + 1 for i, w in enumerate(words)}
    
    @pytest.fixture
    def mock_index_to_word(self, mock_word_to_index: Dict[str, int]) -> Dict[int, str]:
        """Create reverse vocabulary."""
        return {v: k for k, v in mock_word_to_index.items()}
    
    def test_greedy_generation_time(
        self,
        mock_model: MagicMock,
        mock_word_to_index: Dict[str, int],
        mock_index_to_word: Dict[int, str],
    ) -> None:
        """Test that greedy generation completes within time limit."""
        from inference.caption_generator import GreedyGenerator
        
        generator = GreedyGenerator(
            model=mock_model,
            word_to_index=mock_word_to_index,
            index_to_word=mock_index_to_word,
            max_length=20,
        )
        
        # Ensure endseq is predicted eventually
        call_count = [0]
        vocab_size = len(mock_word_to_index) + 1
        
        def mock_predict(inputs, verbose=0):
            call_count[0] += 1
            pred = np.random.rand(vocab_size).astype(np.float32)
            if call_count[0] >= 5:
                pred[mock_word_to_index["endseq"]] = 0.99
            pred = pred / pred.sum()
            return np.array([pred])
        
        mock_model.predict = mock_predict
        
        image_features = np.random.rand(1, 1536).astype(np.float32)
        
        # Measure time
        times = []
        for _ in range(5):
            call_count[0] = 0
            start = time.time()
            _ = generator.generate(image_features)
            times.append(time.time() - start)
        
        avg_time = statistics.mean(times)
        
        # Should complete quickly (< 2s without real model)
        assert avg_time < 2.0, f"Greedy generation too slow: {avg_time:.3f}s"
    
    def test_beam_search_generation_time(
        self,
        mock_model: MagicMock,
        mock_word_to_index: Dict[str, int],
        mock_index_to_word: Dict[int, str],
    ) -> None:
        """Test that beam search completes within time limit."""
        from inference.caption_generator import BeamSearchGenerator
        
        generator = BeamSearchGenerator(
            model=mock_model,
            word_to_index=mock_word_to_index,
            index_to_word=mock_index_to_word,
            max_length=20,
            beam_width=3,
        )
        
        call_count = [0]
        vocab_size = len(mock_word_to_index) + 1
        
        def mock_predict(inputs, verbose=0):
            call_count[0] += 1
            pred = np.random.rand(vocab_size).astype(np.float32)
            if call_count[0] >= 10:
                pred[mock_word_to_index["endseq"]] = 0.99
            pred = pred / pred.sum()
            return np.array([pred])
        
        mock_model.predict = mock_predict
        
        image_features = np.random.rand(1, 1536).astype(np.float32)
        
        # Measure time
        times = []
        for _ in range(5):
            call_count[0] = 0
            start = time.time()
            _ = generator.generate(image_features)
            times.append(time.time() - start)
        
        avg_time = statistics.mean(times)
        
        # Beam search is slower, allow more time
        assert avg_time < 5.0, f"Beam search too slow: {avg_time:.3f}s"
    
    def test_preprocessing_performance(self) -> None:
        """Test image preprocessing performance."""
        from PIL import Image
        import numpy as np
        
        # Create test image
        img_array = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        image = Image.fromarray(img_array)
        
        # Mock feature extractor
        mock_extractor = MagicMock()
        mock_extractor.return_value = np.random.rand(1, 1536).astype(np.float32)
        
        from inference.preprocessing import ImagePreprocessor
        
        preprocessor = ImagePreprocessor(
            feature_extractor=mock_extractor,
            image_size=(300, 300),
        )
        
        # Measure preprocessing time
        times = []
        for _ in range(10):
            start = time.time()
            _ = preprocessor.extract_features(image)
            times.append(time.time() - start)
        
        avg_time = statistics.mean(times)
        
        # Preprocessing should be fast
        assert avg_time < 1.0, f"Preprocessing too slow: {avg_time:.3f}s"


class TestConcurrency:
    """Test concurrent request handling."""
    
    @pytest.fixture
    def mock_bundle(self) -> MagicMock:
        """Create mock model bundle."""
        bundle = MagicMock()
        bundle.max_length = 20
        bundle.vocab_size = 100
        bundle.word_to_index = {"startseq": 1, "endseq": 2, "a": 3, "dog": 4}
        bundle.index_to_word = {1: "startseq", 2: "endseq", 3: "a", 4: "dog"}
        
        # Mock model prediction
        vocab_size = 100
        pred = np.zeros(vocab_size)
        pred[2] = 1.0  # Always predict endseq
        bundle.model.predict = MagicMock(return_value=np.array([pred]))
        
        return bundle
    
    def test_thread_safety(self, mock_bundle: MagicMock) -> None:
        """Test that caption service is thread-safe."""
        from inference.caption_generator import CaptionService
        
        service = CaptionService(mock_bundle)
        
        results: List[str] = []
        errors: List[Exception] = []
        
        def generate_caption():
            try:
                features = np.random.rand(1, 1536).astype(np.float32)
                caption = service.generate_caption(features, algorithm="greedy")
                results.append(caption)
            except Exception as e:
                errors.append(e)
        
        # Run concurrent requests
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(generate_caption) for _ in range(10)]
            for future in futures:
                future.result()
        
        # Should complete without errors
        assert len(errors) == 0, f"Errors during concurrent execution: {errors}"
        assert len(results) == 10
    
    def test_model_registry_singleton(self) -> None:
        """Test that ModelRegistry is a proper singleton."""
        from inference.model_registry import ModelRegistry
        
        registries = []
        
        def get_registry():
            registries.append(ModelRegistry())
        
        # Create from multiple threads
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(get_registry) for _ in range(10)]
            for future in futures:
                future.result()
        
        # All should be the same instance
        assert all(r is registries[0] for r in registries)


class TestMemoryUsage:
    """Test memory management."""
    
    def test_model_unload(self) -> None:
        """Test that models can be unloaded to free memory."""
        from inference.model_registry import ModelRegistry, ModelConfig
        
        registry = ModelRegistry()
        
        # Clear any existing state
        registry.clear()
        
        # Create mock config
        config = MagicMock(spec=ModelConfig)
        config.model_path = "/fake/path"
        config.vocab_path = "/fake/vocab"
        config.max_length = 38
        config.feature_extractor_name = "EfficientNetB3"
        config.image_size = (300, 300)
        
        registry.register("test_model", config)
        
        # Model shouldn't be loaded yet
        assert not registry.is_loaded("test_model")
        
        # Unload (even though not loaded) shouldn't error
        registry.unload("test_model")
        
        # Clear all
        registry.clear()


class TestAPIPerformance:
    """API endpoint performance tests."""
    
    @pytest.fixture
    def test_client(self):
        """Create test client."""
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        
        from fastapi.testclient import TestClient
        from inference.main import app
        
        return TestClient(app)
    
    def test_health_endpoint_latency(self, test_client) -> None:
        """Test that health endpoint responds quickly."""
        times = []
        
        for _ in range(10):
            start = time.time()
            response = test_client.get("/health")
            times.append(time.time() - start)
            assert response.status_code == 200
        
        avg_time = statistics.mean(times)
        
        # Health check should be very fast
        assert avg_time < 0.1, f"Health check too slow: {avg_time:.3f}s"
    
    def test_metrics_endpoint_latency(self, test_client) -> None:
        """Test that metrics endpoint responds quickly."""
        times = []
        
        for _ in range(10):
            start = time.time()
            response = test_client.get("/metrics")
            times.append(time.time() - start)
            assert response.status_code == 200
        
        avg_time = statistics.mean(times)
        
        assert avg_time < 0.1, f"Metrics endpoint too slow: {avg_time:.3f}s"


class TestBenchmarks:
    """Benchmark tests for establishing baselines."""
    
    def test_tokenizer_encode_benchmark(self, fitted_tokenizer) -> None:
        """Benchmark tokenizer encoding speed."""
        texts = [
            "a dog runs in the park",
            "the cat sits on a mat",
            "a bird flies in the sky",
        ] * 100
        
        start = time.time()
        for text in texts:
            _ = fitted_tokenizer.encode(text)
        duration = time.time() - start
        
        ops_per_second = len(texts) / duration
        
        # Should handle at least 1000 encodes/second
        assert ops_per_second > 1000, f"Tokenizer too slow: {ops_per_second:.0f} ops/s"
    
    def test_tokenizer_decode_benchmark(self, fitted_tokenizer) -> None:
        """Benchmark tokenizer decoding speed."""
        # Create encoded sequences
        text = "a dog runs in the park"
        encoded = fitted_tokenizer.encode(text)
        sequences = [encoded] * 300
        
        start = time.time()
        for seq in sequences:
            _ = fitted_tokenizer.decode(seq)
        duration = time.time() - start
        
        ops_per_second = len(sequences) / duration
        
        # Should handle at least 1000 decodes/second
        assert ops_per_second > 1000, f"Decoder too slow: {ops_per_second:.0f} ops/s"
