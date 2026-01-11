"""Integration tests for the inference API."""

from __future__ import annotations

import io
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Skip if dependencies not available
PIL = pytest.importorskip("PIL")
from PIL import Image


class TestAPIIntegration:
    """Integration tests for the FastAPI application."""
    
    @pytest.fixture
    def test_client(self):
        """Create test client with mocked model."""
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        
        from fastapi.testclient import TestClient
        from inference.main import app
        
        return TestClient(app)
    
    @pytest.fixture
    def sample_image_bytes(self) -> bytes:
        """Create sample image bytes."""
        img = Image.new("RGB", (300, 300), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        return buffer.read()
    
    def test_root_endpoint(self, test_client) -> None:
        """Test root endpoint returns expected response."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert data["status"] == "running"
    
    def test_health_endpoint(self, test_client) -> None:
        """Test health endpoint."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "tensorflow_version" in data
    
    def test_ready_endpoint(self, test_client) -> None:
        """Test readiness endpoint."""
        response = test_client.get("/ready")
        
        assert response.status_code == 200
        data = response.json()
        assert "ready" in data
    
    def test_metrics_endpoint(self, test_client) -> None:
        """Test metrics endpoint."""
        response = test_client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "request_count" in data
        assert "error_count" in data
    
    def test_predict_invalid_file(self, test_client) -> None:
        """Test predict endpoint with invalid file."""
        response = test_client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")},
            data={"algorithm": "beam"},
        )
        
        assert response.status_code in [400, 500]
    
    def test_predict_missing_file(self, test_client) -> None:
        """Test predict endpoint without file."""
        response = test_client.post("/predict")
        
        assert response.status_code == 422  # Validation error


class TestModelRegistry:
    """Tests for ModelRegistry."""
    
    def test_singleton_pattern(self) -> None:
        """Test that ModelRegistry is a singleton."""
        from inference.model_registry import ModelRegistry
        
        r1 = ModelRegistry()
        r2 = ModelRegistry()
        
        assert r1 is r2
    
    def test_register_and_retrieve(self) -> None:
        """Test registering and retrieving model configs."""
        from inference.model_registry import ModelRegistry, ModelConfig
        
        registry = ModelRegistry()
        registry.clear()
        
        config = ModelConfig(
            model_path="/path/to/model.keras",
            vocab_path="/path/to/vocab",
            max_length=38,
        )
        
        registry.register("test_model", config)
        
        assert "test_model" in registry._configs
        
        # Cleanup
        registry.clear()
    
    def test_unregister_model(self) -> None:
        """Test unloading models."""
        from inference.model_registry import ModelRegistry
        
        registry = ModelRegistry()
        registry.clear()
        
        # Should not error even if model doesn't exist
        registry.unload("nonexistent")


class TestCaptionService:
    """Tests for CaptionService."""
    
    @pytest.fixture
    def mock_bundle(self) -> MagicMock:
        """Create mock model bundle."""
        bundle = MagicMock()
        bundle.max_length = 20
        bundle.vocab_size = 100
        bundle.word_to_index = {
            "startseq": 1, "endseq": 2, 
            "a": 3, "dog": 4, "runs": 5
        }
        bundle.index_to_word = {
            1: "startseq", 2: "endseq",
            3: "a", 4: "dog", 5: "runs"
        }
        
        # Mock model - always predicts endseq after a few words
        vocab_size = 100
        call_count = [0]
        
        def mock_predict(inputs, verbose=0):
            call_count[0] += 1
            pred = np.zeros(vocab_size, dtype=np.float32)
            if call_count[0] == 1:
                pred[3] = 0.9  # 'a'
            elif call_count[0] == 2:
                pred[4] = 0.9  # 'dog'
            else:
                pred[2] = 0.9  # 'endseq'
            return np.array([pred])
        
        bundle.model.predict = mock_predict
        
        return bundle
    
    def test_caption_service_greedy(self, mock_bundle: MagicMock) -> None:
        """Test caption service with greedy algorithm."""
        from inference.caption_generator import CaptionService
        
        service = CaptionService(mock_bundle)
        
        features = np.random.rand(1, 1536).astype(np.float32)
        caption = service.generate_caption(features, algorithm="greedy")
        
        assert isinstance(caption, str)
        assert len(caption) > 0
    
    def test_caption_service_beam(self, mock_bundle: MagicMock) -> None:
        """Test caption service with beam search algorithm."""
        from inference.caption_generator import CaptionService
        
        service = CaptionService(mock_bundle)
        
        features = np.random.rand(1, 1536).astype(np.float32)
        caption = service.generate_caption(features, algorithm="beam")
        
        assert isinstance(caption, str)
    
    def test_caption_service_invalid_algorithm(self, mock_bundle: MagicMock) -> None:
        """Test caption service with invalid algorithm."""
        from inference.caption_generator import CaptionService
        
        service = CaptionService(mock_bundle)
        
        features = np.random.rand(1, 1536).astype(np.float32)
        
        with pytest.raises(ValueError, match="Unknown algorithm"):
            service.generate_caption(features, algorithm="invalid")
    
    def test_available_algorithms(self, mock_bundle: MagicMock) -> None:
        """Test listing available algorithms."""
        from inference.caption_generator import CaptionService
        
        service = CaptionService(mock_bundle)
        
        algorithms = service.available_algorithms
        assert "greedy" in algorithms
        assert "beam" in algorithms


class TestImagePreprocessor:
    """Tests for ImagePreprocessor."""
    
    @pytest.fixture
    def mock_extractor(self) -> MagicMock:
        """Create mock feature extractor."""
        extractor = MagicMock()
        extractor.return_value = np.random.rand(1, 1536).astype(np.float32)
        return extractor
    
    def test_preprocessing(self, mock_extractor: MagicMock) -> None:
        """Test image preprocessing."""
        from inference.preprocessing import ImagePreprocessor
        
        preprocessor = ImagePreprocessor(
            feature_extractor=mock_extractor,
            image_size=(300, 300),
        )
        
        # Create test image
        img = Image.new("RGB", (640, 480), color="blue")
        
        features = preprocessor.extract_features(img)
        
        assert mock_extractor.called
    
    def test_image_size_property(self, mock_extractor: MagicMock) -> None:
        """Test image_size property."""
        from inference.preprocessing import ImagePreprocessor
        
        preprocessor = ImagePreprocessor(
            feature_extractor=mock_extractor,
            image_size=(224, 224),
        )
        
        assert preprocessor.image_size == (224, 224)
