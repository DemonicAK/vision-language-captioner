# Image Captioning Production

Production-grade image captioning system with attention-based deep learning model.

## Features

- **Training Pipeline**: Structured training with experiment tracking, TensorBoard logging, and model versioning
- **Inference API**: FastAPI-based REST API with lazy model loading and request monitoring
- **Testing**: Comprehensive unit, integration, and performance tests
- **MLOps**: Model versioning, deployment manifests, and metrics tracking

## Project Structure

```
image-captioning-production/
├── training/               # Training pipeline
│   ├── configs/           # Configuration management
│   ├── data/              # Data processing & tokenization
│   ├── features/          # Image feature extraction
│   ├── models/            # Model architecture
│   ├── trainers/          # Training logic with experiment tracking
│   ├── evaluation/        # BLEU metrics & decoders
│   └── utils/             # Experiment tracking, model versioning
├── inference/             # Production inference service
│   ├── main.py           # FastAPI application
│   ├── model_registry.py # Thread-safe model management
│   ├── caption_generator.py # Caption generation algorithms
│   └── preprocessing.py   # Image preprocessing
├── tests/                 # Test suite
│   ├── test_tokenizer.py
│   ├── test_dataset.py
│   ├── test_decoder.py
│   ├── test_config.py
│   ├── test_experiment_tracking.py
│   ├── test_integration.py
│   └── test_performance.py
└── shared/artifacts/      # Model artifacts
```

## Quick Start

### Training

```bash
# Configure paths in training/config.yaml
python -m training.train --config training/config.yaml
```

### Inference

```bash
# Run with Docker
docker-compose up inference

# Or run directly
cd inference && uvicorn main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info |
| `/health` | GET | Health check |
| `/ready` | GET | Readiness check |
| `/metrics` | GET | Application metrics |
| `/predict` | POST | Generate caption |
| `/model/info` | GET | Model metadata |

### Generate Caption

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg" \
  -F "algorithm=beam"
```

## Testing

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=training --cov=inference --cov-report=html

# Run specific test categories
pytest tests/test_tokenizer.py -v
pytest -m "not slow"  # Skip slow tests
pytest -m performance  # Performance tests only
```

## Experiment Tracking

Training automatically generates:

- `run_summary.json`: Complete run metadata, config snapshot, git info
- `metrics.csv`: Epoch-wise training/validation metrics
- `config_snapshot.json`: Configuration used for the run
- `logs/`: TensorBoard logs

```python
from training.utils import ExperimentTracker, set_global_seed

# Set reproducibility seed
set_global_seed(42)

# Track experiment
tracker = ExperimentTracker("experiments/run_001", seed=42)
tracker.log_config(config_dict)
tracker.log_metrics(epoch=1, train_loss=2.5, val_loss=2.8)
tracker.log_bleu_scores(bleu1=0.65, bleu2=0.45, bleu3=0.35, bleu4=0.25)
tracker.finalize()
```

## Model Versioning

```python
from training.utils.model_versioning import ModelVersionManager

manager = ModelVersionManager("models/")

# Register version
manager.register_version(
    version="1.0.0",
    model_path="artifacts/model.keras",
    metrics={"bleu4": 0.25},
)

# Compare versions
comparison = manager.compare_versions("1.0.0", "1.1.0")
```

## Production Deployment

The inference service is production-ready with:

- **Lazy model loading**: Models load on first request, not at import
- **Thread-safe model registry**: Singleton pattern with proper locking
- **Request logging**: All requests logged with timing and request IDs
- **Health/readiness endpoints**: For load balancers and k8s probes
- **Metrics endpoint**: Request counts, error rates, uptime
- **CORS support**: Configurable cross-origin requests

## Architecture Highlights

### Training
- Dataclass-based configuration with validation
- Registry pattern for extensible feature extractors
- Callbacks for custom metrics and progress tracking
- Distributed training support via TensorFlow strategies

### Inference
- Clean separation: `ModelBundle`, `CaptionService`, `ImagePreprocessor`
- No global state or import-time model loading
- Support for greedy and beam search decoding
- Backward-compatible legacy API functions

## License

MIT
