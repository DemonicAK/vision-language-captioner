# Training Module

Production-grade training pipeline for image captioning models with attention mechanisms.

## Architecture

```
training/
├── __init__.py              # Package exports
├── train.py                 # Main entrypoint (TrainingPipeline class)
├── config.yaml              # Default configuration
├── README.md
│
├── configs/                 # Configuration management
│   ├── __init__.py
│   └── training_config.py   # Dataclass-based configs with validation
│
├── data/                    # Data processing
│   ├── __init__.py
│   ├── preprocessor.py      # Text cleaning (TextPreprocessor, CaptionPreprocessor)
│   ├── caption_loader.py    # Caption loading and splitting (CaptionLoader, DataSplit)
│   ├── tokenizer.py         # Vocabulary and embeddings (Tokenizer, GloVeEmbeddings)
│   └── dataset.py           # tf.data pipelines (DatasetBuilder, TrainingSample)
│
├── features/                # Image feature extraction
│   ├── __init__.py
│   ├── base.py              # Abstract base class (BaseFeatureExtractor)
│   ├── efficientnet.py      # EfficientNet/InceptionV3 extractors
│   └── registry.py          # Registry pattern for extractor lookup
│
├── models/                  # Model architecture
│   ├── __init__.py
│   ├── caption_model.py     # Main model (CaptionModel, build_caption_model)
│   └── components/
│       ├── __init__.py
│       ├── attention.py     # MultiHeadCrossAttention, BahdanauAttention
│       └── encoders.py      # ImageProjection, TextEmbedding, LSTMEncoder
│
├── trainers/                # Training logic
│   ├── __init__.py
│   └── trainer.py           # Trainer, DistributedTrainer, CallbackFactory
│
├── evaluation/              # Evaluation and inference
│   ├── __init__.py
│   ├── metrics.py           # BLEUScore, CaptionEvaluator
│   └── inference.py         # GreedyDecoder, BeamSearchDecoder
│
├── callbacks/               # Custom callbacks
│   ├── __init__.py
│   └── metrics.py           # MetricsCallback, ProgressCallback
│
└── utils/                   # Utilities
    ├── __init__.py
    ├── io.py                # File I/O (save_json, load_json, ensure_dir)
    ├── logging.py           # Logging setup
    ├── mixed_precision.py   # GPU and mixed precision configuration
    ├── experiment_tracking.py # Experiment tracking and reproducibility
    └── model_versioning.py  # Model versioning and artifact management
```

## Quick Start


### 1. Configure Paths

Edit `config.yaml` to set your data paths:

```yaml
images_path: "/path/to/flickr8k/Images/"
captions_file: "/path/to/flickr8k/captions.txt"
glove_path: "/path/to/glove.6B.200d.txt"
```

### 2. Run Training

```bash
# From repository root
python -m training.train --config training/config.yaml
```

### 3. Artifacts

Models and tokenizer files are saved to `shared/artifacts/`:
- `image_caption_model_final.keras` - Trained model
- `checkpoint.keras` - Best checkpoint
- `wordtoix.json` - Word to index mapping
- `ixtoword.json` - Index to word mapping
- `features.npy` - Extracted image features
- `run_summary.json` - Experiment run summary
- `metrics.csv` - Epoch-wise metrics
- `config_snapshot.json` - Configuration snapshot
- `logs/` - TensorBoard logs

## Experiment Tracking

The training pipeline automatically tracks experiments with:

### Run Summary (`run_summary.json`)
```json
{
  "metadata": {
    "run_id": "20240115_143022_abc12345",
    "run_name": "experiment_001",
    "git_info": {"commit": "abc123...", "branch": "main"},
    "environment": {"tensorflow_version": "2.15.0"}
  },
  "config": {...},
  "final_metrics": {"epoch": 20, "train_loss": 2.1, "val_loss": 2.3},
  "bleu_scores": {"test_bleu4": 0.25}
}
```

### Epoch Metrics (`metrics.csv`)
```csv
epoch,train_loss,val_loss,learning_rate,epoch_time_seconds
1,4.5,4.2,0.0001,120.5
2,3.8,3.6,0.0001,118.2
...
```

### Usage
```python
from training.utils import ExperimentTracker, set_global_seed

# Set seed for reproducibility
set_global_seed(42)

# Create tracker
tracker = ExperimentTracker(
    output_dir="experiments/run_001",
    run_name="my_experiment",
    seed=42,
)

# Log config
tracker.log_config(config_dict)

# Log metrics during training
tracker.log_metrics(epoch=1, train_loss=2.5, val_loss=2.8)

# Log BLEU scores
tracker.log_bleu_scores(bleu1=0.65, bleu2=0.45, bleu3=0.35, bleu4=0.25)

# Finalize and save summary
tracker.finalize()
```

## Model Versioning

Register and manage model versions:

```python
from training.utils.model_versioning import ModelVersionManager

manager = ModelVersionManager("models/")

# Register a new version
manager.register_version(
    version="1.0.0",
    model_path="artifacts/model.keras",
    vocab_dir="artifacts/",
    run_id="run_001",
    metrics={"bleu4": 0.25, "val_loss": 2.3},
    description="Initial release",
)

# Get latest version
latest = manager.get_latest_version()

# Compare versions
comparison = manager.compare_versions("1.0.0", "1.1.0")
```

## Key Design Patterns

### 1. **Dataclass Configuration**
Type-safe configuration with validation:
```python
from training.configs import load_config
config = load_config("config.yaml")
print(config.model.feature_dim)  # 1536
```

### 2. **Registry Pattern (Feature Extractors)**
Extensible feature extractor registration:
```python
from training.features import get_feature_extractor
extractor = get_feature_extractor("EfficientNetB3", image_size=(300, 300))
```

### 3. **Abstract Base Classes**
Clean interfaces for extensibility:
```python
from training.features.base import BaseFeatureExtractor

class MyCustomExtractor(BaseFeatureExtractor):
    def build_model(self): ...
    def extract_features(self, ...): ...
```

### 4. **Factory Pattern (Callbacks)**
Standardized callback creation:
```python
from training.trainers.trainer import CallbackFactory
checkpoint = CallbackFactory.model_checkpoint("model.keras")
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `feature_extractor` | `EfficientNetB3` | CNN backbone |
| `feature_dim` | `1536` | Feature dimension |
| `embedding_dim` | `200` | GloVe embedding size |
| `hidden_dim` | `256` | LSTM/attention dimension |
| `batch_size` | `64` | Training batch size |
| `epochs` | `20` | Maximum epochs |
| `learning_rate` | `0.0001` | Initial learning rate |
| `use_mixed_precision` | `true` | Enable float16 training |

## Extending the Pipeline

### Add a New Feature Extractor

```python
# training/features/my_extractor.py
from training.features.base import BaseFeatureExtractor

class MyExtractor(BaseFeatureExtractor):
    FEATURE_DIM = 2048
    NAME = "MyModel"
    
    def build_model(self):
        # Build your model
        pass
    
    def extract_features(self, image_keys, images_path, verbose=1):
        # Extract features
        pass

# Register it
from training.features.registry import FeatureExtractorRegistry
registry = FeatureExtractorRegistry()
registry.register("MyModel", MyExtractor)
```

### Custom Training Callback

```python
from training.callbacks.metrics import MetricsCallback

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Custom logic
        pass
```

## Running on Kaggle

Yes! The training pipeline works on Kaggle notebooks. Here's how:

### Setup (One-time)

1. In a Kaggle notebook, add these required datasets:
   - `adityajn105/flickr8k` (Flickr8k images & captions)
   - `incorpes/glove6b200d` (GloVe embeddings)

2. Install dependencies:
```python
!pip install -q pyyaml tqdm nltk
```

### Running Training

```python
# Clone repo (or upload to notebook)
!git clone https://github.com/DemonicAK/image-captioning-production.git
%cd image-captioning-production

# Setup and run
from training.kaggle_utils import setup_kaggle_training, create_kaggle_config
from training.train import TrainingPipeline

# Create Kaggle config (auto-detects dataset paths)
create_kaggle_config()

# Run training
pipeline = TrainingPipeline("training/config.kaggle.yaml")
pipeline.run()
```

### Download Outputs

After training, download from `/kaggle/working/`:
- `image_caption_model_final.keras` - Trained model
- `wordtoix.json`, `ixtoword.json` - Tokenizer files
- `checkpoint.keras` - Best model checkpoint
- `logs/` - TensorBoard logs

### Kaggle Notebook Template

A complete Kaggle notebook example:

```python
# Cell 1: Install dependencies
!pip install -q pyyaml tqdm nltk

# Cell 2: Clone repo and setup
!git clone https://github.com/DemonicAK/image-captioning-production.git
%cd image-captioning-production

# Cell 3: Create Kaggle config
from training.kaggle_utils import create_kaggle_config, setup_kaggle_training
create_kaggle_config()
env_info = setup_kaggle_training(verbose=True)

# Cell 4: Run training
from training.train import TrainingPipeline
pipeline = TrainingPipeline("training/config.kaggle.yaml")
pipeline.run()

# Cell 5: Check outputs
import os
print("\nGenerated files:")
for f in sorted(os.listdir("/kaggle/working")):
    size = os.path.getsize(f"/kaggle/working/{f}") / 1e6
    print(f"  {f}: {size:.1f} MB")
```

### Important Notes for Kaggle

- **Enable GPU** in notebook settings (required for training)
- **Time limit**: Standard Kaggle notebooks have 9-hour runtime. Training 20 epochs takes ~4-6 hours
- **Memory**: Set `batch_size: 32` if running out of memory
- **Datasets**: Make sure both Flickr8k and GloVe datasets are added as input sources

### Customizing for Kaggle

Edit `config.kaggle.yaml` if using different datasets:

```yaml
# Example: Different Flickr dataset
images_path: "/kaggle/input/flickr-image-dataset/flickr30k_images/flickr30k_images/"
captions_file: "/kaggle/input/flickr-image-dataset/results.csv"
```

## Requirements

- Python >= 3.9
- TensorFlow >= 2.10
- NumPy
- PyYAML
- tqdm
- NLTK (for BLEU evaluation)
