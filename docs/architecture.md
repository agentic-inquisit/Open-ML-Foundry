# System Architecture

## Overview

Open ML Foundry is a modular, local-first fine-tuning framework composed of four core layers:

```
┌──────────────────────────────────────┐
│         User Interface (CLI)          │  ← Command-line interface
├──────────────────────────────────────┤
│    Core Services Layer                │
│  ┌──────────────────────────────────┐ │
│  │  Model Registry  Dataset Browser  │ │  ← Data & model management
│  │  Job Tracker     Dashboard        │ │
│  └──────────────────────────────────┘ │
├──────────────────────────────────────┤
│     Training Engine (JAX/PyTorch)     │  ← ML inference & fine-tuning
├──────────────────────────────────────┤
│   Storage & Persistence (SQLite/FS)   │  ← Local data storage
└──────────────────────────────────────┘
```

## Core Components

### 1. CLI Module (`sentinel/cli/`)

**Purpose:** User-facing command interface

**Files:**
- `main.py` — Entry point, Click framework setup
- `commands.py` — All command implementations
- `utils.py` — Helper functions

**Responsibilities:**
- Parse user commands
- Validate input parameters
- Call service layer
- Format and display output
- Handle errors gracefully

### 2. Model Registry

**Purpose:** Manage imported and trained models

**Location:** `~/.sentinel/models/`

**Structure:**
```
models/
├── resnet50/
│   ├── model.pth
│   ├── metadata.json
│   └── config.yaml
├── bert-base/
│   ├── model_*.pth (checkpoints)
│   ├── metadata.json
│   └── tokenizer.json
└── trained_models/
    ├── train_20240101_001/
    │   └── best_model.pth
    └── train_20240101_002/
        └── best_model.pth
```

**Metadata Example:**
```json
{
  "name": "resnet50",
  "framework": "pytorch",
  "task": "image_classification",
  "input_shape": [1, 3, 224, 224],
  "num_classes": 1000,
  "parameters": 25600000,
  "size_mb": 97,
  "downloaded": true,
  "source": "huggingface"
}
```

### 3. Dataset Browser (`sentinel/cli/dataset_browser.py`)

**Purpose:** Automatic dataset structure detection and validation

**Supported Formats:**
- Image classification: `class/image.jpg`
- Object detection: `image.jpg` + `image.xml/json`
- Text files: `text.txt` with optional labels
- CSV: `data.csv` with columns

**Detection Algorithm:**
1. Scan directory recursively
2. Identify file types (images, text, JSON)
3. Infer structure (classification, detection, regression)
4. Check class distribution
5. Generate warnings for imbalance/small samples

**Output:**
```json
{
  "name": "cifar-custom",
  "type": "image_classification",
  "num_samples": 1500,
  "classes": ["cat", "dog"],
  "distribution": {"cat": 750, "dog": 750},
  "warnings": []
}
```

### 4. Job Tracker (`sentinel/cli/job_tracker.py`)

**Purpose:** Manage training job lifecycle and state

**Database:** SQLite at `~/.sentinel/jobs.db`

**Job Lifecycle:**
```
Created → Queued → Running → Completed/Failed
          ↓              ↓
      Validation    Paused/Resumed
```

**Tracked Metrics:**
- Training loss, validation loss
- Accuracy, precision, recall
- GPU/CPU usage, memory
- Training speed (samples/sec)
- Time elapsed, ETA

**Persistence:**
```python
{
  "job_id": "train_20240101_001",
  "model": "resnet50",
  "dataset": "cifar-custom",
  "status": "running",
  "created_at": "2024-01-01T10:00:00Z",
  "metrics": {
    "current_epoch": 5,
    "total_epochs": 20,
    "loss": 0.315,
    "accuracy": 0.845,
    "gpu_memory": 4096,
    "time_elapsed": 8100
  }
}
```

### 5. Dashboard (`sentinel/cli/dashboard.py`)

**Purpose:** Real-time training visualization in terminal

**Updates:** Every 1 second

**Display Elements:**
- Loss curve (ASCII plot)
- Metrics table (accuracy, loss, F1)
- Progress bar (epoch/iteration)
- Hardware usage (GPU %, memory)
- Training speed (samples/sec)
- ETA countdown

## Data Flow

### Training Workflow

```
User Input (CLI)
    ↓
Command Validation
    ↓
Model Registry Lookup → Load pretrained model
    ↓
Dataset Browser → Validate & prepare data
    ↓
Job Tracker → Create job record
    ↓
Training Engine → Run fine-tuning (JAX/PyTorch)
    ↓
Metrics Collector → Record loss/accuracy each epoch
    ↓
Dashboard → Display real-time progress
    ↓
Job Tracker → Update final metrics & status
    ↓
Model Export → Save trained weights
```

### Model Import Workflow

```
User: sentinel model import --model resnet50
    ↓
CLI → HuggingFace/PyTorch API
    ↓
Download pretrained weights
    ↓
Model Registry → Store locally
    ↓
Metadata → Update registry index
    ↓
Display: "Model resnet50 imported ✓"
```

## Storage Layout

```
~/.sentinel/
├── models/                          # Model registry
│   ├── resnet50/
│   ├── yolov5s/
│   └── bert-base/
├── datasets/                        # Dataset cache
│   ├── cifar-custom/
│   └── imagenet-mini/
├── jobs.db                          # Job tracking (SQLite)
├── config.yaml                      # User settings
└── logs/
    ├── job_train_20240101_001.log
    └── job_train_20240101_002.log
```

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| CLI | Click | Command parsing |
| ML Frameworks | PyTorch, JAX/Flax | Fine-tuning engine |
| Model Loading | HuggingFace | Pretrained models |
| Data Processing | PyArrow, Pandas | Dataset handling |
| Storage | SQLite, Filesystem | Persistence |
| Visualization | Rich | Terminal UI |

## Design Principles

1. **Local-First** — All data stays on user's machine
2. **Privacy-Preserving** — No telemetry or remote API calls
3. **Modular** — Each component has single responsibility
4. **Extensible** — Plugins for new models/datasets
5. **Type-Safe** — Type hints throughout
6. **Well-Tested** — 30+ test cases covering critical paths

## Future Architecture Changes

Planned for v1.0:

- **Multi-GPU Training** — Distributed data parallel
- **Model Hub** — Community model sharing
- **Experiment Tracking** — WandB/MLflow integration
- **AutoML** — Hyperparameter optimization
- **Web Dashboard** — Browser-based UI (FastAPI backend)
