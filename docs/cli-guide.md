# CLI Reference Guide

All commands available in sentinel-finetune CLI.

## Model Commands

### model import

Import a pretrained model to local registry.

```bash
sentinel model import --model <name> [--source huggingface|pytorch|local]
```

**Options:**
- `--model` — Model name (resnet50, yolov5s, bert-base, clip-vit-base)
- `--source` — Where to download from (default: huggingface)
- `--path` — Local path if using local source
- `--force` — Overwrite if exists

**Examples:**
```bash
sentinel model import --model resnet50
sentinel model import --model bert-base --source huggingface
sentinel model import --model ./my-model.pth --source local
```

### model list

List available models in registry.

```bash
sentinel model list [--format table|json]
```

**Output:**
```
Name        | Framework | Task              | Size  | Downloaded
------------|-----------|-------------------|-------|------------
resnet50    | PyTorch   | classification    | 97MB  | ✓
yolov5s     | PyTorch   | object-detection  | 14MB  | ✓
bert-base   | PyTorch   | nlp               | 440MB | ✗
```

### model info

Show details of a model.

```bash
sentinel model info --model <name>
```

**Output:**
```
Model: resnet50
Framework: PyTorch
Task: Image Classification
Input Shape: [1, 3, 224, 224]
Output: 1000 classes
Parameters: 25.6M
Downloaded: Yes
Location: ~/.sentinel/models/resnet50/model.pth
```

---

## Dataset Commands

### dataset prepare

Scan and prepare dataset for training.

```bash
sentinel dataset prepare --path <dir> [--name <name>]
```

**Options:**
- `--path` — Directory containing data
- `--name` — Dataset name (default: directory name)
- `--split` — Train/val/test split (default: 80/10/10)
- `--augmentation` — Enable augmentation (true/false)

**Example:**
```bash
sentinel dataset prepare --path ./my-data --name cifar-custom
```

### dataset info

Show dataset statistics.

```bash
sentinel dataset info --dataset <name>
```

**Output:**
```
Dataset: cifar-custom
Classes: 2 (cat, dog)
Total Samples: 1500
Training: 1200 (80%)
Validation: 150 (10%)
Test: 150 (10%)
Class Distribution:
  - cat: 750 (50%)
  - dog: 750 (50%)
```

---

## Training Commands

### train start

Start a new training job.

```bash
sentinel train start \
  --model <model> \
  --dataset <dataset> \
  --epochs <num> \
  [options]
```

**Required:**
- `--model` — Model name
- `--dataset` — Dataset name
- `--epochs` — Number of epochs

**Optional:**
- `--batch-size` — Batch size (default: 32)
- `--learning-rate` — Learning rate (default: 0.001)
- `--device` — cpu/cuda/mps (default: auto-detect)
- `--mixed-precision` — fp16/bf16/none (default: none)
- `--name` — Job name (auto-generated if not provided)

**Example:**
```bash
sentinel train start \
  --model resnet50 \
  --dataset cifar-custom \
  --epochs 20 \
  --batch-size 64 \
  --learning-rate 0.0001 \
  --name my-classifier
```

### train status

Check status of training job.

```bash
sentinel train status --job-id <id>
```

**Output:**
```
Job ID: train_20240101_001
Model: resnet50
Status: Running
Progress: Epoch 5/20 (25%)
Accuracy: 0.845
Loss: 0.315
Time Elapsed: 2h 15m
ETA: 6h 45m
```

### train list

List all training jobs.

```bash
sentinel train list [--format table|json] [--status active|completed|failed]
```

### train stop

Stop a running training job.

```bash
sentinel train stop --job-id <id>
```

### train resume

Resume a paused training job.

```bash
sentinel train resume --job-id <id>
```

---

## Dashboard

### dashboard

Launch real-time training dashboard.

```bash
sentinel dashboard [--job-id <id>] [--port 8000]
```

**Example:**
```bash
sentinel dashboard --job-id train_20240101_001
```

Displays:
- Loss curve (real-time)
- Accuracy metrics
- GPU/CPU usage
- Training speed (samples/sec)
- ETA to completion

---

## Model Export

### model export

Export trained model to different formats.

```bash
sentinel model export --job-id <id> --format <format> [--path <dir>]
```

**Supported Formats:**
- `onnx` — ONNX format (recommended for inference)
- `pth` — PyTorch format
- `tfjs` — TensorFlow.js (browser)
- `tflite` — TensorFlow Lite (mobile)
- `onnx-quantized` — Quantized ONNX

**Example:**
```bash
sentinel model export --job-id train_20240101_001 --format onnx
```

---

## Global Options

Available with all commands:

```bash
sentinel [command] --help              # Show help
sentinel [command] --verbose           # Detailed output
sentinel [command] --quiet             # Minimal output
sentinel [command] --config <path>     # Use config file
```

---

## Configuration File

Create `.sentinel/config.yaml` for defaults:

```yaml
default:
  batch_size: 32
  learning_rate: 0.001
  device: cuda
  mixed_precision: none

models:
  - name: resnet50
    source: huggingface
```

Then use:

```bash
sentinel train start --model resnet50 --dataset cifar-custom
# Uses defaults from config.yaml
```
