# Getting Started

Get up and running in 5 minutes.

## Prerequisites

- Python 3.8+
- Docker & Docker Compose (optional but recommended)
- 2GB disk space

## Installation

### Option 1: pip (Recommended)

```bash
pip install sentinel-finetune
```

### Option 2: Docker Compose

```bash
git clone https://github.com/agentic-inquisit/localml-finetune.git
cd sentinel-finetune
docker-compose up
```

## Your First Fine-Tune

### 1. Import a Model

```bash
sentinel model import --model resnet50
```

Available models:
- `resnet18`, `resnet50` (image classification)
- `yolov5s`, `yolov5m` (object detection)
- `bert-base` (text)
- `clip-vit-base` (vision-language)

### 2. Prepare Your Dataset

Place images in folders by class:

```
my-data/
├── cat/
│   ├── img1.jpg
│   ├── img2.jpg
└── dog/
    ├── img1.jpg
    └── img2.jpg
```

Then prepare:

```bash
sentinel dataset prepare --path ./my-data
```

### 3. Start Training

```bash
sentinel train start \
  --model resnet50 \
  --dataset my-data \
  --epochs 10 \
  --batch-size 32
```

### 4. Monitor Progress

```bash
# View training status
sentinel train status

# List all training jobs
sentinel train list

# Watch live dashboard
sentinel dashboard --job-id <job-id>
```

### 5. Export Your Model

```bash
sentinel model export --job-id <job-id> --format onnx
```

## Next Steps

- Explore [CLI Guide](cli-guide.md) for all commands
- Check out [Tutorials](tutorials/) for specific use cases
- Read [Architecture](architecture.md) to understand how it works
