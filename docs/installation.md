# Installation Guide

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8 | 3.9+ |
| RAM | 4GB | 8GB+ |
| Disk | 2GB | 10GB+ |
| GPU | Optional | NVIDIA (CUDA 11.8+) |

## Installation Methods

### 1. pip (Recommended for Users)

```bash
pip install sentinel-finetune
```

Verify installation:

```bash
sentinel --version
```

### 2. Docker Compose (Recommended for Production)

Clone and run:

```bash
git clone https://github.com/agentic-inquisit/localml-finetune.git
cd sentinel-finetune
docker-compose up
```

Access API at `http://localhost:8000`

### 3. From Source (for Development)

```bash
git clone https://github.com/agentic-inquisit/localml-finetune.git
cd sentinel-finetune
pip install -e ".[dev]"
pre-commit install
```

## Verification

### Check Installation

```bash
# Show version
sentinel --version

# Show help
sentinel --help

# Test model import
sentinel model list
```

### Test Fine-Tuning

```bash
# Download small test dataset
wget https://example.com/tiny-dataset.zip
unzip tiny-dataset.zip

# Quick fine-tune
sentinel train start \
  --model resnet18 \
  --dataset ./tiny-dataset \
  --epochs 1 \
  --batch-size 8
```

## GPU Support

### NVIDIA GPU (CUDA)

```bash
# Install CUDA 11.8
# https://developer.nvidia.com/cuda-11-8-0-download-archive

# Verify
nvidia-smi
```

Sentinel auto-detects CUDA. To force CPU:

```bash
sentinel train start --device cpu
```

### AMD GPU (ROCm)

```bash
# Install ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

### Apple Silicon (Metal)

```bash
# PyTorch with Metal acceleration
pip install --upgrade torch torchvision torchaudio
```

## Troubleshooting

### "Command not found: sentinel"

Install to user path:

```bash
pip install --user sentinel-finetune
export PATH="$HOME/.local/bin:$PATH"
```

### Out of Memory

Reduce batch size:

```bash
sentinel train start --batch-size 8
```

Or use gradient accumulation:

```bash
sentinel train start --gradient-accumulation-steps 4
```

### Slow Training

Enable mixed precision:

```bash
sentinel train start --mixed-precision fp16
```

See [Troubleshooting](troubleshooting.md) for more.
