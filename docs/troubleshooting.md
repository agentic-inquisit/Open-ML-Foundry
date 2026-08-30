# Troubleshooting Guide

## Installation Issues

### "pip: command not found"

**Cause:** Python pip not installed or not in PATH.

**Solution:**
```bash
# Install pip
python3 -m ensurepip --upgrade

# Verify
python3 -m pip --version
```

### "Python 3.8+ required"

**Cause:** Python version too old.

**Solution:**
```bash
# Check version
python --version

# Install Python 3.9+ from python.org or:
brew install python@3.11  # Mac
apt install python3.11    # Linux
```

### Import errors after installation

**Cause:** Dependencies not installed.

**Solution:**
```bash
# Reinstall with dependencies
pip install --upgrade --force-reinstall sentinel-finetune
```

---

## Runtime Issues

### "sentinel: command not found"

**Cause:** Not in PATH.

**Solution:**
```bash
# Add to PATH
export PATH="$HOME/.local/bin:$PATH"

# Or reinstall in user space
pip install --user sentinel-finetune
```

### Import error: "No module named 'torch'"

**Cause:** PyTorch not installed.

**Solution:**
```bash
# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## Dataset Issues

### Dataset validation fails

**Cause:** Invalid directory structure or corrupted images.

**Solution:**
```bash
# Check what's wrong
sentinel dataset info --dataset my-data --verbose

# Fix common issues:
# 1. Ensure images are in subdirectories by class
# 2. Remove corrupted files
# 3. Ensure image formats are JPEG/PNG

find ./my-data -name "*.jpg" -o -name "*.png" | wc -l  # Count images
```

### "Dataset too small" warning

**Cause:** Fewer than 50 images per class.

**Solution:**
- Collect more images
- Use data augmentation: `--augmentation true`
- Use transfer learning with pretrained model

### Class imbalance warning

**Cause:** Unequal number of images per class.

**Solution:**
- Add more images to minority class
- Use weighted sampling (default: enabled)
- Collect balanced data

---

## Training Issues

### "CUDA out of memory"

**Cause:** GPU memory exhausted.

**Solution:**
```bash
# Reduce batch size
sentinel train start --batch-size 8

# Reduce image size
sentinel train start --img-size 160

# Use CPU instead
sentinel train start --device cpu

# Close other GPU applications
nvidia-smi  # Check usage
```

### Training loss is NaN

**Cause:** Exploding gradients, learning rate too high.

**Solution:**
```bash
# Reduce learning rate significantly
sentinel train start --learning-rate 0.00001

# Try gradient clipping
sentinel train start --gradient-clip-value 1.0

# Check data for outliers/corrupted values
```

### Training very slow

**Cause:** 
- Using CPU instead of GPU
- Batch size too small
- Model too large

**Solution:**
```bash
# Enable GPU
sentinel train start --device cuda
nvidia-smi  # Verify GPU usage

# Increase batch size (if memory allows)
sentinel train start --batch-size 64

# Use smaller model
sentinel model import --model resnet18

# Use mixed precision (faster)
sentinel train start --mixed-precision fp16
```

### Accuracy not improving

**Cause:**
- Learning rate too low
- Model capacity insufficient
- Data quality issues
- Insufficient training

**Solution:**
```bash
# Increase learning rate
sentinel train start --learning-rate 0.001

# Train longer
sentinel train start --epochs 50

# Use larger model
sentinel model import --model resnet50

# Check data quality
sentinel dataset info --dataset my-data

# Try data augmentation
sentinel train start --augmentation true
```

### Training crashes/OOM randomly

**Cause:** Batch size too large for available memory.

**Solution:**
```bash
# Start with smaller batch size
sentinel train start --batch-size 16

# Monitor GPU memory
watch nvidia-smi

# Use gradient accumulation
sentinel train start --gradient-accumulation-steps 4
```

### Cannot resume training

**Cause:** Checkpoint file corrupted or job ID wrong.

**Solution:**
```bash
# List all jobs
sentinel train list

# Use correct job ID
sentinel train resume --job-id <exact-id>

# Start fresh if checkpoint corrupted
sentinel train start --model resnet50 --dataset my-data
```

---

## Model Issues

### Model import fails

**Cause:**
- Internet connection issue
- Model doesn't exist
- Insufficient disk space

**Solution:**
```bash
# Check internet
ping huggingface.co

# Verify model name
sentinel model list

# Check disk space
df -h

# Manual download
wget <model-url>
sentinel model import --model ./model.pth --source local
```

### Model export fails

**Cause:**
- Training not completed
- Checkpoint corrupted
- Format not supported

**Solution:**
```bash
# Verify training completed
sentinel train status --job-id <job-id>

# List supported formats
sentinel model export --help

# Try different format
sentinel model export --job-id <id> --format pth
```

### Inference accuracy different from training

**Cause:**
- Model not in eval mode
- Input normalization different
- Quantization loss (if quantized)

**Solution:**
```python
# Ensure eval mode
model.eval()

# Match training preprocessing
# If trained on ImageNet:
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

---

## Docker Issues

### "Cannot connect to Docker daemon"

**Cause:** Docker not running.

**Solution:**
```bash
# Start Docker
systemctl start docker     # Linux
open -a Docker             # Mac
# Or use Docker Desktop

# Verify
docker ps
```

### Container exits immediately

**Cause:** Configuration error or missing dependency.

**Solution:**
```bash
# Check logs
docker logs sentinel-api
docker logs sentinel-edge

# Verify config
cat docker-compose.yml

# Rebuild
docker-compose build --no-cache
docker-compose up
```

### Port already in use

**Cause:** Another process using the port.

**Solution:**
```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>

# Or change port in docker-compose.yml
```

---

## Performance & Optimization

### Inference too slow on edge device

**Solution:**
```bash
# Use smaller model
sentinel model import --model mobilenet

# Quantize model
sentinel model export --format tflite --quantization int8

# Reduce input size
# Change model's expected input from 224×224 to 160×160

# Use TFLite runtime (faster)
pip install tflite-runtime  # Instead of tensorflow
```

### High CPU/GPU usage even when idle

**Cause:** Model loaded in memory.

**Solution:**
```bash
# Stop training
sentinel train stop --job-id <id>

# Unload models
sentinel cache clear

# Monitor
top
nvidia-smi
```

---

## Getting More Help

1. **Check docs:** https://sentinel-finetune.readthedocs.io
2. **GitHub Issues:** Report bugs
3. **GitHub Discussions:** Ask questions
4. **FAQ:** Common answers

When reporting issues include:
```bash
# System info
sentinel --version
python --version
nvidia-smi  # If using GPU
uname -a

# Error message
# Full command you ran
# Dataset info
sentinel dataset info --dataset my-data
```
