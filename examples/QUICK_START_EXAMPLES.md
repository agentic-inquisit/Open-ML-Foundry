# Quick Start Examples

This directory contains example workflows for using Sentinel CLI and REST API.

## Getting Started with CLI

### Example 1: List Built-in Models (30 seconds)

```bash
cd sentinel-cloud-vision-upd

# Activate venv (if not already activated)
source venv/bin/activate

# List all models
sentinel model list
```

Expected output:
```
📦 Available Models:

  Built-in Models:
    ✓ fasterrcnn    - Object detection (80 COCO classes)
    ✓ cnn           - Custom classifier (JAX)
    ✓ clip          - Image-text embeddings (OpenAI)

  Imported Models:
    (none yet - use 'sentinel model import' to add)
```

---

### Example 2: View Model Details (30 seconds)

```bash
# Get details about built-in CNN model
sentinel model info cnn

# Or get details about object detection
sentinel model info fasterrcnn

# Or get details about CLIP embeddings
sentinel model info clip
```

Example output for `sentinel model info cnn`:
```
📋 Model: Custom 3-layer CNN
   Type: Classification
   Framework: JAX/Flax
   Pretrained: No (trains from scratch)
   Latency: 5-10ms
   Input: Images (224x224 default)
   Output: Class predictions, confidence
```

---

### Example 3: Prepare a Dataset (2 minutes)

#### 3a: Create a Class-Based Dataset (Recommended)

```bash
# Create dataset folder structure with classes
mkdir -p training_data/dogs
mkdir -p training_data/cats

# Add some images to each class
# cp ~/Downloads/dog_photos/*.jpg training_data/dogs/
# cp ~/Downloads/cat_photos/*.jpg training_data/cats/

# Prepare with analysis
sentinel dataset prepare --path ./training_data --report --preview
```

Expected output:
```
📂 Scanning dataset: training_data
   Found: 12 images
   Structure: Class Folders

✓ Dataset prepared:
   Train: 10 images (80%)
   Val:   1 images (10%)
   Test:  1 images (10%)

🏷️ Classes (2):
   • dogs: 8 images (67%)
   • cats: 4 images (33%)

⚠️ Warnings:
   ⚠️ Class imbalance detected (ratio 2.0x). Consider balancing: 4 to 8 images per class

📸 Sample images:
   - dogs/dog1.jpg
   - cats/cat1.jpg
   - dogs/dog2.jpg
```

#### 3b: Get Detailed Analysis

```bash
# Show detailed report with distribution
sentinel dataset info ./training_data

# Export metadata for reproducibility
sentinel dataset info ./training_data --export dataset_metadata.json
```

Output:
```
============================================================
📊 Dataset Report: training_data
============================================================

📈 Summary:
   Total images: 12
   Structure: Class Folders
   Number of classes: 2

🏷️ Class Distribution:
   dogs                    8 images  66.7% ████████████
   cats                    4 images  33.3% ██████

⚠️ Warnings:
   ⚠️ Class imbalance detected (ratio 2.0x). Consider balancing: 4 to 8 images per class

============================================================
```

---

### Example 4: Import a Custom Model (2 minutes)

#### Option A: Import a PyTorch model

```bash
# Assuming you have a trained PyTorch model
sentinel model import --path ./my_model.pth --name my_custom_model
```

Expected output:
```
📦 Importing model: my_custom_model
   Path: ./my_model.pth
   Type: pytorch

✓ Model registered: my_custom_model
  Use in training: sentinel train start --model my_custom_model --dataset <path>
```

#### Option B: Import a HuggingFace model

```bash
# Import a model from HuggingFace Hub
sentinel model import \
  --path openai/clip-vit-base-patch32 \
  --name clip_base \
  --type huggingface
```

#### Option C: Import an ONNX model

```bash
# Import an ONNX model for cross-platform compatibility
sentinel model import \
  --path ./quantized_model.onnx \
  --name mobile_detector \
  --type onnx
```

After import, verify it was added:
```bash
sentinel model list
```

---

### Example 5: Start Training (3-5 minutes)

#### Quick Training with Built-in CNN

```bash
# Most basic usage
sentinel train start --model cnn --dataset ./test_dataset --epochs 5
```

#### Advanced Training with GPU and Custom Hyperparameters

```bash
sentinel train start \
  --model cnn \
  --dataset ./test_dataset \
  --epochs 20 \
  --batch-size 16 \
  --lr 0.0005 \
  --gpu
```

#### Training with Custom Model

```bash
# After importing a model, use it for training
sentinel train start \
  --model my_custom_model \
  --dataset ./test_dataset \
  --epochs 10 \
  --gpu
```

#### Training with Live Metrics (Coming in Phase 3)

```bash
# This feature will be available in the live dashboard phase
sentinel train start \
  --model cnn \
  --dataset ./test_dataset \
  --epochs 10 \
  --live
```

---

## Using REST API (Alternative to CLI)

### Setup: Start the servers

```bash
# Terminal 1: Activate venv and start servers
./start.sh
```

### Example 1: Object Detection

```bash
# Detect objects in an image using pre-trained FasterRCNN
curl -X POST http://localhost:8001/detect \
  -F "image=@test_image.jpg"
```

Response:
```json
{
  "detections": [
    {
      "class": "dog",
      "confidence": 0.94,
      "bbox": [10, 20, 150, 180]
    },
    {
      "class": "person",
      "confidence": 0.89,
      "bbox": [160, 10, 300, 220]
    }
  ],
  "processing_time_ms": 12
}
```

### Example 2: Fine-tune via REST API

```bash
# Fine-tune the built-in CNN on your images
curl -X POST http://localhost:8001/finetune \
  -F "dataset=@image1.jpg" \
  -F "dataset=@image2.jpg" \
  -F "dataset=@image3.jpg" \
  -F "target_object=my_object" \
  -F "epochs=5" \
  -F "num_classes=2"
```

Response:
```json
{
  "model_id": "my_object_v1.0",
  "status": "completed",
  "final_accuracy": 0.92,
  "final_loss": 0.23,
  "download_url": "http://localhost:8001/download-model/my_object_v1.0",
  "training_time_seconds": 120
}
```

### Example 3: Export Model for Mobile

```bash
# Export a trained model to TFLite
python -c "
from edge.optimized_inference import OptimizedVisionInference
engine = OptimizedVisionInference()
engine.export_to_tflite('my_model.tflite')
engine.export_to_onnx('my_model.onnx')
print('Models exported successfully')
"
```

This creates:
- `my_model.tflite` (3-10MB) - For iOS/Android
- `my_model.onnx` (5-15MB) - For cross-platform deployment

---

## Complete Workflow: Import, Prepare, Train, Deploy

```bash
# 1. Import a custom model
sentinel model import --path ./my_model.pth --name custom_v1

# 2. Prepare dataset
sentinel dataset prepare --path ./data/train --preview

# 3. Start training
sentinel train start \
  --model custom_v1 \
  --dataset ./data/train \
  --epochs 10 \
  --gpu

# 4. Once training completes, export for mobile
python -c "
from edge.optimized_inference import OptimizedVisionInference
engine = OptimizedVisionInference()
engine.export_to_tflite('custom_v1_mobile.tflite')
"

# 5. Deploy to device (example for Android/iOS)
# Use the .tflite file with TensorFlow Lite interpreter
```

---

## Troubleshooting

### CLI not found: `sentinel: command not found`

**Solution 1: Activate virtual environment**
```bash
source venv/bin/activate
```

**Solution 2: Use wrapper script**
```bash
./sentinel.sh model list
./sentinel.sh dataset prepare --path ./images
```

**Solution 3: Install as editable package**
```bash
pip install -e .
sentinel model list
```

---

### Dataset preparation shows 0 images

**Causes & solutions:**
- Wrong path: `ls ./my_images` should show files
- Wrong format: Supported formats are `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`
- Subdirectories: Command recursively searches subdirectories

**Fix:**
```bash
# Check what's in the folder
ls -la ./my_images

# Flatten if needed
find ./my_images -name "*.jpg" -type f | wc -l

# Prepare and preview
sentinel dataset prepare --path ./my_images --preview
```

---

### Training is slow (no GPU)

**Check:**
```bash
# On Linux
nvidia-smi

# On macOS
system_profiler SPNVMeController

# On Windows
Get-ItemProperty 'HKLM:\System\CurrentControlSet\Services\nvlddmkm'
```

**Use CPU optimizations:**
```bash
# JAX will auto-detect GPU, but you can force CPU:
sentinel train start --model cnn --dataset ./images  # Uses GPU if available

# Or use reduced batch size for memory:
sentinel train start --model cnn --dataset ./images --batch-size 8
```

---

## Next Steps

1. **See [CLI_GUIDE.md](../docs/CLI_GUIDE.md)** for complete command reference
2. **See [MODELS_AND_SAMPLES.md](../docs/MODELS_AND_SAMPLES.md)** for in-depth REST API examples
3. **See [README.md](../README.md)** for architecture and features overview

---

## File Structure for This Example

```
sentinel-cloud-vision-upd/
├── examples/
│   ├── QUICK_START_EXAMPLES.md    ← You are here
│   ├── sample_images/              (optional: test images)
│   └── sample_configs/             (optional: training configs)
├── docs/
│   ├── CLI_GUIDE.md               ← Full CLI command reference
│   └── MODELS_AND_SAMPLES.md      ← REST API examples
└── sentinel/
    └── cli/
        ├── main.py                ← Entry point
        └── commands.py            ← Command implementations
```
