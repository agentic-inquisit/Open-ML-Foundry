# Tutorial: Fine-tune Image Classifier

Learn to fine-tune ResNet for custom image classification.

## Use Case

Classify custom product images: shoes, bags, hats.

## Step 1: Prepare Data

Organize images in folders by class:

```
product-data/
├── shoes/
│   ├── shoe_001.jpg
│   ├── shoe_002.jpg
│   └── ...
├── bags/
│   ├── bag_001.jpg
│   └── ...
└── hats/
    ├── hat_001.jpg
    └── ...
```

**Tips:**
- Use high-quality images (200×200 px minimum)
- Aim for 50+ images per class
- Ensure balanced classes (similar count per class)

## Step 2: Import Base Model

```bash
sentinel model import --model resnet50
```

**Why ResNet50?**
- Fast inference
- Good accuracy-speed tradeoff
- 97MB (reasonable size)
- Well-optimized

## Step 3: Prepare Dataset

```bash
sentinel dataset prepare --path ./product-data --name products
```

**Check dataset stats:**

```bash
sentinel dataset info --dataset products
```

Expected output:
```
Dataset: products
Classes: 3 (shoes, bags, hats)
Total Samples: 450
Training: 360 (80%)
Validation: 45 (10%)
Test: 45 (10%)
Class Distribution:
  - shoes: 150
  - bags: 150
  - hats: 150
```

## Step 4: Start Training

```bash
sentinel train start \
  --model resnet50 \
  --dataset products \
  --epochs 15 \
  --batch-size 32 \
  --learning-rate 0.0001 \
  --name product-classifier
```

**Parameters explained:**
- `--epochs 15` — Train for 15 passes through data
- `--batch-size 32` — Process 32 images per iteration
- `--learning-rate 0.0001` — Conservative LR (pretrained model)
- `--name` — Meaningful job name for tracking

## Step 5: Monitor Training

In another terminal:

```bash
sentinel dashboard --job-id product-classifier
```

Watch real-time metrics:
- Loss should decrease
- Accuracy should increase
- Training speed (images/sec)

**Good signs:**
- Loss: 3.0 → 0.5 (decreasing)
- Accuracy: 33% → 85%+ (increasing)
- No spikes or oscillations

**Bad signs:**
- Loss not decreasing (LR too low or too high)
- Loss NaN (gradient explosion)
- Accuracy flat (model not learning)

## Step 6: Evaluate Results

After training completes:

```bash
sentinel train status --job-id product-classifier
```

Output:
```
Job ID: product-classifier
Model: resnet50
Status: Completed
Final Accuracy: 0.889
Final Loss: 0.312
Training Time: 45 minutes
Best Checkpoint: Epoch 12
```

## Step 7: Export Model

Save for deployment:

```bash
# Export to ONNX (recommended for inference)
sentinel model export --job-id product-classifier --format onnx

# Export to PyTorch
sentinel model export --job-id product-classifier --format pth

# Export to TFLite (mobile)
sentinel model export --job-id product-classifier --format tflite
```

Models saved to `./exports/`

## Troubleshooting

### Training too slow

```bash
# Increase batch size (if GPU memory allows)
sentinel train start --batch-size 64

# Use mixed precision (faster on modern GPUs)
sentinel train start --mixed-precision fp16
```

### Accuracy not improving

```bash
# Check learning rate (might be too high)
sentinel train start --learning-rate 0.00001

# Train longer
sentinel train start --epochs 30

# Check dataset quality
sentinel dataset info --dataset products
```

### Out of memory

```bash
# Reduce batch size
sentinel train start --batch-size 16

# Use smaller model
sentinel model import --model resnet18
```

## Next Steps

- Deploy model to production
- Set up continuous monitoring
- Collect new data and retrain
- Experiment with different architectures
