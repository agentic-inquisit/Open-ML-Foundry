# Frequently Asked Questions

## General

**Q: Is my data sent anywhere?**
A: No. LocalML finetune is completely local. All data stays on your machine. No cloud services, no telemetry, no external API calls.

**Q: Can I use this offline?**
A: Yes, after importing models. Models download on first import, then work offline forever.

**Q: How much storage do I need?**
A: Depends on model + dataset:
- ResNet50: ~100MB
- BERT-base: ~440MB
- Small dataset: ~1GB
- Total: Start with 5GB free space

**Q: Does it support GPU?**
A: Yes. NVIDIA GPU (CUDA), AMD (ROCm), or Apple Silicon (Metal). Auto-detects.

## Installation & Setup

**Q: Why does pip install fail?**
A: Check Python version (need 3.8+):
```bash
python --version
```

If still failing, try:
```bash
pip install --upgrade pip
pip install sentinel-finetune
```

**Q: Can I use on Windows?**
A: Yes. Use Docker Compose (simplest) or Windows Subsystem for Linux (WSL2).

**Q: Can I use on Mac?**
A: Yes. Works on Intel and Apple Silicon. Apple Silicon may be slower without Metal optimization.

## Models

**Q: Can I use my own pretrained model?**
A: Yes:
```bash
sentinel model import --model ./my-model.pth --source local
```

**Q: What model formats are supported?**
A: PyTorch (.pth), HuggingFace, ONNX, TensorFlow SavedModel

**Q: How do I use a model not in the default list?**
A: Download from HuggingFace:
```bash
sentinel model import --model bert-base-chinese
```

## Training

**Q: How long does training take?**
A: Depends on:
- Model size (ResNet18: fast, ResNet50: slower)
- Dataset size (100 images: 5 min, 10k images: hours)
- Hardware (GPU: 10× faster than CPU)

Typical: 15-60 minutes for 1000 images.

**Q: What batch size should I use?**
A: Start with 32. If out of memory, reduce to 16 or 8.

**Q: What learning rate should I use?**
A: For pretrained models: 0.0001 - 0.001 (conservative)
For scratch: 0.001 - 0.01 (more aggressive)

**Q: How many epochs do I need?**
A: Start with 10-20. Watch validation accuracy:
- If increasing → keep training
- If plateauing → stop early

**Q: Can I resume training?**
A: Yes:
```bash
sentinel train resume --job-id <job-id>
```

**Q: Can I stop training and keep the checkpoint?**
A: Yes:
```bash
sentinel train stop --job-id <job-id>
```

Best checkpoint is saved automatically.

## Datasets

**Q: What data formats are supported?**
A: 
- Images: JPEG, PNG
- Text: TXT, CSV
- Annotations: YOLO, COCO, Pascal VOC format

**Q: How many images do I need?**
A: Minimum: 20-50 per class
Recommended: 100+ per class
Better: 1000+ per class

**Q: What if my dataset is imbalanced?**
A: Sentinel handles this automatically with weighted sampling.

**Q: Can I use external datasets (ImageNet, COCO)?**
A: Yes, download and prepare:
```bash
sentinel dataset prepare --path ./imagenet-subset
```

## Performance & Accuracy

**Q: Why is accuracy low?**
A: Common causes:
1. Too few images (get more data)
2. Poor image quality (clean your data)
3. Wrong model (try larger model)
4. Wrong learning rate (try 0.0001)
5. Imbalanced classes (add more minority examples)

**Q: Training is slow. How do I speed up?**
A: Try these:
1. Use GPU: `--device cuda`
2. Increase batch size: `--batch-size 64`
3. Use smaller model: `resnet18` instead of `resnet50`
4. Reduce image resolution
5. Use mixed precision: `--mixed-precision fp16`

**Q: Can I use multiple GPUs?**
A: Not yet in v0.3.0. Planned for v1.0.

**Q: My model overfits (high train, low val accuracy)**
A: Try:
1. Reduce epochs
2. Add regularization: `--weight-decay 0.01`
3. Use data augmentation (default: on)
4. Get more data
5. Use smaller model

## Export & Inference

**Q: What export formats are available?**
A: 
- ONNX (recommended, most compatible)
- PyTorch (.pth)
- TensorFlow Lite (.tflite) (mobile/edge)
- TensorFlow SavedModel

**Q: Which format should I use?**
A: 
- **Desktop inference**: ONNX or PyTorch
- **Web/Browser**: ONNX.js or TFJS
- **Mobile**: TFLite
- **Edge devices**: TFLite or ONNX

**Q: Can I quantize the model?**
A: Yes during export:
```bash
sentinel model export --format tflite --quantization int8
```

Reduces size 4× with minimal accuracy loss.

## Troubleshooting

**Q: "CUDA out of memory"**
A: Reduce batch size:
```bash
sentinel train start --batch-size 8
```

**Q: "Model not found"**
A: Import first:
```bash
sentinel model import --model resnet50
```

**Q: "Dataset validation failed"**
A: Run this to see issues:
```bash
sentinel dataset info --dataset my-data
```

**Q: Training loss is NaN**
A: Learning rate too high. Try:
```bash
sentinel train start --learning-rate 0.00001
```

**Q: Can't export model**
A: Make sure training completed:
```bash
sentinel train status --job-id <job-id>
```

Should show "Completed", not "Running".

## Advanced

**Q: Can I use custom loss functions?**
A: Not in v0.3.0. Planned for future releases.

**Q: Can I do distributed training?**
A: Not in v0.3.0. Multi-GPU support planned for v1.0.

**Q: Can I integrate with Weights & Biases?**
A: Not in v0.3.0. MLflow/WandB integration planned.

**Q: Can I train on my own framework (JAX, TF)?**
A: Not yet. Currently supports PyTorch models.

## Support

**Q: Where do I report bugs?**
A: GitHub Issues: https://github.com/agentic-inquisit/localml-finetune/issues

**Q: How do I contribute?**
A: See CONTRIBUTING.md in repo

**Q: Is there a community forum?**
A: GitHub Discussions: https://github.com/agentic-inquisit/localml-finetune/discussions

**Q: How often is this maintained?**
A: Active development. Updates monthly with bug fixes and features.
