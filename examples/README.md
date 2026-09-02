# Open ML Foundry Examples

Complete, working examples for different use cases.

## Examples Directory

```
examples/
├── image-classification/      # Image classifier fine-tuning
│   ├── train.py               # Training script
│   ├── config.yaml            # Configuration
│   ├── requirements.txt        # Dependencies
│   └── README.md               # Guide
├── object-detection/          # Object detection (YOLO)
│   └── README.md
├── nlp-fine-tuning/          # Text classification (BERT)
│   └── README.md
├── edge-deployment/          # Deploy to edge devices
│   └── README.md
├── docker-compose-example.yml # Full stack deployment
└── README.md                  # This file
```

## Quick Links

### Image Classification
Fine-tune ResNet50 for custom image recognition.

```bash
cd examples/image-classification
python train.py --data ./data --epochs 10
```

**Learn:** [Image Classification Guide](image-classification/README.md)

### Object Detection
Fine-tune YOLOv5 for object detection.

```bash
cd examples/object-detection
python train.py --data ./data --model yolov5s
```

**Learn:** [Object Detection Guide](object-detection/README.md)

### NLP Fine-Tuning
Fine-tune BERT for text classification.

```bash
cd examples/nlp-fine-tuning
python train.py --data ./data --model bert-base
```

**Learn:** [NLP Fine-Tuning Guide](nlp-fine-tuning/README.md)

### Edge Deployment
Deploy models to Raspberry Pi, Jetson Nano, etc.

```bash
cd examples/edge-deployment
python detect_pi.py --model model.tflite --camera /dev/video0
```

**Learn:** [Edge Deployment Guide](edge-deployment/README.md)

## Full Stack Deployment

Run complete Sentinel stack with Docker Compose:

```bash
docker-compose -f docker-compose-example.yml up
```

Services:
- API server (port 8000)
- Edge module (port 8001)
- PostgreSQL (port 5432)
- Redis cache (port 6379)
- pgAdmin (port 5050)

Access: http://localhost:8000

## Workflow: From Training to Deployment

### 1. Train Locally

```bash
cd examples/image-classification
python train.py --data ./my-data --epochs 20
```

### 2. Evaluate

```bash
python eval.py --model model.pth --data ./my-data/test
```

### 3. Export

```bash
sentinel model export \
  --job-id train_001 \
  --format tflite \
  --quantization int8
```

### 4. Deploy to Edge

```bash
scp model.tflite pi@raspberrypi:~/
ssh pi@raspberrypi
python detect_pi.py --model model.tflite
```

### 5. Monitor

```bash
# Via LocalML CLI
sentinel train status --job-id train_001

# Via API
curl http://localhost:8000/train/status/train_001

# Via Web UI
open http://localhost:8000/dashboard
```

## Data Format Reference

### Image Classification
```
data/
├── train/
│   ├── class_1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   └── class_2/
├── val/
│   ├── class_1/
│   └── class_2/
```

### Object Detection (YOLO)
```
data/
├── images/
│   ├── train/
│   │   └── image1.jpg
│   └── val/
│       └── image1.jpg
└── labels/
    ├── train/
    │   └── image1.txt  # <class> <x> <y> <w> <h>
    └── val/
        └── image1.txt
```

### NLP Classification
```
data/
├── train.csv  # text, label columns
└── val.csv
```

## Model Formats Supported

- **PyTorch** — `.pth`, `.pt` files
- **ONNX** — `.onnx` files
- **TensorFlow Lite** — `.tflite` files
- **TensorFlow SavedModel** — SavedModel format
- **HuggingFace** — Transformers format

## Performance Targets

### Image Classification
- Accuracy: 85%+ on 1000-image dataset
- Training: 30 min on GPU (10 epochs)
- Inference: <50ms latency

### Object Detection
- mAP: >0.60 on custom dataset
- Training: 2-4 hours on GPU
- Inference: <100ms per image

### NLP
- Accuracy: 90%+ on balanced dataset
- Training: 30 min on GPU (3 epochs)
- Inference: <200ms per text

### Edge Deployment
- Raspberry Pi: 10+ fps
- Jetson Nano: 20+ fps
- Memory: <500MB

## Common Tasks

### Check Model Size
```bash
ls -lh model.pth
```

### Benchmark Model
```bash
python ../benchmarks/run_benchmarks.py --model resnet50
```

### Convert Model
```bash
sentinel model export --job-id train_001 --format onnx
```

### Quantize Model
```bash
sentinel model export --job-id train_001 --format tflite --quantization int8
```

## Troubleshooting

**Training too slow:**
- Use GPU: Check CUDA availability
- Reduce dataset size initially
- Use smaller model (MobileNet)

**Low accuracy:**
- More training data needed (100+ images/class)
- Train longer (20-50 epochs)
- Check data quality
- Adjust learning rate

**Deployment issues:**
- Check model export format
- Verify input shapes match
- Test on small dataset first
- Check memory available

## Next Steps

1. Choose example matching your use case
2. Prepare dataset following format guide
3. Run training script
4. Evaluate and export model
5. Deploy to target environment
6. Monitor performance

## Resources

- **Tutorials:** [docs/tutorials/](../docs/tutorials/)
- **CLI Guide:** [docs/cli-guide.md](../docs/cli-guide.md)
- **API Reference:** [docs/components/api-serving.md](../docs/components/api-serving.md)
- **Benchmarking:** [benchmarks/README.md](../benchmarks/README.md)

## Support

- Questions? See [FAQ](../docs/faq.md)
- Issues? Check [Troubleshooting](../docs/troubleshooting.md)
- Report bugs: GitHub Issues
- Discuss: GitHub Discussions

---

**Happy fine-tuning!** 🚀
