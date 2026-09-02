# Benchmarking & Performance

Systematic benchmarking suite for Open ML Foundry.

## Directory Structure

```
benchmarks/
├── run_benchmarks.py      # Main benchmarking script
├── models/
│   └── model_configs.yaml # Reference model configurations
├── datasets/              # Small reference datasets (empty for now)
│   └── README.md
├── results/               # Benchmark results (generated)
│   └── .gitkeep
└── README.md              # This file
```

## Usage

### Run Default Benchmark

```bash
python benchmarks/run_benchmarks.py
```

Benchmarks ResNet50 on CPU with:
- 100 inference runs
- 5 training epochs
- Saves results to `benchmarks/results/benchmark_results.json`

### Custom Benchmarks

```bash
# Specific model
python benchmarks/run_benchmarks.py --model yolov5s

# Use GPU
python benchmarks/run_benchmarks.py --model resnet50 --device cuda

# More inference runs
python benchmarks/run_benchmarks.py --inference-runs 1000

# Longer training
python benchmarks/run_benchmarks.py --training-epochs 20

# Custom output
python benchmarks/run_benchmarks.py --output results/my_benchmark.json
```

### Or Use Convenience Script

```bash
./scripts/benchmark.sh
./scripts/benchmark.sh --model yolov5s --gpu
./scripts/benchmark.sh --gpu --output results.json
```

## Supported Models

### Image Classification
- `resnet18` — Lightweight ResNet
- `resnet50` — Standard ResNet
- `resnet101` — Larger ResNet
- `mobilenet` — Mobile-optimized
- `vgg16` — Classic VGG

### Object Detection
- `yolov5s` — YOLOv5 Small
- `yolov5m` — YOLOv5 Medium
- `faster_rcnn` — Faster R-CNN

### NLP
- `bert_base` — BERT Base (110M)
- `bert_large` — BERT Large (340M)
- `distilbert` — DistilBERT

### Vision-Language
- `clip_vit_base` — CLIP ViT-B

See `models/model_configs.yaml` for detailed specs.

## Benchmark Metrics

### Inference Performance
- **Latency (ms):** Time for single inference
  - Mean, median, std dev, min, max
- **Throughput (fps):** Images/sec inference speed
- **Memory (MB):** Peak memory usage during inference

### Training Performance
- **Time per epoch (sec):** Training speed
- **Throughput (samples/sec):** Data processing rate
- **Memory (MB):** Peak memory during training

### Model Characteristics
- **Parameters (millions):** Total model parameters
- **Size (MB):** Model file size

## Benchmark Profiles

### Quick
```bash
python benchmarks/run_benchmarks.py
# 10 inference runs, 1 epoch training
# ~30 seconds
```

### Standard
```bash
python benchmarks/run_benchmarks.py --inference-runs 100 --training-epochs 5
# 100 inference runs, 5 epochs
# ~5 minutes
```

### Thorough
```bash
python benchmarks/run_benchmarks.py --inference-runs 1000 --training-epochs 20
# 1000 inference runs, 20 epochs
# ~30 minutes
```

## Hardware Profiles

### CPU
```bash
python benchmarks/run_benchmarks.py --device cpu
```

### NVIDIA GPU (CUDA)
```bash
python benchmarks/run_benchmarks.py --device cuda
```

### Apple Silicon (Metal)
```bash
python benchmarks/run_benchmarks.py --device mps
```

## Results Format

Results saved as JSON:

```json
{
  "model": "resnet50",
  "device": "cpu",
  "timestamp": "2024-01-01 10:00:00",
  "inference": {
    "latency_ms_mean": 45.23,
    "latency_ms_median": 44.80,
    "latency_ms_std": 2.15,
    "latency_ms_min": 41.50,
    "latency_ms_max": 52.30,
    "throughput_fps": 22.1,
    "memory_mb": 456.2
  },
  "training": {
    "total_time_sec": 125.45,
    "avg_epoch_time_sec": 25.09,
    "samples_per_sec": 39.7,
    "memory_mb": 1024.5
  },
  "model_size": {
    "parameters_millions": 25.6,
    "size_mb": 97.3
  },
  "summary": {
    "inference_latency_ms": 45.23,
    "inference_throughput_fps": 22.1,
    "training_samples_per_sec": 39.7
  }
}
```

## Optimization Targets

### Edge Devices (Raspberry Pi, Jetson)
- Max latency: 100ms
- Min throughput: 10 fps
- Max memory: 500MB

### Mobile (iOS, Android)
- Max latency: 50ms
- Min throughput: 20 fps
- Max memory: 300MB

### Server
- Max latency: 200ms
- Min throughput: 5 fps
- Max memory: 2GB

### Browser (WebGL/WASM)
- Max latency: 500ms
- Min throughput: 2 fps
- Max memory: 100MB

## Regression Testing

Compare against baselines:

```bash
# Save baseline
python benchmarks/run_benchmarks.py --output results/baseline.json

# Run test
python benchmarks/run_benchmarks.py --output results/current.json

# Compare
python benchmarks/compare_results.py results/baseline.json results/current.json
```

Thresholds (from `model_configs.yaml`):
- Latency regression: 10% (alert if worse)
- Throughput regression: 10% (alert if worse)
- Memory regression: 20% (alert if worse)

## Tips for Accurate Benchmarking

1. **Close other applications** — Minimize interference
2. **Multiple runs** — Use 100+ runs for stable averages
3. **Warm up** — First run often slower (JIT compilation)
4. **Consistent environment** — Same OS, drivers, kernel
5. **Report both runs** — Show variation (mean ± std)
6. **Use profiling** — Identify bottlenecks

## Adding New Benchmarks

### Add Model to Config

Edit `models/model_configs.yaml`:

```yaml
my_model:
  name: "My Model"
  framework: "pytorch"
  task: "classification"
  input_shape: [1, 3, 224, 224]
  parameters_millions: 50.0
  size_mb: 200
  recommended_batch_size: 32
  baseline_latency_ms: 75
```

### Add Custom Benchmark

Edit `run_benchmarks.py`:

```python
def benchmark_custom(self) -> Dict[str, float]:
    """Custom benchmark."""
    # Implementation
    return {"custom_metric": 42.0}
```

## Troubleshooting

**Benchmark too slow:**
- Use `--inference-runs 10` for quick test
- Use smaller model: `--model mobilenet`
- Use CPU only: `--device cpu`

**Out of memory:**
- Use smaller model
- Reduce batch size in code
- Close other applications

**GPU not detected:**
- Check CUDA installation: `nvidia-smi`
- Verify PyTorch CUDA support
- Use CPU: `--device cpu`

**Results vary between runs:**
- Normal variation expected
- Use more runs: `--inference-runs 1000`
- Report mean ± std dev

## References

- [PyTorch Benchmarking](https://pytorch.org/tutorials/recipes/recipes/benchmark.html)
- [MLCommons Benchmarks](https://mlcommons.org/)
- [SPEC CPU](https://www.spec.org/cpu/)

## Next Steps

- [ ] Add real datasets to `datasets/`
- [ ] Implement regression testing
- [ ] Add hardware detection
- [ ] Create comparison visualizations
- [ ] Setup automated benchmarking in CI/CD
