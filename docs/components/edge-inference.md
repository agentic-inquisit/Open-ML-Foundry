# Edge Inference Module

Optimized ML inference and fine-tuning for edge devices.

## Overview

Located in `edge/` directory. Provides:
- Lightweight model inference
- XLA compilation & optimization
- On-device fine-tuning
- A/B testing utilities
- Inference caching

## Components

### inference_cache.py — Caching System

Reduces redundant inference calls:

```python
from edge.inference_cache import InferenceCache

cache = InferenceCache(max_size=1000)

# Cache inference results
result = cache.get_or_compute(
    model="resnet50",
    input_hash="abc123",
    compute_fn=lambda: model.predict(image)
)
```

**Benefits:**
- 10× faster repeated inference
- Reduced memory usage
- Automatic eviction of old entries

### benchmark_xla.py — Performance Benchmarking

Measure and optimize inference performance:

```python
from edge.benchmark_xla import BenchmarkXLA

benchmark = BenchmarkXLA()

# Benchmark different configurations
results = benchmark.run(
    model_path="resnet50.pth",
    input_shape=(1, 3, 224, 224),
    num_runs=100
)

print(f"Latency: {results['latency_ms']:.2f}ms")
print(f"Throughput: {results['throughput_fps']:.1f} fps")
print(f"Memory: {results['memory_mb']:.1f}MB")
```

### ab_testing.py — A/B Testing Framework

Compare model versions:

```python
from edge.ab_testing import ABTest

test = ABTest(
    model_a="resnet50_v1.pth",
    model_b="resnet50_v2.pth",
    test_dataset="validation_set.json"
)

# Run comparison
results = test.run(num_samples=1000)

print(f"Model A accuracy: {results['model_a']['accuracy']:.3f}")
print(f"Model B accuracy: {results['model_b']['accuracy']:.3f}")
print(f"Statistical significance: {results['p_value']:.4f}")
```

### finetune.db — Job Database

SQLite database tracking fine-tuning jobs:

```sql
-- Job status
SELECT * FROM finetune_jobs 
WHERE status='running'
ORDER BY created_at DESC;

-- Metrics per epoch
SELECT * FROM training_metrics 
WHERE job_id='train_001'
ORDER BY epoch;
```

### finetuned_models/ — Model Storage

Directory structure:
```
finetuned_models/
├── job_001/
│   ├── model_best.pth
│   ├── model_final.pth
│   ├── metadata.json
│   └── metrics.json
├── job_002/
│   └── ...
```

### finetune_requests/ — Request Queue

Queue for fine-tuning requests from multiple sources:

```
finetune_requests/
├── pending/
│   ├── request_001.json
│   ├── request_002.json
├── processing/
│   └── request_001.json
└── completed/
    └── request_001.json
```

Request format:
```json
{
  "request_id": "req_001",
  "model": "resnet50",
  "dataset_path": "/data/my-dataset",
  "epochs": 10,
  "batch_size": 32,
  "callback_url": "http://my-server/callback"
}
```

## XLA Optimization

Compile models for faster inference:

```python
from edge.benchmark_xla import compile_with_xla
import jax
import jax.numpy as jnp

# Compile function with XLA
@jax.jit
def predict(x):
    return model(x)

# First call: compilation (slow)
# Subsequent calls: very fast (10-100× speedup)
output = predict(input_data)
```

## Fine-Tuning on Edge

Train models locally on edge device:

```python
from edge.finetune import EdgeFinetuner

finetuner = EdgeFinetuner(
    model_path="resnet50.pth",
    device="cuda"  # or 'cpu'
)

# Load data
train_data = load_dataset("my-data")

# Fine-tune
history = finetuner.train(
    train_data=train_data,
    epochs=5,
    learning_rate=0.0001,
    batch_size=16
)

# Save result
finetuner.save("finetuned_models/job_001/")
```

## Docker Deployment

### Building

```bash
docker build -f edge/Dockerfile -t sentinel-edge .
```

### Running

```bash
docker run -p 8001:8001 \
  -v /path/to/models:/models \
  -v /path/to/data:/data \
  sentinel-edge
```

## Performance Specifications

### Inference

| Hardware | Model | Latency | FPS | Memory |
|----------|-------|---------|-----|--------|
| Raspberry Pi 4 | ResNet18 | 400ms | 2.5 | 400MB |
| Jetson Nano | ResNet50 | 120ms | 8.3 | 500MB |
| Intel i7 | ResNet50 | 20ms | 50 | 800MB |

### Fine-Tuning

| Hardware | Dataset | Epochs | Time | Memory |
|----------|---------|--------|------|--------|
| RPi 4 | 1000 imgs | 5 | 2 hours | 800MB |
| Jetson | 10k imgs | 10 | 1 hour | 2GB |
| GPU | 100k imgs | 20 | 10 min | 8GB |

## Usage Examples

### Real-Time Inference with Caching

```python
from edge.inference_cache import InferenceCache
import cv2

cache = InferenceCache()
model = load_model("resnet50")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # Hash image to check cache
    img_hash = hashlib.md5(frame.tobytes()).hexdigest()
    
    # Get prediction (cached or computed)
    prediction = cache.get_or_compute(
        "resnet50",
        img_hash,
        lambda: model.predict(frame)
    )
    
    print(f"Prediction: {prediction}")
```

### Benchmark Before Deployment

```python
from edge.benchmark_xla import BenchmarkXLA

benchmark = BenchmarkXLA()

# Test different configurations
configs = [
    {"dtype": "float32"},
    {"dtype": "float16"},
    {"dtype": "int8"},  # quantized
]

for config in configs:
    results = benchmark.run(
        model_path="resnet50.pth",
        config=config
    )
    print(f"{config}: {results['latency_ms']:.2f}ms")
```

### A/B Test Model Updates

```python
from edge.ab_testing import ABTest

# Compare old vs new model
test = ABTest(
    model_a="resnet50_v1.0.pth",
    model_b="resnet50_v1.1.pth",
    test_dataset="validation_data.json"
)

results = test.run(num_samples=5000)

# Only deploy v1.1 if statistically better
if results['p_value'] < 0.05:
    print("✓ Deploy new model")
else:
    print("✗ Keep current model")
```

## Troubleshooting

**Inference too slow:**
- Enable XLA: `@jax.jit`
- Use smaller model
- Reduce input resolution
- Enable quantization

**Out of memory on edge:**
- Reduce batch size
- Use smaller model (MobileNet)
- Quantize model (int8)
- Use inference caching

**Fine-tuning not converging:**
- Lower learning rate (start with 0.00001)
- Check data quality
- Increase epochs
- Use larger model
