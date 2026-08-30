# Edge Deployment Example

Deploy fine-tuned models to edge devices (Raspberry Pi, Jetson Nano, etc).

## Quick Start

### 1. Prepare Model

```bash
# Export trained model
sentinel model export \
  --job-id train_001 \
  --format tflite \
  --quantization int8
```

### 2. Deploy to Raspberry Pi

```bash
# On Raspberry Pi:
scp model.tflite pi@raspberrypi:~/

# SSH into RPi
ssh pi@raspberrypi

# Install dependencies
pip install tflite-runtime opencv-python numpy

# Run inference
python detect_pi.py --model model.tflite --camera /dev/video0
```

### 3. Deploy to Jetson Nano

```bash
# Similar to RPi but with GPU support
# Build image:
docker build -f Dockerfile.jetson -t sentinel-jetson .

# Run container:
docker run --gpus all -it sentinel-jetson python detect.py
```

## Files

- `detect_pi.py` — Inference on Raspberry Pi
- `detect_jetson.py` — Inference on Jetson Nano
- `Dockerfile.jetson` — Jetson container
- `requirements_pi.txt` — RPi dependencies
- `requirements_jetson.txt` — Jetson dependencies

## Real-Time Inference Example

```python
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# Load model
interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input/output shapes
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare input
    input_data = cv2.resize(frame, (224, 224))
    input_data = input_data.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Display result
    prediction = np.argmax(output_data[0])
    cv2.putText(frame, f"Class: {prediction}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Edge Inference", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Performance Optimization

### Model Quantization
```bash
# Convert to quantized format (4× smaller)
sentinel model export \
  --job-id train_001 \
  --format tflite \
  --quantization int8
```

### Reduce Input Size
```python
# Use 160×160 instead of 224×224
cv2.resize(frame, (160, 160))
```

### Inference Caching
```python
from sentinel.edge.inference_cache import InferenceCache

cache = InferenceCache(max_size=100)
result = cache.get_or_compute(model_name, image_hash, infer_fn)
```

## Deployment Targets

| Device | Latency | FPS | Memory | Cost |
|--------|---------|-----|--------|------|
| Raspberry Pi 4 | 400ms | 2.5 | 400MB | $55 |
| Jetson Nano | 120ms | 8 | 500MB | $99 |
| Intel NUC | 50ms | 20 | 200MB | $300 |

## Troubleshooting

**Slow inference:**
- Use smaller model (MobileNet)
- Reduce resolution (160×160)
- Use quantized model

**Out of memory:**
- Reduce model size
- Use streaming inference
- Close other apps

**Camera not working:**
- Check permissions: `sudo usermod -a -G video pi`
- List cameras: `ls /dev/video*`
- Test: `vcgencmd measure_temp`

## References

- [Raspberry Pi Setup](https://www.raspberrypi.org/documentation/)
- [Jetson Nano Docs](https://docs.nvidia.com/jetson/jetson-nano-devkit/index.html)
- [LocalML Edge Deployment](../../docs/tutorials/edge-deployment.md)
- [TensorFlow Lite](https://www.tensorflow.org/lite)
