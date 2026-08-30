# Tutorial: Edge Deployment

Deploy fine-tuned models to edge devices: Raspberry Pi, Jetson, mobile.

## Use Case

Real-time plant disease detection on Raspberry Pi 4 with camera.

## Step 1: Fine-tune on Desktop

Train model on your powerful machine:

```bash
sentinel train start \
  --model resnet18 \
  --dataset plant-diseases \
  --epochs 20 \
  --batch-size 32 \
  --name plant-detector
```

Use smaller model (ResNet18 instead of ResNet50) for edge compatibility.

## Step 2: Export for Edge

```bash
# Export to TFLite (mobile/embedded)
sentinel model export \
  --job-id plant-detector \
  --format tflite \
  --quantization int8

# OR export to ONNX
sentinel model export \
  --job-id plant-detector \
  --format onnx
```

This creates optimized model files (~5-20 MB).

## Step 3: Prepare Raspberry Pi

### Setup

```bash
# SSH into RPi
ssh pi@raspberrypi.local

# Install dependencies
sudo apt update
sudo apt install python3-pip python3-opencv

# Install sentinel-finetune
pip3 install sentinel-finetune

# Download model (copy from desktop)
scp ~/exports/plant-detector.tflite pi@raspberrypi.local:~/
```

### Verify Setup

```bash
python3 --version  # Should be 3.7+
python3 -c "import cv2; print(cv2.__version__)"
python3 -c "import tensorflow as tf; print(tf.__version__)"
```

## Step 4: Run Inference on RPi

### Live Camera Inference

```python
import cv2
import tflite_runtime.interpreter as tflite
import numpy as np
import time

# Load model
interpreter = tflite.Interpreter(model_path="plant-detector.tflite")
interpreter.allocate_tensors()

# Get input/output shapes
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

class_names = ['healthy', 'diseased']

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess
    input_data = cv2.resize(frame, (224, 224))
    input_data = input_data.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    
    # Inference
    start = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    inference_time = (time.time() - start) * 1000
    
    # Get prediction
    pred = np.argmax(output_data[0])
    confidence = float(output_data[0][pred])
    
    # Display
    label = f"{class_names[pred]}: {confidence:.2f}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"{inference_time:.1f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Plant Disease Detector", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Performance

Expected on Raspberry Pi 4:
- Inference time: 200-500 ms (TFLite)
- FPS: 2-5 fps
- Accuracy: Same as desktop

To improve speed:
- Use smaller model (MobileNet instead of ResNet)
- Reduce image size (160×160 instead of 224×224)
- Use quantized model (int8)

## Step 5: Deploy to Jetson Nano

### Setup

```bash
# Install NVIDIA Jetson SDK
# https://docs.nvidia.com/deeplearning/jetson/jetson-nano-devkit/develop-jetson-nano.html

# Install dependencies
sudo apt install python3-opencv

pip install sentinel-finetune
pip install onnxruntime-gpu  # GPU inference
```

### Run on Jetson

```python
import cv2
import onnxruntime as ort
import numpy as np
import time

# Use GPU provider
sess = ort.InferenceSession("plant-detector.onnx", 
                            providers=['CUDAExecutionProvider'])

# ... rest of inference code (same as RPi)
```

**Performance on Jetson Nano:**
- Inference time: 50-150 ms (GPU)
- FPS: 6-20 fps
- 5-10× faster than RPi

## Step 6: Deploy to Mobile (Optional)

### TensorFlow Lite on Android

1. Convert model to TFLite (already done)
2. Add to Android app:

```kotlin
val interpreter = Interpreter(loadModelFile("plant-detector.tflite"))
val input = arrayOf(inputArray)  // [1, 224, 224, 3]
val output = Array(1) { FloatArray(2) }
interpreter.run(input, output)
```

### iOS with CoreML

1. Convert to CoreML:

```bash
coremltools.converters.convert("plant-detector.onnx", 
                               source="onnx",
                               target="iOS13")
```

2. Use in Xcode:

```swift
import CoreML
let model = try plant_detector()
let output = try model.prediction(image: cgImage)
```

## Step 7: Production Deployment Checklist

- [ ] Model file copied to edge device
- [ ] Dependencies installed (OpenCV, TFLite, ONNX)
- [ ] Inference code tested locally
- [ ] Latency measured and acceptable
- [ ] Accuracy verified on edge device
- [ ] Error handling in place (model load fails, inference fails)
- [ ] Logging configured
- [ ] Power consumption acceptable

## Troubleshooting

### Inference too slow

1. Use smaller model:
   ```bash
   sentinel model import --model resnet18  # Instead of resnet50
   ```

2. Reduce input size:
   ```python
   cv2.resize(frame, (160, 160))  # Instead of 224x224
   ```

3. Use quantized model:
   ```bash
   sentinel model export --format tflite --quantization int8
   ```

### Out of memory on RPi

- Reduce model size (MobileNet)
- Close other applications
- Use lighter framework (TFLite instead of ONNX)

### Inaccurate on edge

- Edge lighting different from training data → retrain with edge data
- Edge camera different → add data augmentation
- Quantization loss → verify with `--quantization none`

## Performance Comparison

| Device | Model | Latency | FPS | Memory |
|--------|-------|---------|-----|--------|
| Raspberry Pi 4 | ResNet18 TFLite | 400ms | 2.5 | 400MB |
| Raspberry Pi 4 | MobileNet TFLite | 100ms | 10 | 200MB |
| Jetson Nano | ResNet18 ONNX GPU | 120ms | 8 | 500MB |
| iPhone 12 | CoreML | 50ms | 20 | 300MB |

## Next Steps

- Set up continuous monitoring
- Collect edge data for retraining
- Implement model update pipeline
- Add error detection and fallbacks
