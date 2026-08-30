# Tutorial: Fine-tune Object Detector

Learn to fine-tune YOLOv5 for custom object detection.

## Use Case

Detect and localize defects in manufacturing quality control.

## Step 1: Prepare Annotated Data

YOLOv5 expects bounding box annotations in YOLO format:

```
data/
├── images/
│   ├── defect_001.jpg
│   ├── defect_002.jpg
│   └── ...
└── labels/
    ├── defect_001.txt
    ├── defect_002.txt
    └── ...
```

**Label format (.txt):**
```
<class_id> <x_center> <y_center> <width> <height>
```

Example `defect_001.txt`:
```
0 0.5 0.3 0.2 0.15
1 0.8 0.6 0.1 0.08
```

**Annotation tools:**
- Roboflow (free)
- LabelImg
- CVAT

## Step 2: Import YOLO Model

```bash
sentinel model import --model yolov5s
```

**Model options:**
- `yolov5n` — Nano (lightweight, fast)
- `yolov5s` — Small (balanced)
- `yolov5m` — Medium (faster)
- `yolov5l` — Large (slower, more accurate)

## Step 3: Create Dataset Config

Create `dataset.yaml`:

```yaml
path: /path/to/data
train: images
val: images

nc: 2  # number of classes
names: ['defect', 'ok']  # class names
```

## Step 4: Prepare Dataset

```bash
sentinel dataset prepare --path ./data --name defects --format yolo
```

Verify:

```bash
sentinel dataset info --dataset defects
```

## Step 5: Train Detector

```bash
sentinel train start \
  --model yolov5s \
  --dataset defects \
  --epochs 50 \
  --batch-size 16 \
  --img-size 640 \
  --patience 10 \
  --name defect-detector
```

**Parameters:**
- `--epochs 50` — More epochs for detection (more complex task)
- `--img-size 640` — Input resolution
- `--patience 10` — Early stopping after 10 epochs without improvement

## Step 6: Monitor Training

```bash
sentinel dashboard --job-id defect-detector
```

**Key metrics for detection:**
- mAP (mean Average Precision) — should increase
- Loss — should decrease
- P/R curve — precision vs recall tradeoff

## Step 7: Evaluate

```bash
sentinel train status --job-id defect-detector
```

Look for:
- mAP@0.5 > 0.60 for good detection
- Balanced precision/recall

## Step 8: Export

```bash
# ONNX for inference
sentinel model export --job-id defect-detector --format onnx

# TFLite for mobile/edge
sentinel model export --job-id defect-detector --format tflite

# PyTorch for PyTorch ecosystem
sentinel model export --job-id defect-detector --format pth
```

## Inference Example

After export to ONNX:

```python
import onnxruntime as ort
import cv2
import numpy as np

# Load model
sess = ort.InferenceSession("defect-detector.onnx")

# Prepare image
img = cv2.imread("test_image.jpg")
img = cv2.resize(img, (640, 640))
img = img[np.newaxis, ...].astype(np.float32) / 255.0

# Predict
outputs = sess.run(None, {'images': img})
detections = outputs[0]  # [x, y, w, h, confidence, class]

# Draw boxes
for det in detections:
    x, y, w, h, conf, cls = det
    if conf > 0.5:  # confidence threshold
        cv2.rectangle(img, (int(x-w/2), int(y-h/2)), 
                      (int(x+w/2), int(y+h/2)), (0, 255, 0), 2)

cv2.imwrite("output.jpg", img)
```

## Tips for Better Results

1. **Quality annotations** — Accurate bounding boxes critical
2. **Balanced dataset** — Similar number of images per class
3. **Diverse backgrounds** — Different lighting, angles, scales
4. **Data augmentation** — Let Sentinel handle it (default: on)
5. **Patient training** — Detection needs more epochs than classification

## Common Issues

**Low mAP:**
- Add more training data
- Improve annotation quality
- Train longer (more epochs)
- Use larger model (yolov5m/l)

**Training slow:**
- Use GPU: `--device cuda`
- Reduce image size: `--img-size 416`
- Increase batch size: `--batch-size 32`

**Memory errors:**
- Reduce batch size: `--batch-size 8`
- Reduce image size: `--img-size 416`
- Use smaller model: `yolov5n`
