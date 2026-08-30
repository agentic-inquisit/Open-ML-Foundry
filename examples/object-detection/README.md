# Object Detection Fine-Tuning Example

Fine-tune YOLOv5 for custom object detection.

## Quick Start

```bash
# Prepare dataset in YOLO format
# data/
# ├── images/
# │   ├── train/
# │   └── val/
# └── labels/
#     ├── train/
#     └── val/

# Train model
python train.py --data ./data --model yolov5s --epochs 50

# Detect objects
python detect.py --model weights/best.pt --source test.jpg
```

## Data Format

YOLO annotation format (.txt):
```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized (0-1).

## Files

- `train.py` — Training script
- `detect.py` — Inference script
- `config.yaml` — Configuration file
- `requirements.txt` — Dependencies

## References

- [YOLOv5 Docs](https://docs.ultralytics.com/)
- [LocalML Object Detection Tutorial](../../docs/tutorials/object-detection.md)

## Common Tasks

### Evaluate Model
```bash
python val.py --model weights/best.pt --data ./data
```

### Export Model
```bash
python export.py --model weights/best.pt --format onnx
```

### Use TensorBoard
```bash
tensorboard --logdir runs/
```
