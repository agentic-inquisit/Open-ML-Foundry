# Image Classification Fine-Tuning Example

Fine-tune a ResNet50 model on a custom image classification dataset.

## Quick Start

### 1. Prepare Dataset

Organize images in folder structure:

```
data/
├── train/
│   ├── class_1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   └── class_2/
│       ├── image1.jpg
│       └── image2.jpg
├── val/
│   ├── class_1/
│   └── class_2/
└── test/
    ├── class_1/
    └── class_2/
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train Model

```bash
python train.py --data ./data --epochs 10 --batch-size 32
```

### 4. Use Trained Model

```python
import torch
from train import ImageClassifier

# Load model
model = ImageClassifier(num_classes=2)
model.model.load_state_dict(torch.load("model.pth"))

# Predict
from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open("test_image.jpg")
image_tensor = transform(image).unsqueeze(0).to("cuda")

with torch.no_grad():
    output = model.model(image_tensor)
    prediction = torch.argmax(output)

print(f"Predicted class: {prediction}")
```

## Configuration

Edit `config.yaml` to customize:
- Model architecture (resnet50, resnet18, etc)
- Learning rate and batch size
- Number of epochs
- Data augmentation options
- Device selection (CPU/GPU)

## Training Parameters

```bash
python train.py \
  --data ./data \
  --model resnet50 \
  --epochs 20 \
  --batch-size 64 \
  --output ./model_final.pth
```

**Options:**
- `--data` — Path to dataset directory
- `--model` — Model architecture
- `--epochs` — Number of training epochs
- `--batch-size` — Batch size for training
- `--output` — Path to save trained model

## Output

After training:
- `model.pth` — Trained model weights
- `model_history.json` — Training history (loss, accuracy)
- Console output with metrics per epoch

## Tips for Better Results

1. **More Data** — 1000+ images per class for good accuracy
2. **Balance Classes** — Similar number of images per class
3. **Higher Resolution** — Higher quality images improve results
4. **Longer Training** — 20-50 epochs often better than 10
5. **Lower Learning Rate** — 0.0001 recommended for fine-tuning
6. **Data Augmentation** — Enabled by default, helps generalization

## Common Issues

### Out of Memory
Reduce batch size:
```bash
python train.py --data ./data --batch-size 8
```

### Low Accuracy
- Collect more data
- Train longer: `--epochs 50`
- Check data quality
- Lower learning rate

### Training Too Slow
Use GPU (if available):
- Make sure CUDA is installed
- Script auto-detects GPU

## Next Steps

- Evaluate on test set
- Export model (ONNX, TFLite)
- Deploy to production
- Fine-tune on more data for better results

## References

- [PyTorch Transfer Learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Open ML Foundry Docs](../../docs/tutorials/fine-tune-classifier.md)
