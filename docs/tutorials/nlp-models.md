# Tutorial: Fine-tune NLP Models

Learn to fine-tune BERT for text classification.

## Use Case

Classify customer support tickets: bug, feature-request, billing.

## Step 1: Prepare Text Data

Create CSV file with text and labels:

```csv
text,label
"This button is broken",bug
"Can you add dark mode?",feature-request
"Why was I charged twice?",billing
"App crashes on startup",bug
...
```

Or organize as text files:

```
data/
├── bug/
│   ├── ticket_001.txt
│   └── ticket_002.txt
├── feature-request/
│   └── ticket_003.txt
└── billing/
    └── ticket_004.txt
```

**Tips:**
- 50+ examples per class minimum
- Clean text (remove special chars if needed)
- Balanced classes

## Step 2: Import BERT Model

```bash
sentinel model import --model bert-base
```

**Model options:**
- `bert-base` — 110M parameters, fast
- `bert-large` — 340M parameters, more accurate
- `roberta-base` — Better robustness
- `distilbert` — Smaller, faster

## Step 3: Prepare Dataset

```bash
sentinel dataset prepare --path ./data --name support-tickets --format text-classification
```

**Or from CSV:**

```bash
sentinel dataset prepare --path ./tickets.csv --dataset-column text --label-column label --name support-tickets
```

Verify:

```bash
sentinel dataset info --dataset support-tickets
```

Output:
```
Dataset: support-tickets
Type: Text Classification
Num Samples: 450
Classes: 3 (bug, feature-request, billing)
Distribution:
  - bug: 200
  - feature-request: 150
  - billing: 100
Avg Text Length: 42 words
```

## Step 4: Train Classifier

```bash
sentinel train start \
  --model bert-base \
  --dataset support-tickets \
  --epochs 3 \
  --batch-size 16 \
  --learning-rate 2e-5 \
  --max-seq-length 128 \
  --name ticket-classifier
```

**Parameters:**
- `--epochs 3` — NLP usually needs fewer epochs (high-capacity model)
- `--learning-rate 2e-5` — Conservative for pretrained NLP models
- `--max-seq-length 128` — Truncate/pad sequences to this length

## Step 5: Monitor Training

```bash
sentinel dashboard --job-id ticket-classifier
```

Watch:
- Training loss (should decrease)
- Validation accuracy
- Validation loss

## Step 6: Evaluate

```bash
sentinel train status --job-id ticket-classifier
```

Expected:
- Accuracy > 85% for balanced dataset
- Per-class metrics (precision, recall, F1)

## Step 7: Export Model

```bash
# ONNX for inference
sentinel model export --job-id ticket-classifier --format onnx

# PyTorch for HuggingFace ecosystem
sentinel model export --job-id ticket-classifier --format pth
```

## Inference Example

Using exported ONNX model:

```python
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
sess = ort.InferenceSession("ticket-classifier.onnx")

# Prepare input
text = "This button is broken"
inputs = tokenizer.encode_plus(text, return_tensors="np", max_length=128, padding=True)

# Predict
outputs = sess.run(None, dict(inputs))
logits = outputs[0]
predicted_class = np.argmax(logits)

class_names = ['bug', 'feature-request', 'billing']
print(f"Predicted: {class_names[predicted_class]}")
```

## Batch Inference

```python
texts = [
    "App crashes on startup",
    "Add multi-language support",
    "Can't use my discount code"
]

for text in texts:
    inputs = tokenizer.encode_plus(text, return_tensors="np", max_length=128, padding=True)
    outputs = sess.run(None, dict(inputs))
    pred = np.argmax(outputs[0])
    print(f"{text} → {class_names[pred]}")
```

## Advanced: Fine-tune with Domain Data

For better accuracy on specific domain:

```bash
# First phase: general fine-tuning
sentinel train start --model bert-base --dataset support-tickets --epochs 3

# Second phase: domain-specific (if you have more data)
sentinel train start --model bert-base --dataset support-tickets-v2 --epochs 2 --learning-rate 1e-5
```

## Troubleshooting

**Overfitting (high train accuracy, low val accuracy):**
```bash
# Add dropout/regularization
sentinel train start --dropout 0.2 --weight-decay 0.01
```

**Underfitting (both accuracies low):**
```bash
# Train longer
sentinel train start --epochs 5

# Increase learning rate
sentinel train start --learning-rate 5e-5
```

**Out of memory with BERT:**
```bash
# Reduce batch size
sentinel train start --batch-size 8

# Use smaller model
sentinel model import --model distilbert
```

## Production Deployment

Export and serve with FastAPI:

```python
from fastapi import FastAPI
from transformers import AutoTokenizer
import onnxruntime as ort

app = FastAPI()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
sess = ort.InferenceSession("ticket-classifier.onnx")
class_names = ['bug', 'feature-request', 'billing']

@app.post("/classify")
async def classify(text: str):
    inputs = tokenizer.encode_plus(text, return_tensors="np", max_length=128, padding=True)
    outputs = sess.run(None, dict(inputs))
    pred = int(np.argmax(outputs[0]))
    return {"class": class_names[pred]}
```

Run server:
```bash
uvicorn app:app --port 8000
```

Query:
```bash
curl -X POST "http://localhost:8000/classify?text=App%20crashes%20on%20startup"
```
