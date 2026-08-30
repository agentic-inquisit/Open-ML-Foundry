# NLP Fine-Tuning Example

Fine-tune BERT for text classification (sentiment analysis).

## Quick Start

```bash
# Prepare dataset (CSV with text and labels)
# data/train.csv
# text,label
# "Great product!",positive
# "Terrible experience",negative

# Train model
python train.py --data ./data --model bert-base --epochs 3

# Classify text
python predict.py --model ./model --text "This is amazing!"
```

## Data Format

CSV file with columns:
- `text` — Input text
- `label` — Classification label

Example:
```csv
text,label
"Love this product!",positive
"Not happy with purchase",negative
"Product works as described",positive
```

## Files

- `train.py` — Training script
- `predict.py` — Inference script
- `config.yaml` — Configuration
- `requirements.txt` — Dependencies

## Supported Models

- `bert-base` — BERT Base (110M parameters)
- `bert-large` — BERT Large (340M)
- `distilbert` — DistilBERT (lightweight)
- `roberta-base` — RoBERTa variant

## Training

```bash
python train.py \
  --data ./data \
  --model bert-base \
  --epochs 3 \
  --batch-size 32 \
  --learning-rate 2e-5
```

## Inference

```python
from predict import TextClassifier

classifier = TextClassifier(model_path="./model")
prediction = classifier.predict("This is great!")
print(f"Sentiment: {prediction}")  # positive, negative, neutral
```

## Tips

- Use 3 epochs for BERT (high-capacity model)
- Learning rate: 2e-5 to 5e-5 for fine-tuning
- Minimum 100 examples per class
- Longer sequences need more memory

## References

- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [LocalML NLP Tutorial](../../docs/tutorials/nlp-models.md)
