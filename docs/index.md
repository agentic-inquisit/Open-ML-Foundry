# LocalML finetune Documentation

Welcome to **LocalML finetune** — a local fine-tuning framework for edge ML with privacy-first design.

## Quick Navigation

- **[Getting Started](getting-started.md)** — 5-minute setup guide
- **[Installation](installation.md)** — Detailed setup options
- **[CLI Guide](cli-guide.md)** — Command reference
- **[Architecture](architecture.md)** — System design overview

## Core Features

- **Local Training** — No cloud required, keep data private
- **Multi-Model Support** — Fine-tune ResNet, YOLO, BERT, CLIP
- **Automatic Dataset Detection** — Scan and validate datasets automatically
- **Live Dashboard** — Real-time training metrics
- **Model Registry** — Version and track your trained models
- **Docker Ready** — Single `docker-compose up` deployment

## Who This Is For

- **ML Researchers** — Reproducible, extensible framework
- **Data Scientists** — Easy-to-use CLI, batteries included
- **Edge ML Developers** — Optimize for deployment, resource-aware
- **Privacy-First Teams** — No data leaving your infrastructure

## Start Here

```bash
# Install
pip install sentinel-finetune

# First fine-tune
sentinel model import --model resnet50
sentinel dataset prepare --path ./my-data
sentinel train start --name my-first-model
```

## Examples

- [Image Classification](tutorials/fine-tune-classifier.md)
- [Object Detection](tutorials/object-detection.md)
- [NLP Models](tutorials/nlp-models.md)
- [Edge Deployment](tutorials/edge-deployment.md)

## Resources

- [FAQ](faq.md)
- [Troubleshooting](troubleshooting.md)
- [Contributing](../CONTRIBUTING.md)
- [Security Policy](../SECURITY.md)
