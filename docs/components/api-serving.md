# API Serving Module

REST API endpoints for fine-tuning and model management.

## Overview

Located in `serving/` directory. Provides FastAPI-based REST API for:
- Model training jobs
- Inference requests
- Admin operations
- Feature management

## Components

### main.py — Main API Server

Core FastAPI application with endpoints:

```python
# Health check
GET /health

# Training
POST /train/start
GET /train/status/{job_id}
GET /train/list
POST /train/cancel/{job_id}

# Models
GET /models/list
POST /models/import
GET /models/export/{job_id}

# Datasets
POST /datasets/prepare
GET /datasets/info/{dataset_id}

# Inference
POST /infer
GET /infer/batch
```

### admin_app.py — Admin Endpoints

Administrative operations:

```python
# User management
GET /admin/users
POST /admin/users/create
POST /admin/users/delete/{user_id}

# System stats
GET /admin/stats
GET /admin/logs
POST /admin/config/update
```

### features_api.py — Feature Endpoints

ML feature management:

```python
# Features
POST /features/extract
GET /features/list
POST /features/validate

# Metrics
GET /metrics/training
GET /metrics/inference
```

### handler.py — Request Handlers

Utility functions for request processing:
- Input validation
- Error handling
- Response formatting
- Request logging

### config.properties — Configuration

Server configuration:
- Port settings
- Database connection
- Authentication keys
- Model paths

## Docker Deployment

### Building

```bash
docker build -f serving/Dockerfile -t sentinel-api .
```

### Running

```bash
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://... \
  -e MODEL_PATH=/models \
  sentinel-api
```

## Usage Examples

### Start Training via API

```bash
curl -X POST http://localhost:8000/train/start \
  -H "Content-Type: application/json" \
  -d '{
    "model": "resnet50",
    "dataset": "my-data",
    "epochs": 10,
    "batch_size": 32
  }'
```

Response:
```json
{
  "job_id": "train_20240101_001",
  "status": "queued",
  "created_at": "2024-01-01T10:00:00Z"
}
```

### Check Training Status

```bash
curl http://localhost:8000/train/status/train_20240101_001
```

Response:
```json
{
  "job_id": "train_20240101_001",
  "status": "running",
  "epoch": 5,
  "total_epochs": 10,
  "loss": 0.315,
  "accuracy": 0.845
}
```

### List Models

```bash
curl http://localhost:8000/models/list
```

Response:
```json
{
  "models": [
    {
      "name": "resnet50",
      "framework": "pytorch",
      "task": "classification",
      "downloaded": true
    }
  ]
}
```

## Authentication

API uses JWT tokens:

```bash
# Get token
curl -X POST http://localhost:8000/auth/token \
  -d "username=user&password=pass"

# Use token in requests
curl http://localhost:8000/models/list \
  -H "Authorization: Bearer <token>"
```

## Integration with CLI

CLI tool can connect to running API:

```bash
sentinel config set --api-url http://localhost:8000
sentinel config set --api-token <jwt-token>

# Now all commands go through API
sentinel train start --model resnet50 --dataset my-data
```

## Performance

- **Latency**: <200ms per request (excluding training)
- **Throughput**: 100+ requests/second
- **Concurrency**: Handles 50+ concurrent users
- **Scalability**: Stateless, can run multiple instances

## Development

### Running Locally

```bash
cd serving
pip install -r ../requirements.txt
python main.py
```

Server runs on `http://localhost:8000`
Docs available at `http://localhost:8000/docs` (Swagger UI)

### Adding New Endpoints

```python
from fastapi import APIRouter

router = APIRouter()

@router.get("/my-feature")
def my_feature():
    """Description of endpoint."""
    return {"result": "value"}

# In main.py:
app.include_router(router)
```

### Testing

```bash
pytest tests/test_serving_*.py -v
```

## Deployment

### Docker Compose

See `docker-compose.yml` for running all services:

```bash
docker-compose up
# API available at http://localhost:8000
```

### Kubernetes

Example deployment manifest:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentinel-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: sentinel-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: sentinel-secret
              key: database-url
```

## Troubleshooting

**API won't start:**
```bash
# Check port 8000 is available
lsof -i :8000

# Check database connection
python -c "import sqlalchemy; print('OK')"
```

**High latency:**
- Check database connection pool
- Monitor GPU utilization
- Check network bandwidth

**Authentication errors:**
- Verify JWT secret in config
- Check token expiration
- Ensure Authorization header format: `Bearer <token>`
