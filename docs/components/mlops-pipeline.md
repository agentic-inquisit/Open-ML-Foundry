# MLOps Pipeline

ML operations and advanced features.

## Overview

Located in `mlops/` directory. Provides:
- Embedding extraction and management
- RAG (Retrieval-Augmented Generation) orchestration
- Feature extraction pipelines
- Model ensemble management

## Components

### embedding_service.py — Embedding Extraction

Generate and manage embeddings for vectors search:

```python
from mlops.embedding_service import EmbeddingService

service = EmbeddingService(model="bert-base")

# Extract embeddings
embeddings = service.embed_texts([
    "fine-tune resnet50",
    "train yolo detector",
    "bert classification"
])

# embeddings shape: (3, 768) for BERT

# Save embeddings for retrieval
service.save_embeddings(embeddings, "queries_db")

# Similarity search
similar_queries = service.find_similar(
    "train image classifier",
    top_k=5
)
```

**Supports:**
- Text embeddings (BERT, RoBERTa)
- Image embeddings (CLIP, ResNet)
- Cross-modal embeddings
- Custom embedding models

### rag_orchestrator.py — RAG Pipeline

Retrieval-Augmented Generation for enhanced responses:

```python
from mlops.rag_orchestrator import RAGOrchestrator

rag = RAGOrchestrator(
    embedding_model="bert-base",
    llm_model="gpt2",
    vector_db="faiss"
)

# Index knowledge base
rag.index_documents([
    "Fine-tuning tutorial...",
    "Best practices for ResNet...",
    "Edge deployment guide..."
])

# Retrieve and generate
response = rag.generate(
    query="How do I fine-tune ResNet?",
    top_k=3  # Retrieve top 3 similar docs
)

print(response)  # AI-generated response with context
```

**Features:**
- Document indexing
- Semantic search
- Context-aware generation
- Multi-source retrieval
- Ranking and reranking

## Use Cases

### 1. Smart Documentation

Use RAG to answer user questions from docs:

```python
# Index all documentation
rag.index_documents(read_markdown_files("docs/"))

# Answer user query
user_query = "I get CUDA out of memory error"
response = rag.generate(user_query)
# Returns: "Based on the troubleshooting guide, try..."
```

### 2. Feature Extraction

Extract features from training data:

```python
from mlops.feature_extraction import FeatureExtractor

extractor = FeatureExtractor(model="resnet50")

# Extract features from images
features = extractor.extract(image_paths)
# shape: (num_images, 2048)

# Use features for downstream tasks
clustering_result = cluster(features)
anomaly_detection_result = detect_anomalies(features)
```

### 3. Model Ensemble

Combine multiple model predictions:

```python
from mlops.ensemble import ModelEnsemble

ensemble = ModelEnsemble(models=[
    load_model("resnet50_v1"),
    load_model("resnet50_v2"),
    load_model("vgg16")
])

# Ensemble prediction
prediction = ensemble.predict(image)
# Combines predictions from all 3 models

# Weighted ensemble
prediction = ensemble.predict(
    image,
    weights=[0.5, 0.3, 0.2]
)
```

## Architecture

```
MLOps Pipeline
├── Data Ingestion
│   ├── Load documents
│   ├── Load images
│   └── Load training data
├── Embedding Generation
│   ├── Text embeddings
│   ├── Image embeddings
│   └── Cross-modal
├── Vector Storage (FAISS/Pinecone)
│   ├── Index embeddings
│   ├── Store metadata
│   └── Enable retrieval
├── Retrieval
│   ├── Semantic search
│   ├── Reranking
│   └── Context assembly
└── Generation (LLM)
    ├── Prompt engineering
    ├── Context injection
    └── Response generation
```

## Integration with Fine-Tuning

### Automatic Dataset Preparation

Use RAG to help prepare datasets:

```python
# Query: "Show me best practices for dataset preparation"
advice = rag.generate("best practices dataset preparation")

# Apply advice to dataset
dataset = prepare_dataset("my-data", advice)
```

### Training Optimization

Use embeddings to find similar past trainings:

```python
# Find similar past training runs
similar_runs = rag.find_similar_trainings(
    current_config=config,
    top_k=5
)

# Copy hyperparameters from best run
best_config = similar_runs[0]['config']
recommended_lr = best_config['learning_rate']
```

## Deployment

### Docker Compose

MLOps services run in separate container:

```yaml
mlops:
  build:
    context: .
    dockerfile: mlops/Dockerfile
  ports:
    - "8002:8002"
  volumes:
    - ./docs:/app/docs
    - ./mlops_cache:/app/cache
```

### Usage

```bash
# Via API
curl -X POST http://localhost:8002/embed \
  -d '{"text": "fine-tune resnet50"}'

# Via Python
from mlops.embedding_service import EmbeddingService
service = EmbeddingService()
embedding = service.embed_text("fine-tune resnet50")
```

## Performance

| Operation | Time | Memory |
|-----------|------|--------|
| Embed 100 texts | 2 sec | 500MB |
| Index 1000 docs | 5 sec | 1GB |
| Retrieve + Rank | 100ms | 200MB |
| Generate response | 2 sec | 2GB |

## Configuration

Create `mlops/config.yaml`:

```yaml
embedding:
  model: "bert-base-uncased"
  cache_size: 1000
  batch_size: 32

vector_db:
  type: "faiss"
  dimension: 768
  index_type: "IVF"

llm:
  model: "gpt2"
  max_length: 200
  temperature: 0.7
```

## Troubleshooting

**Embeddings slow to compute:**
- Use smaller model (DistilBERT)
- Enable GPU: `device='cuda'`
- Batch processing: `embed_texts()` with batch_size

**Vector search returning wrong results:**
- Rebuild index: `service.rebuild_index()`
- Check embedding quality
- Increase search `top_k`

**Out of memory:**
- Reduce vector db size
- Use smaller embedding model
- Clear cache: `service.clear_cache()`
