# Components Overview

LocalML finetune consists of 9 major components working together.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                      │
├─────────────────────────────────────────────────────────────┤
│  CLI (sentinel/)          │    Web UI (web/)                 │
│  - Commands               │    - Dashboard                   │
│  - Dataset prep           │    - Job monitoring              │
│  - Model import           │    - Settings                    │
├─────────────────────────────────────────────────────────────┤
│                   API & Services Layer                       │
├─────────────────────────────────────────────────────────────┤
│  REST API (serving/)      │    Edge Module (edge/)           │
│  - /train/start           │    - Inference                   │
│  - /models/list           │    - Fine-tuning                 │
│  - /infer                 │    - Caching                     │
├─────────────────────────────────────────────────────────────┤
│                   ML Operations Layer                        │
├─────────────────────────────────────────────────────────────┤
│  Embeddings (mlops/)      │    Data & Persistence            │
│  - Vector search          │    - Database (migrations/)      │
│  - RAG pipeline           │    - Model storage               │
│  - Feature extraction     │    - Request queue               │
└─────────────────────────────────────────────────────────────┘
```

## Components Guide

### 1. CLI Module (`sentinel/cli/`)

**Purpose:** Command-line interface for users

**Key Files:**
- `main.py` — Click framework setup
- `commands.py` — All CLI commands (model, dataset, train, dashboard)
- `dataset_browser.py` — Auto-detect dataset structure
- `job_tracker.py` — Manage training jobs
- `dashboard.py` — Terminal UI for metrics

**User Interaction:**
```bash
sentinel model import --model resnet50
sentinel dataset prepare --path ./my-data
sentinel train start --model resnet50 --dataset my-data
```

**Learn More:** [docs/cli-guide.md](../cli-guide.md)

---

### 2. REST API (`serving/`)

**Purpose:** HTTP API for programmatic access

**Key Files:**
- `main.py` — FastAPI server with all endpoints
- `admin_app.py` — Administrative operations
- `features_api.py` — ML feature endpoints
- `handler.py` — Request processing utilities
- `config.properties` — Server configuration
- `Dockerfile` — Container image

**API Endpoints:**
```
POST   /train/start
GET    /train/status/{job_id}
GET    /models/list
POST   /models/import
POST   /infer
```

**Learn More:** [docs/components/api-serving.md](./api-serving.md)

---

### 3. Edge Module (`edge/`)

**Purpose:** Optimized inference & local fine-tuning

**Key Files:**
- `inference_cache.py` — Cache inference results
- `benchmark_xla.py` — Performance profiling
- `ab_testing.py` — Compare model versions
- `finetune.db` — SQLite job tracking
- `finetuned_models/` — Trained model storage
- `finetune_requests/` — Request queue

**Capabilities:**
- 10× faster inference with caching
- XLA compilation for speed
- Local fine-tuning on edge devices
- Automatic benchmarking

**Learn More:** [docs/components/edge-inference.md](./edge-inference.md)

---

### 4. Web UI (`web/`)

**Purpose:** Browser-based dashboard

**Key Files:**
- `index.html` — Main interface
- `globals.css` — Global styling
- `theme.css` — Theming system
- `LAYOUT_CHANGES.md` — Layout documentation
- `THEME_MIGRATION.md` — Theme upgrade guide

**Features:**
- Real-time training dashboard
- Model management
- Dataset browser
- Job monitoring
- System settings

**Learn More:** [docs/components/web-ui.md](./web-ui.md)

---

### 5. ML Operations (`mlops/`)

**Purpose:** Advanced ML capabilities

**Key Files:**
- `embedding_service.py` — Text/image embeddings
- `rag_orchestrator.py` — Retrieval-Augmented Generation

**Features:**
- Semantic search over documentation
- Smart question answering
- Feature extraction
- Model ensemble management

**Learn More:** [docs/components/mlops-pipeline.md](./mlops-pipeline.md)

---

### 6. Database (`migrations/`)

**Purpose:** Data persistence

**Key Files:**
- `001_init_all_features.sql` — Database schema

**Tables:**
- `finetune_jobs` — Training job records
- `training_metrics` — Epoch-by-epoch metrics
- `models` — Model registry
- `datasets` — Dataset catalog
- `users` — User accounts & permissions

---

### 7. Examples (`examples/`)

**Purpose:** Usage documentation & quick starts

**Key Files:**
- `QUICK_START_EXAMPLES.md` — Code examples

**Includes:**
- CLI examples
- API examples
- Python SDK examples
- Docker examples

---

### 8. Tests (`tests/`)

**Purpose:** Automated testing

**Key Files:**
- `test_cli_structure.py` — CLI module tests
- `test_dataset_browser.py` — Dataset detection tests
- `test_job_tracker.py` — Job management tests
- `test_security.py` — Security validation
- `conftest.py` — Pytest configuration

**Coverage:** 30+ test cases across all modules

---

### 9. GitHub (`/.github/`)

**Purpose:** CI/CD & community standards

**Key Files:**
- `workflows/tests.yml` — Automated testing
- `ISSUE_TEMPLATE/` — Issue templates
- `PULL_REQUEST_TEMPLATE.md` — PR guidelines

**Automation:**
- Test all platforms (Ubuntu, macOS, Windows)
- Python 3.8-3.11 compatibility
- Code linting & formatting
- Security checks

---

## Data Flow

### Training Workflow

```
CLI Input
  ↓
API Server
  ↓
Edge Module (fine-tuning)
  ↓
Job Database (track progress)
  ↓
Model Storage (save results)
  ↓
Dashboard (display metrics)
```

### Inference Workflow

```
API Request
  ↓
Inference Cache (check if seen before)
  ↓
Edge Module (run model)
  ↓
Result Cache (store for next time)
  ↓
Response to client
```

### RAG Workflow

```
User Query
  ↓
Embedding Service (text → vector)
  ↓
Vector Search (find similar docs)
  ↓
RAG Orchestrator (generate response)
  ↓
Response to user
```

## Deployment Scenarios

### Single Machine (Development)

```bash
make docker-up
# All services in one docker-compose.yml
```

### Distributed (Production)

```yaml
# Separate containers:
- sentinel-api (API server)
- sentinel-edge (Edge inference)
- sentinel-web (Web UI)
- sentinel-db (PostgreSQL)
- sentinel-cache (Redis)
- sentinel-mlops (Embeddings/RAG)
```

### Edge Only

```bash
# Just edge module on Raspberry Pi / Jetson
docker run -v /models:/models sentinel-edge
```

## Component Dependencies

```
CLI
  ├── depends on: API Server
  └── depends on: Local filesystem

API Server
  ├── depends on: Edge Module
  ├── depends on: Database
  └── depends on: Model Storage

Edge Module
  ├── depends on: Inference Cache
  ├── depends on: Job Database
  └── depends on: Model files

Web UI
  ├── depends on: API Server (REST)
  └── depends on: API Server (WebSocket)

MLOps
  ├── depends on: Embedding Service
  └── depends on: Vector Database
```

## Development Workflow

### Adding a New Feature

1. **CLI** — Update `sentinel/cli/commands.py`
2. **API** — Update `serving/main.py` endpoint
3. **Backend** — Update `edge/` module logic
4. **Tests** — Add tests in `tests/`
5. **Docs** — Update relevant docs
6. **UI** — Update `web/index.html` if needed

### Testing Locally

```bash
make dev-install
make test              # Run all tests
make lint              # Check code quality
make format            # Auto-format code
make docker-up         # Start services
```

## Performance & Scalability

| Component | Throughput | Latency | Scalability |
|-----------|-----------|---------|------------|
| API Server | 100+ req/s | <200ms | Horizontal |
| Edge (inference) | 10 fps | 100ms | Vertical |
| Edge (fine-tune) | 1 job/GPU | Hours | Horizontal |
| Web Dashboard | 50 concurrent | 1s update | Horizontal |
| Vector Search | 1000 queries/s | <100ms | Horizontal |

## Next Steps

1. **[Get Started](../getting-started.md)** — Quick setup
2. **[CLI Guide](../cli-guide.md)** — Learn commands
3. **[Architecture](../architecture.md)** — System design
4. **[Tutorials](../tutorials/)** — Practical examples
5. **[API Serving](./api-serving.md)** — REST integration
6. **[Edge Inference](./edge-inference.md)** — Edge deployment
7. **[MLOps](./mlops-pipeline.md)** — Advanced features
