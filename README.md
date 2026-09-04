<p align="center">
  <img src="assets/icon.ico" alt="Open ML Foundry" width="520">
</p>

> **Open-source multi-modal fine-tuning. LLMs and vision models, fully local, accelerate edge deployments**

All-in-one framework for fine-tuning large language models and vision models on your machine. Session-based training for Qwen3.8, GLM-5.3-Flash, Kimi K3, MiniMax-H3, DeepSeek-V4, Gemma 4 with LoRA/QLoRA. Plus vision models (ResNet, YOLO, CLIP). Privacy-first, runs completely local, deploy to edge devices. No cloud required.

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue)
![JAX/Flax](https://img.shields.io/badge/JAX%2FFlax-0.4.13+-purple)
![Alpha](https://img.shields.io/badge/Status-Alpha-orange)

---

## 🎯 Use Cases (What You Can Do Today)

**LLM Fine-Tuning (Session-Based) - v0.3.0**
| Use Case | Feature | Models |
|----------|---------|--------|
| **Fine-tune LLM** | LoRA/QLoRA with session history | Qwen, Gemma, DeepSeek, GLM, MiniMax |
| **Test inference** | Chat-like interface during training | All LLM models |
| **Model selection** | Choose from HF Hub or GGUF models | HF Hub + GGUF |
| **Multi-turn training** | Train on dialogue/conversation data | All LLM models |

See [Known Limitations](#-known-limitations) — this hasn't been run end-to-end yet.

**Vision Fine-Tuning**
| Use Case | Feature | Status |
|----------|---------|--------|
| **Custom image classifier** | ResNet/CNN + your data (demo-scale training loop) | ✅ Working |
| **Object detection** | FasterRCNN ResNet50 (COCO) | ✅ Working |
| **Mobile export** | TFLite/ONNX + quantization | ✅ Working |
| **Edge inference** | 8-10ms latency | ✅ Working |
| **Image search** | CLIP embeddings | ✅ Working |
| **Model versioning** | Auto-versioning + rollback | ✅ Working |
| **Test a fine-tuned checkpoint** | Load checkpoint → predict | ✅ Working (`FinetuneInference` loads real params and runs a real forward pass) |

---

## 📊 Comparison with Existing Tools

**LLM Fine-Tuning Focus**
| Feature | Open ML Foundry | Unsloth | LLaMA-Factory | Axolotl |
|---------|---------|---------|--------------|---------|
| **LoRA/QLoRA** | ✅ | ✅⭐ (fastest) | ✅ | ✅ |
| **Session-based UI** | ✅ Chat history | ⚠️ Unsloth Studio (web UI + desktop app, has session features) | ❌ CLI/Web basic | ❌ CLI only |
| **Model switching** | ❌ New session per model (`model_name` is fixed at session creation, no code path changes it) | ❌ | ⚠️ Restart | ⚠️ Restart |
| **Local-only (no cloud)** | ✅ | ✅ | ✅ | ✅ |
| **Popular models** | Qwen, Gemma, DeepSeek, GLM | ⭐ Optimized | All HF models | All HF models |
| **Inference UI** | ✅ Chat interface | ⚠️ Unsloth Studio (chat + side-by-side model comparison) | ⚠️ Basic | ❌ |
| **Vision + LLM** | ✅ Both | ❌ LLM only | ❌ LLM only | ❌ LLM only |

**Vision Fine-Tuning (Legacy Support)**
| Feature | Open ML Foundry | PyTorch Lightning | FastAI |
|---------|---------|--------------|--------|
| **Image classification** | ✅ ResNet/CNN | ✅ | ✅ |
| **Object detection** | ✅ FasterRCNN | ❌ | ⚠️ |
| **Edge deployment** | ✅ TFLite/ONNX | ⚠️ | ❌ |
| **CLIP embeddings** | ✅ | ❌ | ❌ |

**TL;DR - Choose based on use case:**
- **Unsloth if:** You want FASTEST LLM training speed (2-5x)
- **Open ML Foundry if:** You need LLM + vision + edge deployment + session-based UI
- **LLaMA-Factory if:** You want all HuggingFace models with advanced config
- **PyTorch Lightning if:** You need distributed/multi-GPU framework

---

## 📖 Documentation (Streamlined for Users)

Start experimenting with **built-in models immediately** — no setup needed beyond `pip install` and `docker-compose up`.

### Essential Guides
1. **[Models & Sample Experiments](docs/MODELS_AND_SAMPLES.md)** ← Start here
   - Object detection on any image
   - Train custom classifier in 5-10s
   - Semantic image search
   - 5 quick experiment ideas

2. **[CLI Guide](docs/CLI_GUIDE.md)** ← Command-line interface (Phase 1)
   - `sentinel model import` - Import custom models
   - `sentinel dataset prepare` - Prepare training data
   - `sentinel train start` - Train with built-in or custom models
   - Usage examples for all commands

3. **[Getting Started (5 mins)](docs/01_Setup_QuickStart_20260821.md)**
   - Install → Start → Use

4. **[API Reference](docs/11_API_EndpointsReference_20260821.md)**
   - All endpoints
   - Request/response examples

5. **[Architecture Overview](docs/05_Architecture_SystemOverview_20260821.md)**
   - How the system works

6. **[Training Features](docs/20_Training_FeaturesOverview_20260821.md)**
   - Fine-tuning capabilities
   - Model versioning
   - Performance metrics

7. **[Inference Optimization](docs/30_Inference_OptimizationDetails_20260821.md)**
   - XLA speedups
   - Mobile export
   - Benchmark results

8. **[Security & Auth](docs/14_Security_RBACImplementation_20260821.md)**
   - JWT tokens
   - Role-based access
   - Production hardening

---

## 🚀 Get Started (5 Minutes)

### 1. Clone & Setup
```bash
git clone https://github.com/agentic-inquisit/open-ml-foundry.git
cd open-ml-foundry

# Run setup (checks dependencies, creates venv, installs packages)
./setup.sh
```

### 2. Session-Based Fine-Tuning (LLM + Vision, Chat UI)
```bash
# Start the API + session UI
uvicorn serving.main:app --reload --port 8000

# Open http://localhost:8000/sessions in a browser:
# 1. "+ New Session" → pick LLM (Qwen, Gemma, DeepSeek, GLM, MiniMax) or Vision
# 2. Point at a dataset (JSONL for LLM, image path for vision)
# 3. Start Training → progress streams into the session as chat events
# 4. Test tab → send a prompt/image, see the model's reply in the same thread
# 5. Every session keeps its full history — switch sessions from the sidebar
```
Or drive the same flow over the REST API directly — see [Core API Endpoints](#-core-api-endpoints) below.

### 3. Vision Fine-Tuning (Direct API, no session)
```bash
# Object detection (pretrained, no fine-tuning needed)
curl http://localhost:8001/detect -F "image=@test.jpg"

# One-shot fine-tune (legacy endpoint, still works outside sessions)
curl -X POST http://localhost:8001/finetune -F "dataset=@image.jpg" -F "target_object=my_class"
```

### 4. Full Server Setup
```bash
# Terminal 1: Start all services
./start.sh

# Terminal 2: Test LLM inference
curl -X POST http://localhost:8000/llm/inference \
  -H "Content-Type: application/json" \
  -d '{"session_id": "my-session", "prompt": "Hello, how are you?"}'

# Test vision inference
curl http://localhost:8001/detect -F "image=@test.jpg"
```

---

## ⚡ Core Features

### 💬 LLM Fine-Tuning (Session-Based) - v0.3.0
Implemented in `core/session_store.py`, `llm/`, `serving/session_api.py`, `serving/static/sessions_chat.html`. Not yet run end-to-end — see [Known Limitations](#-known-limitations).
- **Session management** - Each fine-tuning is a session with persistent, chat-like history (SQLite)
- **LoRA/QLoRA** - via `peft` + `transformers`; QLoRA needs `bitsandbytes` + CUDA GPU (not installed by default)
- **Model support** - Qwen, Gemma, DeepSeek, GLM, MiniMax, Kimi (via HF Hub — see `llm/supported_models.py` for repo id verification status)
- **Model formats** - HuggingFace Hub + GGUF (GGUF needs `llama-cpp-python`, not installed by default — build-tool dependency)
- **Inference in-session** - Test the model mid-conversation from the same chat UI
- **Multi-turn support** - JSONL datasets with `messages` (chat) or `prompt`/`completion` shape
- **Checkpointing** - LoRA adapter saved per session under `training_outputs/llm/<session_id>/`

### 🖼️ Vision Fine-Tuning (Maintained)
- **Image classification** - Fine-tune ResNet/CNN on your images
- **Object detection** - FasterRCNN ResNet50+FPN (COCO pretrained)
- **Quick training** - 5-10s for demo datasets, scales to hours
- **Validation & early stopping** - Automatic train/val split with patience
- **Auto-versioning** - v1.0, v1.1, v2.0 format with rollback

### 📊 Model Management
- **Performance tracking** - Loss, accuracy, metrics per epoch in SQLite
- **Model comparison** - Compare versions side-by-side
- **Session history** - View all training runs like conversations
- **Checkpoint storage** - Save/restore at any epoch

### 🚀 Optimization & Export
- **LoRA adapter export** - Lightweight 1-50MB models
- **TFLite export** - 3-10MB quantized vision models
- **ONNX export** - Cross-platform deployment
- **GGUF quantization** - Efficient LLM inference on CPU

### 🔐 Search & Embeddings
- **CLIP embeddings** - Image-text similarity locally
- **RAG orchestration** - Semantic image search
- **Vector DB ready** - For semantic retrieval

### 🔧 Infrastructure
- **Prometheus metrics** - Track latency, throughput
- **User auth** - JWT-based with role-based access
- **No cloud required** - Fully local, privacy-first
- **Docker support** - docker-compose for quick setup

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│              CLIENT/USER LAYER                  │
│    Web UI / API / Mobile / Command Line         │
└──────────────────┬──────────────────────────────┘
                   │
      ┌────────────┼────────────┐
      │            │            │
┌─────▼───┐  ┌─────▼───┐  ┌────▼────┐
│  USER   │  │  EDGE   │  │ SERVING  │
│  :8000  │  │  :8001  │  │ :8000    │
└─────┬───┘  └─────┬───┘  └────┬─────┘
      │ Auth       │ Training   │ Inference
      │            │  Inference │
      └────────────┼────────────┘
                   │
      ┌────────────▼────────────┐
      │   JAX/Flax + XLA        │
      │  • JIT compilation      │
      │  • Graph optimization   │
      │  • 30%+ speedup         │
      └────────────┬────────────┘
                   │
      ┌────────────▼────────────┐
      │  EXPORT PIPELINE        │
      │  TFLite | ONNX | SavedModel
      └────────────┬────────────┘
                   │
      ┌────────────▼────────────┐
      │   DEPLOYMENT            │
      │  Mobile | Edge | Server  │
      └─────────────────────────┘
```

**Key Layers:**
- **User (8000):** Signup, login, video generation, user endpoints
- **Edge (8001):** Fine-tuning, detection, validation, inference
- **Serving (8000):** FastAPI gateway, auth, metrics
- **Storage:** SQLite model registry + checkpoint files

---

## 📁 Project Structure

```
open-ml-foundry/
├── core/
│   └── session_store.py           # ✅ SQLite session + chat-event storage
│
├── llm/                            # LLM Fine-Tuning
│   ├── supported_models.py        # ✅ Qwen, Gemma, DeepSeek, GLM, MiniMax, Kimi registry
│   ├── model_loader.py            # ✅ HF Hub + GGUF loading
│   ├── lora_trainer.py            # ✅ LoRA/QLoRA training (peft + transformers)
│   └── inference_engine.py        # ✅ Chat-style generation (HF + GGUF)
│
├── edge/                           # Vision Fine-Tuning (Maintained)
│   ├── jax_train.py                # ✅ Training loop + CNN (run_finetuning)
│   ├── vision_session_adapter.py   # ✅ Wraps jax_train for session use
│   ├── model_registry.py           # ✅ Versioning + checkpoint storage
│   ├── vision_module.py            # ✅ FastAPI server (8001) — legacy one-shot API
│   └── ...                         # Other vision services
│
├── serving/
│   ├── main.py                     # ✅ FastAPI (8000), mounts session_api + /sessions UI
│   ├── session_api.py              # ✅ Unified session endpoints (LLM + vision)
│   ├── features_api.py             # ✅ Image gallery, datasets, training jobs, model registry, A/B testing
│   └── static/sessions_chat.html   # ✅ Chat-style session UI (vanilla JS, no build step)
│
├── sessions.db                     # SQLite session store (gitignored, created on first run)
├── requirements.txt                # ✅ Added: transformers, peft, accelerate, huggingface-hub
├── docker-compose.yml              # ✅ Multi-service setup
│
├── mlops/                          # ML Operations
│   ├── embedding_service.py       # ✅ CLIP embeddings
│   └── rag_orchestrator.py        # ✅ Semantic search
│
├── docs/                           # Documentation
└── tests/                          # Unit tests
```

**v0.3.0 Status:**
- ✅ **Implemented:** core/ (sessions), llm/ (LoRA training + inference wiring), edge/vision_session_adapter.py, serving/session_api.py, chat UI at `/sessions`
- ⏳ **Planned (v0.4.0):** Multi-infrastructure support (Kubernetes, AWS, GCP, Azure, edge clusters)

See [Known Limitations](#-known-limitations) before deploying any of this.

---

## 🔌 Core API Endpoints

**Sessions (New — unified LLM + vision)**
- `GET /sessions` - Chat-style session UI (web)
- `GET /api/v1/models` - List supported LLM + vision models
- `POST /api/v1/sessions` - Create session `{name, model_type, model_name, model_format}`
- `GET /api/v1/sessions` - List sessions
- `GET /api/v1/sessions/{id}` - Session details + full chat history
- `POST /api/v1/sessions/{id}/train` - Start training (LoRA for LLM, `run_finetuning` for vision)
- `POST /api/v1/sessions/{id}/inference` - Test the model, appended to history
- `POST /api/v1/sessions/{id}/note` - Add a freeform note to the transcript
- `DELETE /api/v1/sessions/{id}` - Delete session

**Vision (Legacy, one-shot, no session)**
- `POST /finetune` - Train custom CNN on images
- `GET /detect` - Real-time object detection
- `GET /download-model/{id}` - Download trained checkpoint

**Model Management:**
- `GET /models-versions` - View all model versions
- `GET /model-validation` - Validation dashboard (web)
- `POST /validate-model/{model_id}` - Run K-fold CV against a model's recorded training dataset

**System:**
- `GET /metrics` - Prometheus metrics

No authentication layer — this is a local, single-user tool (see [Known Limitations](#-known-limitations)).

**Note:** v0.3.0 is local-only. Multi-infrastructure support (Kubernetes, AWS, GCP, Azure) is planned for v0.4.0+.

---

## 📤 Deployment

**Local Server** (Fastest)
```bash
docker-compose up
# Services on :8000 (API) and :8001 (Fine-tuning)
# 8-10ms inference latency
```

**Mobile/Embedded** (Offline)
```bash
# Export fine-tuned model
python -c "from edge.optimized_inference import OptimizedVisionInference; \
          engine = OptimizedVisionInference(); \
          engine.export_to_tflite('model.tflite'); \
          engine.export_to_onnx('model.onnx')"

# Result: 3-10MB quantized models for Android/iOS/Coral/RPi
```

**Docker**
```bash
docker build -t sentinel .
docker run -p 8000:8000 -p 8001:8001 sentinel
```

## 📈 Performance

| Metric | Value |
|--------|-------|
| Inference latency | 8-10ms (GPU), 20-40ms (mobile) |
| XLA speedup | 30%+ throughput increase |
| Batch throughput | 200 images/sec (32-batch) |
| Model size (TFLite) | 3-10MB (quantized) |
| Training time | 5s-hours (depends on data) |

## ⚙️ Configuration

**Environment Variables:**
```bash
export JAX_PLATFORM_NAME=gpu      # Use GPU
export JAX_ENABLE_X64=False       # Use float32
export BATCH_SIZE=32              # Batch size
export DETECTION_THRESHOLD=0.7    # Object detection confidence
```

**Dependencies:**
- JAX 0.4.13+, Flax 0.7+, PyTorch 2.0+, TensorFlow 2.12+, OpenCV 4.7+
- Optional: pycoral (EdgeTPU), tf2onnx (ONNX), pyspark (Spark)

## 🧪 Testing & Benchmarking

```bash
# Run tests
pytest tests/test_security.py -v

# Benchmark XLA optimization
python edge/benchmark_xla.py

# Profile inference latency
python edge/optimized_inference.py
```

## 🤝 Contributing

Contributions welcome! Areas needing work:
- ❌ Model import UI (Streamlit/React)
- ❌ Dataset browser & preview
- ⚠️ Live training dashboard
- ⚠️ Confusion matrix visualization
- ✅ Inference optimization (help improve latency)
- ✅ Mobile export testing

See CONTRIBUTING.md for guidelines.

## 🎯 Roadmap

**v0.3.0 (Current) - Multi-Modal Foundations**
- ⚠️ LLM fine-tuning with LoRA/QLoRA (Qwen, Gemma, DeepSeek, GLM, MiniMax) — see Known Limitations
- ⚠️ Session-based training with chat history UI — see Known Limitations
- ⚠️ HuggingFace Hub + GGUF model support — see Known Limitations
- ✅ Vision models maintained (ResNet, YOLO, CLIP)
- ✅ Chat UI shipped as a static page (`serving/static/sessions_chat.html`) — plain HTML/JS, not React/Vue

**v0.4.0 - Enhanced Sessions**
- [ ] Multi-turn dialogue training (dataset format already designed, needs real-world testing)
- [ ] Inference optimization for local LLMs
- [ ] Advanced hyperparameter tuning UI
- [ ] Training progress streaming (WebSocket instead of 2s polling)
- [ ] Model comparison dashboard

**v0.5.0 - Multi-Modal Training**
- [ ] Unified JobSpec format (LLM + vision)
- [ ] Multi-infrastructure support (Kubernetes, edge clusters)
- [ ] Distributed training across devices
- [ ] Cost tracking & optimization

**v1.0.0 - Enterprise**
- [ ] Federated learning for privacy
- [ ] Multi-cloud orchestration
- [ ] Advanced audit logging & compliance
- [ ] Custom backend plugins

No committed dates — this is a volunteer-driven open-source project; see `.reserve/ROADMAP.txt` for effort estimates per phase.

## 🔐 Security

- **Authentication**: JWT tokens with role-based access
- **Audit logging**: Track all model training and inference
- **Input validation**: Model & dataset shape checking
- **On-device inference**: No data leaves your machine (optional)

See [SECURITY.md](SECURITY.md) for details.

## ⚠️ Known Limitations

- **LLM sessions (`core/`, `llm/`, `serving/session_api.py`) have not been run end-to-end.** The code has been reviewed but not executed against real dependencies — treat it as "should work" until someone runs it and confirms.
- **No authentication layer.** This is a local, single-user tool by design — there's no login, no per-user access control, and no adversary model between "you" and "you." Don't expose it on a public network interface without adding your own access control in front of it.
- **LLM model repo ids are not all confirmed.** See `llm/supported_models.py` — each entry's `verified` field tracks whether its HuggingFace Hub repo id has actually been checked.
- **QLoRA and GGUF are optional installs.** `bitsandbytes` (QLoRA, needs CUDA) and `llama-cpp-python` (GGUF) are commented out in `requirements.txt` because they need build tools or GPU hardware not present by default.

## 📝 License

MIT - see [LICENSE](LICENSE)

## 🙏 Thanks To

JAX/Flax, XLA, PyTorch, TensorFlow, ONNX, and the open source ML community.

---

**Fast local model fine-tuning with zero cloud dependency** 🚀
