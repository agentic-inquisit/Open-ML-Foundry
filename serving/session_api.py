"""
Session API — the fine-tuning flow reworked as chat-like sessions, for both
LLM and vision models.

Every action inside a session (train, test-inference, notes) is appended as
a timestamped event so the frontend can render the whole run as a chat
transcript instead of a one-shot API call with no history.

Mounted at /api/v1/sessions... in serving/main.py, alongside a static chat
UI served at GET /sessions.
"""

from __future__ import annotations  # PEP 585 generics (dict[X,Y]) on Python 3.8

import threading
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core import session_store as store
from llm import supported_models

router = APIRouter(prefix="/api/v1", tags=["sessions"])

# In-memory registry of loaded models per session, so a session doesn't
# reload/re-download the base model on every train/inference call. Cleared
# when the session is deleted or the process restarts (acceptable for a
# local-first, single-process tool; not meant to survive across workers).
_loaded_models: dict[str, object] = {}
_loaded_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    name: str
    model_type: str            # 'llm' | 'vision'
    model_name: str            # display name, HF repo id, or local path
    model_format: str = "huggingface"  # 'huggingface' | 'gguf' | 'builtin'


class TrainRequest(BaseModel):
    dataset_path: str
    # LLM knobs
    epochs: int = 3
    learning_rate: float = 1e-4
    batch_size: int = 1
    lora_r: int = 8
    lora_alpha: int = 16
    # Vision knobs — dataset_path must be a directory of class subfolders
    # (dataset_path/<class_name>/<image files>) for vision sessions.
    target_object: Optional[str] = None
    num_classes: Optional[int] = None  # None = derive from class subfolders
    image_size: int = 224
    enable_validation: bool = True


class InferenceRequest(BaseModel):
    prompt: Optional[str] = None       # LLM
    image_path: Optional[str] = None   # Vision
    max_tokens: int = 512
    temperature: float = 0.7


class NoteRequest(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------

@router.get("/models")
async def list_models():
    """Vision training only ever runs the built-in CNN
    (edge/jax_train.py has one architecture) — registered/imported/trained
    models from ModelRegistry are listed for reference (so they're visible
    as session targets) but are flagged so picking one doesn't imply a
    different architecture gets trained."""
    from edge.model_registry import ModelRegistry

    registry = ModelRegistry()
    registered = [
        {
            "display_name": f"{entry['name']} ({v['version']})",
            "arch": "cnn",
            "registered": True,
        }
        for entry in registry.get_all_models()
        for v in entry["versions"]
    ]

    return {
        "llm": supported_models.list_models(),
        "vision": [{"display_name": "CNN (built-in)", "arch": "cnn"}] + registered,
    }


# ---------------------------------------------------------------------------
# Sessions CRUD
# ---------------------------------------------------------------------------

@router.post("/sessions")
async def create_session(req: CreateSessionRequest):
    if req.model_type not in ("llm", "vision"):
        raise HTTPException(400, "model_type must be 'llm' or 'vision'")
    session = store.create_session(
        name=req.name, model_type=req.model_type,
        model_name=req.model_name, model_format=req.model_format,
    )
    return _session_dict(session)


@router.get("/sessions")
async def list_sessions(model_type: Optional[str] = None):
    return [_session_dict(s) for s in store.list_sessions(model_type)]


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return {**_session_dict(session), "history": store.get_history(session_id)}


@router.get("/sessions/{session_id}/history")
async def get_history(session_id: str):
    if not store.get_session(session_id):
        raise HTTPException(404, "Session not found")
    return store.get_history(session_id)


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if not store.get_session(session_id):
        raise HTTPException(404, "Session not found")
    with _loaded_lock:
        _loaded_models.pop(session_id, None)
    store.delete_session(session_id)
    return {"deleted": session_id}


@router.post("/sessions/{session_id}/note")
async def add_note(session_id: str, req: NoteRequest):
    """Freeform user note — keeps the chat transcript usable for
    annotations ('tried lr=2e-4, worse') without triggering any action."""
    if not store.get_session(session_id):
        raise HTTPException(404, "Session not found")
    event = store.add_event(session_id, "user_message", role="user", data={"text": req.text})
    return event


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@router.post("/sessions/{session_id}/train")
async def start_training(session_id: str, req: TrainRequest):
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.status == "training":
        raise HTTPException(409, "Session is already training")
    dataset_path = Path(req.dataset_path)
    if not dataset_path.exists():
        raise HTTPException(400, f"Dataset not found: {req.dataset_path}")
    if session.model_type == "vision" and not dataset_path.is_dir():
        raise HTTPException(
            400,
            f"Vision dataset_path must be a directory of class subfolders "
            f"(dataset_path/<class_name>/<image files>), got a file: {req.dataset_path}",
        )

    store.update_session(session_id, status="training", dataset_path=req.dataset_path)
    store.add_event(session_id, "train_started", role="user", data=req.model_dump())

    thread = threading.Thread(target=_run_training, args=(session, req), daemon=True)
    thread.start()

    return {"status": "training", "session_id": session_id}


def _run_training(session: store.Session, req: TrainRequest) -> None:
    # Also tracked through JobTracker (sentinel/cli/job_tracker.py), keyed
    # by session_id, so `sentinel train status <session_id>` shows sessions
    # started from the web UI too, not just ones started from the CLI.
    from sentinel.cli.job_tracker import JobTracker

    session_id = session.id
    tracker = JobTracker()
    tracker.create_job(
        job_id=session_id, model_name=session.model_name, dataset_path=req.dataset_path,
        epochs=req.epochs, batch_size=req.batch_size, learning_rate=req.learning_rate,
    )
    tracker.start_job(session_id)

    try:
        if session.model_type == "llm":
            _train_llm(session, req, tracker)
        else:
            _train_vision(session, req, tracker)
        store.update_session(session_id, status="completed")
        tracker.complete_job(session_id)
    except Exception as e:
        store.update_session(session_id, status="error")
        store.add_event(session_id, "train_failed", role="system", data={"error": str(e)})
        tracker.fail_job(session_id, str(e))


def _train_llm(session: store.Session, req: TrainRequest, tracker) -> None:
    from llm import model_loader, lora_trainer

    loaded = model_loader.load(session.model_name, session.model_format or "huggingface")
    with _loaded_lock:
        _loaded_models[session.id] = loaded

    def on_progress(update: dict) -> None:
        store.add_event(session.id, "train_progress", role="assistant", data=update)
        # HF Trainer's .log() only reports loss without an eval_dataset
        # configured — there's no honest accuracy/val number to attach, so
        # track progress (real fractional epoch) without a full
        # TrainingMetrics entry, same as the CLI's train_start.
        epoch = update.get("epoch")
        if epoch is not None:
            tracker.set_progress(session.id, epoch)

    config = lora_trainer.LoRAConfig(
        r=req.lora_r, lora_alpha=req.lora_alpha, epochs=req.epochs,
        learning_rate=req.learning_rate, batch_size=req.batch_size,
        output_dir=f"training_outputs/llm/{session.id}",
    )
    result = lora_trainer.train(loaded, req.dataset_path, config, on_progress=on_progress)

    store.update_session(session.id, metrics=result, checkpoint_path=result.get("adapter_path"))
    store.add_event(session.id, "train_completed", role="assistant", data=result)


def _train_vision(session: store.Session, req: TrainRequest, tracker) -> None:
    from datetime import datetime
    from edge import vision_session_adapter
    from sentinel.cli.job_tracker import TrainingMetrics

    config = {
        "epochs": req.epochs, "batch_size": req.batch_size,
        "num_classes": req.num_classes, "image_size": req.image_size,
        "enable_validation": req.enable_validation,
    }

    def on_progress(update: dict) -> None:
        store.add_event(session.id, "train_progress", role="assistant", data=update)
        # run_finetuning() has no per-epoch callback of its own — update is
        # its full post-hoc history in one shot. Replay it into JobTracker
        # so `train status`/`--live` reflect real per-epoch metrics.
        train_losses = update.get("train_loss_history", [])
        train_accs = update.get("train_acc_history", [])
        val_losses = update.get("val_loss_history", [])
        val_accs = update.get("val_acc_history", [])
        for i in range(len(train_losses)):
            metric = TrainingMetrics(
                epoch=i + 1,
                loss=train_losses[i],
                accuracy=train_accs[i] if i < len(train_accs) else 0.0,
                val_loss=val_losses[i] if i < len(val_losses) else 0.0,
                val_accuracy=val_accs[i] if i < len(val_accs) else 0.0,
                timestamp=datetime.utcnow().isoformat(),
            )
            tracker.add_metrics(session.id, metric)

    result = vision_session_adapter.train(
        image_path=req.dataset_path, target_object=req.target_object or session.model_name,
        config=config, on_progress=on_progress,
    )

    store.update_session(session.id, metrics=result, checkpoint_path=result.get("checkpoint_path"))
    store.add_event(session.id, "train_completed", role="assistant", data=result)


# ---------------------------------------------------------------------------
# Inference ("test the model" — the chat reply half of the UI)
# ---------------------------------------------------------------------------

@router.post("/sessions/{session_id}/inference")
async def run_inference(session_id: str, req: InferenceRequest):
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    if session.model_type == "llm":
        if not req.prompt:
            raise HTTPException(400, "prompt is required for LLM sessions")
        store.add_event(session_id, "inference_request", role="user", data={"prompt": req.prompt})
        result = await _infer_llm(session, req)
    else:
        if not req.image_path:
            raise HTTPException(400, "image_path is required for vision sessions")
        store.add_event(session_id, "inference_request", role="user",
                        data={"image_path": req.image_path})
        result = await _infer_vision(session, req)

    store.add_event(session_id, "inference_result", role="assistant", data=result)
    return result


async def _infer_llm(session: store.Session, req: InferenceRequest) -> dict:
    from llm import model_loader, inference_engine

    with _loaded_lock:
        loaded = _loaded_models.get(session.id)
    if loaded is None:
        loaded = model_loader.load(session.model_name, session.model_format or "huggingface")
        with _loaded_lock:
            _loaded_models[session.id] = loaded

    return inference_engine.generate(
        loaded, req.prompt, max_tokens=req.max_tokens, temperature=req.temperature,
        adapter_path=session.checkpoint_path,
    )


async def _infer_vision(session: store.Session, req: InferenceRequest) -> dict:
    from edge import vision_session_adapter

    if not session.checkpoint_path:
        raise HTTPException(400, "No trained checkpoint yet — run /train first")

    return vision_session_adapter.infer(
        image_path=req.image_path, checkpoint_path=session.checkpoint_path,
    )


def _session_dict(session: store.Session) -> dict:
    return {
        "id": session.id, "name": session.name, "model_type": session.model_type,
        "model_name": session.model_name, "model_format": session.model_format,
        "status": session.status, "dataset_path": session.dataset_path,
        "training_config": session.training_config, "metrics": session.metrics,
        "checkpoint_path": session.checkpoint_path,
        "created_at": session.created_at, "updated_at": session.updated_at,
    }
