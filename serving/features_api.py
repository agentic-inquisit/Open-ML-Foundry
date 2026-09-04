"""
Open ML Foundry - Feature API Endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Query, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import logging
import sys
import os
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core import image_store, session_store, dataset_store
from edge.model_registry import ModelRegistry
from edge.ab_testing import ABTestingService
from edge.labeling_service import LabelingService
from serving.session_api import TrainRequest, _run_training

logger = logging.getLogger(__name__)
_registry = ModelRegistry()
_ab_testing = ABTestingService()
_labeling = LabelingService()

# ============================================================================
# ROUTER SETUP
# ============================================================================

router = APIRouter(prefix="/api/v1", tags=["features"])

# ============================================================================
# CURRENT USER — local-first, single-user tool
# ============================================================================

LOCAL_OWNER = "local"

class TokenPayload(BaseModel):
    sub: str
    role: str
    exp: datetime

async def get_current_user() -> TokenPayload:
    return TokenPayload(sub=LOCAL_OWNER, role="admin", exp=datetime.utcnow() + timedelta(hours=24))

# ============================================================================
# 1. IMAGE GALLERY ENDPOINTS
# ============================================================================

class ImageUploadResponse(BaseModel):
    id: int
    filename: str
    file_path: str
    size_mb: float
    uploaded_at: datetime
    tags: Optional[str] = None

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
MAX_IMAGE_SIZE_MB = 25

@router.post("/images/upload", response_model=ImageUploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    tags: Optional[str] = Form(None),
    current_user: TokenPayload = Depends(get_current_user)
):
    """Upload image to gallery"""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext or 'unknown'}")

    content = await file.read()
    if len(content) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File exceeds {MAX_IMAGE_SIZE_MB}MB limit")

    record = image_store.save_image(
        owner_sub=current_user.sub, filename=file.filename, content=content, tags=tags,
    )

    return ImageUploadResponse(
        id=record.id,
        filename=record.filename,
        file_path=record.file_path,
        size_mb=round(record.size_mb, 3),
        uploaded_at=datetime.fromisoformat(record.uploaded_at),
        tags=record.tags,
    )

@router.get("/images/gallery")
async def list_images(
    skip: int = Query(0),
    limit: int = Query(10),
    tags: Optional[str] = None,
    current_user: TokenPayload = Depends(get_current_user)
):
    """List user's image gallery"""
    records, total = image_store.list_images(current_user.sub, skip=skip, limit=limit, tags=tags)
    return {
        "images": [
            {
                "id": r.id, "filename": r.filename, "file_path": r.file_path,
                "size_mb": round(r.size_mb, 3), "width": r.width, "height": r.height,
                "uploaded_at": r.uploaded_at,
            }
            for r in records
        ],
        "total": total,
        "skip": skip,
        "limit": limit,
    }

@router.delete("/images/{image_id}")
async def delete_image(
    image_id: int,
    current_user: TokenPayload = Depends(get_current_user)
):
    """Delete image from gallery"""
    if not image_store.delete_image(image_id, owner_sub=current_user.sub):
        raise HTTPException(status_code=404, detail=f"Image {image_id} not found")
    return {"message": f"Image {image_id} deleted"}

# ============================================================================
# 2. MODEL REGISTRY ENDPOINTS
# ============================================================================

class RegistryModelResponse(BaseModel):
    id: int
    name: str
    version: str
    status: str
    access_level: str
    accuracy: Optional[float] = None
    created_at: str

@router.get("/registry/models", response_model=List[RegistryModelResponse])
async def list_registry_models(
    current_user: TokenPayload = Depends(get_current_user)
):
    """List available ML models"""
    flat = []
    for model in _registry.get_all_models():
        for v in model["versions"]:
            flat.append(RegistryModelResponse(
                id=v["model_id"], name=v["name"], version=v["version"],
                status=v["status"], access_level=v["access_level"],
                accuracy=v["metadata"]["accuracy_on_test_set"],
                created_at=v["created_at"],
            ))
    return flat

@router.post("/models/{model_id}/deploy")
async def deploy_model(
    model_id: int,
    current_user: TokenPayload = Depends(get_current_user)
):
    """Deploy model to production (Admin only) — sets access_level to public,
    the closest concept the registry actually has."""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    _registry.set_access_level(model_id, "public")
    return {"message": f"Model {model_id} access_level set to public"}

@router.post("/models/upload")
async def upload_custom_model(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    description: str = Form(""),
    current_user: TokenPayload = Depends(get_current_user)
):
    """Upload custom trained model (Admin)"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    result = _registry.register_model(model_name, description=description, owner=current_user.sub)
    model_id = result["model_id"]

    checkpoint_dir = os.path.join("finetuned_models", model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, file.filename)
    content = await file.read()
    with open(checkpoint_path, "wb") as f:
        f.write(content)

    _registry.save_checkpoint(model_id, epoch=0, checkpoint_path=checkpoint_path,
                              train_loss=0.0, val_loss=0.0, is_best=True)

    return {
        "id": model_id,
        "name": model_name,
        "version": result["version"],
        "file_path": checkpoint_path,
    }

# ============================================================================
# 3. DATASET MANAGEMENT ENDPOINTS
# ============================================================================

class DatasetResponse(BaseModel):
    id: int
    name: str
    total_images: int
    status: str
    created_at: str

@router.post("/datasets/create", response_model=DatasetResponse)
async def create_dataset(
    name: str = Form(...),
    description: str = Form(""),
    current_user: TokenPayload = Depends(get_current_user)
):
    """Create new dataset"""
    d = dataset_store.create_dataset(owner_sub=current_user.sub, name=name, description=description)
    return DatasetResponse(id=d.id, name=d.name, total_images=d.total_images,
                           status=d.status, created_at=d.created_at)

@router.get("/datasets")
async def list_datasets(
    current_user: TokenPayload = Depends(get_current_user)
):
    """List user's datasets"""
    return {"datasets": [
        {"id": d.id, "name": d.name, "total_images": d.total_images,
         "status": d.status, "created_at": d.created_at}
        for d in dataset_store.list_datasets(current_user.sub)
    ]}

@router.post("/datasets/{dataset_id}/upload-images")
async def add_images_to_dataset(
    dataset_id: int,
    files: List[UploadFile] = File(...),
    current_user: TokenPayload = Depends(get_current_user)
):
    """Bulk upload images to dataset — saves each file via
    core/image_store.py, then links the resulting image ids to the dataset."""
    if not dataset_store.get_dataset(dataset_id):
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

    image_ids = []
    for file in files:
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in ALLOWED_IMAGE_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext or 'unknown'}")
        content = await file.read()
        record = image_store.save_image(owner_sub=current_user.sub, filename=file.filename, content=content)
        image_ids.append(record.id)

    total = dataset_store.add_images(dataset_id, image_ids)

    return {
        "dataset_id": dataset_id,
        "uploaded": len(image_ids),
        "total_in_dataset": total,
    }

# ============================================================================
# 4. TRAINING JOB ENDPOINTS 
# ============================================================================

@router.post("/training/start")
async def start_training_job(
    model_name: str,
    model_type: str,
    dataset_path: str,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    current_user: TokenPayload = Depends(get_current_user)
):
    """Start new training job"""
    if model_type not in ("llm", "vision"):
        raise HTTPException(status_code=400, detail="model_type must be 'llm' or 'vision'")
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=400, detail=f"Dataset not found: {dataset_path}")

    session = session_store.create_session(name=f"{model_name} training",
                                            model_type=model_type, model_name=model_name)
    session_store.update_session(session.id, status="training", dataset_path=dataset_path)
    session_store.add_event(session.id, "train_started", role="user", data={
        "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
    })

    req = TrainRequest(dataset_path=dataset_path, epochs=epochs,
                       batch_size=batch_size, learning_rate=learning_rate)
    threading.Thread(target=_run_training, args=(session, req), daemon=True).start()

    return {"job_id": session.id, "status": "training"}

@router.get("/training/jobs")
async def list_training_jobs(
    current_user: TokenPayload = Depends(get_current_user)
):
    """List user's training jobs"""
    return {"jobs": [
        {"id": s.id, "status": s.status, "model_name": s.model_name,
         "model_type": s.model_type, "metrics": s.metrics, "created_at": s.created_at}
        for s in session_store.list_sessions()
    ]}

@router.get("/training/jobs/{job_id}/logs")
async def get_training_logs(
    job_id: str,
    current_user: TokenPayload = Depends(get_current_user)
):
    """Get training logs"""
    if not session_store.get_session(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    return {"logs": session_store.get_history(job_id)}

@router.get("/training/jobs/{job_id}/metrics")
async def get_training_metrics(
    job_id: str,
    current_user: TokenPayload = Depends(get_current_user)
):
    """Get training metrics"""
    session = session_store.get_session(job_id)
    if not session:
        raise HTTPException(status_code=404, detail="Job not found")
    return session.metrics

# ============================================================================
# 5. ANNOTATION TOOL ENDPOINTS
# ============================================================================

@router.post("/annotations/create")
async def create_annotation(
    image_id: int,
    label: str,
    data: Optional[dict] = None,
    current_user: TokenPayload = Depends(get_current_user)
):
    """Label an image (data may include bbox: {x1,y1,x2,y2})"""
    label_id = _labeling.label_image(
        image_id=image_id, class_name=label, labeled_by=current_user.sub,
        bbox=data.get("bbox") if data else None,
    )
    return {"id": label_id, "image_id": image_id, "label": label}

@router.get("/annotations/pending")
async def get_pending_annotations(
    limit: int = 10,
    current_user: TokenPayload = Depends(get_current_user)
):
    """Get images needing annotation"""
    return {"images": _labeling.get_unlabeled_images(limit=limit)}

@router.post("/annotations/{annotation_id}/verify")
async def verify_annotation(
    annotation_id: int,
    approved: bool,
    current_user: TokenPayload = Depends(get_current_user)
):
    """Verify annotation quality (Admin)"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    # Not wired: edge/labeling_service.py's labels table has no
    # verified/approved column, so there's nothing real to write to yet.
    raise HTTPException(status_code=501, detail="Not implemented")

# ============================================================================
# 6. MODEL VERSIONING ENDPOINTS
# ============================================================================

@router.get("/models/{model_id}/versions")
async def list_model_versions(
    model_id: int,
    current_user: TokenPayload = Depends(get_current_user)
):
    """Get all versions of a model"""
    model_name = _registry.get_model_name(model_id)
    if not model_name:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return {"model_id": model_id, "versions": _registry.get_model_versions(model_name)}

@router.post("/models/{model_id}/versions/{version_id}/deploy")
async def deploy_model_version(
    model_id: int,
    version_id: int,
    current_user: TokenPayload = Depends(get_current_user)
):
    """Deploy specific model version (Admin) — each version is its own row
    in the registry, so version_id is deployed directly."""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    _registry.set_access_level(version_id, "public")
    return {"message": f"Model {model_id} version {version_id} access_level set to public"}

# ============================================================================
# 7. A/B TESTING ENDPOINTS 
# ============================================================================

@router.post("/experiments/create")
async def create_ab_experiment(
    name: str,
    model_a_id: int,
    model_b_id: int,
    current_user: TokenPayload = Depends(get_current_user)
):
    """Create A/B testing experiment"""
    model_a_path = _registry.get_latest_checkpoint(model_a_id)
    model_b_path = _registry.get_latest_checkpoint(model_b_id)
    if not model_a_path or not model_b_path:
        raise HTTPException(
            status_code=400,
            detail="Both models need at least one saved checkpoint before an A/B test can run",
        )
    return _ab_testing.create_test(
        name=name, model_a_id=model_a_id, model_b_id=model_b_id,
        model_a_path=model_a_path, model_b_path=model_b_path, owner=current_user.sub,
    )

@router.get("/experiments/{exp_id}/results")
async def get_experiment_results(
    exp_id: int,
    current_user: TokenPayload = Depends(get_current_user)
):
    """Get A/B test results and winner"""
    results = _ab_testing.get_results(exp_id)
    if results.get("status") == "error":
        raise HTTPException(status_code=404, detail=results["message"])
    return results

# ============================================================================
# 8. CUSTOM TRAINING ENDPOINTS 
# ============================================================================

@router.post("/training/custom")
async def train_custom_model(
    name: str,
    dataset_path: str,
    architecture: str,
    hyperparameters: dict,
    current_user: TokenPayload = Depends(get_current_user)
):
    """Start custom model training — same flow as /training/start, with
    `architecture` as the model name and hyperparameters pulled from the dict."""
    return await start_training_job(
        model_name=architecture,
        model_type=hyperparameters.get("model_type", "vision"),
        dataset_path=dataset_path,
        epochs=hyperparameters.get("epochs", 10),
        batch_size=hyperparameters.get("batch_size", 32),
        learning_rate=hyperparameters.get("learning_rate", 0.001),
        current_user=current_user,
    )

@router.get("/training/custom/{job_id}/download")
async def download_trained_model(
    job_id: str,
    current_user: TokenPayload = Depends(get_current_user)
):
    """Download trained model weights"""
    session = session_store.get_session(job_id)
    if not session:
        raise HTTPException(status_code=404, detail="Job not found")
    if not session.checkpoint_path or not os.path.exists(session.checkpoint_path):
        raise HTTPException(status_code=404, detail="No checkpoint saved for this job yet")
    return FileResponse(session.checkpoint_path)

