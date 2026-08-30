"""
SQLAlchemy Models for LocalML
Complete database schema for all 20 features
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, JSON, ForeignKey, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

# ============================================================================
# CORE USER & TEAM MODELS
# ============================================================================

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(255), unique=True, index=True)
    email = Column(String(255), unique=True, index=True)
    password_hash = Column(String(255))
    role = Column(String(50), default="user")  # 'admin', 'user'
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    # Relationships
    images = relationship("ImageAsset", back_populates="owner")
    datasets = relationship("Dataset", back_populates="owner")
    training_jobs = relationship("TrainingJob", back_populates="user")
    api_keys = relationship("APIKey", back_populates="user")
    notifications = relationship("Notification", back_populates="user")
    team_memberships = relationship("TeamMember", back_populates="user")


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True)
    description = Column(Text)
    owner_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    members = relationship("TeamMember", back_populates="team", cascade="all, delete-orphan")


class TeamMember(Base):
    __tablename__ = "team_members"

    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey("teams.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    role = Column(String(50), default="member")  # 'owner', 'admin', 'member', 'viewer'
    joined_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    team = relationship("Team", back_populates="members")
    user = relationship("User", back_populates="team_memberships")


# ============================================================================
# 1. IMAGE & ASSET MODELS
# ============================================================================

class ImageAsset(Base):
    __tablename__ = "image_assets"

    id = Column(Integer, primary_key=True)
    filename = Column(String(255))
    file_path = Column(String(512))
    s3_key = Column(String(512))  # For cloud storage
    owner_id = Column(Integer, ForeignKey("users.id"))
    size_mb = Column(Float)
    format = Column(String(50))  # 'jpg', 'png', 'tiff'
    width = Column(Integer)
    height = Column(Integer)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed = Column(Boolean, default=False)
    metadata = Column(JSON)  # EXIF, camera info, etc.
    tags = Column(String(500))  # Comma-separated

    # Relationships
    owner = relationship("User", back_populates="images")
    annotations = relationship("Annotation", back_populates="image")


# ============================================================================
# 2. MODEL & REGISTRY MODELS
# ============================================================================

class MLModel(Base):
    __tablename__ = "ml_models"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True)
    description = Column(Text)
    model_type = Column(String(50))  # 'detection', 'segmentation', 'classification'
    framework = Column(String(50))  # 'jax', 'tensorflow', 'pytorch'
    version = Column(String(50))
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    inference_time_ms = Column(Float)
    model_size_mb = Column(Float)
    is_active = Column(Boolean, default=True)
    is_production = Column(Boolean, default=False)
    file_path = Column(String(512))
    s3_key = Column(String(512))
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)

    # Relationships
    versions = relationship("ModelVersion", back_populates="model")
    inference_results = relationship("InferenceResult", back_populates="model")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey("ml_models.id"))
    version = Column(String(50))
    file_path = Column(String(512))
    s3_key = Column(String(512))
    accuracy = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    deployed_at = Column(DateTime)
    is_production = Column(Boolean, default=False)
    changelog = Column(Text)

    # Relationships
    model = relationship("MLModel", back_populates="versions")


# ============================================================================
# 3. DATASET MODELS
# ============================================================================

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    description = Column(Text)
    owner_id = Column(Integer, ForeignKey("users.id"))
    total_images = Column(Integer, default=0)
    total_annotations = Column(Integer, default=0)
    size_mb = Column(Float, default=0.0)
    status = Column(String(50), default="collecting")  # 'collecting', 'ready', 'archived'
    created_at = Column(DateTime, default=datetime.utcnow)
    split_train = Column(Float, default=0.8)
    split_val = Column(Float, default=0.1)
    split_test = Column(Float, default=0.1)
    metadata = Column(JSON)

    # Relationships
    owner = relationship("User", back_populates="datasets")
    training_jobs = relationship("TrainingJob", back_populates="dataset")
    batch_jobs = relationship("BatchJob", back_populates="dataset")


# ============================================================================
# 4. TRAINING JOB MODELS
# ============================================================================

class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    model_type = Column(String(50))  # 'detection', 'classification', etc.
    job_name = Column(String(255))
    status = Column(String(50), default="queued")  # 'queued', 'running', 'completed', 'failed'
    epochs = Column(Integer)
    batch_size = Column(Integer)
    learning_rate = Column(Float)
    optimizer = Column(String(50), default="adam")
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    accuracy = Column(Float)
    loss = Column(Float)
    val_accuracy = Column(Float)
    val_loss = Column(Float)
    output_model_id = Column(Integer, ForeignKey("ml_models.id"))
    logs = Column(Text)  # Training logs
    error_message = Column(Text)

    # Relationships
    user = relationship("User", back_populates="training_jobs")
    dataset = relationship("Dataset", back_populates="training_jobs")


# ============================================================================
# 5. BATCH PROCESSING MODELS
# ============================================================================

class BatchJob(Base):
    __tablename__ = "batch_jobs"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    model_id = Column(Integer, ForeignKey("ml_models.id"))
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    status = Column(String(50), default="pending")  # 'pending', 'processing', 'completed', 'failed'
    total_items = Column(Integer)
    processed_items = Column(Integer, default=0)
    failed_items = Column(Integer, default=0)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    results_location = Column(String(512))  # S3 path or local path
    metadata = Column(JSON)


# ============================================================================
# 6. ANNOTATION MODELS
# ============================================================================

class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("image_assets.id"))
    annotator_id = Column(Integer, ForeignKey("users.id"))
    annotation_type = Column(String(50))  # 'bbox', 'polygon', 'point', 'classification'
    label = Column(String(255))
    confidence = Column(Float)
    data = Column(JSON)  # Coordinates, class, etc.
    created_at = Column(DateTime, default=datetime.utcnow)
    verified = Column(Boolean, default=False)
    verified_by = Column(Integer)
    verified_at = Column(DateTime)

    # Relationships
    image = relationship("ImageAsset", back_populates="annotations")


# ============================================================================
# 7. INFERENCE & ANALYTICS MODELS
# ============================================================================

class InferenceResult(Base):
    __tablename__ = "inference_results"

    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey("ml_models.id"))
    image_id = Column(Integer, ForeignKey("image_assets.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    predictions = Column(JSON)  # Detection results, classifications, etc.
    confidence_score = Column(Float)
    inference_time_ms = Column(Float)
    processed_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    model = relationship("MLModel", back_populates="inference_results")


class PerformanceMetric(Base):
    __tablename__ = "performance_metrics"

    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey("ml_models.id"))
    metric_name = Column(String(100))  # 'accuracy', 'precision', 'recall', 'f1'
    value = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    dataset_id = Column(Integer)
    metadata = Column(JSON)


# ============================================================================
# 8. A/B TESTING MODELS
# ============================================================================

class ABExperiment(Base):
    __tablename__ = "ab_experiments"

    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    description = Column(Text)
    model_a_id = Column(Integer, ForeignKey("ml_models.id"))
    model_b_id = Column(Integer, ForeignKey("ml_models.id"))
    status = Column(String(50), default="running")  # 'running', 'completed'
    sample_size = Column(Integer)
    confidence_level = Column(Float, default=0.95)
    total_predictions = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    winner_model_id = Column(Integer)
    p_value = Column(Float)
    results = Column(JSON)


# ============================================================================
# 9. NOTIFICATION MODELS
# ============================================================================

class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    type = Column(String(50))  # 'job_completed', 'model_deployed', 'error', etc.
    title = Column(String(255))
    message = Column(Text)
    read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    data = Column(JSON)  # Additional context (job_id, model_id, etc.)

    # Relationships
    user = relationship("User", back_populates="notifications")


# ============================================================================
# 10. API KEY MODELS
# ============================================================================

class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    key_hash = Column(String(255), unique=True, index=True)
    key_prefix = Column(String(10))  # First 10 chars for display
    name = Column(String(255))
    permissions = Column(JSON)  # ['read:inference', 'write:training', etc.]
    rate_limit = Column(Integer, default=100)  # requests per minute
    last_used = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    is_active = Column(Boolean, default=True)

    # Relationships
    user = relationship("User", back_populates="api_keys")


# ============================================================================
# 11. WEBHOOK MODELS
# ============================================================================

class Webhook(Base):
    __tablename__ = "webhooks"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    url = Column(String(512))
    events = Column(JSON)  # ['training.completed', 'inference.finished', etc.]
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_triggered = Column(DateTime)
    failure_count = Column(Integer, default=0)


class WebhookDelivery(Base):
    __tablename__ = "webhook_deliveries"

    id = Column(Integer, primary_key=True)
    webhook_id = Column(Integer, ForeignKey("webhooks.id"))
    event = Column(String(100))
    payload = Column(JSON)
    response_status = Column(Integer)
    response_body = Column(Text)
    delivered_at = Column(DateTime, default=datetime.utcnow)
    retry_count = Column(Integer, default=0)


# ============================================================================
# 12. SCHEDULED JOB MODELS
# ============================================================================

class ScheduledJob(Base):
    __tablename__ = "scheduled_jobs"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String(255))
    job_type = Column(String(50))  # 'inference', 'training', 'export', 'cleanup'
    cron_expression = Column(String(100))  # "0 9 * * *" (9am daily)
    config = Column(JSON)  # Job parameters
    is_active = Column(Boolean, default=True)
    last_run = Column(DateTime)
    next_run = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_result = Column(String(50))  # 'success', 'failed'
    last_error = Column(Text)


# ============================================================================
# 13. COST TRACKING MODELS
# ============================================================================

class CostMetric(Base):
    __tablename__ = "cost_metrics"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    operation = Column(String(50))  # 'inference', 'training', 'storage', 'bandwidth'
    cost = Column(Float, default=0.0)
    units = Column(Integer)  # images processed, training hours, GB stored
    unit_cost = Column(Float)  # Cost per unit
    date = Column(Date, default=datetime.utcnow)
    metadata = Column(JSON)


# ============================================================================
# 14. ANOMALY DETECTION MODELS
# ============================================================================

class AnomalyDetection(Base):
    __tablename__ = "anomaly_detections"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    image_id = Column(Integer, ForeignKey("image_assets.id"))
    model_id = Column(Integer, ForeignKey("ml_models.id"))
    anomaly_score = Column(Float)
    is_anomaly = Column(Boolean)
    heatmap_path = Column(String(512))
    detected_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)


# ============================================================================
# 15. EXPLAINABILITY (XAI) MODELS
# ============================================================================

class ExplainabilityReport(Base):
    __tablename__ = "explainability_reports"

    id = Column(Integer, primary_key=True)
    inference_id = Column(Integer, ForeignKey("inference_results.id"))
    method = Column(String(50))  # 'grad-cam', 'saliency', 'lime', 'shap'
    explanation_data = Column(JSON)  # Feature importance, attribution, etc.
    heatmap_path = Column(String(512))
    generated_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)


# ============================================================================
# 16. AUDIT LOG MODELS
# ============================================================================

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    action = Column(String(100))  # 'LIST_USERS', 'DELETE_USER', 'DEPLOY_MODEL', etc.
    resource = Column(String(255))  # What was modified
    resource_id = Column(Integer)
    status_code = Column(Integer)
    ip_address = Column(String(50))
    user_agent = Column(String(500))
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)


# ============================================================================
# 17. EXPORT JOB MODELS
# ============================================================================

class ExportJob(Base):
    __tablename__ = "export_jobs"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    source_type = Column(String(50))  # 'inference', 'dataset', 'training'
    source_id = Column(Integer)
    format = Column(String(50))  # 'json', 'csv', 'xml', 'parquet'
    status = Column(String(50), default="pending")  # 'pending', 'processing', 'completed', 'failed'
    output_path = Column(String(512))
    s3_key = Column(String(512))
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    file_size_mb = Column(Float)


# ============================================================================
# 18. STREAMING SESSION MODELS
# ============================================================================

class StreamingSession(Base):
    __tablename__ = "streaming_sessions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    model_id = Column(Integer, ForeignKey("ml_models.id"))
    source = Column(String(255))  # 'webcam', 'rtsp://...', 'http://...'
    status = Column(String(50), default="inactive")  # 'active', 'paused', 'stopped'
    frames_processed = Column(Integer, default=0)
    detections = Column(Integer, default=0)
    started_at = Column(DateTime)
    stopped_at = Column(DateTime)
    avg_fps = Column(Float)
    metadata = Column(JSON)


# ============================================================================
# 19. CUSTOM MODEL TRAINING MODELS
# ============================================================================

class CustomModelTraining(Base):
    __tablename__ = "custom_model_training"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String(255))
    description = Column(Text)
    base_model_id = Column(Integer, ForeignKey("ml_models.id"))
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    architecture = Column(String(255))  # 'resnet50', 'mobilenet', 'custom'
    hyperparameters = Column(JSON)
    status = Column(String(50), default="pending")  # 'pending', 'training', 'completed', 'failed'
    accuracy = Column(Float)
    loss = Column(Float)
    output_model_id = Column(Integer, ForeignKey("ml_models.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)


# ============================================================================
# 20. INVOICE & BILLING MODELS
# ============================================================================

class Invoice(Base):
    __tablename__ = "invoices"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    invoice_number = Column(String(50), unique=True)
    period_start = Column(Date)
    period_end = Column(Date)
    total_cost = Column(Float)
    subtotal = Column(Float)
    tax = Column(Float)
    status = Column(String(50), default="draft")  # 'draft', 'sent', 'paid', 'overdue'
    due_date = Column(Date)
    paid_date = Column(Date)
    issued_at = Column(DateTime, default=datetime.utcnow)
    items = Column(JSON)  # Line items
    metadata = Column(JSON)
