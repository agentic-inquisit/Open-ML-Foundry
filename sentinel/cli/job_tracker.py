"""
Job Tracker Module
Tracks training jobs, metrics, and status
"""

import json
from datetime import datetime
from pathlib import Path
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import time


class JobStatus(Enum):
    """Training job status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingMetrics:
    """Single epoch metrics"""
    epoch: int
    loss: float
    accuracy: float
    val_loss: float
    val_accuracy: float
    timestamp: str

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainingJob:
    """Training job information"""
    job_id: str
    model_name: str
    dataset_path: str
    status: JobStatus
    epochs: int
    batch_size: int
    learning_rate: float
    gpu_enabled: bool
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    current_epoch: int = 0
    total_epochs: int = 0
    progress_percent: float = 0.0
    metrics: List[TrainingMetrics] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = []

    def to_dict(self):
        return {
            "job_id": self.job_id,
            "model_name": self.model_name,
            "dataset_path": self.dataset_path,
            "status": self.status.value,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "gpu_enabled": self.gpu_enabled,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "progress_percent": self.progress_percent,
            "metrics": [m.to_dict() for m in self.metrics]
        }


class JobTracker:
    """Tracks training jobs and metrics"""

    def __init__(self, storage_dir: Optional[str] = None):
        """Initialize job tracker"""
        self.storage_dir = Path(storage_dir) if storage_dir else Path.home() / ".sentinel" / "jobs"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: Dict[str, TrainingJob] = {}
        self._load_jobs()

    def create_job(
        self,
        job_id: str,
        model_name: str,
        dataset_path: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        gpu_enabled: bool = False
    ) -> TrainingJob:
        """Create a new training job"""
        job = TrainingJob(
            job_id=job_id,
            model_name=model_name,
            dataset_path=dataset_path,
            status=JobStatus.PENDING,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            gpu_enabled=gpu_enabled,
            created_at=datetime.now().isoformat(),
            total_epochs=epochs
        )
        self.jobs[job_id] = job
        self._save_job(job)
        return job

    def start_job(self, job_id: str) -> TrainingJob:
        """Mark job as started"""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")

        job = self.jobs[job_id]
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now().isoformat()
        self._save_job(job)
        return job

    def set_progress(self, job_id: str, current_epoch: float) -> TrainingJob:
        """Update progress TrainingMetrics record.
        """
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")

        job = self.jobs[job_id]
        job.current_epoch = current_epoch
        job.progress_percent = (
            min(100.0, (current_epoch / job.total_epochs) * 100.0) if job.total_epochs else 0.0
        )
        self._save_job(job)
        return job

    def add_metrics(self, job_id: str, metrics: TrainingMetrics) -> TrainingJob:
        """Add metrics for current epoch"""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")

        job = self.jobs[job_id]
        job.metrics.append(metrics)
        job.current_epoch = metrics.epoch
        job.progress_percent = (metrics.epoch / job.total_epochs) * 100.0
        self._save_job(job)
        return job

    def complete_job(self, job_id: str) -> TrainingJob:
        """Mark job as completed"""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")

        job = self.jobs[job_id]
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now().isoformat()
        job.progress_percent = 100.0
        self._save_job(job)
        return job

    def fail_job(self, job_id: str, error: str) -> TrainingJob:
        """Mark job as failed"""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")

        job = self.jobs[job_id]
        job.status = JobStatus.FAILED
        job.completed_at = datetime.now().isoformat()
        # Store error in metrics or separate field
        self._save_job(job)
        return job

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get job by ID"""
        return self.jobs.get(job_id)

    def list_jobs(self, status: Optional[JobStatus] = None) -> List[TrainingJob]:
        """List all jobs, optionally filtered by status"""
        jobs = list(self.jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def _save_job(self, job: TrainingJob):
        """Save job to disk"""
        job_file = self.storage_dir / f"{job.job_id}.json"
        job_file.write_text(json.dumps(job.to_dict(), indent=2))

    def _load_jobs(self):
        """Load jobs from disk"""
        if not self.storage_dir.exists():
            return

        for job_file in self.storage_dir.glob("*.json"):
            try:
                data = json.loads(job_file.read_text())
                # Reconstruct job object
                metrics = [TrainingMetrics(**m) for m in data.get("metrics", [])]
                data["metrics"] = metrics
                data["status"] = JobStatus(data["status"])
                job = TrainingJob(**data)
                self.jobs[job.job_id] = job
            except Exception as e:
                print(f"Warning: Could not load job {job_file}: {e}")

    def get_latest_job(self) -> Optional[TrainingJob]:
        """Get most recent job"""
        jobs = self.list_jobs()
        return jobs[0] if jobs else None

    def get_running_job(self) -> Optional[TrainingJob]:
        """Get currently running job"""
        running = self.list_jobs(status=JobStatus.RUNNING)
        return running[0] if running else None


class MetricsCollector:
    """Collects and aggregates training metrics"""

    def __init__(self):
        """Initialize metrics collector"""
        self.metrics: List[TrainingMetrics] = []

    def add_metric(self, epoch: int, loss: float, accuracy: float,
                   val_loss: float, val_accuracy: float):
        """Add metric for epoch"""
        metric = TrainingMetrics(
            epoch=epoch,
            loss=loss,
            accuracy=accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            timestamp=datetime.now().isoformat()
        )
        self.metrics.append(metric)
        return metric

    def get_metrics(self) -> List[TrainingMetrics]:
        """Get all metrics"""
        return self.metrics

    def get_latest_metric(self) -> Optional[TrainingMetrics]:
        """Get most recent metric"""
        return self.metrics[-1] if self.metrics else None

    def get_best_loss(self) -> float:
        """Get best (lowest) loss seen"""
        if not self.metrics:
            return float('inf')
        return min(m.loss for m in self.metrics)

    def get_best_accuracy(self) -> float:
        """Get best (highest) accuracy seen"""
        if not self.metrics:
            return 0.0
        return max(m.accuracy for m in self.metrics)

    def get_trend(self, metric_name: str, window: int = 5) -> str:
        """Get trend (improving/stable/declining) for metric"""
        if len(self.metrics) < 2:
            return "→ stable"

        recent = self.metrics[-window:]
        if metric_name == "loss":
            values = [m.loss for m in recent]
        elif metric_name == "accuracy":
            values = [m.accuracy for m in recent]
        else:
            return "→ unknown"

        # Calculate trend
        if len(values) < 2:
            return "→ stable"

        trend = values[-1] - values[0]
        if metric_name == "loss":
            if trend < -0.01:
                return "↓ improving"
            elif trend > 0.01:
                return "↑ declining"
        else:  # accuracy
            if trend > 0.01:
                return "↑ improving"
            elif trend < -0.01:
                return "↓ declining"

        return "→ stable"

    def estimate_time_remaining(self, start_time: float, current_epoch: int,
                               total_epochs: int) -> str:
        """Estimate time remaining based on epoch duration"""
        if current_epoch == 0:
            return "calculating..."

        elapsed = time.time() - start_time
        time_per_epoch = elapsed / current_epoch
        remaining_epochs = total_epochs - current_epoch
        remaining_seconds = time_per_epoch * remaining_epochs

        minutes = int(remaining_seconds / 60)
        seconds = int(remaining_seconds % 60)

        if minutes > 60:
            hours = minutes // 60
            mins = minutes % 60
            return f"{hours}h {mins}m remaining"
        elif minutes > 0:
            return f"{minutes}m {seconds}s remaining"
        else:
            return f"{seconds}s remaining"
