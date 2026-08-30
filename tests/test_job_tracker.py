"""
Test suite for Job Tracker and Dashboard modules
"""

import sys
from pathlib import Path
import tempfile
import shutil
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from sentinel.cli.job_tracker import (
    JobTracker, JobStatus, TrainingJob, TrainingMetrics, MetricsCollector
)
from sentinel.cli.dashboard import TerminalDashboard, MetricsFormatter, ProgressIndicator


def test_create_job():
    """Test creating a training job"""
    tracker = JobTracker()
    job = tracker.create_job(
        job_id="test_001",
        model_name="cnn",
        dataset_path="./images",
        epochs=10,
        batch_size=32,
        learning_rate=0.001,
        gpu_enabled=True
    )

    assert job.job_id == "test_001"
    assert job.model_name == "cnn"
    assert job.status == JobStatus.PENDING
    assert job.epochs == 10
    assert job.gpu_enabled == True
    print("✓ Job creation works")


def test_job_status_transitions():
    """Test job status transitions"""
    tracker = JobTracker()
    job = tracker.create_job(
        job_id="test_002",
        model_name="cnn",
        dataset_path="./images",
        epochs=5,
        batch_size=32,
        learning_rate=0.001
    )

    # Start job
    tracker.start_job("test_002")
    assert job.status == JobStatus.RUNNING
    assert job.started_at is not None

    # Complete job
    tracker.complete_job("test_002")
    assert job.status == JobStatus.COMPLETED
    assert job.completed_at is not None
    assert job.progress_percent == 100.0

    print("✓ Job status transitions work")


def test_add_metrics():
    """Test adding metrics to job"""
    tracker = JobTracker()
    tracker.create_job(
        job_id="test_003",
        model_name="cnn",
        dataset_path="./images",
        epochs=5,
        batch_size=32,
        learning_rate=0.001
    )

    tracker.start_job("test_003")

    # Add metrics for first epoch
    metric1 = TrainingMetrics(
        epoch=1,
        loss=2.5,
        accuracy=0.3,
        val_loss=2.4,
        val_accuracy=0.32,
        timestamp="2026-08-30T10:00:00"
    )

    job = tracker.add_metrics("test_003", metric1)

    assert len(job.metrics) == 1
    assert job.current_epoch == 1
    assert job.progress_percent == 20.0  # 1/5 epochs

    # Add second epoch
    metric2 = TrainingMetrics(
        epoch=2,
        loss=2.0,
        accuracy=0.5,
        val_loss=1.9,
        val_accuracy=0.52,
        timestamp="2026-08-30T10:01:00"
    )

    job = tracker.add_metrics("test_003", metric2)

    assert len(job.metrics) == 2
    assert job.current_epoch == 2
    assert job.progress_percent == 40.0  # 2/5 epochs

    print("✓ Adding metrics works")


def test_metrics_collector():
    """Test metrics collector"""
    collector = MetricsCollector()

    # Add metrics
    collector.add_metric(epoch=1, loss=2.5, accuracy=0.3, val_loss=2.4, val_accuracy=0.32)
    collector.add_metric(epoch=2, loss=2.0, accuracy=0.5, val_loss=1.9, val_accuracy=0.52)
    collector.add_metric(epoch=3, loss=1.5, accuracy=0.7, val_loss=1.4, val_accuracy=0.72)

    assert len(collector.get_metrics()) == 3

    # Test best metrics
    assert collector.get_best_loss() == 1.5
    assert collector.get_best_accuracy() == 0.7

    # Test trend detection
    trend = collector.get_trend("loss")
    assert "improving" in trend or "declining" in trend

    print("✓ Metrics collector works")


def test_job_persistence():
    """Test job persistence to disk"""
    tmpdir = tempfile.mkdtemp(prefix="test_jobs_")
    try:
        tracker = JobTracker(storage_dir=tmpdir)

        # Create job
        tracker.create_job(
            job_id="persist_001",
            model_name="cnn",
            dataset_path="./images",
            epochs=5,
            batch_size=32,
            learning_rate=0.001
        )

        # Verify file exists
        job_file = Path(tmpdir) / "persist_001.json"
        assert job_file.exists()

        # Create new tracker and load
        tracker2 = JobTracker(storage_dir=tmpdir)
        job = tracker2.get_job("persist_001")

        assert job is not None
        assert job.model_name == "cnn"

        print("✓ Job persistence works")

    finally:
        shutil.rmtree(tmpdir)


def test_job_listing():
    """Test listing jobs"""
    tracker = JobTracker()

    # Create multiple jobs
    for i in range(3):
        tracker.create_job(
            job_id=f"list_test_{i}",
            model_name="cnn",
            dataset_path="./images",
            epochs=5,
            batch_size=32,
            learning_rate=0.001
        )

    jobs = tracker.list_jobs()
    assert len(jobs) >= 3

    # Filter by status
    pending_jobs = tracker.list_jobs(status=JobStatus.PENDING)
    assert len(pending_jobs) >= 3

    # Start one job
    tracker.start_job("list_test_0")
    running_jobs = tracker.list_jobs(status=JobStatus.RUNNING)
    assert len(running_jobs) >= 1

    print("✓ Job listing works")


def test_terminal_dashboard():
    """Test terminal dashboard rendering"""
    tracker = JobTracker()
    job = tracker.create_job(
        job_id="dash_001",
        model_name="cnn",
        dataset_path="./images",
        epochs=5,
        batch_size=32,
        learning_rate=0.001
    )

    tracker.start_job("dash_001")

    # Add metrics
    from sentinel.cli.job_tracker import MetricsCollector
    collector = MetricsCollector()
    collector.add_metric(epoch=1, loss=2.5, accuracy=0.3, val_loss=2.4, val_accuracy=0.32)

    tracker.add_metrics("dash_001", collector.get_metrics()[0])

    # Create dashboard
    dashboard = TerminalDashboard(job, collector)

    # Test rendering
    header = dashboard.render_header()
    assert "Dashboard" in header

    info = dashboard.render_job_info()
    assert "dash_001" in info
    assert "cnn" in info

    progress = dashboard.render_progress_bar()
    assert "[" in progress and "]" in progress

    metrics = dashboard.render_metrics()
    assert "Current Metrics" in metrics or "No metrics" in metrics

    full = dashboard.render_full(show_chart=False)
    assert len(full) > 100

    print("✓ Terminal dashboard rendering works")


def test_metrics_formatter():
    """Test metrics formatting"""
    assert MetricsFormatter.format_metric(0.12345, decimals=2) == "0.12"
    assert MetricsFormatter.format_percentage(0.5, decimals=1) == "50.0%"
    assert "s" in MetricsFormatter.format_time(30)
    assert "m" in MetricsFormatter.format_time(300)

    print("✓ Metrics formatter works")


def test_progress_indicator():
    """Test progress indicator"""
    indicator = ProgressIndicator(total=100, desc="Training")

    indicator.update(25)
    progress = indicator.render(width=20)
    assert "25" in progress

    indicator.update(100)
    progress = indicator.render()
    assert "100" in progress

    print("✓ Progress indicator works")


def test_error_handling():
    """Test error handling"""
    tracker = JobTracker()

    # Try to get nonexistent job
    job = tracker.get_job("nonexistent")
    assert job is None

    # Try to add metrics to nonexistent job
    try:
        metric = TrainingMetrics(
            epoch=1, loss=2.5, accuracy=0.3,
            val_loss=2.4, val_accuracy=0.32,
            timestamp="2026-08-30T10:00:00"
        )
        tracker.add_metrics("nonexistent", metric)
        assert False, "Should raise ValueError"
    except ValueError:
        pass

    print("✓ Error handling works")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Job Tracker and Dashboard Modules")
    print("="*60 + "\n")

    tests = [
        test_create_job,
        test_job_status_transitions,
        test_add_metrics,
        test_metrics_collector,
        test_job_persistence,
        test_job_listing,
        test_terminal_dashboard,
        test_metrics_formatter,
        test_progress_indicator,
        test_error_handling,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")

    sys.exit(0 if failed == 0 else 1)
