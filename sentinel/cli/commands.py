"""
CLI Commands for Open ML Foundry
"""

import click
from pathlib import Path
import json
from datetime import datetime
import uuid


# ============================================================================
# MODEL COMMANDS
# ============================================================================

@click.group(name="model")
def model_group():
    """Manage imported models"""
    pass


@model_group.command(name="import")
@click.option("--path", "-p", required=True, type=click.Path(exists=True),
              help="Path to model file (.pth, .pt, or HuggingFace model ID)")
@click.option("--name", "-n", required=True,
              help="Model name for reference (e.g., 'custom_resnet')")
@click.option("--type", "-t", type=click.Choice(["pytorch", "huggingface", "onnx", "tensorflow"]),
              default="pytorch", help="Model type (default: pytorch)")
def import_model(path, name, type):
    """
    Register an externally-sourced model in the model registry so it shows
    up in `sentinel model list`/`info` and can be referenced by name.

    Backed by edge/model_registry.py's ModelRegistry (model_registry.db) —
    the same store `/finetune` (edge/vision_module.py) and `sentinel train
    start` write completed training runs into, so imported and trained
    models both show up in one place. This records the import; it doesn't
    validate or convert the model file, and `sentinel train start --model
    <name>` still needs a real training path (currently: the built-in CNN
    for vision, or an LLM catalog entry) to actually use it.

    Examples:
        sentinel model import --path ./my_model.pth --name custom_resnet
        sentinel model import --path openai/clip-vit-base-patch32 --name clip_v1 --type huggingface
    """
    from edge.model_registry import ModelRegistry

    click.echo(f"\n📦 Importing model: {name}")
    click.echo(f"   Path: {path}")
    click.echo(f"   Type: {type}")

    registry = ModelRegistry()
    reg_result = registry.register_model(
        model_name=name,
        description=f"Imported {type} model from {path}",
        owner="cli",
        access_level="private",
    )
    registry.add_history_event(
        reg_result["model_id"], "model_imported", f"path={path}, type={type}", "cli"
    )

    click.echo(f"\n✓ Registered in model_registry.db: {name} {reg_result['version']} "
              f"(id={reg_result['model_id']})")
    click.echo(f"  Use in training: sentinel train start --model {name} --dataset <path>")


@model_group.command(name="list")
def list_models():
    """List models — what `sentinel train start` can actually run, plus
    everything on record in the model registry."""
    from edge.model_registry import ModelRegistry
    from llm import supported_models

    click.echo("\n📦 Built-in (vision, trains from scratch via JAX/Flax):")
    click.echo("    cnn                  — 3-conv-layer classifier")

    click.echo("\n📦 LLM catalog (LoRA/QLoRA fine-tuning):")
    for m in supported_models.list_models():
        tag = "verified" if m["verified"] else "repo id unconfirmed"
        click.echo(f"    {m['display_name']:20s} — {m['family']}, {m['size_hint']} ({tag})")

    registry = ModelRegistry()
    registered = registry.get_all_models()
    click.echo(f"\n📦 Registered (model_registry.db — imports and completed training runs):")
    if not registered:
        click.echo("    (none yet — 'sentinel model import' or a completed training run adds one)")
    else:
        for entry in registered:
            latest = entry["versions"][0] if entry["versions"] else None
            if latest:
                click.echo(f"    {entry['name']:20s} — {entry['version_count']} version(s), "
                          f"latest {latest['version']} ({latest['status']})")


@model_group.command(name="info")
@click.argument("model_name")
def model_info(model_name):
    """Show model details — checks the built-in CNN, then the LLM catalog,
    then model_registry.db, in that order."""
    from edge.model_registry import ModelRegistry
    from llm import supported_models

    if model_name == "cnn":
        click.echo(f"\n📋 Model: cnn (built-in)")
        click.echo(f"   Type: Classification")
        click.echo(f"   Framework: JAX/Flax")
        click.echo(f"   Architecture: 3x (conv → relu → avg_pool) → dense(256) → dense(num_classes)")
        click.echo(f"   Pretrained: No — trains from scratch each run")
        click.echo(f"   Input: Images, default 224x224 (--image-size to change)")
        click.echo(f"   Latency: not benchmarked")
        return

    llm_entry = next(
        (m for m in supported_models.list_models()
         if model_name in (m["display_name"], m["repo_id"])),
        None,
    )
    if llm_entry:
        tag = "verified" if llm_entry["verified"] else "unconfirmed — check on Hugging Face before use"
        click.echo(f"\n📋 Model: {llm_entry['display_name']}")
        click.echo(f"   Family: {llm_entry['family']}")
        click.echo(f"   Repo ID: {llm_entry['repo_id']} ({tag})")
        click.echo(f"   Size: {llm_entry['size_hint']}")
        click.echo(f"   GGUF support: {'Yes' if llm_entry['supports_gguf'] else 'No'}")
        if llm_entry["notes"]:
            click.echo(f"   Notes: {llm_entry['notes']}")
        return

    registry = ModelRegistry()
    versions = registry.get_model_versions(model_name)
    if not versions:
        click.echo(f"❌ Model '{model_name}' not found (checked built-ins, the LLM catalog, "
                  f"and model_registry.db)")
        return

    click.echo(f"\n📋 Model: {model_name} — {len(versions)} version(s) in model_registry.db")
    for v in versions:
        click.echo(f"\n   {v['version']}  (id={v['model_id']}, status={v['status']})")
        click.echo(f"     Created: {v['created_at']}")
        if v["description"]:
            click.echo(f"     Description: {v['description']}")
        meta = v["metadata"]
        if meta.get("epochs_trained"):
            click.echo(f"     Trained: {meta['epochs_trained']} epochs, "
                      f"final_train_loss={meta['final_train_loss']}, "
                      f"best_val_loss={meta['best_val_loss']}")
        ds = v["dataset"]
        if ds.get("total_images"):
            click.echo(f"     Dataset: {ds['total_images']} images, classes={ds['classes']}")


# ============================================================================
# DATASET COMMANDS
# ============================================================================

from sentinel.cli.dataset_browser import DatasetBrowser

@click.group(name="dataset")
def dataset_group():
    """Manage datasets"""
    pass


@dataset_group.command(name="prepare")
@click.option("--path", "-p", required=True, type=click.Path(exists=True),
              help="Path to dataset folder")
@click.option("--split", "-s", nargs=3, type=float, default=[0.8, 0.1, 0.1],
              help="Train/val/test split (default: 0.8 0.1 0.1)")
@click.option("--preview", is_flag=True, help="Show sample images")
@click.option("--report", is_flag=True, help="Show detailed analysis")
def prepare_dataset(path, split, preview, report):
    """
    Prepare dataset for training with advanced validation

    Examples:
        sentinel dataset prepare --path ./my_images
        sentinel dataset prepare --path ./my_images --report --preview
        sentinel dataset prepare --path ./my_images --split 0.7 0.15 0.15
    """
    try:
        # Use DatasetBrowser for detailed analysis
        browser = DatasetBrowser(path)
        browser.scan()

        # Show report if requested
        if report:
            browser.print_report()

        # Standard output
        summary = browser.get_summary()

        click.echo(f"\n📂 Scanning dataset: {Path(path).name}")
        click.echo(f"   Found: {summary['total_images']} images")
        click.echo(f"   Structure: {summary['structure'].replace('_', ' ').title()}")

        # Calculate splits
        total = summary['total_images']
        train_count = int(total * split[0])
        val_count = int(total * split[1])
        test_count = total - train_count - val_count

        click.echo(f"\n✓ Dataset prepared:")
        click.echo(f"   Train: {train_count} images ({split[0]*100:.0f}%)")
        click.echo(f"   Val:   {val_count} images ({split[1]*100:.0f}%)")
        click.echo(f"   Test:  {test_count} images ({split[2]*100:.0f}%)")

        # Show class distribution if multiple classes
        if summary['num_classes'] > 1:
            click.echo(f"\n🏷️ Classes ({summary['num_classes']}):")
            for cls, count in summary['classes'].items():
                percentage = (count / total) * 100
                click.echo(f"   • {cls}: {count} images ({percentage:.1f}%)")

        # Show warnings
        if summary['warnings']:
            click.echo(f"\n⚠️ Warnings:")
            for warning in summary['warnings']:
                click.echo(f"   {warning}")

        if preview:
            samples = browser.get_sample_images(3)
            click.echo(f"\n📸 Sample images:")
            for img in samples:
                click.echo(f"   - {Path(img).name}")

    except FileNotFoundError as e:
        click.echo(f"\n❌ Error: {e}", err=True)
    except ValueError as e:
        click.echo(f"\n❌ Error: {e}", err=True)


@dataset_group.command(name="info")
@click.argument("path", type=click.Path(exists=True))
@click.option("--export", "-e", type=click.Path(), help="Export metadata to JSON file")
def dataset_info(path, export):
    """
    Show detailed dataset information and statistics

    Examples:
        sentinel dataset info ./my_dataset
        sentinel dataset info ./my_dataset --export dataset.json
    """
    try:
        browser = DatasetBrowser(path)
        browser.scan()
        browser.print_report()

        if export:
            export_path = browser.export_metadata(export)
            click.echo(f"✓ Metadata exported to: {export_path}")

    except FileNotFoundError as e:
        click.echo(f"\n❌ Error: {e}", err=True)
    except ValueError as e:
        click.echo(f"\n❌ Error: {e}", err=True)


# ============================================================================
# TRAINING COMMANDS
# ============================================================================

@click.group(name="train")
def train_group():
    """Train and fine-tune models"""
    pass


@train_group.command(name="start")
@click.option("--model", "-m", required=True,
              help="'cnn' (built-in vision), an LLM catalog name/repo id, or a name "
                   "already in model_registry.db")
@click.option("--dataset", "-d", required=True,
              help="Vision: directory of class subfolders (dataset/<class>/<images>). "
                   "LLM: JSONL file.")
@click.option("--epochs", "-e", type=int, default=5, help="Number of epochs (default: 5)")
@click.option("--batch-size", "-b", type=int, default=4, help="Batch size (default: 4)")
@click.option("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
@click.option("--gpu", is_flag=True, help="Record GPU intent on the job (device placement "
                                          "is chosen automatically either way)")
@click.option("--num-classes", type=int, default=None,
              help="Vision only. Omit to derive from the dataset's class subfolders.")
@click.option("--image-size", type=int, default=224, help="Vision only (default: 224)")
@click.option("--target-object", default=None, help="Vision only. Defaults to --model.")
@click.option("--model-format", type=click.Choice(["huggingface", "gguf"]), default="huggingface",
              help="LLM only (default: huggingface)")
@click.option("--enable-validation/--no-validation", default=True,
              help="Vision only: hold out part of the dataset for validation (default: on)")
@click.option("--live", is_flag=True, help="Render the full terminal dashboard after training completes")
@click.option("--job-id", default=None, help="Custom job ID (optional)")
def train_start(model, dataset, epochs, batch_size, lr, gpu, num_classes, image_size,
                target_object, model_format, enable_validation, live, job_id):
    """
    Vision models train via edge/jax_train.py against a dataset directory of 
    class subfolders. LLM catalog models train via llm/lora_trainer.py (LoRA/QLoRA) 
    against a JSONL dataset. 

    Examples:
        sentinel train start --model cnn --dataset ./images --epochs 10
        sentinel train start --model Qwen3.8 --dataset ./chat.jsonl --live
        sentinel train start --model cnn --dataset ./images --job-id exp_001
    """
    from sentinel.cli.job_tracker import JobTracker, MetricsCollector, TrainingMetrics
    from sentinel.cli.dashboard import TerminalDashboard
    from llm import supported_models

    if not job_id:
        job_id = f"job_{int(uuid.uuid4().int / 1e10)}"

    llm_entry = next(
        (m for m in supported_models.list_models()
         if model in (m["display_name"], m["repo_id"])),
        None,
    )
    kind = "llm" if llm_entry else "vision"

    click.echo(f"\n🚀 Starting training")
    click.echo(f"   Job ID: {job_id}")
    click.echo(f"   Model: {model} ({kind})")
    click.echo(f"   Dataset: {dataset}")
    click.echo(f"   Epochs: {epochs}")
    click.echo(f"   Batch size: {batch_size}")
    click.echo(f"   Learning rate: {lr}")

    tracker = JobTracker()
    job = tracker.create_job(
        job_id=job_id, model_name=model, dataset_path=dataset,
        epochs=epochs, batch_size=batch_size, learning_rate=lr, gpu_enabled=gpu,
    )
    click.echo(f"\n✓ Job created and tracked")
    click.echo(f"   Job stored in: {tracker.storage_dir / f'{job_id}.json'}")

    tracker.start_job(job_id)
    metrics_collector = MetricsCollector()

    try:
        if kind == "llm":
            from llm import model_loader, lora_trainer

            click.echo(f"\n⏳ Loading {model} ({model_format})...")
            loaded = model_loader.load(model, model_format)

            def on_progress(update: dict) -> None:
                loss = update.get("loss")
                if loss is None:
                    return
                epoch = update.get("epoch") or 0.0
                tracker.set_progress(job_id, epoch)
                click.echo(f"   step {update.get('step')} | epoch {epoch:.2f} | loss {loss:.4f}")

            config = lora_trainer.LoRAConfig(
                epochs=epochs, learning_rate=lr, batch_size=batch_size,
                output_dir=f"training_outputs/lora/{job_id}",
            )
            result = lora_trainer.train(loaded, dataset, config, on_progress=on_progress)

            click.echo(f"\n✓ Training complete")
            click.echo(f"   Final loss: {result.get('final_loss')}")
            click.echo(f"   Adapter saved to: {result.get('adapter_path')}")

        else:
            from edge import jax_train

            if model != "cnn":
                click.echo(f"\n⚠ '{model}' isn't the built-in cnn or an LLM catalog entry — "
                          f"edge/jax_train.py only implements one vision architecture (the "
                          f"built-in CNN), so this trains that, not whatever '{model}' was "
                          f"registered/imported as. The job/registry record its name as "
                          f"'{model}' for tracking, not as a claim it's a different architecture.")

            click.echo(f"\n⏳ Training (progress prints below as each epoch finishes)...")
            result = jax_train.run_finetuning(
                dataset_dir=dataset,
                target_object=target_object or model,
                steps=epochs, batch_size=batch_size,
                num_classes=num_classes, image_size=image_size,
                enable_validation=enable_validation,
                checkpoint_dir=f"training_outputs/vision/{job_id}",
            )
            if result.get("status") == "error":
                raise RuntimeError(result.get("message", "Vision training failed"))

            train_losses = result.get("train_loss_history", [])
            train_accs = result.get("train_acc_history", [])
            val_losses = result.get("val_loss_history", [])
            val_accs = result.get("val_acc_history", [])
            for i in range(len(train_losses)):
                metric = TrainingMetrics(
                    epoch=i + 1,
                    loss=train_losses[i],
                    accuracy=train_accs[i] if i < len(train_accs) else 0.0,
                    val_loss=val_losses[i] if i < len(val_losses) else 0.0,
                    val_accuracy=val_accs[i] if i < len(val_accs) else 0.0,
                    timestamp=datetime.now().isoformat(),
                )
                tracker.add_metrics(job_id, metric)
                metrics_collector.metrics.append(metric)

            click.echo(f"\n✓ Training complete")
            click.echo(f"   Final train loss: {result.get('final_train_loss')}")
            click.echo(f"   Best val loss: {result.get('best_val_loss')}")
            click.echo(f"   Checkpoint: {result.get('best_checkpoint')}")

        tracker.complete_job(job_id)

    except Exception as e:
        tracker.fail_job(job_id, str(e))
        click.echo(f"\n❌ Training failed: {e}", err=True)
        return

    if live:
        if kind == "llm":
            click.echo(f"\n(--live shows the full metrics dashboard for vision jobs only — "
                      f"LLM training here only reports loss, not accuracy, so there's no "
                      f"honest chart to draw. Loss was streamed above as training ran.)")
        else:
            job = tracker.get_job(job_id)  # reload with final state
            dashboard = TerminalDashboard(job, metrics_collector)
            click.echo(dashboard.render_full(show_chart=True))

    click.echo(f"\n   View job: sentinel train status {job_id}")
    click.echo(f"   List jobs: sentinel train list")


@train_group.command(name="status")
@click.argument("job_id")
def train_status(job_id):
    """
    Check training job status (Phase 3)

    Examples:
        sentinel train status job_12345
        sentinel train status exp_001
    """
    from sentinel.cli.job_tracker import JobTracker
    from sentinel.cli.dashboard import TerminalDashboard, MetricsFormatter
    from sentinel.cli.job_tracker import MetricsCollector

    tracker = JobTracker()
    job = tracker.get_job(job_id)

    if not job:
        click.echo(f"\n❌ Job {job_id} not found")
        click.echo(f"\n📋 Available jobs:")
        for j in tracker.list_jobs()[:5]:
            click.echo(f"   • {j.job_id} ({j.status.value}) - {j.model_name}")
        return

    click.echo(f"\n📊 Job Status")
    click.echo(f"   Job ID:   {job.job_id}")
    click.echo(f"   Status:   {job.status.value.upper()}")
    click.echo(f"   Model:    {job.model_name}")
    click.echo(f"   Dataset:  {job.dataset_path}")
    click.echo(f"   Epoch:    {job.current_epoch}/{job.total_epochs}")
    click.echo(f"   Progress: {job.progress_percent:.1f}%")

    if job.metrics:
        latest = job.metrics[-1]
        click.echo(f"\n📈 Latest Metrics:")
        click.echo(f"   Loss:     {latest.loss:.4f}")
        click.echo(f"   Accuracy: {MetricsFormatter.format_percentage(latest.accuracy)}")
        click.echo(f"   Val Loss: {latest.val_loss:.4f}")
        click.echo(f"   Val Acc:  {MetricsFormatter.format_percentage(latest.val_accuracy)}")

    if job.started_at:
        click.echo(f"\n⏱️ Timing:")
        click.echo(f"   Started: {job.started_at}")
        if job.completed_at:
            click.echo(f"   Completed: {job.completed_at}")


@train_group.command(name="list")
@click.option("--status", "-s", type=click.Choice(["pending", "running", "completed", "failed"]),
              default=None, help="Filter by status")
@click.option("--limit", "-l", type=int, default=10, help="Limit results (default: 10)")
def train_list(status, limit):
    """
    List all training jobs (Phase 3)

    Examples:
        sentinel train list
        sentinel train list --status running
        sentinel train list --status completed --limit 5
    """
    from sentinel.cli.job_tracker import JobTracker, JobStatus

    tracker = JobTracker()

    # Filter by status if requested
    filter_status = JobStatus(status) if status else None
    jobs = tracker.list_jobs(status=filter_status)[:limit]

    if not jobs:
        click.echo(f"\n📭 No jobs found")
        if status:
            click.echo(f"   (with status: {status})")
        return

    click.echo(f"\n📋 Training Jobs ({len(jobs)} of {len(tracker.jobs)})")
    click.echo("="*70)

    for job in jobs:
        # Status symbol
        status_symbol = {
            "pending": "⏳",
            "running": "🔄",
            "completed": "✓",
            "failed": "✗",
            "cancelled": "⊘"
        }.get(job.status.value, "?")

        click.echo(f"{status_symbol} {job.job_id}")
        click.echo(f"   Model:    {job.model_name}")
        click.echo(f"   Dataset:  {job.dataset_path}")
        click.echo(f"   Progress: {job.current_epoch}/{job.total_epochs} epochs ({job.progress_percent:.0f}%)")

        if job.metrics:
            latest = job.metrics[-1]
            click.echo(f"   Latest:   Loss={latest.loss:.4f}, Acc={latest.accuracy*100:.1f}%")

        click.echo()


# ============================================================================
# HELP COMMANDS
# ============================================================================

@click.command()
def examples():
    """Show usage examples"""
    click.echo("""
    Open ML Foundry - CLI Examples

    1. List available models:
       sentinel model list

    2. Import a PyTorch model:
       sentinel model import --path ./my_model.pth --name custom_resnet

    3. Import a HuggingFace model:
       sentinel model import --path openai/clip-vit-base-patch32 --name clip_v1 --type huggingface

    4. Prepare a dataset:
       sentinel dataset prepare --path ./images --split 0.8 0.1 0.1 --preview

    5. Train the built-in vision model (dataset = one subfolder per class):
       sentinel train start --model cnn --dataset ./images --epochs 10 --live

    6. Train an LLM from the catalog (dataset = a JSONL file):
       sentinel train start --model Qwen3.8 --dataset ./chat.jsonl --epochs 3

    For more help:
       sentinel <command> --help
    """)
