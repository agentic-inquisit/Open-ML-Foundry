"""
Vision Session Adapter — thin wrapper around the existing, working
`run_finetuning` in edge/jax_train.py so vision fine-tuning can plug into
the same session/chat interface as LLM fine-tuning, without duplicating the
JAX training loop.
"""

from pathlib import Path
from typing import Callable, Optional

from edge.jax_train import run_finetuning, FinetuneInference

ProgressCallback = Callable[[dict], None]


def train(image_path: str, target_object: str, config: dict,
          on_progress: Optional[ProgressCallback] = None) -> dict:
    """Run vision fine-tuning for a session and normalize the result shape
    to match what the LLM trainer returns, so session_api.py can treat both
    uniformly.

    `image_path` must be a directory laid out as one subfolder per class
    (dataset_dir/<class_name>/<image files>) — see
    edge.jax_train.load_image_dataset. `num_classes` in config is optional;
    when omitted it's derived from the class subfolders found.

    `run_finetuning` itself doesn't take a progress callback (it just
    prints), so we call it once and report a single "training complete"
    progress event with the full history rather than faking per-epoch
    streaming — the alternative would be duplicating the training loop,
    which isn't worth it for what the callback buys here.
    """
    result = run_finetuning(
        dataset_dir=image_path,
        target_object=target_object,
        steps=config.get("epochs", 5),
        batch_size=config.get("batch_size", 4),
        num_classes=config.get("num_classes"),
        image_size=config.get("image_size", 224),
        enable_validation=config.get("enable_validation", True),
        validation_split=config.get("validation_split", 0.2),
        enable_early_stopping=config.get("enable_early_stopping", False),
        patience=config.get("patience", 3),
        checkpoint_interval=config.get("checkpoint_interval", 1),
        checkpoint_dir=config.get("checkpoint_dir", "training_outputs/vision"),
    )

    if on_progress:
        on_progress(result)

    if result.get("status") == "error":
        raise RuntimeError(result.get("message", "Vision training failed"))

    return {
        "final_loss": result.get("final_train_loss"),
        "final_accuracy": result.get("final_train_accuracy"),
        "checkpoint_path": result.get("best_checkpoint"),
        "epochs": result.get("epochs_trained", config.get("epochs", 5)),
        "raw": result,
    }


def infer(image_path: str, checkpoint_path: str, num_classes: int = 10,
          image_size: int = 224) -> dict:
    """Test a fine-tuned vision checkpoint against an image.

    num_classes/image_size are only fallbacks — load_checkpoint() reads the
    checkpoint's own sidecar metadata (written by run_finetuning) and uses
    that instead when present, so a mismatched default here doesn't break
    a real checkpoint.
    """
    image_bytes = Path(image_path).read_bytes()

    engine = FinetuneInference(num_classes=num_classes, input_size=image_size)
    if not engine.load_checkpoint(checkpoint_path):
        return {"status": "error", "message": f"Failed to load checkpoint: {checkpoint_path}"}
    return engine.predict(image_bytes)
