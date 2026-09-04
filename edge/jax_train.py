import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import serialization as flax_serialization
from flax.training import train_state
import optax
import numpy as np
import cv2
# Grain imports for efficient data loading in JAX pipelines
# Grain provides true parallelism and prefetching, unlike Python's DataLoader
try:
    import grain
    GRAIN_AVAILABLE = True
except ImportError:
    GRAIN_AVAILABLE = False
    print("Warning: Grain not installed. Install with: pip install grain-ml")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Custom dataset class implementing Grain's data source interface
# This allows Grain to efficiently manage batching, shuffling, and multiprocess data loading
class ImageDataset:
    """
    Dataset wrapper for Grain integration.
    Stores preprocessed images and labels for batch construction.
    Grain will call __getitem__ in parallel across multiple workers.
    """
    def __init__(self, images, labels):
        # Store normalized image arrays (N, 28, 28, 3) - already preprocessed
        self.images = images
        # Store integer labels (N,) for classification targets
        self.labels = labels

    def __len__(self):
        # Required by Grain: total number of samples in dataset
        return len(self.images)

    def __getitem__(self, idx):
        # Called by Grain workers to fetch individual samples
        # Returns single example; Grain handles batching in separate process
        return {
            'image': jnp.array(self.images[idx], dtype=jnp.float32),  # Convert to JAX array
            'label': int(self.labels[idx])  # Keep as Python int for label encoding
        }

class CNN(nn.Module):
    num_classes: int = 10
    input_size: int = 224  # Configurable: 28 (MNIST), 64, 128, 224 (ImageNet-compatible)

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=128, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_classes)(x)  # Parameterized output classes
        return x

def preprocess_image(image_bytes, target_size=224, preserve_aspect=True):
    """
    Decode and normalize image from byte stream.
    Grain workers call this independently in parallel for each image.
    Returns normalized numpy array suitable for model input.

    Args:
        image_bytes: Raw image bytes (JPEG/PNG)
        target_size: Target dimension (will be target_size x target_size)
        preserve_aspect: If True, pad image; if False, stretch image
    """
    # Decode JPEG/PNG bytes to numpy array using OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Handle decode failures gracefully
    if img is None:
        raise ValueError("Failed to decode image from bytes")

    h, w = img.shape[:2]

    if preserve_aspect:
        # Aspect-ratio-preserving resize with padding (letterbox)
        scale = min(target_size / w, target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))

        # Create target-sized image with zero padding
        padded = np.zeros((target_size, target_size, 3), dtype=img.dtype)
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img
        img = padded
    else:
        # Direct resize (may distort)
        img = cv2.resize(img, (target_size, target_size))

    # Normalize to [0, 1] range: divide by 255 (uint8 max)
    # Normalized inputs accelerate training convergence
    img = img.astype(np.float32) / 255.0

    return img


def _scan_class_folders(dataset_dir):
    """Walk a dataset_dir/<class_name>/<image files> tree and return
    (raw_image_bytes, labels, class_names) without decoding anything.
    Shared by load_image_dataset() (which decodes) and load_raw_dataset()
    (which hands raw bytes to callers that decode later, e.g. K-fold CV
    feeding edge.jax_train.run_finetuning per fold).

    Raises ValueError if the directory doesn't exist, has no class
    subfolders, or contains no image files — callers should treat this as a
    real failure rather than falling back to synthetic data.
    """
    root = Path(dataset_dir)
    if not root.is_dir():
        raise ValueError(f"Dataset directory not found: {dataset_dir}")

    class_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not class_dirs:
        raise ValueError(
            f"No class subfolders in {dataset_dir} — expected "
            f"{dataset_dir}/<class_name>/<image files>"
        )

    class_names = {idx: d.name for idx, d in enumerate(class_dirs)}
    raw_images = []
    labels = []

    for label_idx, class_dir in enumerate(class_dirs):
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            raw_images.append(path.read_bytes())
            labels.append(label_idx)

    if not raw_images:
        raise ValueError(f"No image files found under {dataset_dir}")

    return raw_images, labels, class_names


def load_raw_dataset(dataset_dir):
    """Load a dataset_dir/<class_name>/<image files> tree as raw bytes —
    for callers (like ValidationService.kfold_cross_validation) that decode
    images themselves rather than needing a preprocessed array up front.

    Returns (images: list[bytes], labels: list[int], class_names).
    """
    return _scan_class_folders(dataset_dir)


def load_image_dataset(dataset_dir, image_size=224, preserve_aspect=True):
    """
    Load an image classification dataset laid out as one subfolder per class:

        dataset_dir/
          class_a/*.jpg
          class_b/*.jpg
          ...

    Returns (images, labels, class_names, skipped) where images is an
    (N, image_size, image_size, 3) float32 array, labels is an (N,) int
    array, class_names maps label index -> folder name (sorted
    alphabetically, so label assignment is deterministic across runs on the
    same directory), and skipped lists (path, error) pairs for files that
    couldn't be decoded.

    Raises ValueError if the directory doesn't exist, has no class
    subfolders, or no images could be decoded — callers should treat this as
    a real failure rather than falling back to synthetic data.
    """
    raw_images, raw_labels, class_names = _scan_class_folders(dataset_dir)

    images = []
    labels = []
    skipped = []
    for img_bytes, label_idx in zip(raw_images, raw_labels):
        try:
            img = preprocess_image(img_bytes, target_size=image_size,
                                   preserve_aspect=preserve_aspect)
        except Exception as e:
            skipped.append((label_idx, str(e)))
            continue
        images.append(img)
        labels.append(label_idx)

    if not images:
        raise ValueError(f"No decodable images found under {dataset_dir}")

    return (
        np.stack(images).astype(np.float32),
        np.array(labels, dtype=np.int32),
        class_names,
        skipped,
    )


def create_train_state(rng, learning_rate, input_shape, num_classes=10, input_size=224):
    """Creates initial `TrainState`.

    Args:
        rng: JAX random key
        learning_rate: Learning rate for Adam optimizer
        input_shape: Shape of input (batch_size, height, width, channels)
        num_classes: Number of output classes
        input_size: Target image size (224, 128, 64, 28, etc.)
    """
    cnn = CNN(num_classes=num_classes, input_size=input_size)
    params = cnn.init(rng, jnp.ones(input_shape))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=cnn.apply, params=params, tx=tx)

def train_step(state, batch):
    """
    Train for a single step on a batch of data.
    Designed to work with Grain-batched data (batch['image'] shape: [batch_size, image_size, image_size, 3]).
    Returns: (state, loss, accuracy)
    """
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch['label']).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    preds = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(preds == batch['label'])
    return state, loss, accuracy


def eval_step(state, batch):
    """
    Evaluation step: compute loss + accuracy on batch without updating parameters.
    Returns: (loss, accuracy)
    """
    logits = state.apply_fn({'params': state.params}, batch['image'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']).mean()
    preds = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(preds == batch['label'])
    return loss, accuracy


class FinetuneInference:
    """
    Wrapper for running inference with fine-tuned JAX/Flax models.
    Handles model loading, preprocessing, and prediction.
    """

    def __init__(self, num_classes=10, input_size=224):
        """
        Args:
            num_classes: Number of output classes (may be overridden by the
                checkpoint's own metadata once load_checkpoint() runs)
            input_size: Input image dimension (224, 128, 64, 28)
        """
        self.num_classes = num_classes
        self.input_size = input_size
        self.model = CNN(num_classes=num_classes, input_size=input_size)
        self.params = None
        self.checkpoint_path = None
        self.class_names = None

    def load_checkpoint(self, checkpoint_path, class_mapping=None):
        """
        Load a fine-tuned checkpoint saved by run_finetuning().

        Args:
            checkpoint_path: Path to the .ckpt file (msgpack-serialized
                flax params). A sidecar `<checkpoint_path>.meta.json` with
                num_classes/input_size/class_names is read if present so the
                model architecture matches what was actually trained.
            class_mapping: Dict mapping class indices to names — overrides
                whatever the sidecar metadata (or default) provides.

        Returns:
            True on success. False if the file is missing or can't be
            deserialized — self.params is left as None in that case, so
            predict() will report the error rather than fabricate a result.
        """
        path = Path(checkpoint_path)
        if not path.is_file():
            print(f"Error loading checkpoint: file not found: {checkpoint_path}")
            return False

        meta_path = Path(str(path) + ".meta.json")
        meta = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception as e:
                print(f"Warning: could not read checkpoint metadata {meta_path}: {e}")

        num_classes = meta.get("num_classes", self.num_classes)
        input_size = meta.get("input_size", self.input_size)
        if num_classes != self.num_classes or input_size != self.input_size:
            self.num_classes = num_classes
            self.input_size = input_size
            self.model = CNN(num_classes=num_classes, input_size=input_size)

        try:
            rng = jax.random.PRNGKey(0)
            param_skeleton = self.model.init(
                rng, jnp.ones([1, self.input_size, self.input_size, 3])
            )["params"]
            self.params = flax_serialization.from_bytes(param_skeleton, path.read_bytes())
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            self.params = None
            return False

        self.checkpoint_path = str(path)
        class_names_meta = {int(k): v for k, v in meta.get("class_names", {}).items()}
        self.class_names = class_mapping or class_names_meta or {
            i: f"class_{i}" for i in range(self.num_classes)
        }
        return True

    def predict(self, image_bytes, return_top_k=3):
        """
        Run inference on image.

        Args:
            image_bytes: Raw image data
            return_top_k: Return top K predictions

        Returns:
            Dict with predictions, confidences, class names. Returns a
            status='error' dict if no checkpoint has been loaded yet.
        """
        if self.params is None:
            return {
                "status": "error",
                "message": "No checkpoint loaded — call load_checkpoint() first",
            }

        try:
            # Preprocess image
            img = preprocess_image(image_bytes, target_size=self.input_size, preserve_aspect=True)
            img_batch = jnp.expand_dims(img, axis=0)  # Add batch dimension

            logits = self.model.apply({'params': self.params}, img_batch)
            probs = jax.nn.softmax(logits[0])
            top_k_indices = np.argsort(-probs)[:return_top_k]

            predictions = []
            for idx in top_k_indices:
                predictions.append({
                    'class_index': int(idx),
                    'class_name': self.class_names.get(int(idx), f"class_{idx}"),
                    'confidence': float(probs[idx]),
                    'percentage': float(probs[idx] * 100)
                })

            return {
                'status': 'success',
                'predictions': predictions,
                'image_size': self.input_size,
                'num_classes': self.num_classes,
                'checkpoint': self.checkpoint_path if hasattr(self, 'checkpoint_path') else None
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

def _save_checkpoint(state, checkpoint_path, num_classes, input_size, class_names):
    """Write real params to disk (msgpack via flax.serialization) plus a
    sidecar `<checkpoint_path>.meta.json` describing the architecture, so
    FinetuneInference.load_checkpoint() can rebuild the right model shape
    without the caller having to already know it."""
    Path(checkpoint_path).write_bytes(flax_serialization.to_bytes(state.params))
    meta = {
        "num_classes": num_classes,
        "input_size": input_size,
        "class_names": {str(k): v for k, v in class_names.items()},
    }
    Path(str(checkpoint_path) + ".meta.json").write_text(json.dumps(meta))


def run_finetuning(dataset_dir, target_object="unknown", steps=5, batch_size=4,
                  num_classes=None, image_size=224,
                  enable_validation=True, validation_split=0.2,
                  enable_early_stopping=False, patience=3,
                  checkpoint_interval=1, checkpoint_dir="finetuned_models"):
    """
    Finetuning loop with validation, early stopping, and checkpointing.

    Args:
        dataset_dir: Path to a directory laid out as one subfolder per
            class (dataset_dir/<class_name>/<image files>). Loaded via
            load_image_dataset() — there is no synthetic-data fallback, a
            missing or empty dataset is a real error.
        target_object: Used to name checkpoint files
        steps: Number of epochs
        batch_size: Samples per batch
        num_classes: Number of output classes. None (default) derives it
            from the number of class subfolders actually found; pass an
            explicit value only to force a larger output layer than the
            current dataset covers (e.g. incremental training).
        image_size: Target image dimension
        enable_validation: If True, split data into train/val (80/20 by default)
        validation_split: Fraction of data for validation (default 0.2)
        enable_early_stopping: If True, stop if validation loss doesn't improve
        patience: Number of epochs to wait before early stopping
        checkpoint_interval: Save a checkpoint every N epochs (0 = no periodic checkpoints)
        checkpoint_dir: Directory to save checkpoints

    Returns:
        Dict with training status, loss history, validation history, best model info.
    """
    try:
        # === STEP 1: LOAD AND SPLIT REAL DATA ===
        images, labels_array, class_names, skipped = load_image_dataset(
            dataset_dir, image_size=image_size, preserve_aspect=True
        )
        if skipped:
            print(f"⚠ Skipped {len(skipped)} unreadable file(s) in {dataset_dir}")

        derived_num_classes = len(class_names)
        if num_classes is None:
            num_classes = derived_num_classes
        elif num_classes < derived_num_classes:
            return {
                "status": "error",
                "message": (
                    f"num_classes={num_classes} but {derived_num_classes} class "
                    f"folders were found in {dataset_dir}"
                ),
            }

        num_samples = len(images)
        if num_samples < batch_size:
            return {
                "status": "error",
                "message": (
                    f"Only {num_samples} image(s) found in {dataset_dir}, need at "
                    f"least batch_size={batch_size}"
                ),
            }

        # Split into train/val if validation enabled
        if enable_validation:
            split_idx = max(1, int(num_samples * (1 - validation_split)))
            split_idx = min(split_idx, num_samples - 1) if num_samples > 1 else num_samples
            train_images, val_images = images[:split_idx], images[split_idx:]
            train_labels, val_labels = labels_array[:split_idx], labels_array[split_idx:]
            print(f"✓ Train/Val split: {len(train_labels)} train, {len(val_labels)} val")
            if len(val_labels) == 0:
                enable_validation = False
                val_images, val_labels = None, None
        else:
            train_images, val_images = images, None
            train_labels, val_labels = labels_array, None

        # === STEP 2: CREATE DATASETS ===
        train_dataset = ImageDataset(train_images, train_labels)
        val_dataset = ImageDataset(val_images, val_labels) if enable_validation else None

        # === STEP 3: INITIALIZE MODEL ===
        rng = jax.random.PRNGKey(42)
        rng, init_rng = jax.random.split(rng)
        state = create_train_state(init_rng, learning_rate=1e-3,
                                  input_shape=[batch_size, image_size, image_size, 3],
                                  num_classes=num_classes, input_size=image_size)

        # === STEP 4: CREATE DATALOADERS ===
        if GRAIN_AVAILABLE:
            train_loader = grain.DataLoader(
                train_dataset, batch_size=batch_size, num_workers=2,
                prefetch_size=1, drop_remainder=False, seed=42
            )
            val_loader = grain.DataLoader(
                val_dataset, batch_size=batch_size, num_workers=1,
                prefetch_size=1, drop_remainder=False, seed=42
            ) if enable_validation else None
            print(f"✓ Grain DataLoader initialized")
        else:
            def create_simple_loader(dataset, batch_sz):
                def loader():
                    for i in range(0, len(dataset), batch_sz):
                        batch_indices = list(range(i, min(i + batch_sz, len(dataset))))
                        batch = {
                            'image': jnp.stack([dataset[idx]['image'] for idx in batch_indices]),
                            'label': jnp.array([dataset[idx]['label'] for idx in batch_indices])
                        }
                        yield batch
                return loader()

            train_loader = create_simple_loader(train_dataset, batch_size)
            val_loader = create_simple_loader(val_dataset, batch_size) if enable_validation else None

        # === STEP 5: CHECKPOINT SETUP ===
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_val_loss = float('inf')
        best_checkpoint_path = None
        best_epoch = None
        patience_counter = 0

        # === STEP 6: TRAINING LOOP WITH VALIDATION & CHECKPOINTING ===
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        stopping_epoch = None

        for epoch in range(steps):
            # Training phase
            epoch_train_losses = []
            epoch_train_accs = []
            for batch in train_loader:
                state, loss, acc = train_step(state, batch)
                epoch_train_losses.append(float(loss))
                epoch_train_accs.append(float(acc))

            avg_train_loss = np.mean(epoch_train_losses)
            avg_train_acc = float(np.mean(epoch_train_accs))
            train_losses.append(avg_train_loss)
            train_accuracies.append(avg_train_acc)
            print(f"Epoch {epoch+1}/{steps} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f}", end="")

            # Validation phase
            if enable_validation and val_loader:
                epoch_val_losses = []
                epoch_val_accs = []
                for val_batch in val_loader:
                    val_loss, val_acc = eval_step(state, val_batch)
                    epoch_val_losses.append(float(val_loss))
                    epoch_val_accs.append(float(val_acc))

                avg_val_loss = np.mean(epoch_val_losses)
                avg_val_acc = float(np.mean(epoch_val_accs))
                val_losses.append(avg_val_loss)
                val_accuracies.append(avg_val_acc)
                print(f" | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}", end="")

                # Best-so-far tracking (used for early stopping when enabled,
                # and to pick which checkpoint is "best" either way)
                is_best = avg_val_loss < best_val_loss
                if is_best:
                    best_val_loss = avg_val_loss

                if enable_early_stopping:
                    if is_best:
                        patience_counter = 0
                        print(" ✓ (Best)", end="")
                    else:
                        patience_counter += 1
                        print(f" (Patience: {patience_counter}/{patience})", end="")

                        if patience_counter >= patience:
                            print(f"\n⏸ Early stopping triggered at epoch {epoch+1}")
                            print(f"   Best validation loss: {best_val_loss:.4f}")
                            stopping_epoch = epoch + 1
                            break

                if is_best:
                    best_path = os.path.join(checkpoint_dir, f"{target_object}_best.ckpt")
                    _save_checkpoint(state, best_path, num_classes, image_size, class_names)
                    best_checkpoint_path = best_path
                    best_epoch = epoch + 1

            print()  # Newline

            # Periodic checkpoint (independent of the "best" one above)
            if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
                checkpoint_name = f"{target_object}_epoch_{epoch+1:03d}.ckpt"
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
                _save_checkpoint(state, checkpoint_path, num_classes, image_size, class_names)
                print(f"  💾 Checkpoint saved: {checkpoint_path}")
                if not enable_validation:
                    best_checkpoint_path = checkpoint_path

        # === STEP 7: RESULTS ===
        result = {
            "status": "success",
            "final_train_loss": train_losses[-1] if train_losses else None,
            "avg_train_loss": float(np.mean(train_losses)) if train_losses else None,
            "train_loss_history": train_losses,
            "final_train_accuracy": train_accuracies[-1] if train_accuracies else None,
            "best_train_accuracy": max(train_accuracies) if train_accuracies else None,
            "train_acc_history": train_accuracies,
            "validation_enabled": enable_validation,
            "early_stopping_enabled": enable_early_stopping,
            "stopping_epoch": stopping_epoch,
            "epochs_trained": len(train_losses),
            "best_checkpoint": best_checkpoint_path,
        }

        if enable_validation:
            result.update({
                "final_val_loss": val_losses[-1] if val_losses else None,
                "best_val_loss": best_val_loss if best_val_loss != float('inf') else None,
                "val_loss_history": val_losses,
                "final_val_accuracy": val_accuracies[-1] if val_accuracies else None,
                "best_val_accuracy": max(val_accuracies) if val_accuracies else None,
                "val_acc_history": val_accuracies,
                "best_epoch": best_epoch,
            })

        result.update({
            "checkpoint_interval": checkpoint_interval,
            "checkpoint_dir": checkpoint_dir,
            "num_classes": num_classes,
            "class_names": class_names,
            "total_images": num_samples,
            "train_images": len(train_labels),
            "val_images": len(val_labels) if enable_validation and val_labels is not None else 0,
            "samples_trained": len(train_labels) * len(train_losses),
            "skipped_files": len(skipped),
            "using_grain": GRAIN_AVAILABLE
        })

        return result

    except Exception as e:
        import traceback
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }

def run_inference(image_bytes, checkpoint_path=None, num_classes=10, image_size=224):
    """
    Run inference through FinetuneInference. Thin wrapper kept for
    serving/main.py's older /jax-inference endpoint, which predates the
    session API and doesn't carry a checkpoint reference of its own.

    Returns a status='error' dict (never a fabricated prediction) if no
    checkpoint_path is given or it can't be loaded.
    """
    if not checkpoint_path:
        return {
            "status": "error",
            "message": "No checkpoint_path provided — train a model first",
        }

    engine = FinetuneInference(num_classes=num_classes, input_size=image_size)
    if not engine.load_checkpoint(checkpoint_path):
        return {
            "status": "error",
            "message": f"Failed to load checkpoint: {checkpoint_path}",
        }
    return engine.predict(image_bytes)
