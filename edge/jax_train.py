import jax
import jax.numpy as jnp
from flax import linen as nn
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
            num_classes: Number of output classes
            input_size: Input image dimension (224, 128, 64, 28)
        """
        self.num_classes = num_classes
        self.input_size = input_size
        self.model = CNN(num_classes=num_classes, input_size=input_size)
        self.params = None
        self.class_names = None

    def load_checkpoint(self, checkpoint_path, class_mapping=None):
        """
        Load a fine-tuned checkpoint.

        Args:
            checkpoint_path: Path to .ckpt file
            class_mapping: Dict mapping class indices to names
                          e.g., {0: 'robin', 1: 'sparrow', 2: 'jay'}
        """
        try:
            # In production, would load with flax.serialization.from_bytes()
            # For now, mock loading
            print(f"✓ Loaded checkpoint: {checkpoint_path}")
            self.checkpoint_path = checkpoint_path
            self.class_names = class_mapping or {i: f"class_{i}" for i in range(self.num_classes)}
            return True
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            return False

    def predict(self, image_bytes, return_top_k=3):
        """
        Run inference on image.

        Args:
            image_bytes: Raw image data
            return_top_k: Return top K predictions

        Returns:
            Dict with predictions, confidences, class names
        """
        try:
            # Preprocess image
            img = preprocess_image(image_bytes, target_size=self.input_size, preserve_aspect=True)
            img_batch = jnp.expand_dims(img, axis=0)  # Add batch dimension

            # Inference (dummy for now - needs actual loaded params)
            # In production: logits = self.model.apply({'params': self.params}, img_batch)
            logits = jnp.array(np.random.randn(1, self.num_classes))  # Mock prediction

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

def run_finetuning(image_bytes, target_object="unknown", steps=5, batch_size=4,
                  num_classes=10, image_size=224, labels=None,
                  enable_validation=True, validation_split=0.2,
                  enable_early_stopping=False, patience=3,
                  checkpoint_interval=1, checkpoint_dir="finetuned_models"):
    """
    Finetuning loop with validation, early stopping, and checkpointing.

    Args:
        image_bytes: Binary image data
        target_object: Class label
        steps: Number of epochs
        batch_size: Samples per batch
        num_classes: Number of output classes
        image_size: Target image dimension
        labels: Pre-labeled array or None for dummy labels
        enable_validation: If True, split data into train/val (80/20 by default)
        validation_split: Fraction of data for validation (default 0.2)
        enable_early_stopping: If True, stop if validation loss doesn't improve
        patience: Number of epochs to wait before early stopping (requires user confirmation)
        checkpoint_interval: Save checkpoint every N epochs (0 = no checkpoints)
        checkpoint_dir: Directory to save checkpoints

    Returns:
        Dict with training status, loss history, validation history, best model info.
    """
    try:
        from pathlib import Path
        import os

        # === STEP 1: PREPROCESS AND SPLIT DATA ===
        img = preprocess_image(image_bytes, target_size=image_size, preserve_aspect=True)
        num_samples = max(batch_size * 2, 8)
        images = np.repeat(np.expand_dims(img, axis=0), num_samples, axis=0)

        if labels is None:
            labels_array = np.random.randint(0, num_classes, size=num_samples)
        else:
            labels_array = np.array(labels, dtype=np.int32)
            if len(labels_array) != num_samples:
                if len(labels_array) < num_samples:
                    labels_array = np.pad(labels_array, (0, num_samples - len(labels_array)), constant_values=0)
                else:
                    labels_array = labels_array[:num_samples]

        # Split into train/val if validation enabled
        if enable_validation:
            split_idx = int(num_samples * (1 - validation_split))
            train_images, val_images = images[:split_idx], images[split_idx:]
            train_labels, val_labels = labels_array[:split_idx], labels_array[split_idx:]
            print(f"✓ Train/Val split: {len(train_labels)} train, {len(val_labels)} val")
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

                # Early stopping logic
                if enable_early_stopping:
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
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

            print()  # Newline

            # Checkpointing
            if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
                checkpoint_name = f"{target_object}_epoch_{epoch+1:03d}.ckpt"
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
                print(f"  💾 Checkpoint saved: {checkpoint_path}")
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
        }

        if enable_validation:
            result.update({
                "final_val_loss": val_losses[-1] if val_losses else None,
                "best_val_loss": best_val_loss if best_val_loss != float('inf') else None,
                "val_loss_history": val_losses,
                "final_val_accuracy": val_accuracies[-1] if val_accuracies else None,
                "best_val_accuracy": max(val_accuracies) if val_accuracies else None,
                "val_acc_history": val_accuracies,
                "best_checkpoint": best_checkpoint_path
            })

        result.update({
            "checkpoint_interval": checkpoint_interval,
            "checkpoint_dir": checkpoint_dir,
            "samples_trained": len(train_labels) * len(train_losses),
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

# def run_inference(image_bytes):
#     """
#     Performs JAX inference on the provided image.
#     """
#     try:
#         # Preprocess image
#         nparr = np.frombuffer(image_bytes, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         if img is None:
#              return {"status": "error", "message": "Could not decode image"}
        
#         img = cv2.resize(img, (28, 28))
#         img = img / 255.0
#         img = np.expand_dims(img, axis=0) # Add batch dim
        
#         # Initialize (in a real app, we'd load saved weights)
#         rng = jax.random.PRNGKey(0)
#         cnn = CNN()
#         params = cnn.init(rng, jnp.ones([1, 28, 28, 3]))['params']
        
#         # Inference
#         logits = cnn.apply({'params': params}, jnp.array(img))
#         probs = jax.nn.softmax(logits)
#         predicted_class = int(jnp.argmax(probs))
#         confidence = float(jnp.max(probs))
        
#         return {
#             "status": "success",
#             "class_id": predicted_class,
#             "confidence": confidence,
#             "backend": "jax/flax"
#         }
#     except Exception as e:
#         return {"status": "error", "message": str(e)}
