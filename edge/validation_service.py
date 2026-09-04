"""
Model Validation & Performance Comparison Service
Handles K-fold cross-validation, parameter tracking, and rollback capability
"""

import json
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime


def _write_class_folder_dataset(images: List[bytes], labels: List[int], base_dir: str) -> Path:
    """Write an in-memory (images, labels) pair out as a
    dataset_dir/<label>/<i>.jpg tree — the layout edge.jax_train.run_finetuning
    expects. Used to bridge callers (like K-fold CV) that already hold real
    image bytes and labels in memory rather than on disk."""
    root = Path(base_dir)
    for i, (img_bytes, label) in enumerate(zip(images, labels)):
        class_dir = root / str(int(label))
        class_dir.mkdir(parents=True, exist_ok=True)
        (class_dir / f"{i:05d}.jpg").write_bytes(img_bytes)
    return root


class ValidationService:
    """Service for pre-deployment validation and change tracking."""

    def __init__(self, registry, jax_train_module):
        """
        Initialize validation service.

        Args:
            registry: ModelRegistry instance
            jax_train_module: jax_train module for training
        """
        self.registry = registry
        self.jax_train = jax_train_module

    def kfold_cross_validation(
        self,
        images: List[bytes],
        labels: List[int],
        num_classes: int,
        image_size: int = 224,
        num_folds: int = 5,
        epochs: int = 5,
        batch_size: int = 4,
        learning_rate: float = 0.001,
        **training_kwargs
    ) -> Dict:
        """
        Perform K-fold cross-validation on image dataset.

        Args:
            images: List of image bytes
            labels: List of class labels (0-indexed)
            num_classes: Number of classes
            image_size: Input image size
            num_folds: Number of folds (default 5)
            epochs: Epochs per fold
            batch_size: Batch size
            learning_rate: Learning rate
            **training_kwargs: Additional training parameters

        Returns:
            Dict with fold results, mean, std, and confidence intervals
        """
        if len(images) < num_folds:
            return {
                "status": "error",
                "message": f"Need at least {num_folds} images for {num_folds}-fold CV, got {len(images)}"
            }

        # Prepare data
        indices = np.arange(len(images))
        np.random.seed(42)  # Reproducible
        np.random.shuffle(indices)

        fold_results = []
        fold_size = len(indices) // num_folds

        for fold_num in range(num_folds):
            # Split data
            val_start = fold_num * fold_size
            val_end = val_start + fold_size if fold_num < num_folds - 1 else len(indices)

            val_indices = indices[val_start:val_end]
            train_indices = np.concatenate([indices[:val_start], indices[val_end:]])

            # Prepare fold data
            train_images = [images[i] for i in train_indices]
            train_labels = np.array([labels[i] for i in train_indices])
            val_images = [images[i] for i in val_indices]
            val_labels = np.array([labels[i] for i in val_indices])

            try:
                # Train on fold — jax_train_module is only ever handed to this
                # service when JAX/Flax actually imported (see vision_module.py's
                # JAX_AVAILABLE gate), so run_finetuning is always present here.
                fold_kwargs = {k: v for k, v in training_kwargs.items() if k != "checkpoint_dir"}
                with tempfile.TemporaryDirectory(prefix="cv_fold_") as tmp_dir:
                    fold_dataset_dir = _write_class_folder_dataset(train_images, train_labels, tmp_dir)
                    training_result = self.jax_train.run_finetuning(
                        dataset_dir=str(fold_dataset_dir),
                        target_object=f"cv_fold_{fold_num}",
                        steps=epochs,
                        batch_size=batch_size,
                        num_classes=num_classes,
                        image_size=image_size,
                        enable_validation=True,
                        validation_split=0.2,
                        checkpoint_dir=tmp_dir,
                        **fold_kwargs
                    )

                    if training_result.get("status") == "error":
                        return {
                            "status": "error",
                            "message": f"Error in fold {fold_num + 1}: {training_result.get('message')}",
                        }

                    fold_accuracy = self._evaluate_fold(
                        val_images, val_labels,
                        training_result.get("best_checkpoint"),
                        num_classes, image_size
                    )
                    if fold_accuracy is None:
                        return {
                            "status": "error",
                            "message": f"Fold {fold_num + 1}: could not evaluate checkpoint "
                                       f"{training_result.get('best_checkpoint')}",
                        }

                fold_result = {
                    "train_loss": training_result.get("final_train_loss", 0.0),
                    "val_loss": training_result.get("best_val_loss", 0.0),
                    "accuracy": fold_accuracy
                }

                fold_results.append(fold_result)

            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Error in fold {fold_num + 1}: {str(e)}"
                }

        # Compute statistics
        accuracies = [f["accuracy"] for f in fold_results if f.get("accuracy") is not None]
        train_losses = [f["train_loss"] for f in fold_results if f.get("train_loss") is not None]
        val_losses = [f["val_loss"] for f in fold_results if f.get("val_loss") is not None]

        if not accuracies:
            return {"status": "error", "message": "No valid fold results"}

        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        ci_lower = mean_accuracy - 1.96 * std_accuracy / np.sqrt(num_folds)
        ci_upper = mean_accuracy + 1.96 * std_accuracy / np.sqrt(num_folds)

        return {
            "status": "success",
            "fold_results": fold_results,
            "num_folds": num_folds,
            "mean_accuracy": float(mean_accuracy),
            "std_accuracy": float(std_accuracy),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "mean_train_loss": float(np.mean(train_losses)) if train_losses else None,
            "mean_val_loss": float(np.mean(val_losses)) if val_losses else None
        }

    def _evaluate_fold(self, images: List[bytes], labels: np.ndarray,
                       checkpoint_path: Optional[str], num_classes: int,
                       image_size: int) -> Optional[float]:
        """Evaluate a checkpoint on fold validation set. Returns None (not a
        fabricated number) if there's no checkpoint to evaluate or loading it
        fails — callers must treat None as "could not evaluate this fold"."""
        if not checkpoint_path or not images:
            return None

        inference = self.jax_train.FinetuneInference(
            num_classes=num_classes, input_size=image_size
        )
        if not inference.load_checkpoint(checkpoint_path):
            return None

        correct = 0
        for img_bytes, true_label in zip(images, labels):
            result = inference.predict(img_bytes, return_top_k=1)
            predictions = result.get("predictions", [])
            if predictions:
                pred_idx = predictions[0].get("class_index", -1)
                if pred_idx == true_label:
                    correct += 1

        return correct / len(images)

    def track_parameter_changes(
        self,
        from_model_id: Optional[int],
        to_model_id: int,
        from_params: Optional[Dict],
        to_params: Dict
    ) -> Dict:
        """
        Track what parameters changed between model versions.

        Returns:
            Dict with {changed_params, unchanged_params, param_impact_estimate}
        """
        if not from_params:
            return {
                "changed_params": {},
                "unchanged_params": {},
                "param_impact_estimate": {}
            }

        changed = {}
        unchanged = {}

        for key in to_params:
            if key not in from_params:
                changed[key] = {"from": None, "to": to_params[key]}
            elif from_params[key] != to_params[key]:
                changed[key] = {"from": from_params[key], "to": to_params[key]}
            else:
                unchanged[key] = to_params[key]

        # Estimate impact of key parameter changes
        impact_estimate = {}
        if "epochs" in changed:
            old_epochs = changed["epochs"]["from"]
            new_epochs = changed["epochs"]["to"]
            if old_epochs and new_epochs > old_epochs:
                impact_estimate["epochs"] = "potentially_increased_overfitting"
            elif old_epochs and new_epochs < old_epochs:
                impact_estimate["epochs"] = "potentially_reduced_training"

        if "learning_rate" in changed:
            old_lr = changed["learning_rate"]["from"]
            new_lr = changed["learning_rate"]["to"]
            if old_lr and new_lr > old_lr:
                impact_estimate["learning_rate"] = "potentially_faster_learning_larger_steps"
            elif old_lr and new_lr < old_lr:
                impact_estimate["learning_rate"] = "potentially_slower_learning_smaller_steps"

        if "batch_size" in changed:
            old_bs = changed["batch_size"]["from"]
            new_bs = changed["batch_size"]["to"]
            if old_bs and new_bs > old_bs:
                impact_estimate["batch_size"] = "potentially_noisier_gradients_faster"
            elif old_bs and new_bs < old_bs:
                impact_estimate["batch_size"] = "potentially_smoother_gradients_slower"

        return {
            "changed_params": changed,
            "unchanged_params": unchanged,
            "param_impact_estimate": impact_estimate
        }

    def track_dataset_changes(
        self,
        from_model_id: Optional[int],
        to_model_id: int,
        from_dataset: Optional[Dict],
        to_dataset: Dict
    ) -> Dict:
        """
        Track what dataset changed between model versions.

        Returns:
            Dict with {changed_aspects, added_images, removed_images, changes}
        """
        if not from_dataset:
            return {
                "changed_aspects": {},
                "added_images": to_dataset.get("total_images", 0),
                "removed_images": 0
            }

        changes = {}
        for key in to_dataset:
            if key not in from_dataset:
                changes[key] = {"from": None, "to": to_dataset[key]}
            elif from_dataset[key] != to_dataset[key]:
                changes[key] = {"from": from_dataset[key], "to": to_dataset[key]}

        added = 0
        removed = 0

        if "total_images" in changes:
            from_total = changes["total_images"]["from"]
            to_total = changes["total_images"]["to"]
            if from_total and to_total:
                if to_total > from_total:
                    added = to_total - from_total
                else:
                    removed = from_total - to_total

        # Estimate impact
        impact_estimate = {}
        if "class_distribution" in changes:
            impact_estimate["class_distribution"] = "changed_class_balance_may_affect_bias"
        if added > 0:
            impact_estimate["added_images"] = f"added_{added}_new_samples_more_diversity"
        if removed > 0:
            impact_estimate["removed_images"] = f"removed_{removed}_samples_less_data"

        return {
            "changed_aspects": changes,
            "added_images": added,
            "removed_images": removed,
            "impact_estimate": impact_estimate
        }

    def compare_validation_results(
        self,
        new_kfold_result: Dict,
        previous_kfold_result: Optional[Dict],
        ab_test_result: Optional[Dict] = None
    ) -> Dict:
        """
        Compare new model validation results against previous best.

        Returns:
            Decision: PASS or FAIL with detailed breakdown
        """
        decision = {
            "status": "PASS",
            "reasons": [],
            "warnings": [],
            "metrics_comparison": {}
        }

        new_mean = new_kfold_result.get("mean_accuracy", 0.0)
        new_std = new_kfold_result.get("std_accuracy", 0.0)
        new_ci_lower = new_kfold_result.get("ci_lower", 0.0)

        if previous_kfold_result:
            prev_mean = previous_kfold_result.get("mean_accuracy", 0.0)
            prev_ci_lower = previous_kfold_result.get("ci_lower", 0.0)

            decision["metrics_comparison"]["new_mean_accuracy"] = new_mean
            decision["metrics_comparison"]["previous_mean_accuracy"] = prev_mean
            decision["metrics_comparison"]["difference"] = new_mean - prev_mean

            # Check if new model is significantly worse
            if new_ci_lower < prev_ci_lower - 0.05:  # 5% margin
                decision["status"] = "FAIL"
                decision["reasons"].append(
                    f"New model accuracy ({new_mean:.3f}) significantly lower than "
                    f"previous ({prev_mean:.3f}) with 95% confidence"
                )

            # Check if new model is equal or better
            elif new_mean >= prev_mean:
                decision["reasons"].append(
                    f"New model matches or improves previous: "
                    f"{new_mean:.3f} vs {prev_mean:.3f}"
                )

            # Check if high variance (less stable)
            if new_std > prev_kfold_result.get("std_accuracy", 0.0) + 0.05:
                decision["warnings"].append(
                    f"New model has higher variance (std: {new_std:.3f}), "
                    f"less stable than previous"
                )
        else:
            decision["reasons"].append(
                f"First model: K-fold mean accuracy {new_mean:.3f}, "
                f"std {new_std:.3f}"
            )

        # Add A/B test result
        if ab_test_result:
            if ab_test_result.get("overall_winner") == "A":
                decision["reasons"].append("A/B test: New model wins")
            elif ab_test_result.get("overall_winner") == "B":
                decision["warnings"].append("A/B test: Previous model wins")

        return decision

    def suggest_rollback_strategy(
        self,
        param_changes: Dict,
        dataset_changes: Dict,
        performance_degradation: float
    ) -> Dict:
        """
        Suggest which aspects to rollback based on changes and degradation.

        Args:
            param_changes: Parameter change tracking dict
            dataset_changes: Dataset change tracking dict
            performance_degradation: Accuracy drop (e.g., 0.05 for 5% drop)

        Returns:
            Rollback strategy suggestion
        """
        suggestions = {
            "primary_suspect": None,
            "rollback_options": [],
            "recommended": None
        }

        param_changed = len(param_changes.get("changed_params", {})) > 0
        dataset_changed = len(dataset_changes.get("changed_aspects", {})) > 0

        if not param_changed and not dataset_changed:
            suggestions["primary_suspect"] = "Unknown (no tracked changes)"
            suggestions["rollback_options"].append("Manually inspect training logs")
            return suggestions

        # Heuristics for root cause
        if param_changed and not dataset_changed:
            suggestions["primary_suspect"] = "Hyperparameter change"
            suggestions["rollback_options"].append("Revert to previous parameter set")
            suggestions["recommended"] = "revert_params"

        elif dataset_changed and not param_changed:
            suggestions["primary_suspect"] = "Dataset change (quality/balance)"
            suggestions["rollback_options"].append("Revert to previous dataset")
            suggestions["recommended"] = "revert_dataset"

        else:  # Both changed
            if performance_degradation > 0.1:  # > 10% drop
                suggestions["primary_suspect"] = "Likely dataset quality issue"
                suggestions["rollback_options"].append("Revert to previous dataset first")
                suggestions["recommended"] = "revert_dataset"
            else:
                suggestions["primary_suspect"] = "Likely hyperparameter issue"
                suggestions["rollback_options"].append("Revert to previous parameters first")
                suggestions["recommended"] = "revert_params"

            suggestions["rollback_options"].append("Test parameter change isolation")
            suggestions["rollback_options"].append("Test dataset change isolation")

        return suggestions
