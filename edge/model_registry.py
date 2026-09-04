# Model Registry and Versioning System
# Tracks all fine-tuned models with versioning, metadata, and access control

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import os


class ModelRegistry:
    """
    Central registry for fine-tuned models with versioning, metadata, and access control.

    Features:
    - Automatic version numbering (v1.0, v1.1, v2.0, etc)
    - Full metadata tracking (parameters, dataset, performance)
    - Access control (public/private/shared)
    - Comparison between versions
    - Rollback capability
    """

    def __init__(self, db_path: str = "model_registry.db", models_dir: str = "finetuned_models"):
        """
        Initialize model registry.

        Args:
            db_path: Path to SQLite database
            models_dir: Directory to store model files
        """
        self.db_path = db_path
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create database schema."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Models table: core model information
        c.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                version TEXT UNIQUE,
                major_version INTEGER,
                minor_version INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                description TEXT,
                status TEXT DEFAULT 'active',
                access_level TEXT DEFAULT 'private',
                owner TEXT,
                UNIQUE(model_name, major_version, minor_version)
            )
        """)

        # Model metadata table: training parameters and performance
        c.execute("""
            CREATE TABLE IF NOT EXISTS model_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                num_classes INTEGER,
                image_size INTEGER,
                epochs_trained INTEGER,
                batch_size INTEGER,
                learning_rate REAL,
                validation_split REAL,
                early_stopping_enabled BOOLEAN,
                checkpoint_interval INTEGER,
                final_train_loss REAL,
                best_val_loss REAL,
                accuracy_on_test_set REAL,
                training_duration_seconds INTEGER,
                FOREIGN KEY(model_id) REFERENCES models(id)
            )
        """)

        # Dataset metadata table: what data was used for training
        c.execute("""
            CREATE TABLE IF NOT EXISTS dataset_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                total_images INTEGER,
                training_images INTEGER,
                validation_images INTEGER,
                test_images INTEGER,
                classes TEXT,
                class_distribution TEXT,
                data_source TEXT,
                preprocessing_steps TEXT,
                FOREIGN KEY(model_id) REFERENCES models(id)
            )
        """)

        # Checkpoints table: track which epochs/checkpoints are saved
        c.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                epoch INTEGER,
                checkpoint_path TEXT UNIQUE,
                file_size_mb REAL,
                train_loss REAL,
                val_loss REAL,
                is_best BOOLEAN DEFAULT 0,
                created_at TEXT,
                FOREIGN KEY(model_id) REFERENCES models(id)
            )
        """)

        # Access control table: manage who can use each model
        c.execute("""
            CREATE TABLE IF NOT EXISTS access_control (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                access_level TEXT,
                allowed_users TEXT,
                shared_with TEXT,
                can_download BOOLEAN DEFAULT 1,
                can_inference BOOLEAN DEFAULT 1,
                can_finetune BOOLEAN DEFAULT 0,
                FOREIGN KEY(model_id) REFERENCES models(id)
            )
        """)

        # Model history table: track changes/improvements
        c.execute("""
            CREATE TABLE IF NOT EXISTS model_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                event_type TEXT,
                event_description TEXT,
                changed_by TEXT,
                changed_at TEXT,
                FOREIGN KEY(model_id) REFERENCES models(id)
            )
        """)

        # Parameter sets table: reusable hyperparameter configurations
        c.execute("""
            CREATE TABLE IF NOT EXISTS parameter_sets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                epochs INTEGER,
                batch_size INTEGER,
                learning_rate REAL,
                validation_split REAL,
                early_stopping_enabled BOOLEAN,
                patience INTEGER,
                checkpoint_interval INTEGER,
                created_at TEXT NOT NULL,
                owner TEXT
            )
        """)

        # K-fold results table: validation metrics per fold
        c.execute("""
            CREATE TABLE IF NOT EXISTS kfold_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                fold_number INTEGER,
                fold_train_loss REAL,
                fold_val_loss REAL,
                fold_accuracy REAL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(model_id) REFERENCES models(id)
            )
        """)

        # Validation gates table: pre-deployment validation records
        c.execute("""
            CREATE TABLE IF NOT EXISTS validation_gates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                new_model_id INTEGER NOT NULL,
                best_previous_model_id INTEGER,
                kfold_mean_score REAL,
                kfold_std_dev REAL,
                kfold_ci_lower REAL,
                kfold_ci_upper REAL,
                ab_test_winner TEXT,
                ab_test_confidence REAL,
                params_changed TEXT,
                dataset_changed TEXT,
                passed BOOLEAN,
                reason_if_failed TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(new_model_id) REFERENCES models(id),
                FOREIGN KEY(best_previous_model_id) REFERENCES models(id)
            )
        """)

        # Change tracking table: parameter/dataset changes between versions
        c.execute("""
            CREATE TABLE IF NOT EXISTS change_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_model_id INTEGER,
                to_model_id INTEGER NOT NULL,
                param_changes TEXT,
                dataset_changes TEXT,
                impact_analysis TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(from_model_id) REFERENCES models(id),
                FOREIGN KEY(to_model_id) REFERENCES models(id)
            )
        """)

        conn.commit()
        conn.close()

    def register_model(self, model_name: str, description: str = "",
                      owner: str = "system", access_level: str = "private") -> Dict:
        """
        Register a new model version.

        Args:
            model_name: Name of the model (e.g., "bird_classifier")
            description: Human-readable description
            owner: Owner/creator of the model
            access_level: "private", "shared", or "public"

        Returns:
            Dict with model_id and version string (e.g., "v1.0")
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Get next version
        c.execute("""
            SELECT MAX(major_version) FROM models WHERE model_name = ?
        """, (model_name,))
        result = c.fetchone()
        max_major = result[0] if result[0] is not None else 0

        major_version = max_major + 1
        minor_version = 0
        version_str = f"v{major_version}.{minor_version}"

        # Insert model
        c.execute("""
            INSERT INTO models (model_name, version, major_version, minor_version,
                              created_at, description, owner, access_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (model_name, version_str, major_version, minor_version,
              datetime.now().isoformat(), description, owner, access_level))

        model_id = c.lastrowid

        # Initialize access control
        c.execute("""
            INSERT INTO access_control (model_id, access_level)
            VALUES (?, ?)
        """, (model_id, access_level))

        conn.commit()
        conn.close()

        return {
            "status": "success",
            "model_id": model_id,
            "model_name": model_name,
            "version": version_str,
            "message": f"Registered {model_name} {version_str}"
        }

    def add_metadata(self, model_id: int, metadata: Dict) -> bool:
        """
        Add training metadata to a model.

        Args:
            model_id: Model ID from registration
            metadata: Dict with keys:
                - num_classes, image_size, epochs_trained, batch_size
                - final_train_loss, best_val_loss, accuracy_on_test_set
                - training_duration_seconds, etc
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("""
            INSERT INTO model_metadata (
                model_id, num_classes, image_size, epochs_trained,
                batch_size, learning_rate, validation_split,
                early_stopping_enabled, checkpoint_interval,
                final_train_loss, best_val_loss, accuracy_on_test_set,
                training_duration_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_id,
            metadata.get("num_classes"),
            metadata.get("image_size"),
            metadata.get("epochs_trained"),
            metadata.get("batch_size"),
            metadata.get("learning_rate"),
            metadata.get("validation_split"),
            metadata.get("early_stopping_enabled", False),
            metadata.get("checkpoint_interval"),
            metadata.get("final_train_loss"),
            metadata.get("best_val_loss"),
            metadata.get("accuracy_on_test_set"),
            metadata.get("training_duration_seconds")
        ))

        conn.commit()
        conn.close()
        return True

    def add_dataset_info(self, model_id: int, dataset_info: Dict) -> bool:
        """
        Add dataset information.

        Args:
            model_id: Model ID
            dataset_info: Dict with:
                - total_images, training_images, validation_images, test_images
                - classes (list or comma-separated string)
                - class_distribution (dict or JSON string)
                - data_source, preprocessing_steps
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Convert classes to string if needed
        classes = dataset_info.get("classes", "")
        if isinstance(classes, list):
            classes = ",".join(classes)

        # Convert class_distribution to JSON if needed
        class_dist = dataset_info.get("class_distribution", {})
        if isinstance(class_dist, dict):
            class_dist = json.dumps(class_dist)

        c.execute("""
            INSERT INTO dataset_metadata (
                model_id, total_images, training_images,
                validation_images, test_images, classes,
                class_distribution, data_source, preprocessing_steps
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_id,
            dataset_info.get("total_images"),
            dataset_info.get("training_images"),
            dataset_info.get("validation_images"),
            dataset_info.get("test_images"),
            classes,
            class_dist,
            dataset_info.get("data_source"),
            dataset_info.get("preprocessing_steps")
        ))

        conn.commit()
        conn.close()
        return True

    def save_checkpoint(self, model_id: int, epoch: int, checkpoint_path: str,
                       train_loss: float, val_loss: float, is_best: bool = False) -> bool:
        """
        Register a checkpoint.

        Args:
            model_id: Model ID
            epoch: Epoch number
            checkpoint_path: File path to checkpoint
            train_loss: Training loss at this epoch
            val_loss: Validation loss at this epoch
            is_best: Whether this is the best checkpoint
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Get file size
        try:
            file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
        except:
            file_size = 0

        c.execute("""
            INSERT INTO checkpoints (
                model_id, epoch, checkpoint_path, file_size_mb,
                train_loss, val_loss, is_best, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (model_id, epoch, checkpoint_path, file_size,
              train_loss, val_loss, is_best, datetime.now().isoformat()))

        # If this is best, mark others as not best
        if is_best:
            c.execute("""
                UPDATE checkpoints SET is_best = 0
                WHERE model_id = ? AND id != last_insert_rowid()
            """, (model_id,))

        conn.commit()
        conn.close()
        return True

    def get_model_versions(self, model_name: str) -> List[Dict]:
        """
        Get all versions of a model.

        Args:
            model_name: Model name

        Returns:
            List of model versions with full metadata
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("""
            SELECT m.*, mm.*, dm.*, ac.access_level
            FROM models m
            LEFT JOIN model_metadata mm ON m.id = mm.model_id
            LEFT JOIN dataset_metadata dm ON m.id = dm.model_id
            LEFT JOIN access_control ac ON m.id = ac.model_id
            WHERE m.model_name = ?
            ORDER BY m.major_version DESC, m.minor_version DESC
        """, (model_name,))

        models = []
        for row in c.fetchall():
            models.append({
                "model_id": row["id"],
                "name": row["model_name"],
                "version": row["version"],
                "created_at": row["created_at"],
                "description": row["description"],
                "owner": row["owner"],
                "status": row["status"],
                "access_level": row["access_level"],
                "metadata": {
                    "num_classes": row["num_classes"],
                    "image_size": row["image_size"],
                    "epochs_trained": row["epochs_trained"],
                    "batch_size": row["batch_size"],
                    "learning_rate": row["learning_rate"],
                    "final_train_loss": row["final_train_loss"],
                    "best_val_loss": row["best_val_loss"],
                    "accuracy_on_test_set": row["accuracy_on_test_set"],
                    "training_duration_seconds": row["training_duration_seconds"]
                },
                "dataset": {
                    "total_images": row["total_images"],
                    "training_images": row["training_images"],
                    "validation_images": row["validation_images"],
                    "test_images": row["test_images"],
                    "classes": row["classes"].split(",") if row["classes"] else [],
                    "class_distribution": json.loads(row["class_distribution"]) if row["class_distribution"] else {},
                    "data_source": row["data_source"]
                }
            })

        conn.close()
        return models

    def get_all_models(self) -> List[Dict]:
        """Get all registered models across all versions."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("""
            SELECT DISTINCT model_name FROM models ORDER BY model_name
        """)

        all_models = []
        for row in c.fetchall():
            versions = self.get_model_versions(row["model_name"])
            all_models.append({
                "name": row["model_name"],
                "version_count": len(versions),
                "versions": versions
            })

        conn.close()
        return all_models

    def get_model_name(self, model_id: int) -> Optional[str]:
        """Resolve a model row's id to its model_name, for callers that
        only have the id (get_model_versions takes model_name, not id)."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT model_name FROM models WHERE id = ?", (model_id,))
        row = c.fetchone()
        conn.close()
        return row[0] if row else None

    def get_latest_checkpoint(self, model_id: int) -> Optional[str]:
        """Return the most recent checkpoint_path saved for a model_id, or
        None if no checkpoint has been saved yet."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "SELECT checkpoint_path FROM checkpoints WHERE model_id = ? "
            "ORDER BY epoch DESC LIMIT 1",
            (model_id,),
        )
        row = c.fetchone()
        conn.close()
        return row[0] if row else None

    def get_best_version(self, model_name: str) -> Optional[Dict]:
        """Get the best (latest active) version of a model."""
        versions = self.get_model_versions(model_name)
        for v in versions:
            if v["status"] == "active":
                return v
        return versions[0] if versions else None

    def set_access_level(self, model_id: int, access_level: str,
                        allowed_users: str = "", shared_with: str = "") -> bool:
        """
        Set access control for a model.

        Args:
            model_id: Model ID
            access_level: "private", "shared", or "public"
            allowed_users: Comma-separated user IDs (for private)
            shared_with: Comma-separated user/org IDs (for shared)
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("""
            UPDATE access_control
            SET access_level = ?, allowed_users = ?, shared_with = ?
            WHERE model_id = ?
        """, (access_level, allowed_users, shared_with, model_id))

        conn.commit()
        conn.close()
        return True

    def compare_versions(self, model_name: str) -> Dict:
        """Compare all versions of a model side-by-side."""
        versions = self.get_model_versions(model_name)

        return {
            "model_name": model_name,
            "total_versions": len(versions),
            "versions": versions,
            "comparison": {
                "by_accuracy": sorted(versions,
                    key=lambda v: v["metadata"]["accuracy_on_test_set"] or 0, reverse=True),
                "by_loss": sorted(versions,
                    key=lambda v: v["metadata"]["best_val_loss"] or float('inf')),
                "by_date": sorted(versions,
                    key=lambda v: v["created_at"], reverse=True)
            }
        }

    def add_history_event(self, model_id: int, event_type: str,
                         description: str, changed_by: str = "system"):
        """Log a change to the model."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("""
            INSERT INTO model_history (model_id, event_type, event_description,
                                      changed_by, changed_at)
            VALUES (?, ?, ?, ?, ?)
        """, (model_id, event_type, description, changed_by, datetime.now().isoformat()))

        conn.commit()
        conn.close()

    def get_model_history(self, model_id: int) -> List[Dict]:
        """Get change history for a model."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("""
            SELECT * FROM model_history
            WHERE model_id = ?
            ORDER BY changed_at DESC
        """, (model_id,))

        history = []
        for row in c.fetchall():
            history.append({
                "event_type": row["event_type"],
                "description": row["event_description"],
                "changed_by": row["changed_by"],
                "timestamp": row["changed_at"]
            })

        conn.close()
        return history

    def save_parameter_set(self, name: str, description: str = "", owner: str = "system",
                          epochs: int = 5, batch_size: int = 4, learning_rate: float = 0.001,
                          validation_split: float = 0.2, early_stopping_enabled: bool = True,
                          patience: int = 3, checkpoint_interval: int = 1) -> Dict:
        """Save a reusable hyperparameter configuration."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        try:
            c.execute("""
                INSERT INTO parameter_sets (
                    name, description, epochs, batch_size, learning_rate,
                    validation_split, early_stopping_enabled, patience,
                    checkpoint_interval, created_at, owner
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, description, epochs, batch_size, learning_rate,
                  validation_split, early_stopping_enabled, patience,
                  checkpoint_interval, datetime.now().isoformat(), owner))

            set_id = c.lastrowid
            conn.commit()
            return {"status": "success", "set_id": set_id, "name": name}
        except sqlite3.IntegrityError:
            return {"status": "error", "message": f"Parameter set '{name}' already exists"}
        finally:
            conn.close()

    def get_parameter_sets(self) -> List[Dict]:
        """Get all saved parameter sets."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("SELECT * FROM parameter_sets ORDER BY created_at DESC")
        sets = []
        for row in c.fetchall():
            sets.append({
                "id": row["id"],
                "name": row["name"],
                "description": row["description"],
                "epochs": row["epochs"],
                "batch_size": row["batch_size"],
                "learning_rate": row["learning_rate"],
                "validation_split": row["validation_split"],
                "early_stopping_enabled": row["early_stopping_enabled"],
                "patience": row["patience"],
                "checkpoint_interval": row["checkpoint_interval"],
                "created_at": row["created_at"],
                "owner": row["owner"]
            })

        conn.close()
        return sets

    def save_kfold_results(self, model_id: int, fold_results: List[Dict]) -> bool:
        """Save K-fold validation results."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        for fold_num, fold_data in enumerate(fold_results, 1):
            c.execute("""
                INSERT INTO kfold_results (
                    model_id, fold_number, fold_train_loss, fold_val_loss,
                    fold_accuracy, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (model_id, fold_num, fold_data.get("train_loss"),
                  fold_data.get("val_loss"), fold_data.get("accuracy"),
                  datetime.now().isoformat()))

        conn.commit()
        conn.close()
        return True

    def get_kfold_results(self, model_id: int) -> Dict:
        """Get K-fold results for a model."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("""
            SELECT fold_number, fold_train_loss, fold_val_loss, fold_accuracy
            FROM kfold_results WHERE model_id = ? ORDER BY fold_number
        """, (model_id,))

        results = []
        for row in c.fetchall():
            results.append({
                "fold": row["fold_number"],
                "train_loss": row["fold_train_loss"],
                "val_loss": row["fold_val_loss"],
                "accuracy": row["fold_accuracy"]
            })

        if not results:
            conn.close()
            return None

        accuracies = [r["accuracy"] for r in results if r["accuracy"]]
        train_losses = [r["train_loss"] for r in results if r["train_loss"]]

        conn.close()
        return {
            "folds": results,
            "num_folds": len(results),
            "mean_accuracy": sum(accuracies) / len(accuracies) if accuracies else None,
            "std_accuracy": (sum((x - (sum(accuracies)/len(accuracies)))**2 for x in accuracies) / len(accuracies))**0.5 if len(accuracies) > 1 else 0,
            "mean_train_loss": sum(train_losses) / len(train_losses) if train_losses else None
        }

    def save_validation_gate(self, new_model_id: int, best_prev_model_id: Optional[int],
                            kfold_mean: float, kfold_std: float, kfold_ci_lower: float,
                            kfold_ci_upper: float, ab_winner: str, ab_confidence: float,
                            params_changed: Dict, dataset_changed: Dict, passed: bool,
                            reason: str = "") -> Dict:
        """Record a pre-deployment validation gate result."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("""
            INSERT INTO validation_gates (
                new_model_id, best_previous_model_id, kfold_mean_score, kfold_std_dev,
                kfold_ci_lower, kfold_ci_upper, ab_test_winner, ab_test_confidence,
                params_changed, dataset_changed, passed, reason_if_failed, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (new_model_id, best_prev_model_id, kfold_mean, kfold_std,
              kfold_ci_lower, kfold_ci_upper, ab_winner, ab_confidence,
              json.dumps(params_changed), json.dumps(dataset_changed),
              passed, reason, datetime.now().isoformat()))

        gate_id = c.lastrowid
        conn.commit()
        conn.close()

        return {"status": "success", "gate_id": gate_id}

    def get_validation_gate(self, gate_id: int) -> Optional[Dict]:
        """Get a validation gate record."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("SELECT * FROM validation_gates WHERE id = ?", (gate_id,))
        row = c.fetchone()
        conn.close()

        if not row:
            return None

        return {
            "id": row["id"],
            "new_model_id": row["new_model_id"],
            "best_previous_model_id": row["best_previous_model_id"],
            "kfold_mean_score": row["kfold_mean_score"],
            "kfold_std_dev": row["kfold_std_dev"],
            "kfold_ci": [row["kfold_ci_lower"], row["kfold_ci_upper"]],
            "ab_test_winner": row["ab_test_winner"],
            "ab_test_confidence": row["ab_test_confidence"],
            "params_changed": json.loads(row["params_changed"]),
            "dataset_changed": json.loads(row["dataset_changed"]),
            "passed": row["passed"],
            "reason_if_failed": row["reason_if_failed"],
            "created_at": row["created_at"]
        }

    def save_change_tracking(self, from_model_id: Optional[int], to_model_id: int,
                            param_changes: Dict, dataset_changes: Dict,
                            impact_analysis: Dict) -> bool:
        """Track what changed between model versions."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("""
            INSERT INTO change_tracking (
                from_model_id, to_model_id, param_changes, dataset_changes,
                impact_analysis, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (from_model_id, to_model_id, json.dumps(param_changes),
              json.dumps(dataset_changes), json.dumps(impact_analysis),
              datetime.now().isoformat()))

        conn.commit()
        conn.close()
        return True

    def get_change_history(self, model_id: int) -> List[Dict]:
        """Get change history for a model."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("""
            SELECT * FROM change_tracking WHERE to_model_id = ?
            ORDER BY created_at DESC
        """, (model_id,))

        changes = []
        for row in c.fetchall():
            changes.append({
                "from_model_id": row["from_model_id"],
                "to_model_id": row["to_model_id"],
                "param_changes": json.loads(row["param_changes"]),
                "dataset_changes": json.loads(row["dataset_changes"]),
                "impact_analysis": json.loads(row["impact_analysis"]),
                "created_at": row["created_at"]
            })

        conn.close()
        return changes
