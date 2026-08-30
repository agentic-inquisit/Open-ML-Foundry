"""
A/B Testing Service for Model Comparison
Compares two model versions on the same test dataset
"""

import sqlite3
import json
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class ABTestingService:
    """Service for managing A/B tests between model versions."""

    def __init__(self, db_path: str = "model_registry.db"):
        """Initialize ABTestingService with shared database."""
        self.db_path = db_path
        self._init_tables()

    def _init_tables(self) -> None:
        """Create ab_tests, ab_results, ab_summary tables if not present."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS ab_tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                model_a_id INTEGER NOT NULL,
                model_b_id INTEGER NOT NULL,
                model_a_path TEXT,
                model_b_path TEXT,
                split_ratio REAL DEFAULT 0.5,
                deployment_env TEXT DEFAULT 'development',
                dataset_description TEXT,
                owner TEXT DEFAULT 'system',
                status TEXT DEFAULT 'active',
                created_at TEXT NOT NULL,
                completed_at TEXT,
                winner_model_id INTEGER,
                CONSTRAINT different_models CHECK (model_a_id != model_b_id)
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS ab_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id INTEGER NOT NULL,
                image_hash TEXT NOT NULL,
                image_filename TEXT,
                model_id INTEGER NOT NULL,
                model_label TEXT NOT NULL,
                top_prediction TEXT,
                top_confidence REAL,
                all_predictions TEXT,
                latency_ms REAL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(test_id) REFERENCES ab_tests(id)
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS ab_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id INTEGER NOT NULL,
                model_id INTEGER NOT NULL,
                model_label TEXT NOT NULL,
                total_requests INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                avg_confidence REAL DEFAULT 0.0,
                avg_latency_ms REAL DEFAULT 0.0,
                last_updated TEXT,
                FOREIGN KEY(test_id) REFERENCES ab_tests(id),
                UNIQUE(test_id, model_label)
            )
        """)

        conn.commit()
        conn.close()

    def create_test(
        self,
        name: str,
        model_a_id: int,
        model_b_id: int,
        model_a_path: str,
        model_b_path: str,
        split_ratio: float = 0.5,
        deployment_env: str = "development",
        description: str = "",
        dataset_description: str = "",
        owner: str = "system"
    ) -> Dict:
        """Create a new A/B test."""
        if model_a_id == model_b_id:
            raise ValueError("Model A and B must be different")

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        try:
            c.execute("""
                INSERT INTO ab_tests (
                    name, description, model_a_id, model_b_id,
                    model_a_path, model_b_path, split_ratio,
                    deployment_env, dataset_description, owner, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                name, description, model_a_id, model_b_id,
                model_a_path, model_b_path, split_ratio,
                deployment_env, dataset_description, owner,
                datetime.now().isoformat()
            ))

            test_id = c.lastrowid

            # Initialize summary rows
            c.execute("""
                INSERT INTO ab_summary (test_id, model_id, model_label, last_updated)
                VALUES (?, ?, ?, ?)
            """, (test_id, model_a_id, 'A', datetime.now().isoformat()))

            c.execute("""
                INSERT INTO ab_summary (test_id, model_id, model_label, last_updated)
                VALUES (?, ?, ?, ?)
            """, (test_id, model_b_id, 'B', datetime.now().isoformat()))

            conn.commit()

            return {
                "status": "success",
                "test_id": test_id,
                "name": name,
                "message": f"Created A/B test '{name}' with ID {test_id}"
            }
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                return {"status": "error", "message": f"Test name '{name}' already exists"}
            raise
        finally:
            conn.close()

    def get_all_tests(self) -> List[Dict]:
        """Get all A/B tests with summary stats."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("""
            SELECT t.*, a.wins as a_wins, a.total_requests as a_requests,
                   a.avg_confidence as a_confidence, a.avg_latency_ms as a_latency,
                   b.wins as b_wins, b.total_requests as b_requests,
                   b.avg_confidence as b_confidence, b.avg_latency_ms as b_latency
            FROM ab_tests t
            LEFT JOIN ab_summary a ON t.id = a.test_id AND a.model_label = 'A'
            LEFT JOIN ab_summary b ON t.id = b.test_id AND b.model_label = 'B'
            ORDER BY t.created_at DESC
        """)

        tests = []
        for row in c.fetchall():
            tests.append({
                "test_id": row["id"],
                "name": row["name"],
                "description": row["description"],
                "model_a_id": row["model_a_id"],
                "model_b_id": row["model_b_id"],
                "model_a_path": row["model_a_path"],
                "model_b_path": row["model_b_path"],
                "split_ratio": row["split_ratio"],
                "deployment_env": row["deployment_env"],
                "dataset_description": row["dataset_description"],
                "owner": row["owner"],
                "status": row["status"],
                "created_at": row["created_at"],
                "completed_at": row["completed_at"],
                "winner_model_id": row["winner_model_id"],
                "summary": {
                    "A": {
                        "wins": row["a_wins"],
                        "total_requests": row["a_requests"],
                        "avg_confidence": row["a_confidence"],
                        "avg_latency_ms": row["a_latency"]
                    },
                    "B": {
                        "wins": row["b_wins"],
                        "total_requests": row["b_requests"],
                        "avg_confidence": row["b_confidence"],
                        "avg_latency_ms": row["b_latency"]
                    }
                }
            })

        conn.close()
        return tests

    def get_test_by_id(self, test_id: int) -> Optional[Dict]:
        """Get test by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("SELECT * FROM ab_tests WHERE id = ?", (test_id,))
        row = c.fetchone()
        conn.close()

        if not row:
            return None

        return dict(row)

    def _assign_model(self, image_bytes: bytes, split_ratio: float) -> str:
        """Deterministic model assignment using image hash."""
        digest = hashlib.sha256(image_bytes).hexdigest()
        bucket = int(digest[:8], 16) % 100
        return 'A' if bucket < int(split_ratio * 100) else 'B'

    def run_inference_single(
        self,
        image_bytes: bytes,
        model_path: str,
        model_label: str,
        image_hash: str,
        image_filename: str,
        test_id: int
    ) -> Dict:
        """Run inference on single model and store result."""
        try:
            import jax_train
            if not hasattr(jax_train, 'FinetuneInference'):
                raise ImportError("FinetuneInference not available")

            inference = jax_train.FinetuneInference(num_classes=10, input_size=224)
            loaded = inference.load_checkpoint(model_path)

            if not loaded:
                status = "load_error"
                top_pred = "LOAD_ERROR"
                top_conf = 0.0
                predictions = []
                latency = 0.0
            else:
                t0 = time.time()
                result = inference.predict(image_bytes, return_top_k=3)
                latency = (time.time() - t0) * 1000

                status = result.get("status", "error")
                preds = result.get("predictions", [])
                top_pred = preds[0]["class_name"] if preds else "UNKNOWN"
                top_conf = preds[0]["confidence"] if preds else 0.0
                predictions = preds
        except Exception as e:
            status = "mock"
            top_pred = "MOCK"
            top_conf = 0.5
            predictions = [{"class_name": "MOCK", "confidence": 0.5}]
            latency = 0.0

        # Store result
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("""
            INSERT INTO ab_results (
                test_id, image_hash, image_filename, model_id, model_label,
                top_prediction, top_confidence, all_predictions, latency_ms, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            test_id, image_hash, image_filename, -1, model_label,
            top_pred, top_conf, json.dumps(predictions), latency,
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

        return {
            "model_label": model_label,
            "top_prediction": top_pred,
            "top_confidence": top_conf,
            "latency_ms": latency,
            "all_predictions": predictions,
            "status": status
        }

    def run_batch(
        self,
        test_id: int,
        images: List[Tuple[bytes, str]]
    ) -> Dict:
        """Run batch test: all images through both models."""
        test = self.get_test_by_id(test_id)
        if not test:
            return {"status": "error", "message": "Test not found"}

        results = []
        for image_bytes, filename in images:
            image_hash = hashlib.sha256(image_bytes).hexdigest()

            result_a = self.run_inference_single(
                image_bytes, test["model_a_path"], 'A', image_hash, filename, test_id
            )
            result_b = self.run_inference_single(
                image_bytes, test["model_b_path"], 'B', image_hash, filename, test_id
            )

            results.append({
                "filename": filename,
                "image_hash": image_hash,
                "model_a": result_a,
                "model_b": result_b
            })

        # Recompute summary stats
        self._recompute_summary(test_id)

        return {
            "status": "success",
            "test_id": test_id,
            "images_processed": len(images),
            "results": results
        }

    def _recompute_summary(self, test_id: int) -> None:
        """Recompute aggregate stats for test."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        for model_label in ['A', 'B']:
            # Get all results for this model in this test
            c.execute("""
                SELECT DISTINCT image_hash FROM ab_results
                WHERE test_id = ? AND model_label = ?
            """, (test_id, model_label))

            image_hashes = [row[0] for row in c.fetchall()]
            total_requests = len(image_hashes)

            # Count wins: for each image, compare confidences
            wins = 0
            total_confidence = 0.0
            total_latency = 0.0

            for image_hash in image_hashes:
                c.execute("""
                    SELECT top_confidence, latency_ms FROM ab_results
                    WHERE test_id = ? AND image_hash = ? AND model_label = ?
                    LIMIT 1
                """, (test_id, image_hash, model_label))
                row = c.fetchone()
                if row:
                    confidence = row[0] if row[0] else 0.0
                    latency = row[1] if row[1] else 0.0
                    total_confidence += confidence
                    total_latency += latency

                # Check if this model wins this image
                c.execute("""
                    SELECT model_label, top_confidence FROM ab_results
                    WHERE test_id = ? AND image_hash = ?
                    ORDER BY top_confidence DESC
                    LIMIT 1
                """, (test_id, image_hash))
                winner_row = c.fetchone()
                if winner_row and winner_row[0] == model_label:
                    wins += 1

            avg_confidence = total_confidence / total_requests if total_requests > 0 else 0.0
            avg_latency = total_latency / total_requests if total_requests > 0 else 0.0

            # Update summary
            c.execute("""
                UPDATE ab_summary
                SET total_requests = ?, wins = ?, avg_confidence = ?,
                    avg_latency_ms = ?, last_updated = ?
                WHERE test_id = ? AND model_label = ?
            """, (
                total_requests, wins, avg_confidence, avg_latency,
                datetime.now().isoformat(), test_id, model_label
            ))

        conn.commit()
        conn.close()

    def get_results(self, test_id: int) -> Dict:
        """Get detailed results for a test."""
        test = self.get_test_by_id(test_id)
        if not test:
            return {"status": "error", "message": "Test not found"}

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        # Get summary
        c.execute("""
            SELECT model_label, total_requests, wins, avg_confidence, avg_latency_ms
            FROM ab_summary WHERE test_id = ?
        """, (test_id,))

        summary = {}
        for row in c.fetchall():
            summary[row["model_label"]] = {
                "wins": row["wins"],
                "total_requests": row["total_requests"],
                "avg_confidence": row["avg_confidence"],
                "avg_latency_ms": row["avg_latency_ms"]
            }

        # Get per-image results
        c.execute("""
            SELECT DISTINCT image_hash, image_filename FROM ab_results
            WHERE test_id = ?
            ORDER BY image_filename
        """, (test_id,))

        per_image = []
        for row in c.fetchall():
            image_hash = row["image_hash"]
            filename = row["image_filename"]

            c.execute("""
                SELECT model_label, top_prediction, top_confidence, latency_ms
                FROM ab_results WHERE test_id = ? AND image_hash = ?
            """, (test_id, image_hash))

            results_by_model = {}
            confidences = {}

            for res_row in c.fetchall():
                label = res_row["model_label"]
                results_by_model[label] = {
                    "top_prediction": res_row["top_prediction"],
                    "top_confidence": res_row["top_confidence"],
                    "latency_ms": res_row["latency_ms"]
                }
                confidences[label] = res_row["top_confidence"]

            # Determine winner
            if 'A' in confidences and 'B' in confidences:
                if confidences['A'] > confidences['B']:
                    winner = 'A'
                elif confidences['B'] > confidences['A']:
                    winner = 'B'
                else:
                    winner = 'tie'
            else:
                winner = 'tie'

            per_image.append({
                "image_hash": image_hash,
                "filename": filename,
                "model_a": results_by_model.get('A', {}),
                "model_b": results_by_model.get('B', {}),
                "winner": winner
            })

        # Determine overall winner
        if summary:
            a_wins = summary.get('A', {}).get('wins', 0) or 0
            b_wins = summary.get('B', {}).get('wins', 0) or 0

            if a_wins > b_wins:
                overall_winner = 'A'
            elif b_wins > a_wins:
                overall_winner = 'B'
            elif a_wins == 0 and b_wins == 0:
                overall_winner = 'no_results'
            else:
                overall_winner = 'tie'
        else:
            overall_winner = 'no_results'

        conn.close()

        return {
            "test": {
                "id": test["id"],
                "name": test["name"],
                "description": test["description"],
                "deployment_env": test["deployment_env"],
                "owner": test["owner"],
                "created_at": test["created_at"],
                "status": test["status"]
            },
            "summary": summary,
            "per_image": per_image,
            "overall_winner": overall_winner
        }

    def promote_winner(
        self,
        test_id: int,
        winner_model_id: int,
        promoted_by: str = "system"
    ) -> Dict:
        """Mark test as completed with a winner."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("""
            UPDATE ab_tests
            SET status = 'completed', winner_model_id = ?, completed_at = ?
            WHERE id = ?
        """, ('completed', datetime.now().isoformat(), test_id))

        conn.commit()
        conn.close()

        return {
            "status": "success",
            "test_id": test_id,
            "winner_model_id": winner_model_id,
            "message": f"Test {test_id} completed with winner model {winner_model_id}"
        }
