"""
Model Inference Cache Service
Pre-loads and caches model checkpoints for fast inference
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json


class ModelInferenceCache:
    """
    Caches loaded models in memory for fast inference serving.
    Manages model lifecycle: preload, cache, evict, monitoring.
    """

    def __init__(self, jax_train_module):
        """
        Initialize cache service.

        Args:
            jax_train_module: jax_train module for creating FinetuneInference objects
        """
        self.jax_train = jax_train_module
        self.cache = {}  # {model_path: FinetuneInference object}
        self.cache_metadata = {}  # {model_path: {loaded_at, hits, misses}}
        self.max_cache_size = 5  # Max models to keep in memory

    def preload_model(self, model_path: str, num_classes: int = 10,
                     image_size: int = 224) -> Dict:
        """
        Pre-load a model checkpoint into memory.

        Args:
            model_path: Full path to model checkpoint
            num_classes: Number of classes
            image_size: Input image size

        Returns:
            Dict with status and metadata
        """
        if not os.path.exists(model_path):
            return {"status": "error", "message": f"Model not found: {model_path}"}

        if model_path in self.cache:
            self.cache_metadata[model_path]["hits"] += 1
            return {
                "status": "success",
                "message": f"Model already cached: {model_path}",
                "cached": True
            }

        try:
            # Check cache size
            if len(self.cache) >= self.max_cache_size:
                self._evict_least_used()

            # Create inference object
            if not hasattr(self.jax_train, 'FinetuneInference'):
                return {"status": "error", "message": "FinetuneInference not available"}

            inference = self.jax_train.FinetuneInference(
                num_classes=num_classes, input_size=image_size
            )

            # Load checkpoint
            loaded = inference.load_checkpoint(model_path)
            if not loaded:
                return {"status": "error", "message": f"Failed to load: {model_path}"}

            # Cache it
            self.cache[model_path] = inference
            self.cache_metadata[model_path] = {
                "loaded_at": datetime.now().isoformat(),
                "num_classes": num_classes,
                "image_size": image_size,
                "hits": 0,
                "misses": 0,
                "file_size_mb": os.path.getsize(model_path) / (1024 * 1024)
            }

            return {
                "status": "success",
                "message": f"Model pre-loaded: {Path(model_path).name}",
                "path": model_path,
                "cached": True
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_model(self, model_path: str) -> Optional[object]:
        """
        Get a model from cache (or load if not cached).

        Args:
            model_path: Full path to model checkpoint

        Returns:
            FinetuneInference object or None if failed
        """
        if model_path in self.cache:
            self.cache_metadata[model_path]["hits"] += 1
            return self.cache[model_path]

        # Try to load and cache
        result = self.preload_model(model_path)
        if result["status"] == "success":
            self.cache_metadata[model_path]["misses"] += 1
            return self.cache.get(model_path)

        self.cache_metadata.setdefault(model_path, {})["misses"] = \
            self.cache_metadata.get(model_path, {}).get("misses", 0) + 1
        return None

    def _evict_least_used(self) -> str:
        """
        Evict least-used model from cache.

        Returns:
            Evicted model path
        """
        if not self.cache:
            return None

        # Find least-used (by hits)
        least_used_path = min(
            self.cache.keys(),
            key=lambda p: self.cache_metadata.get(p, {}).get("hits", 0)
        )

        del self.cache[least_used_path]
        return least_used_path

    def get_cache_status(self) -> Dict:
        """
        Get current cache status and statistics.

        Returns:
            Dict with cache info and statistics
        """
        total_hits = sum(
            meta.get("hits", 0) for meta in self.cache_metadata.values()
        )
        total_misses = sum(
            meta.get("misses", 0) for meta in self.cache_metadata.values()
        )
        total_requests = total_hits + total_misses
        hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0

        cached_models = []
        for path, meta in self.cache_metadata.items():
            cached_models.append({
                "path": path,
                "filename": Path(path).name,
                "loaded_at": meta.get("loaded_at"),
                "hits": meta.get("hits", 0),
                "misses": meta.get("misses", 0),
                "file_size_mb": meta.get("file_size_mb", 0),
                "in_memory": path in self.cache
            })

        return {
            "status": "success",
            "cache_size": len(self.cache),
            "max_cache_size": self.max_cache_size,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "hit_rate_percent": hit_rate,
            "total_memory_mb": sum(m.get("file_size_mb", 0) for m in self.cache_metadata.values()),
            "cached_models": cached_models
        }

    def clear_cache(self) -> Dict:
        """
        Clear all cached models from memory.

        Returns:
            Status dict
        """
        cleared_count = len(self.cache)
        self.cache.clear()
        return {
            "status": "success",
            "message": f"Cleared {cleared_count} models from cache",
            "cleared_count": cleared_count
        }

    def remove_from_cache(self, model_path: str) -> Dict:
        """
        Remove a specific model from cache.

        Args:
            model_path: Full path to model

        Returns:
            Status dict
        """
        if model_path in self.cache:
            del self.cache[model_path]
            return {
                "status": "success",
                "message": f"Removed from cache: {Path(model_path).name}"
            }

        return {"status": "error", "message": "Model not in cache"}

    def warmup_cache(self, model_paths: List[str], num_classes: int = 10,
                     image_size: int = 224) -> Dict:
        """
        Pre-load multiple models into cache at once.

        Args:
            model_paths: List of model checkpoint paths
            num_classes: Number of classes
            image_size: Input image size

        Returns:
            Dict with results
        """
        results = {
            "status": "success",
            "total": len(model_paths),
            "loaded": 0,
            "failed": 0,
            "details": []
        }

        for path in model_paths:
            result = self.preload_model(path, num_classes, image_size)
            if result["status"] == "success":
                results["loaded"] += 1
            else:
                results["failed"] += 1

            results["details"].append({
                "path": path,
                "status": result["status"],
                "message": result.get("message", "")
            })

        return results
