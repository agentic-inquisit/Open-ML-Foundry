"""
Multi-Model Inference Service
Run inference on multiple models and combine results
"""

import time
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime


class MultiModelInference:
    """
    Service for running inference on multiple models and combining results.
    Supports comparison, voting, and averaging.
    """

    def __init__(self, cache_service):
        """
        Initialize multi-model inference service.

        Args:
            cache_service: ModelInferenceCache instance for fast model access
        """
        self.cache = cache_service

    def inference_all(self, image_bytes: bytes, model_paths: List[str],
                      top_k: int = 3) -> Dict:
        """
        Run inference on all models and return individual results.

        Args:
            image_bytes: Image data
            model_paths: List of model checkpoint paths
            top_k: Top K predictions per model

        Returns:
            Dict with results from all models
        """
        if not model_paths:
            return {"status": "error", "message": "No models provided"}

        results = {
            "status": "success",
            "image_size": len(image_bytes),
            "num_models": len(model_paths),
            "timestamp": datetime.now().isoformat(),
            "predictions": {}
        }

        total_latency = 0

        for model_path in model_paths:
            try:
                t0 = time.time()

                # Get model from cache
                inference = self.cache.get_model(model_path)
                if not inference:
                    results["predictions"][model_path] = {
                        "status": "error",
                        "message": "Failed to load model"
                    }
                    continue

                # Run inference
                pred_result = inference.predict(image_bytes, return_top_k=top_k)
                latency = (time.time() - t0) * 1000

                results["predictions"][model_path] = {
                    "status": "success",
                    "predictions": pred_result.get("predictions", []),
                    "latency_ms": latency,
                    "top_prediction": pred_result.get("predictions", [{}])[0].get("class_name", "unknown"),
                    "confidence": pred_result.get("predictions", [{}])[0].get("confidence", 0.0)
                }

                total_latency += latency

            except Exception as e:
                results["predictions"][model_path] = {
                    "status": "error",
                    "message": str(e)
                }

        results["total_latency_ms"] = total_latency
        results["avg_latency_ms"] = total_latency / len(model_paths) if model_paths else 0

        return results

    def inference_comparison(self, image_bytes: bytes, model_a_path: str,
                            model_b_path: str, top_k: int = 3) -> Dict:
        """
        Compare inference results from two models (A/B comparison).

        Args:
            image_bytes: Image data
            model_a_path: Path to model A
            model_b_path: Path to model B
            top_k: Top K predictions

        Returns:
            Dict with side-by-side comparison
        """
        results = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "model_a": {},
            "model_b": {},
            "comparison": {}
        }

        # Run inference on both
        t0 = time.time()
        inference_a = self.cache.get_model(model_a_path)
        if not inference_a:
            return {"status": "error", "message": f"Failed to load model A: {model_a_path}"}

        result_a = inference_a.predict(image_bytes, return_top_k=top_k)
        latency_a = (time.time() - t0) * 1000

        t0 = time.time()
        inference_b = self.cache.get_model(model_b_path)
        if not inference_b:
            return {"status": "error", "message": f"Failed to load model B: {model_b_path}"}

        result_b = inference_b.predict(image_bytes, return_top_k=top_k)
        latency_b = (time.time() - t0) * 1000

        # Extract results
        preds_a = result_a.get("predictions", [])
        preds_b = result_b.get("predictions", [])

        results["model_a"] = {
            "path": model_a_path,
            "predictions": preds_a,
            "top_prediction": preds_a[0]["class_name"] if preds_a else "none",
            "confidence": preds_a[0]["confidence"] if preds_a else 0.0,
            "latency_ms": latency_a
        }

        results["model_b"] = {
            "path": model_b_path,
            "predictions": preds_b,
            "top_prediction": preds_b[0]["class_name"] if preds_b else "none",
            "confidence": preds_b[0]["confidence"] if preds_b else 0.0,
            "latency_ms": latency_b
        }

        # Comparison
        conf_a = preds_a[0]["confidence"] if preds_a else 0.0
        conf_b = preds_b[0]["confidence"] if preds_b else 0.0

        results["comparison"] = {
            "winner": "A" if conf_a > conf_b else "B" if conf_b > conf_a else "tie",
            "confidence_diff": abs(conf_a - conf_b),
            "latency_diff_ms": abs(latency_a - latency_b),
            "faster_model": "A" if latency_a < latency_b else "B",
            "agree_on_prediction": (preds_a[0]["class_name"] if preds_a else None) ==
                                  (preds_b[0]["class_name"] if preds_b else None)
        }

        return results

    def inference_ensemble(self, image_bytes: bytes, model_paths: List[str],
                          ensemble_method: str = "voting", top_k: int = 3) -> Dict:
        """
        Run inference on multiple models and combine via ensemble.

        Args:
            image_bytes: Image data
            model_paths: List of model paths (minimum 3 for voting)
            ensemble_method: "voting" (majority), "averaging" (confidence average),
                           "max" (highest confidence), "min" (lowest confidence)
            top_k: Top K predictions

        Returns:
            Dict with ensemble prediction
        """
        if len(model_paths) < 2:
            return {"status": "error", "message": "Need at least 2 models for ensemble"}

        if ensemble_method not in ["voting", "averaging", "max", "min"]:
            return {"status": "error", "message": f"Unknown ensemble method: {ensemble_method}"}

        # Run all models
        all_results = self.inference_all(image_bytes, model_paths, top_k)

        if all_results["status"] != "success":
            return all_results

        # Extract predictions
        predictions_by_model = {}
        confidences_by_model = {}

        for model_path, result in all_results["predictions"].items():
            if result.get("status") != "success":
                continue

            top_pred = result.get("top_prediction")
            confidence = result.get("confidence", 0.0)

            if top_pred:
                predictions_by_model[model_path] = top_pred
                confidences_by_model[top_pred] = confidences_by_model.get(top_pred, [])
                confidences_by_model[top_pred].append({
                    "model": model_path,
                    "confidence": confidence
                })

        if not predictions_by_model:
            return {"status": "error", "message": "No valid predictions from any model"}

        # Ensemble
        ensemble_result = {
            "status": "success",
            "ensemble_method": ensemble_method,
            "num_models_used": len(predictions_by_model),
            "individual_predictions": predictions_by_model,
            "timestamp": datetime.now().isoformat()
        }

        if ensemble_method == "voting":
            # Majority voting
            prediction_counts = {}
            for pred in predictions_by_model.values():
                prediction_counts[pred] = prediction_counts.get(pred, 0) + 1

            winner = max(prediction_counts.items(), key=lambda x: x[1])
            ensemble_result["ensemble_prediction"] = winner[0]
            ensemble_result["ensemble_confidence"] = winner[1] / len(predictions_by_model)
            ensemble_result["voting_counts"] = prediction_counts

        elif ensemble_method == "averaging":
            # Average confidence for each class
            class_confidences = {}
            for class_name, confidences in confidences_by_model.items():
                avg_conf = sum(c["confidence"] for c in confidences) / len(confidences)
                class_confidences[class_name] = avg_conf

            winner = max(class_confidences.items(), key=lambda x: x[1])
            ensemble_result["ensemble_prediction"] = winner[0]
            ensemble_result["ensemble_confidence"] = winner[1]
            ensemble_result["class_confidences"] = class_confidences

        elif ensemble_method == "max":
            # Highest confidence
            all_preds_with_conf = []
            for model_path, result in all_results["predictions"].items():
                if result.get("status") == "success":
                    all_preds_with_conf.append({
                        "model": model_path,
                        "prediction": result.get("top_prediction"),
                        "confidence": result.get("confidence", 0.0)
                    })

            if all_preds_with_conf:
                winner = max(all_preds_with_conf, key=lambda x: x["confidence"])
                ensemble_result["ensemble_prediction"] = winner["prediction"]
                ensemble_result["ensemble_confidence"] = winner["confidence"]
                ensemble_result["highest_confidence_model"] = winner["model"]

        elif ensemble_method == "min":
            # Lowest confidence (conservative)
            all_preds_with_conf = []
            for model_path, result in all_results["predictions"].items():
                if result.get("status") == "success":
                    all_preds_with_conf.append({
                        "model": model_path,
                        "prediction": result.get("top_prediction"),
                        "confidence": result.get("confidence", 0.0)
                    })

            if all_preds_with_conf:
                winner = min(all_preds_with_conf, key=lambda x: x["confidence"])
                ensemble_result["ensemble_prediction"] = winner["prediction"]
                ensemble_result["ensemble_confidence"] = winner["confidence"]
                ensemble_result["lowest_confidence_model"] = winner["model"]

        return ensemble_result

    def inference_with_fallback(self, image_bytes: bytes,
                               primary_model: str,
                               fallback_models: List[str],
                               top_k: int = 3) -> Dict:
        """
        Run inference with fallback: try primary, if fails try fallback models.

        Args:
            image_bytes: Image data
            primary_model: Primary model path
            fallback_models: List of fallback model paths (in order of preference)
            top_k: Top K predictions

        Returns:
            Dict with prediction and which model was used
        """
        all_models = [primary_model] + fallback_models

        for i, model_path in enumerate(all_models):
            try:
                inference = self.cache.get_model(model_path)
                if not inference:
                    continue

                t0 = time.time()
                result = inference.predict(image_bytes, return_top_k=top_k)
                latency = (time.time() - t0) * 1000

                return {
                    "status": "success",
                    "model_used": model_path,
                    "model_rank": i + 1,
                    "is_fallback": i > 0,
                    "predictions": result.get("predictions", []),
                    "latency_ms": latency,
                    "top_prediction": result.get("predictions", [{}])[0].get("class_name"),
                    "confidence": result.get("predictions", [{}])[0].get("confidence", 0.0)
                }

            except Exception as e:
                if i == len(all_models) - 1:
                    # Last one failed
                    return {
                        "status": "error",
                        "message": f"All models failed. Last error: {str(e)}"
                    }
                # Try next fallback
                continue

        return {"status": "error", "message": "No models available"}
