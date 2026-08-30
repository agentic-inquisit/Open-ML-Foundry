"""
Traffic Router Service
Routes requests to different model versions based on configuration
"""

import hashlib
import random
from typing import Dict, List, Tuple
from datetime import datetime
from enum import Enum


class RoutingStrategy(Enum):
    """Routing strategy options"""
    PERCENTAGE_SPLIT = "percentage_split"
    HASH_BASED = "hash_based"
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    CANARY = "canary"


class TrafficRouter:
    """
    Routes inference requests to different model versions.
    Supports multiple routing strategies for A/B testing and canary deployments.
    """

    def __init__(self):
        """Initialize traffic router"""
        self.routes = {}  # {route_id: route_config}
        self.request_counter = 0
        self.request_history = {}  # {route_id: {model_path: count}}

    def create_route(self, route_id: str, model_paths: Dict[str, float],
                     strategy: str = "percentage_split",
                     description: str = "") -> Dict:
        """
        Create a new routing configuration.

        Args:
            route_id: Unique identifier for this route
            model_paths: {model_path: weight}
                        For percentage_split: weight = percentage (0-100)
                        For weighted: weight = any positive number
            strategy: Routing strategy name
            description: Route description

        Returns:
            Status dict
        """
        if route_id in self.routes:
            return {"status": "error", "message": f"Route {route_id} already exists"}

        if not model_paths or len(model_paths) < 1:
            return {"status": "error", "message": "Need at least 1 model"}

        if strategy == "percentage_split":
            total = sum(model_paths.values())
            if not (99 <= total <= 101):
                return {"status": "error", "message": f"Percentages must sum to 100, got {total}"}

        elif strategy == "canary":
            if len(model_paths) != 2:
                return {"status": "error", "message": "Canary requires exactly 2 models (stable + canary)"}
            stable_pct = list(model_paths.values())[0]
            canary_pct = list(model_paths.values())[1]
            if not (stable_pct > canary_pct):
                return {"status": "error", "message": "Canary strategy: first model should have higher percentage"}

        self.routes[route_id] = {
            "model_paths": model_paths,
            "strategy": strategy,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "request_count": 0,
            "round_robin_counter": 0
        }

        self.request_history[route_id] = {path: 0 for path in model_paths.keys()}

        return {
            "status": "success",
            "route_id": route_id,
            "message": f"Created route with strategy {strategy}",
            "models": list(model_paths.keys())
        }

    def route_request(self, route_id: str, request_data: bytes = b"") -> Dict:
        """
        Route a request to a model version based on strategy.

        Args:
            route_id: Routing configuration ID
            request_data: Request data (for hash-based routing)

        Returns:
            Dict with {selected_model_path, route_info}
        """
        if route_id not in self.routes:
            return {"status": "error", "message": f"Route not found: {route_id}"}

        route = self.routes[route_id]
        strategy = route["strategy"]
        model_paths = list(route["model_paths"].keys())
        weights = list(route["model_paths"].values())

        selected_model = None

        if strategy == "percentage_split":
            # Random selection based on percentages
            rand = random.uniform(0, 100)
            cumulative = 0
            for model, pct in route["model_paths"].items():
                cumulative += pct
                if rand <= cumulative:
                    selected_model = model
                    break
            selected_model = selected_model or model_paths[-1]

        elif strategy == "hash_based":
            # Deterministic: same request always routes to same model
            hash_value = int(hashlib.md5(request_data).hexdigest(), 16) % 100
            cumulative = 0
            for model, pct in route["model_paths"].items():
                cumulative += pct
                if hash_value <= cumulative:
                    selected_model = model
                    break
            selected_model = selected_model or model_paths[-1]

        elif strategy == "round_robin":
            # Rotate through models
            route["round_robin_counter"] = (route["round_robin_counter"] + 1) % len(model_paths)
            selected_model = model_paths[route["round_robin_counter"]]

        elif strategy == "weighted":
            # Weighted random selection
            selected_model = random.choices(model_paths, weights=weights, k=1)[0]

        elif strategy == "canary":
            # Canary: route small % to new version, rest to stable
            stable_model = model_paths[0]
            canary_model = model_paths[1]
            stable_pct = route["model_paths"][stable_model]

            rand = random.uniform(0, 100)
            selected_model = canary_model if rand > stable_pct else stable_model

        else:
            selected_model = model_paths[0]

        # Track
        route["request_count"] += 1
        self.request_history[route_id][selected_model] += 1
        self.request_counter += 1

        return {
            "status": "success",
            "route_id": route_id,
            "strategy": strategy,
            "selected_model": selected_model,
            "weights": route["model_paths"],
            "request_number": route["request_count"]
        }

    def get_route_stats(self, route_id: str) -> Dict:
        """
        Get statistics for a route.

        Args:
            route_id: Route ID

        Returns:
            Dict with route statistics
        """
        if route_id not in self.routes:
            return {"status": "error", "message": f"Route not found: {route_id}"}

        route = self.routes[route_id]
        history = self.request_history[route_id]

        stats = {
            "status": "success",
            "route_id": route_id,
            "strategy": route["strategy"],
            "description": route["description"],
            "total_requests": route["request_count"],
            "created_at": route["created_at"],
            "model_distribution": {}
        }

        total = sum(history.values())
        for model, count in history.items():
            actual_pct = (count / total * 100) if total > 0 else 0
            expected_pct = route["model_paths"].get(model, 0)

            stats["model_distribution"][model] = {
                "count": count,
                "actual_percentage": actual_pct,
                "expected_percentage": expected_pct if route["strategy"] == "percentage_split" else None,
                "difference_from_expected": (actual_pct - expected_pct) if route["strategy"] == "percentage_split" else None
            }

        return stats

    def update_route(self, route_id: str, model_paths: Dict[str, float],
                     strategy: str = None) -> Dict:
        """
        Update an existing route configuration.

        Args:
            route_id: Route ID
            model_paths: New model paths and weights
            strategy: New strategy (optional)

        Returns:
            Status dict
        """
        if route_id not in self.routes:
            return {"status": "error", "message": f"Route not found: {route_id}"}

        route = self.routes[route_id]

        if model_paths:
            route["model_paths"] = model_paths
            self.request_history[route_id] = {path: 0 for path in model_paths.keys()}

        if strategy:
            route["strategy"] = strategy

        return {
            "status": "success",
            "message": f"Updated route {route_id}",
            "models": list(route["model_paths"].keys())
        }

    def delete_route(self, route_id: str) -> Dict:
        """
        Delete a routing configuration.

        Args:
            route_id: Route ID

        Returns:
            Status dict
        """
        if route_id not in self.routes:
            return {"status": "error", "message": f"Route not found: {route_id}"}

        del self.routes[route_id]
        if route_id in self.request_history:
            del self.request_history[route_id]

        return {
            "status": "success",
            "message": f"Deleted route {route_id}"
        }

    def list_routes(self) -> Dict:
        """
        List all active routes.

        Returns:
            Dict with all routes and their stats
        """
        routes_list = []
        for route_id, route in self.routes.items():
            stats = self.get_route_stats(route_id)
            routes_list.append({
                "route_id": route_id,
                "strategy": route["strategy"],
                "description": route["description"],
                "models": list(route["model_paths"].keys()),
                "requests": route["request_count"],
                **stats.get("model_distribution", {})
            })

        return {
            "status": "success",
            "total_routes": len(routes_list),
            "total_requests": self.request_counter,
            "routes": routes_list
        }

    def promote_model(self, route_id: str, model_path: str,
                      new_weight: float = 100.0) -> Dict:
        """
        Promote a model in a route (increase its traffic percentage).

        Args:
            route_id: Route ID
            model_path: Model path to promote
            new_weight: New weight for this model (others scale down)

        Returns:
            Status dict
        """
        if route_id not in self.routes:
            return {"status": "error", "message": f"Route not found: {route_id}"}

        route = self.routes[route_id]

        if model_path not in route["model_paths"]:
            return {"status": "error", "message": f"Model not in route: {model_path}"}

        # Scale other models proportionally
        other_models = {m: w for m, w in route["model_paths"].items() if m != model_path}

        if other_models:
            other_total = sum(other_models.values())
            remaining_weight = 100.0 - new_weight
            scale_factor = remaining_weight / other_total if other_total > 0 else 0

            for model in other_models:
                other_models[model] *= scale_factor

            route["model_paths"][model_path] = new_weight
            route["model_paths"].update(other_models)
        else:
            route["model_paths"][model_path] = 100.0

        return {
            "status": "success",
            "message": f"Promoted {model_path} to {new_weight}%",
            "new_weights": route["model_paths"]
        }

    def get_recommended_next_step(self, route_id: str) -> Dict:
        """
        Analyze route statistics and recommend next action.

        Args:
            route_id: Route ID

        Returns:
            Dict with recommendation
        """
        stats = self.get_route_stats(route_id)
        if stats["status"] != "success":
            return stats

        route = self.routes[route_id]
        total_requests = stats["total_requests"]

        if total_requests < 100:
            return {
                "status": "success",
                "recommendation": "insufficient_data",
                "message": f"Need more requests for reliable comparison ({total_requests}/100+)",
                "progress_percent": min(100, total_requests / 100 * 100)
            }

        distribution = stats["model_distribution"]

        # Analyze for canary deployment
        if route["strategy"] == "canary":
            models = list(distribution.keys())
            if len(models) == 2:
                canary = models[1]
                stable = models[0]

                canary_perf = distribution[canary].get("actual_percentage", 0)
                stable_perf = distribution[stable].get("actual_percentage", 0)

                if canary_perf > stable_perf:
                    return {
                        "status": "success",
                        "recommendation": "promote_canary",
                        "message": f"Canary outperforming stable ({canary_perf:.1f}% vs {stable_perf:.1f}%). Ready to promote.",
                        "next_action": f"Increase {canary} weight to 100%"
                    }
                else:
                    return {
                        "status": "success",
                        "recommendation": "rollback",
                        "message": f"Canary underperforming ({canary_perf:.1f}% vs {stable_perf:.1f}%). Recommend rollback.",
                        "next_action": f"Revert to 100% {stable}"
                    }

        return {
            "status": "success",
            "recommendation": "monitor",
            "message": "Continue monitoring traffic split"
        }
