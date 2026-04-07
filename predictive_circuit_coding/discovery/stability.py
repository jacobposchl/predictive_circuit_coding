from __future__ import annotations

from typing import Any


def estimate_clustering_stability(*, cluster_stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "silhouette_score": cluster_stats.get("silhouette_score"),
        "non_noise_fraction": cluster_stats.get("non_noise_fraction", 0.0),
        "cluster_persistence_mean": cluster_stats.get("cluster_persistence_mean"),
        "cluster_persistence_min": cluster_stats.get("cluster_persistence_min"),
        "cluster_persistence_max": cluster_stats.get("cluster_persistence_max"),
        "cluster_persistence_by_cluster": cluster_stats.get("cluster_persistence_by_cluster", {}),
    }
