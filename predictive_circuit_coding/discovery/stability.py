from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any

import numpy as np

from predictive_circuit_coding.discovery.clustering import cluster_candidate_tokens
from predictive_circuit_coding.training.contracts import CandidateTokenRecord


def _bootstrap_candidate_indices(
    *,
    candidate_count: int,
    min_cluster_size: int,
    rng: np.random.Generator,
    sample_fraction: float,
) -> np.ndarray:
    sample_size = max(int(round(float(candidate_count) * sample_fraction)), int(min_cluster_size), 2)
    sample_size = min(sample_size, candidate_count)
    return np.sort(rng.choice(candidate_count, size=sample_size, replace=False))


def _bootstrap_cluster_agreement(
    *,
    base_candidates: tuple[CandidateTokenRecord, ...],
    min_cluster_size: int,
    stability_rounds: int,
    seed: int,
    sample_fraction: float = 0.8,
) -> dict[int, float | None]:
    cluster_members: dict[int, set[str]] = {}
    for candidate in base_candidates:
        if int(candidate.cluster_id) == -1:
            continue
        cluster_members.setdefault(int(candidate.cluster_id), set()).add(str(candidate.candidate_id))
    if not cluster_members:
        return {}

    candidate_lookup = {str(candidate.candidate_id): candidate for candidate in base_candidates}
    candidate_ids = [str(candidate.candidate_id) for candidate in base_candidates]
    rng = np.random.default_rng(seed)
    agreement_values: dict[int, list[float]] = defaultdict(list)

    for _ in range(int(stability_rounds)):
        subset_indices = _bootstrap_candidate_indices(
            candidate_count=len(base_candidates),
            min_cluster_size=min_cluster_size,
            rng=rng,
            sample_fraction=sample_fraction,
        )
        subset_candidates = tuple(base_candidates[int(index)] for index in subset_indices.tolist())
        reclustered_candidates, _ = cluster_candidate_tokens(
            candidates=subset_candidates,
            min_cluster_size=min_cluster_size,
        )
        recluster_map = {
            str(candidate.candidate_id): int(candidate.cluster_id)
            for candidate in reclustered_candidates
        }
        subset_ids = {candidate_ids[int(index)] for index in subset_indices.tolist()}
        for cluster_id, members in cluster_members.items():
            observed_members = sorted(members & subset_ids)
            if len(observed_members) < 2:
                continue
            pair_scores: list[float] = []
            for left_id, right_id in combinations(observed_members, 2):
                left_cluster = recluster_map.get(left_id, -1)
                right_cluster = recluster_map.get(right_id, -1)
                pair_scores.append(1.0 if left_cluster != -1 and left_cluster == right_cluster else 0.0)
            if pair_scores:
                agreement_values[cluster_id].append(float(np.mean(pair_scores)))

    return {
        cluster_id: (float(np.mean(values)) if values else None)
        for cluster_id, values in sorted(agreement_values.items())
    }


def estimate_clustering_stability(
    *,
    candidates: tuple[CandidateTokenRecord, ...],
    cluster_stats: dict[str, Any],
    min_cluster_size: int,
    stability_rounds: int,
    seed: int,
) -> dict[str, Any]:
    bootstrap_stability = _bootstrap_cluster_agreement(
        base_candidates=candidates,
        min_cluster_size=min_cluster_size,
        stability_rounds=stability_rounds,
        seed=seed,
    )
    stability_values = [value for value in bootstrap_stability.values() if value is not None]
    return {
        "silhouette_score": cluster_stats.get("silhouette_score"),
        "non_noise_fraction": cluster_stats.get("non_noise_fraction", 0.0),
        "cluster_persistence_mean": cluster_stats.get("cluster_persistence_mean"),
        "cluster_persistence_min": cluster_stats.get("cluster_persistence_min"),
        "cluster_persistence_max": cluster_stats.get("cluster_persistence_max"),
        "cluster_persistence_by_cluster": cluster_stats.get("cluster_persistence_by_cluster", {}),
        "stability_rounds": int(stability_rounds),
        "bootstrap_cluster_agreement_by_cluster": bootstrap_stability,
        "bootstrap_cluster_agreement_mean": (float(np.mean(stability_values)) if stability_values else None),
        "bootstrap_cluster_agreement_min": (float(np.min(stability_values)) if stability_values else None),
        "bootstrap_cluster_agreement_max": (float(np.max(stability_values)) if stability_values else None),
        "bootstrap_sample_fraction": 0.8,
        "stability_estimate_available": bool(stability_values),
    }
