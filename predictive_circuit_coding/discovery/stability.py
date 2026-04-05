from __future__ import annotations

import random

from predictive_circuit_coding.discovery.clustering import cluster_candidate_tokens
from predictive_circuit_coding.training.contracts import CandidateTokenRecord


def estimate_clustering_stability(
    *,
    candidates: tuple[CandidateTokenRecord, ...],
    similarity_threshold: float,
    min_cluster_size: int,
    rounds: int,
    seed: int,
) -> dict[str, float]:
    if not candidates:
        return {"mean_cluster_count": 0.0, "mean_non_noise_fraction": 0.0}
    rng = random.Random(seed)
    cluster_counts: list[float] = []
    non_noise_fractions: list[float] = []
    candidates_list = list(candidates)
    for _ in range(rounds):
        shuffled = candidates_list[:]
        rng.shuffle(shuffled)
        clustered, stats = cluster_candidate_tokens(
            candidates=tuple(shuffled),
            similarity_threshold=similarity_threshold,
            min_cluster_size=min_cluster_size,
        )
        cluster_counts.append(stats["cluster_count"])
        non_noise_fractions.append(
            sum(1 for candidate in clustered if candidate.cluster_id != -1) / max(1, len(clustered))
        )
    return {
        "mean_cluster_count": float(sum(cluster_counts) / len(cluster_counts)),
        "mean_non_noise_fraction": float(sum(non_noise_fractions) / len(non_noise_fractions)),
    }
