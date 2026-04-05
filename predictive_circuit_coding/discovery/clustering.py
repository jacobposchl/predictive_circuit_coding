from __future__ import annotations

import math

import torch

from predictive_circuit_coding.training.contracts import CandidateTokenRecord


def _cosine_similarity(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    lhs = lhs / lhs.norm().clamp_min(1.0e-8)
    rhs = rhs / rhs.norm().clamp_min(1.0e-8)
    return float(torch.dot(lhs, rhs).item())


def cluster_candidate_tokens(
    *,
    candidates: tuple[CandidateTokenRecord, ...],
    similarity_threshold: float,
    min_cluster_size: int,
) -> tuple[tuple[CandidateTokenRecord, ...], dict[str, float]]:
    centroids: list[torch.Tensor] = []
    cluster_members: list[list[int]] = []
    assigned_cluster_ids: list[int] = []
    embeddings = [torch.tensor(candidate.embedding, dtype=torch.float32) for candidate in candidates]

    for index, embedding in enumerate(embeddings):
        best_cluster = -1
        best_similarity = -math.inf
        for cluster_id, centroid in enumerate(centroids):
            similarity = _cosine_similarity(embedding, centroid)
            if similarity >= similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster_id
        if best_cluster == -1:
            centroids.append(embedding.clone())
            cluster_members.append([index])
            assigned_cluster_ids.append(len(centroids) - 1)
        else:
            cluster_members[best_cluster].append(index)
            centroids[best_cluster] = torch.stack([embeddings[item] for item in cluster_members[best_cluster]], dim=0).mean(dim=0)
            assigned_cluster_ids.append(best_cluster)

    valid_clusters = {cluster_id for cluster_id, members in enumerate(cluster_members) if len(members) >= min_cluster_size}
    clustered: list[CandidateTokenRecord] = []
    for candidate, cluster_id in zip(candidates, assigned_cluster_ids):
        final_cluster_id = cluster_id if cluster_id in valid_clusters else -1
        clustered.append(
            CandidateTokenRecord(
                candidate_id=candidate.candidate_id,
                cluster_id=final_cluster_id,
                recording_id=candidate.recording_id,
                session_id=candidate.session_id,
                subject_id=candidate.subject_id,
                unit_id=candidate.unit_id,
                unit_region=candidate.unit_region,
                unit_depth_um=candidate.unit_depth_um,
                patch_index=candidate.patch_index,
                patch_start_s=candidate.patch_start_s,
                patch_end_s=candidate.patch_end_s,
                window_start_s=candidate.window_start_s,
                window_end_s=candidate.window_end_s,
                label=candidate.label,
                score=candidate.score,
                embedding=candidate.embedding,
            )
        )
    cluster_count = len(valid_clusters)
    return tuple(clustered), {
        "cluster_count": float(cluster_count),
        "noise_count": float(sum(1 for candidate in clustered if candidate.cluster_id == -1)),
    }
