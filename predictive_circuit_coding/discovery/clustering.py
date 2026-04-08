from __future__ import annotations

from collections.abc import Sequence

import hdbscan
import numpy as np
from sklearn.metrics import silhouette_score

from predictive_circuit_coding.training.contracts import CandidateTokenRecord


def _normalized_embeddings(candidates: Sequence[CandidateTokenRecord]) -> np.ndarray:
    embeddings = np.asarray([candidate.embedding for candidate in candidates], dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1.0e-8, a_max=None)
    return embeddings / norms


def _persistence_by_cluster(labels: np.ndarray, clusterer: hdbscan.HDBSCAN) -> dict[int, float]:
    valid_cluster_ids = sorted(int(cluster_id) for cluster_id in np.unique(labels) if int(cluster_id) != -1)
    persistences = getattr(clusterer, "cluster_persistence_", None)
    if persistences is None:
        return {}
    return {
        cluster_id: float(persistences[index])
        for index, cluster_id in enumerate(valid_cluster_ids)
        if index < len(persistences)
    }


def cluster_candidate_tokens(
    *,
    candidates: tuple[CandidateTokenRecord, ...],
    min_cluster_size: int,
) -> tuple[tuple[CandidateTokenRecord, ...], dict[str, float | dict[int, float] | None]]:
    if int(min_cluster_size) < 2:
        raise ValueError("Discovery clustering requires min_cluster_size >= 2.")
    if not candidates:
        return tuple(), {
            "cluster_count": 0.0,
            "noise_count": 0.0,
            "non_noise_fraction": 0.0,
            "silhouette_score": None,
            "cluster_persistence_mean": None,
            "cluster_persistence_min": None,
            "cluster_persistence_max": None,
            "cluster_persistence_by_cluster": {},
        }

    embeddings = _normalized_embeddings(candidates)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        metric="euclidean",
        algorithm="best",
        allow_single_cluster=True,
    )
    labels = clusterer.fit_predict(embeddings)
    persistence_by_cluster = _persistence_by_cluster(labels, clusterer)

    clustered = tuple(
        CandidateTokenRecord(
            candidate_id=candidate.candidate_id,
            cluster_id=int(cluster_id),
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
        for candidate, cluster_id in zip(candidates, labels.tolist(), strict=True)
    )

    non_noise_mask = labels != -1
    non_noise_labels = labels[non_noise_mask]
    cluster_count = len({int(cluster_id) for cluster_id in non_noise_labels.tolist()})
    non_noise_fraction = float(non_noise_mask.mean()) if len(labels) else 0.0
    if cluster_count >= 2 and int(non_noise_mask.sum()) >= 2:
        silhouette = float(silhouette_score(embeddings[non_noise_mask], non_noise_labels, metric="cosine"))
    else:
        silhouette = None

    persistence_values = list(persistence_by_cluster.values())
    return clustered, {
        "cluster_count": float(cluster_count),
        "noise_count": float(int((labels == -1).sum())),
        "non_noise_fraction": non_noise_fraction,
        "silhouette_score": silhouette,
        "cluster_persistence_mean": (float(np.mean(persistence_values)) if persistence_values else None),
        "cluster_persistence_min": (float(np.min(persistence_values)) if persistence_values else None),
        "cluster_persistence_max": (float(np.max(persistence_values)) if persistence_values else None),
        "cluster_persistence_by_cluster": persistence_by_cluster,
    }
