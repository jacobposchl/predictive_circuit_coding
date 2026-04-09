from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch

from predictive_circuit_coding.training.contracts import CandidateTokenRecord, write_json_payload


def _normalized_features(features: torch.Tensor) -> torch.Tensor:
    tensor = features.detach().cpu().to(dtype=torch.float32)
    if tensor.ndim != 2:
        raise ValueError("features must be a 2D tensor")
    return tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)


def _neighbor_indices(features: torch.Tensor, *, neighbor_k: int) -> tuple[torch.Tensor, int]:
    normalized = _normalized_features(features)
    sample_count = int(normalized.shape[0])
    if sample_count <= 1:
        return torch.empty((sample_count, 0), dtype=torch.long), 0
    actual_k = max(0, min(int(neighbor_k), sample_count - 1))
    if actual_k == 0:
        return torch.empty((sample_count, 0), dtype=torch.long), 0
    similarity = normalized @ normalized.T
    similarity.fill_diagonal_(float("-inf"))
    return torch.topk(similarity, k=actual_k, dim=1).indices, actual_k


def _attribute_metric(*, neighbor_indices: torch.Tensor, values: tuple[str, ...]) -> dict[str, float | None]:
    sample_count = len(values)
    if sample_count <= 1:
        return {
            "mean_neighbor_match_fraction": None,
            "global_base_rate": None,
            "enrichment_over_base": None,
        }
    actual_k = int(neighbor_indices.shape[1]) if neighbor_indices.ndim == 2 else 0
    if actual_k == 0:
        return {
            "mean_neighbor_match_fraction": 0.0,
            "global_base_rate": 0.0,
            "enrichment_over_base": None,
        }

    mean_neighbor_matches: list[float] = []
    mean_base_rates: list[float] = []
    for index, value in enumerate(values):
        same_attribute_count = sum(1 for other in values if other == value) - 1
        mean_base_rates.append(float(same_attribute_count) / float(sample_count - 1))
        matches = 0
        for neighbor_index in neighbor_indices[index].tolist():
            if values[int(neighbor_index)] == value:
                matches += 1
        mean_neighbor_matches.append(float(matches) / float(actual_k))

    mean_neighbor_match_fraction = sum(mean_neighbor_matches) / float(len(mean_neighbor_matches))
    global_base_rate = sum(mean_base_rates) / float(len(mean_base_rates))
    enrichment = (
        mean_neighbor_match_fraction / global_base_rate
        if global_base_rate > 0.0
        else None
    )
    return {
        "mean_neighbor_match_fraction": mean_neighbor_match_fraction,
        "global_base_rate": global_base_rate,
        "enrichment_over_base": enrichment,
    }


def summarize_neighbor_geometry(
    *,
    features: torch.Tensor,
    attributes: dict[str, tuple[str, ...]],
    neighbor_k: int,
) -> dict[str, Any]:
    sample_count = int(features.shape[0])
    neighbor_indices, actual_k = _neighbor_indices(features, neighbor_k=neighbor_k)
    metrics = {
        str(name): _attribute_metric(neighbor_indices=neighbor_indices, values=tuple(str(value) for value in values))
        for name, values in attributes.items()
    }
    return {
        "sample_count": sample_count,
        "neighbor_k": actual_k,
        "metrics": metrics,
    }


def summarize_candidate_neighbor_geometry(
    *,
    candidates: tuple[CandidateTokenRecord, ...],
    neighbor_k: int,
) -> dict[str, Any]:
    if not candidates:
        return {
            "sample_count": 0,
            "neighbor_k": 0,
            "metrics": {
                "session_id": {
                    "mean_neighbor_match_fraction": None,
                    "global_base_rate": None,
                    "enrichment_over_base": None,
                },
                "subject_id": {
                    "mean_neighbor_match_fraction": None,
                    "global_base_rate": None,
                    "enrichment_over_base": None,
                },
                "unit_region": {
                    "mean_neighbor_match_fraction": None,
                    "global_base_rate": None,
                    "enrichment_over_base": None,
                },
            },
        }
    features = torch.tensor([candidate.embedding for candidate in candidates], dtype=torch.float32)
    return summarize_neighbor_geometry(
        features=features,
        attributes={
            "session_id": tuple(candidate.session_id for candidate in candidates),
            "subject_id": tuple(candidate.subject_id for candidate in candidates),
            "unit_region": tuple(candidate.unit_region for candidate in candidates),
        },
        neighbor_k=neighbor_k,
    )


def write_neighbor_geometry_json(summary: dict[str, Any], path: str | Path) -> Path:
    return write_json_payload(summary, path)


def write_neighbor_geometry_csv(summary: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "attribute",
                "sample_count",
                "neighbor_k",
                "mean_neighbor_match_fraction",
                "global_base_rate",
                "enrichment_over_base",
            ],
        )
        writer.writeheader()
        for attribute, metrics in (summary.get("metrics") or {}).items():
            writer.writerow(
                {
                    "attribute": attribute,
                    "sample_count": summary.get("sample_count"),
                    "neighbor_k": summary.get("neighbor_k"),
                    "mean_neighbor_match_fraction": metrics.get("mean_neighbor_match_fraction"),
                    "global_base_rate": metrics.get("global_base_rate"),
                    "enrichment_over_base": metrics.get("enrichment_over_base"),
                }
            )
    return target


def _whiten_session_features(features: torch.Tensor, *, epsilon: float = 1.0e-5) -> torch.Tensor:
    tensor = features.detach().cpu().to(dtype=torch.float32)
    if tensor.ndim != 2:
        raise ValueError("features must be a 2D tensor")
    sample_count, feature_dim = tensor.shape
    if sample_count == 0 or feature_dim == 0:
        return tensor.clone()
    centered = tensor - tensor.mean(dim=0, keepdim=True)
    if sample_count <= 1:
        return centered
    covariance = (centered.T @ centered) / float(max(sample_count - 1, 1))
    covariance = covariance + torch.eye(feature_dim, dtype=torch.float32) * float(epsilon)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    inverse_sqrt = eigenvectors @ torch.diag(eigenvalues.clamp_min(float(epsilon)).rsqrt()) @ eigenvectors.T
    return centered @ inverse_sqrt


def _orthogonal_procrustes(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.shape != target.shape:
        raise ValueError("source and target must have matching shapes")
    if source.ndim != 2:
        raise ValueError("source and target must be 2D matrices")
    cross_covariance = source.T @ target
    left, _, right_t = torch.linalg.svd(cross_covariance, full_matrices=False)
    rotation = left @ right_t
    if torch.det(rotation) < 0.0:
        left = left.clone()
        left[:, -1] *= -1.0
        rotation = left @ right_t
    return rotation


def _safe_cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left_norm = float(left.norm().item())
    right_norm = float(right.norm().item())
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    return float(torch.dot(left, right).item() / (left_norm * right_norm))


def _session_alignment_anchors(features: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, int]]:
    positive_mask = labels > 0.5
    negative_mask = ~positive_mask
    positive_count = int(positive_mask.sum().item())
    negative_count = int(negative_mask.sum().item())
    if positive_count <= 0 or negative_count <= 0:
        raise ValueError("Each session must contain at least one positive and one negative window for alignment.")
    positive_mean = features[positive_mask].mean(dim=0)
    negative_mean = features[negative_mask].mean(dim=0)
    label_axis = positive_mean - negative_mean
    anchors = torch.stack([positive_mean, negative_mean, label_axis], dim=0)
    return _normalized_features(anchors), {
        "positive_window_count": positive_count,
        "negative_window_count": negative_count,
    }


def summarize_session_alignment_geometry(
    *,
    features: torch.Tensor,
    labels: torch.Tensor,
    session_ids: tuple[str, ...],
    subject_ids: tuple[str, ...],
    neighbor_k: int,
    reference_session_id: str | None = None,
) -> dict[str, Any]:
    tensor = features.detach().cpu().to(dtype=torch.float32)
    label_tensor = labels.detach().cpu().to(dtype=torch.float32).reshape(-1)
    if tensor.ndim != 2:
        raise ValueError("features must be a 2D tensor")
    sample_count = int(tensor.shape[0])
    if int(label_tensor.shape[0]) != sample_count:
        raise ValueError("labels must align with features")
    if len(session_ids) != sample_count or len(subject_ids) != sample_count:
        raise ValueError("session_ids and subject_ids must align with features")
    if sample_count == 0:
        empty_summary = summarize_neighbor_geometry(
            features=torch.empty((0, 0), dtype=torch.float32),
            attributes={
                "label": (),
                "session_id": (),
                "subject_id": (),
            },
            neighbor_k=neighbor_k,
        )
        return {
            "sample_count": 0,
            "session_count": 0,
            "reference_session_id": None,
            "geometry_original": empty_summary,
            "geometry_whitened": empty_summary,
            "geometry_aligned": empty_summary,
            "aggregate_metrics": {},
            "per_session_metrics": [],
        }

    session_to_indices: dict[str, list[int]] = {}
    session_to_subject: dict[str, str] = {}
    for index, session_id in enumerate(session_ids):
        session_to_indices.setdefault(str(session_id), []).append(index)
        session_to_subject.setdefault(str(session_id), str(subject_ids[index]))

    valid_session_ids = [
        session_id
        for session_id, indices in sorted(session_to_indices.items())
        if len(indices) >= 2
    ]
    if not valid_session_ids:
        raise ValueError("At least one session with two or more selected windows is required for alignment.")
    resolved_reference_session_id = (
        str(reference_session_id)
        if reference_session_id is not None
        else max(valid_session_ids, key=lambda session_id: len(session_to_indices[session_id]))
    )
    if resolved_reference_session_id not in session_to_indices:
        raise ValueError(f"reference_session_id '{resolved_reference_session_id}' was not found in the selected sessions.")

    whitened_features = torch.empty_like(tensor)
    aligned_features = torch.empty_like(tensor)
    session_anchor_map: dict[str, torch.Tensor] = {}
    session_window_counts: dict[str, dict[str, int]] = {}
    for session_id in valid_session_ids:
        session_indices = torch.tensor(session_to_indices[session_id], dtype=torch.long)
        session_features = tensor.index_select(0, session_indices)
        session_labels = label_tensor.index_select(0, session_indices)
        whitened = _whiten_session_features(session_features)
        whitened_features.index_copy_(0, session_indices, whitened)
        anchors, counts = _session_alignment_anchors(whitened, session_labels)
        session_anchor_map[session_id] = anchors
        session_window_counts[session_id] = {
            "window_count": int(session_indices.numel()),
            **counts,
        }

    reference_anchors = session_anchor_map[resolved_reference_session_id]
    per_session_rows: list[dict[str, Any]] = []
    label_axis_before_values: list[float] = []
    label_axis_after_values: list[float] = []
    positive_before_values: list[float] = []
    positive_after_values: list[float] = []
    negative_before_values: list[float] = []
    negative_after_values: list[float] = []
    anchor_rmse_values: list[float] = []

    for session_id in valid_session_ids:
        session_indices = torch.tensor(session_to_indices[session_id], dtype=torch.long)
        session_whitened = whitened_features.index_select(0, session_indices)
        session_anchors = session_anchor_map[session_id]
        if session_id == resolved_reference_session_id:
            rotation = torch.eye(session_whitened.shape[1], dtype=torch.float32)
        else:
            rotation = _orthogonal_procrustes(session_anchors, reference_anchors)
        aligned = session_whitened @ rotation
        aligned_features.index_copy_(0, session_indices, aligned)
        aligned_anchors = session_anchors @ rotation

        label_before = _safe_cosine_similarity(session_anchors[2], reference_anchors[2])
        label_after = _safe_cosine_similarity(aligned_anchors[2], reference_anchors[2])
        positive_before = _safe_cosine_similarity(session_anchors[0], reference_anchors[0])
        positive_after = _safe_cosine_similarity(aligned_anchors[0], reference_anchors[0])
        negative_before = _safe_cosine_similarity(session_anchors[1], reference_anchors[1])
        negative_after = _safe_cosine_similarity(aligned_anchors[1], reference_anchors[1])
        anchor_rmse = float(torch.sqrt(torch.mean((aligned_anchors - reference_anchors) ** 2)).item())

        if session_id != resolved_reference_session_id:
            if label_before is not None:
                label_axis_before_values.append(label_before)
            if label_after is not None:
                label_axis_after_values.append(label_after)
            if positive_before is not None:
                positive_before_values.append(positive_before)
            if positive_after is not None:
                positive_after_values.append(positive_after)
            if negative_before is not None:
                negative_before_values.append(negative_before)
            if negative_after is not None:
                negative_after_values.append(negative_after)
            anchor_rmse_values.append(anchor_rmse)

        per_session_rows.append(
            {
                "session_id": session_id,
                "subject_id": session_to_subject[session_id],
                "window_count": session_window_counts[session_id]["window_count"],
                "positive_window_count": session_window_counts[session_id]["positive_window_count"],
                "negative_window_count": session_window_counts[session_id]["negative_window_count"],
                "is_reference_session": session_id == resolved_reference_session_id,
                "label_axis_cosine_to_reference_before": label_before,
                "label_axis_cosine_to_reference_after": label_after,
                "positive_centroid_cosine_to_reference_before": positive_before,
                "positive_centroid_cosine_to_reference_after": positive_after,
                "negative_centroid_cosine_to_reference_before": negative_before,
                "negative_centroid_cosine_to_reference_after": negative_after,
                "anchor_rmse_after_alignment": anchor_rmse,
            }
        )

    label_values = tuple("positive" if float(value) > 0.5 else "negative" for value in label_tensor.tolist())
    geometry_attributes = {
        "label": label_values,
        "session_id": tuple(str(value) for value in session_ids),
        "subject_id": tuple(str(value) for value in subject_ids),
    }
    original_geometry = summarize_neighbor_geometry(
        features=tensor,
        attributes=geometry_attributes,
        neighbor_k=neighbor_k,
    )
    whitened_geometry = summarize_neighbor_geometry(
        features=whitened_features,
        attributes=geometry_attributes,
        neighbor_k=neighbor_k,
    )
    aligned_geometry = summarize_neighbor_geometry(
        features=aligned_features,
        attributes=geometry_attributes,
        neighbor_k=neighbor_k,
    )

    def _mean_or_none(values: list[float]) -> float | None:
        return float(sum(values) / len(values)) if values else None

    return {
        "sample_count": sample_count,
        "session_count": len(valid_session_ids),
        "reference_session_id": resolved_reference_session_id,
        "neighbor_k": int(original_geometry.get("neighbor_k") or 0),
        "geometry_original": original_geometry,
        "geometry_whitened": whitened_geometry,
        "geometry_aligned": aligned_geometry,
        "aggregate_metrics": {
            "mean_label_axis_cosine_before": _mean_or_none(label_axis_before_values),
            "mean_label_axis_cosine_after": _mean_or_none(label_axis_after_values),
            "mean_positive_centroid_cosine_before": _mean_or_none(positive_before_values),
            "mean_positive_centroid_cosine_after": _mean_or_none(positive_after_values),
            "mean_negative_centroid_cosine_before": _mean_or_none(negative_before_values),
            "mean_negative_centroid_cosine_after": _mean_or_none(negative_after_values),
            "mean_anchor_rmse_after_alignment": _mean_or_none(anchor_rmse_values),
        },
        "per_session_metrics": per_session_rows,
    }


def write_session_alignment_json(summary: dict[str, Any], path: str | Path) -> Path:
    return write_json_payload(summary, path)


def write_session_alignment_csv(summary: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "session_id",
                "subject_id",
                "window_count",
                "positive_window_count",
                "negative_window_count",
                "is_reference_session",
                "label_axis_cosine_to_reference_before",
                "label_axis_cosine_to_reference_after",
                "positive_centroid_cosine_to_reference_before",
                "positive_centroid_cosine_to_reference_after",
                "negative_centroid_cosine_to_reference_before",
                "negative_centroid_cosine_to_reference_after",
                "anchor_rmse_after_alignment",
            ],
        )
        writer.writeheader()
        for row in summary.get("per_session_metrics") or []:
            writer.writerow(row)
    return target
