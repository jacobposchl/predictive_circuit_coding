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
