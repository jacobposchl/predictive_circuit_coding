from __future__ import annotations

import json
from pathlib import Path
import random

import torch

from predictive_circuit_coding.decoding import extract_frozen_tokens, fit_additive_probe
from predictive_circuit_coding.training.artifacts import load_training_checkpoint
from predictive_circuit_coding.training.config import ExperimentConfig
from predictive_circuit_coding.training.contracts import ValidationSummary


def _load_discovery_artifact(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _candidate_centroids(candidates: list[dict]) -> list[torch.Tensor]:
    grouped: dict[int, list[torch.Tensor]] = {}
    for candidate in candidates:
        cluster_id = int(candidate["cluster_id"])
        if cluster_id == -1:
            continue
        grouped.setdefault(cluster_id, []).append(torch.tensor(candidate["embedding"], dtype=torch.float32))
    return [torch.stack(values, dim=0).mean(dim=0) for _, values in sorted(grouped.items())]


def _cosine_similarity(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    lhs = lhs / lhs.norm().clamp_min(1.0e-8)
    rhs = rhs / rhs.norm().clamp_min(1.0e-8)
    return float(torch.dot(lhs, rhs).item())


def _provenance_issues(candidates: list[dict]) -> tuple[str, ...]:
    issues: list[str] = []
    for candidate in candidates:
        for key in ("recording_id", "session_id", "subject_id", "unit_id"):
            if not candidate.get(key):
                issues.append(f"missing_{key}:{candidate.get('candidate_id', 'unknown')}")
        if float(candidate["patch_end_s"]) <= float(candidate["patch_start_s"]):
            issues.append(f"bad_patch_interval:{candidate['candidate_id']}")
        if float(candidate["window_end_s"]) <= float(candidate["window_start_s"]):
            issues.append(f"bad_window_interval:{candidate['candidate_id']}")
    return tuple(issues)


def validate_discovery_artifact(
    *,
    experiment_config: ExperimentConfig,
    data_config_path: str | Path,
    checkpoint_path: str | Path,
    discovery_artifact_path: str | Path,
) -> ValidationSummary:
    artifact = _load_discovery_artifact(discovery_artifact_path)
    if not artifact.get("candidates"):
        raise ValueError(
            "Validation cannot run because the discovery artifact contains no candidate tokens."
        )
    discovery_collection = extract_frozen_tokens(
        experiment_config=experiment_config,
        data_config_path=data_config_path,
        checkpoint_path=checkpoint_path,
        split_name=experiment_config.splits.discovery,
        max_batches=experiment_config.discovery.max_batches,
    )
    real_fit = fit_additive_probe(
        tokens=discovery_collection.tokens,
        token_mask=discovery_collection.token_mask,
        labels=discovery_collection.labels,
        epochs=experiment_config.discovery.probe_epochs,
        learning_rate=experiment_config.discovery.probe_learning_rate,
    )
    rng = random.Random(experiment_config.discovery.shuffle_seed)
    shuffled_labels = discovery_collection.labels.clone()
    permutation = list(range(len(shuffled_labels)))
    rng.shuffle(permutation)
    shuffled_labels = shuffled_labels[torch.tensor(permutation, dtype=torch.long)]
    shuffled_fit = fit_additive_probe(
        tokens=discovery_collection.tokens,
        token_mask=discovery_collection.token_mask,
        labels=shuffled_labels,
        epochs=experiment_config.discovery.probe_epochs,
        learning_rate=experiment_config.discovery.probe_learning_rate,
    )

    test_collection = extract_frozen_tokens(
        experiment_config=experiment_config,
        data_config_path=data_config_path,
        checkpoint_path=checkpoint_path,
        split_name=experiment_config.splits.test,
        max_batches=experiment_config.evaluation.max_batches,
    )
    centroids = _candidate_centroids(artifact["candidates"])
    positive_test_records = [record for record in test_collection.records if record.label == 1]
    recurrence_hits = 0
    if centroids:
        for record in positive_test_records:
            embedding = torch.tensor(record.embedding, dtype=torch.float32)
            if max((_cosine_similarity(embedding, centroid) for centroid in centroids), default=-1.0) >= experiment_config.discovery.recurrence_similarity_threshold:
                recurrence_hits += 1

    checkpoint_state = load_training_checkpoint(checkpoint_path, map_location="cpu")
    metadata = checkpoint_state.get("metadata", {})
    cluster_count = len({int(candidate["cluster_id"]) for candidate in artifact["candidates"] if int(candidate["cluster_id"]) != -1})
    return ValidationSummary(
        dataset_id=experiment_config.dataset_id,
        checkpoint_path=str(checkpoint_path),
        discovery_artifact_path=str(discovery_artifact_path),
        real_label_metrics=real_fit.metrics,
        shuffled_label_metrics=shuffled_fit.metrics,
        baseline_sensitivity_summary={
            "evaluated_baseline_type": metadata.get("continuation_baseline_type", experiment_config.objective.continuation_baseline_type),
            "comparison_available": False,
        },
        candidate_count=len(artifact["candidates"]),
        cluster_count=cluster_count,
        stability_summary=artifact.get("stability_summary", {}),
        recurrence_summary={
            "positive_test_record_count": len(positive_test_records),
            "recurrence_hit_count": recurrence_hits,
            "recurrence_rate": (recurrence_hits / len(positive_test_records)) if positive_test_records else 0.0,
        },
        provenance_issues=_provenance_issues(artifact["candidates"]),
    )
