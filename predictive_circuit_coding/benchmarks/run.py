from __future__ import annotations

from dataclasses import replace
import json
import random
import shutil
from pathlib import Path
from typing import Any
from typing import Callable

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from predictive_circuit_coding.benchmarks.contracts import (
    BenchmarkArmSpec,
    BenchmarkTaskSpec,
    MotifBenchmarkResult,
)
from predictive_circuit_coding.benchmarks.features import (
    BenchmarkWindowCollection,
    apply_collection_transform,
    extract_benchmark_selected_windows,
    extract_benchmark_split_collection,
    fit_global_pca_transform,
    normalize_feature_rows,
    normalize_token_tensor,
    write_collection_token_shards,
)
from predictive_circuit_coding.benchmarks.reports import write_single_row_summary
from predictive_circuit_coding.data import resolve_runtime_dataset_view
from predictive_circuit_coding.decoding import (
    apply_session_linear_transforms_to_features,
    apply_session_linear_transforms_to_tokens,
    build_session_stratified_holdout_split,
    fit_additive_probe_features,
    fit_session_alignment_transforms,
    fit_session_whitening_transforms,
    summarize_neighbor_geometry,
)
from predictive_circuit_coding.decoding.scoring import select_candidate_tokens_from_shards
from predictive_circuit_coding.discovery.clustering import cluster_candidate_tokens
from predictive_circuit_coding.discovery.reporting import (
    build_discovery_cluster_report,
    write_discovery_cluster_report_csv,
    write_discovery_cluster_report_json,
)
from predictive_circuit_coding.discovery.run import (
    _ensure_binary_label_coverage,
    prepare_discovery_collection,
    write_discovery_artifact,
)
from predictive_circuit_coding.discovery.stability import estimate_clustering_stability
from predictive_circuit_coding.training.config import ExperimentConfig
from predictive_circuit_coding.training.contracts import CandidateTokenRecord, DecoderSummary, DiscoveryArtifact
from predictive_circuit_coding.utils.notebook import BenchmarkProgressEvent


_SIMILARITY_CHUNK_SIZE = 512
BenchmarkProgressCallback = Callable[[BenchmarkProgressEvent], None]


def _maybe_emit_benchmark_progress(
    progress_callback: BenchmarkProgressCallback | None,
    event: BenchmarkProgressEvent,
) -> None:
    if progress_callback is not None:
        progress_callback(event)


def default_benchmark_task_specs(
) -> tuple[BenchmarkTaskSpec, ...]:
    return (
        BenchmarkTaskSpec(name="stimulus_change", target_label="stimulus_change"),
        BenchmarkTaskSpec(name="trials_go", target_label="trials.go"),
    )


def default_motif_arm_specs() -> tuple[BenchmarkArmSpec, ...]:
    return (
        BenchmarkArmSpec("encoder_raw", "raw"),
        BenchmarkArmSpec("encoder_token_normalized", "token_normalized"),
        BenchmarkArmSpec("encoder_probe_weighted", "raw", candidate_geometry_mode="probe_weighted"),
        BenchmarkArmSpec(
            "encoder_aligned_oracle",
            "aligned_oracle",
            claim_safe=False,
            supervision_level="oracle_eval_labels",
        ),
    )


def _task_config(experiment_config: ExperimentConfig, task: BenchmarkTaskSpec) -> ExperimentConfig:
    return replace(
        experiment_config,
        discovery=replace(
            experiment_config.discovery,
            target_label=task.target_label,
            target_label_mode=task.target_label_mode,
            target_label_match_value=task.target_label_match_value,
        ),
    )


def _probe_logits_from_features(*, state_dict: dict[str, Any], features: torch.Tensor) -> torch.Tensor:
    weight = state_dict["linear.weight"].detach().cpu().reshape(-1).to(dtype=torch.float32)
    bias = float(state_dict["linear.bias"].detach().cpu().item())
    feature_tensor = features.detach().cpu().to(dtype=torch.float32)
    return (feature_tensor @ weight) + bias


def _probe_metrics_from_logits(*, sample_logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float | None]:
    probabilities = torch.sigmoid(sample_logits)
    predictions = (probabilities >= 0.5).to(dtype=labels.dtype)
    accuracy = float((predictions == labels).to(dtype=torch.float32).mean().item())
    bce = float(torch.nn.functional.binary_cross_entropy_with_logits(sample_logits, labels).item())
    labels_np = labels.detach().cpu().numpy()
    probabilities_np = probabilities.detach().cpu().numpy()
    roc_auc = None
    pr_auc = None
    if len({int(value) for value in labels_np.tolist()}) >= 2:
        roc_auc = float(roc_auc_score(labels_np, probabilities_np))
        pr_auc = float(average_precision_score(labels_np, probabilities_np))
    return {
        "probe_accuracy": accuracy,
        "probe_bce": bce,
        "probe_roc_auc": roc_auc,
        "probe_pr_auc": pr_auc,
        "positive_rate": float(labels.mean().item()),
    }


def _fit_shuffled_probe_features(
    *,
    features: torch.Tensor,
    labels: torch.Tensor,
    epochs: int,
    learning_rate: float,
    seed: int,
    label_name: str,
):
    shuffled_labels = labels.clone()
    permutation = list(range(len(shuffled_labels)))
    random.Random(int(seed)).shuffle(permutation)
    shuffled_labels = shuffled_labels[torch.tensor(permutation, dtype=torch.long)]
    return fit_additive_probe_features(
        features=features,
        labels=shuffled_labels,
        epochs=epochs,
        learning_rate=learning_rate,
        label_name=label_name,
    )


def _candidate_centroids(candidates: tuple[CandidateTokenRecord, ...]) -> list[torch.Tensor]:
    grouped: dict[int, list[torch.Tensor]] = {}
    for candidate in candidates:
        cluster_id = int(candidate.cluster_id)
        if cluster_id == -1:
            continue
        grouped.setdefault(cluster_id, []).append(torch.tensor(candidate.embedding, dtype=torch.float32))
    return [torch.stack(values, dim=0).mean(dim=0) for _, values in sorted(grouped.items())]


def _window_similarity_scores(
    *,
    tokens: torch.Tensor,
    token_mask: torch.Tensor,
    centroids: list[torch.Tensor],
) -> torch.Tensor:
    if tokens.numel() == 0 or token_mask.numel() == 0:
        return torch.empty((0,), dtype=torch.float32)
    if not centroids:
        return torch.zeros((tokens.shape[0],), dtype=torch.float32)
    centroid_tensor = torch.stack(
        [centroid / centroid.norm().clamp_min(1.0e-8) for centroid in centroids],
        dim=0,
    )
    scores = torch.empty((tokens.shape[0],), dtype=torch.float32)
    for start in range(0, tokens.shape[0], _SIMILARITY_CHUNK_SIZE):
        end = min(start + _SIMILARITY_CHUNK_SIZE, tokens.shape[0])
        chunk = tokens[start:end]
        normalized_chunk = chunk / chunk.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        chunk_sims = torch.einsum("wtd,cd->wtc", normalized_chunk, centroid_tensor)
        chunk_sims = chunk_sims.masked_fill(~token_mask[start:end].unsqueeze(-1), float("-inf"))
        scores[start:end] = chunk_sims.amax(dim=(1, 2)).to(dtype=torch.float32)
    return scores


def _held_out_similarity_summary(
    *,
    labels: torch.Tensor,
    window_session_ids: tuple[str, ...],
    window_scores: torch.Tensor,
) -> dict[str, Any]:
    labels_np = labels.detach().cpu().numpy()
    scores_np = window_scores.detach().cpu().numpy()
    if len({int(value) for value in labels_np.tolist()}) < 2:
        return {
            "window_roc_auc": None,
            "window_pr_auc": None,
            "positive_window_count": int((labels > 0.0).sum().item()),
            "negative_window_count": int((labels <= 0.0).sum().item()),
            "per_session_roc_auc": {},
            "comparison_available": False,
            "failure_reason": "held_out_split_missing_one_class",
        }
    per_session_roc_auc: dict[str, float] = {}
    for session_id in sorted(set(window_session_ids)):
        indices = [index for index, value in enumerate(window_session_ids) if value == session_id]
        session_labels = labels_np[indices]
        if len({int(value) for value in session_labels.tolist()}) < 2:
            continue
        per_session_roc_auc[session_id] = float(roc_auc_score(session_labels, scores_np[indices]))
    return {
        "window_roc_auc": float(roc_auc_score(labels_np, scores_np)),
        "window_pr_auc": float(average_precision_score(labels_np, scores_np)),
        "positive_window_count": int((labels > 0.0).sum().item()),
        "negative_window_count": int((labels <= 0.0).sum().item()),
        "per_session_roc_auc": per_session_roc_auc,
        "comparison_available": True,
    }


def _encoder_arm_metadata(arm: BenchmarkArmSpec) -> dict[str, str | bool]:
    return {
        "encoder_training_status": "trained",
        "encoder_checkpoint_loaded": True,
        "claim_safe": bool(arm.claim_safe),
        "supervision_level": arm.supervision_level,
    }


def _transform_summary_row(
    *,
    experiment_config: ExperimentConfig,
    task: BenchmarkTaskSpec,
    arm: BenchmarkArmSpec,
    pca_summary: dict[str, Any] | None,
    whitening_summary: dict[str, Any] | None,
    test_whitening_summary: dict[str, Any] | None,
    token_normalization_summary: dict[str, Any] | None = None,
    alignment_summary: dict[str, Any] | None = None,
    candidate_geometry_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    aggregate = (whitening_summary or {}).get("aggregate_metrics") or {}
    encoder_metadata = _encoder_arm_metadata(arm)
    return {
        "task_name": task.name,
        "target_label": task.target_label,
        "target_label_match_value": task.target_label_match_value,
        "arm_name": arm.name,
        "feature_family": "encoder",
        **encoder_metadata,
        "geometry_mode": arm.geometry_mode,
        "candidate_geometry_mode": arm.candidate_geometry_mode,
        "variant_name": experiment_config.experiment.variant_name,
        "whitening_session_count": (whitening_summary or {}).get("session_count"),
        "whitening_reference_session_id": (whitening_summary or {}).get("reference_session_id"),
        "whitening_mean_label_axis_cosine_before": aggregate.get("mean_label_axis_cosine_before"),
        "whitening_mean_label_axis_cosine_after": aggregate.get("mean_label_axis_cosine_after"),
        "test_whitening_session_count": (test_whitening_summary or {}).get("session_count"),
        "token_normalization_applied": token_normalization_summary is not None,
        "alignment_session_count": (alignment_summary or {}).get("session_count"),
        "candidate_geometry_mode": arm.candidate_geometry_mode,
        "candidate_geometry_transform": (candidate_geometry_summary or {}).get("transform_type"),
    }


def _apply_session_whitening_to_collection(
    *,
    collection: BenchmarkWindowCollection,
    fit_indices: torch.Tensor,
) -> tuple[BenchmarkWindowCollection, dict[str, Any], dict[str, Any]]:
    transforms, summary = fit_session_whitening_transforms(
        features=collection.pooled_features,
        session_ids=collection.window_session_ids,
        fit_indices=fit_indices,
    )
    transformed = BenchmarkWindowCollection(
        feature_family=collection.feature_family,
        pooled_features=apply_session_linear_transforms_to_features(
            features=collection.pooled_features,
            session_ids=collection.window_session_ids,
            transforms=transforms,
        ),
        token_tensors=apply_session_linear_transforms_to_tokens(
            tokens=collection.token_tensors,
            session_ids=collection.window_session_ids,
            transforms=transforms,
        ),
        token_mask=collection.token_mask,
        labels=collection.labels,
        window_recording_ids=collection.window_recording_ids,
        window_session_ids=collection.window_session_ids,
        window_subject_ids=collection.window_subject_ids,
        window_start_s=collection.window_start_s,
        window_end_s=collection.window_end_s,
        shard_paths=collection.shard_paths,
        coverage_summary=collection.coverage_summary,
    )
    return transformed, summary, transforms


def _representation_row(
    *,
    experiment_config: ExperimentConfig,
    task: BenchmarkTaskSpec,
    arm: BenchmarkArmSpec,
    status: str,
    failure_reason: str | None,
    discovery_fit_metrics: dict[str, Any] | None,
    discovery_shuffled_metrics: dict[str, Any] | None,
    discovery_heldout_metrics: dict[str, Any] | None,
    test_metrics: dict[str, Any] | None,
    geometry_summary: dict[str, Any] | None,
    summary_path: Path,
    transform_summary_path: Path,
) -> dict[str, Any]:
    geometry_metrics = (geometry_summary or {}).get("metrics") or {}
    label_geometry = geometry_metrics.get("label") or {}
    session_geometry = geometry_metrics.get("session_id") or {}
    subject_geometry = geometry_metrics.get("subject_id") or {}
    encoder_metadata = _encoder_arm_metadata(arm)
    return {
        "task_name": task.name,
        "target_label": task.target_label,
        "target_label_match_value": task.target_label_match_value,
        "arm_name": arm.name,
        "feature_family": "encoder",
        **encoder_metadata,
        "geometry_mode": arm.geometry_mode,
        "candidate_geometry_mode": arm.candidate_geometry_mode,
        "variant_name": experiment_config.experiment.variant_name,
        "status": status,
        "failure_reason": failure_reason,
        "discovery_fit_probe_accuracy": (discovery_fit_metrics or {}).get("probe_accuracy"),
        "discovery_fit_probe_bce": (discovery_fit_metrics or {}).get("probe_bce"),
        "discovery_shuffled_probe_accuracy": (discovery_shuffled_metrics or {}).get("probe_accuracy"),
        "discovery_shuffled_probe_bce": (discovery_shuffled_metrics or {}).get("probe_bce"),
        "within_session_probe_accuracy": (discovery_heldout_metrics or {}).get("probe_accuracy"),
        "within_session_probe_bce": (discovery_heldout_metrics or {}).get("probe_bce"),
        "within_session_probe_roc_auc": (discovery_heldout_metrics or {}).get("probe_roc_auc"),
        "within_session_probe_pr_auc": (discovery_heldout_metrics or {}).get("probe_pr_auc"),
        "test_probe_accuracy": (test_metrics or {}).get("probe_accuracy"),
        "test_probe_bce": (test_metrics or {}).get("probe_bce"),
        "test_probe_roc_auc": (test_metrics or {}).get("probe_roc_auc"),
        "test_probe_pr_auc": (test_metrics or {}).get("probe_pr_auc"),
        "label_neighbor_enrichment": label_geometry.get("enrichment_over_base"),
        "session_neighbor_enrichment": session_geometry.get("enrichment_over_base"),
        "subject_neighbor_enrichment": subject_geometry.get("enrichment_over_base"),
        "summary_path": str(summary_path),
        "transform_summary_path": str(transform_summary_path),
    }


def _motif_row(
    *,
    experiment_config: ExperimentConfig,
    task: BenchmarkTaskSpec,
    arm: BenchmarkArmSpec,
    status: str,
    failure_reason: str | None,
    fit_metrics: dict[str, Any] | None,
    shuffled_metrics: dict[str, Any] | None,
    test_metrics: dict[str, Any] | None,
    held_out_similarity_summary: dict[str, Any] | None,
    cluster_quality_summary: dict[str, Any] | None,
    candidate_count: int,
    cluster_count: int,
    cluster_report: dict[str, Any],
    candidate_selection_fallback_used: bool,
    candidate_selection_effective_min_score: float | None,
    summary_path: Path,
    cluster_summary_path: Path,
    discovery_artifact_path: Path,
    transform_summary_path: Path,
) -> dict[str, Any]:
    clusters = cluster_report.get("clusters", []) if isinstance(cluster_report, dict) else []
    max_session_spread = max((int(cluster.get("session_count", 0)) for cluster in clusters), default=0)
    max_subject_spread = max((int(cluster.get("subject_count", 0)) for cluster in clusters), default=0)
    encoder_metadata = _encoder_arm_metadata(arm)
    return {
        "task_name": task.name,
        "target_label": task.target_label,
        "target_label_match_value": task.target_label_match_value,
        "arm_name": arm.name,
        "feature_family": "encoder",
        **encoder_metadata,
        "geometry_mode": arm.geometry_mode,
        "variant_name": experiment_config.experiment.variant_name,
        "status": status,
        "failure_reason": failure_reason,
        "candidate_count": candidate_count,
        "cluster_count": cluster_count,
        "candidate_selection_fallback_used": candidate_selection_fallback_used,
        "candidate_selection_effective_min_score": candidate_selection_effective_min_score,
        "fit_probe_accuracy": (fit_metrics or {}).get("probe_accuracy"),
        "fit_probe_bce": (fit_metrics or {}).get("probe_bce"),
        "shuffled_probe_accuracy": (shuffled_metrics or {}).get("probe_accuracy"),
        "shuffled_probe_bce": (shuffled_metrics or {}).get("probe_bce"),
        "held_out_test_probe_accuracy": (test_metrics or {}).get("probe_accuracy"),
        "held_out_test_probe_bce": (test_metrics or {}).get("probe_bce"),
        "held_out_test_probe_roc_auc": (test_metrics or {}).get("probe_roc_auc"),
        "held_out_test_probe_pr_auc": (test_metrics or {}).get("probe_pr_auc"),
        "held_out_similarity_roc_auc": (held_out_similarity_summary or {}).get("window_roc_auc"),
        "held_out_similarity_pr_auc": (held_out_similarity_summary or {}).get("window_pr_auc"),
        "cluster_persistence_mean": (cluster_quality_summary or {}).get("cluster_persistence_mean"),
        "silhouette_score": (cluster_quality_summary or {}).get("silhouette_score"),
        "max_cluster_session_spread": max_session_spread,
        "max_cluster_subject_spread": max_subject_spread,
        "summary_path": str(summary_path),
        "cluster_summary_path": str(cluster_summary_path),
        "discovery_artifact_path": str(discovery_artifact_path),
        "transform_summary_path": str(transform_summary_path),
    }


def _all_indices(count: int) -> torch.Tensor:
    return torch.arange(int(count), dtype=torch.long)


def _task_direct_field_inventory(
    *,
    dataset_view,
    split_name: str,
    target_label: str,
) -> dict[str, Any]:
    import numpy as np

    from predictive_circuit_coding.data import load_temporaldata_session
    from predictive_circuit_coding.windowing.dataset import split_session_ids

    namespace, dot, field = str(target_label).partition(".")
    if not dot or not namespace or not field or "." in field:
        return {
            "available": True,
            "reason": None,
            "value_counts": {},
            "field_checked": False,
        }
    session_ids = split_session_ids(dataset_view.split_manifest, split_name)
    value_counts: dict[str, int] = {}
    sessions_with_field = 0
    for session_id in session_ids:
        session_path = dataset_view.workspace.brainset_prepared_root / f"{session_id}.h5"
        if not session_path.is_file():
            continue
        session = load_temporaldata_session(session_path, lazy=False)
        interval = getattr(session, namespace, None)
        raw_values = getattr(interval, field, None) if interval is not None else None
        if raw_values is None:
            continue
        sessions_with_field += 1
        for raw_value in np.asarray(raw_values, dtype=object).reshape(-1):
            if raw_value is None:
                continue
            if isinstance(raw_value, bytes):
                value = raw_value.decode("utf-8", errors="replace").strip()
            else:
                value = str(raw_value).strip()
            if not value or value.lower() == "nan":
                continue
            value_counts[value] = value_counts.get(value, 0) + 1
    if sessions_with_field <= 0:
        return {
            "available": False,
            "reason": "skipped_missing_field",
            "value_counts": {},
            "field_checked": True,
        }
    return {
        "available": True,
        "reason": None,
        "value_counts": value_counts,
        "field_checked": True,
    }


def _task_skip_status(
    *,
    task: BenchmarkTaskSpec,
    dataset_view,
    discovery_split_name: str,
) -> tuple[str | None, str | None]:
    if not task.optional:
        return None, None
    inventory = _task_direct_field_inventory(
        dataset_view=dataset_view,
        split_name=discovery_split_name,
        target_label=task.target_label,
    )
    if not inventory["available"]:
        return str(inventory["reason"]), inventory["reason"]
    if task.target_label_match_value is None:
        return "skipped_missing_match_value", "optional_task_requires_target_label_match_value"
    if inventory["field_checked"]:
        value_counts = inventory["value_counts"]
        if task.target_label_match_value not in value_counts:
            return "skipped_missing_match_value", (
                f"target_label_match_value='{task.target_label_match_value}' not found in discovery split"
            )
    return None, None


def _collection_subset_metrics(
    *,
    state_dict: dict[str, Any],
    collection: BenchmarkWindowCollection,
    indices: torch.Tensor,
) -> dict[str, Any]:
    subset_features = collection.pooled_features.index_select(0, indices)
    subset_labels = collection.labels.index_select(0, indices)
    logits = _probe_logits_from_features(state_dict=state_dict, features=subset_features)
    return _probe_metrics_from_logits(sample_logits=logits, labels=subset_labels)


def _fit_optional_pca(
    *,
    arm: BenchmarkArmSpec,
    discovery_collection: BenchmarkWindowCollection,
    test_collection: BenchmarkWindowCollection,
    fit_indices: torch.Tensor,
) -> tuple[BenchmarkWindowCollection, BenchmarkWindowCollection, Any, dict[str, Any]]:
    if not arm.use_pca:
        return discovery_collection, test_collection, None, {
            "transform_type": "pca",
            "applied": False,
            "components": None,
            "input_dim": int(discovery_collection.pooled_features.shape[1]) if discovery_collection.pooled_features.ndim == 2 else 0,
            "explained_variance_ratio_sum": None,
        }
    transform, pca_summary = fit_global_pca_transform(
        features=discovery_collection.pooled_features,
        fit_indices=fit_indices,
        max_components=arm.pca_components,
    )
    return (
        apply_collection_transform(collection=discovery_collection, transform=transform),
        apply_collection_transform(collection=test_collection, transform=transform),
        transform,
        pca_summary,
    )


def _build_whitened_test_collection(
    *,
    collection: BenchmarkWindowCollection,
) -> tuple[BenchmarkWindowCollection, dict[str, Any], dict[str, Any]]:
    fit_indices = _all_indices(collection.pooled_features.shape[0])
    return _apply_session_whitening_to_collection(collection=collection, fit_indices=fit_indices)


def _normalize_collection_tokens(collection: BenchmarkWindowCollection) -> tuple[BenchmarkWindowCollection, dict[str, Any]]:
    normalized = BenchmarkWindowCollection(
        feature_family=collection.feature_family,
        pooled_features=normalize_feature_rows(collection.pooled_features),
        token_tensors=normalize_token_tensor(collection.token_tensors, collection.token_mask),
        token_mask=collection.token_mask,
        labels=collection.labels,
        window_recording_ids=collection.window_recording_ids,
        window_session_ids=collection.window_session_ids,
        window_subject_ids=collection.window_subject_ids,
        window_start_s=collection.window_start_s,
        window_end_s=collection.window_end_s,
        shard_paths=collection.shard_paths,
        coverage_summary=collection.coverage_summary,
    )
    zero_feature_count = int((collection.pooled_features.norm(dim=-1) <= 1.0e-8).sum().item())
    return normalized, {
        "transform_type": "token_normalized",
        "feature_count": int(collection.pooled_features.shape[0]),
        "zero_feature_count": zero_feature_count,
    }


def _apply_oracle_alignment_to_collection(
    *,
    collection: BenchmarkWindowCollection,
) -> tuple[BenchmarkWindowCollection, dict[str, Any], dict[str, Any]]:
    fit_indices = _all_indices(collection.pooled_features.shape[0])
    transforms, summary = fit_session_alignment_transforms(
        features=collection.pooled_features,
        labels=collection.labels,
        session_ids=collection.window_session_ids,
        fit_indices=fit_indices,
    )
    transformed = BenchmarkWindowCollection(
        feature_family=collection.feature_family,
        pooled_features=apply_session_linear_transforms_to_features(
            features=collection.pooled_features,
            session_ids=collection.window_session_ids,
            transforms=transforms,
        ),
        token_tensors=apply_session_linear_transforms_to_tokens(
            tokens=collection.token_tensors,
            session_ids=collection.window_session_ids,
            transforms=transforms,
        ),
        token_mask=collection.token_mask,
        labels=collection.labels,
        window_recording_ids=collection.window_recording_ids,
        window_session_ids=collection.window_session_ids,
        window_subject_ids=collection.window_subject_ids,
        window_start_s=collection.window_start_s,
        window_end_s=collection.window_end_s,
        shard_paths=collection.shard_paths,
        coverage_summary=collection.coverage_summary,
    )
    return transformed, summary, transforms


def _probe_weight_scale(state_dict: dict[str, Any]) -> torch.Tensor:
    weight = state_dict["linear.weight"].detach().cpu().reshape(-1).to(dtype=torch.float32).abs()
    return weight / weight.mean().clamp_min(1.0e-8)


def _apply_probe_weight_to_candidates(
    candidates: tuple[CandidateTokenRecord, ...],
    *,
    state_dict: dict[str, Any],
) -> tuple[tuple[CandidateTokenRecord, ...], dict[str, Any]]:
    from dataclasses import replace

    scale = _probe_weight_scale(state_dict)
    transformed: list[CandidateTokenRecord] = []
    for candidate in candidates:
        embedding = torch.tensor(candidate.embedding, dtype=torch.float32)
        transformed.append(replace(candidate, embedding=tuple((embedding * scale).tolist())))
    return tuple(transformed), {
        "transform_type": "probe_weighted",
        "dimension": int(scale.numel()),
        "scale_min": float(scale.min().item()) if scale.numel() else None,
        "scale_max": float(scale.max().item()) if scale.numel() else None,
        "scale_mean": float(scale.mean().item()) if scale.numel() else None,
    }


def _apply_probe_weight_to_tokens(tokens: torch.Tensor, *, state_dict: dict[str, Any]) -> torch.Tensor:
    scale = _probe_weight_scale(state_dict).view(1, 1, -1)
    return tokens.detach().cpu().to(dtype=torch.float32) * scale


def _default_cluster_stats() -> dict[str, Any]:
    return {
        "cluster_count": 0.0,
        "noise_count": 0.0,
        "non_noise_fraction": 0.0,
        "silhouette_score": None,
        "cluster_persistence_mean": None,
        "cluster_persistence_min": None,
        "cluster_persistence_max": None,
        "cluster_persistence_by_cluster": {},
    }


def _empty_similarity_summary(reason: str) -> dict[str, Any]:
    return {
        "window_roc_auc": None,
        "window_pr_auc": None,
        "positive_window_count": None,
        "negative_window_count": None,
        "per_session_roc_auc": {},
        "comparison_available": False,
        "failure_reason": str(reason),
    }


def _removed_feature_matrix(
    *,
    experiment_config: ExperimentConfig,
    data_config_path: str | Path,
    checkpoint_path: str | Path,
    output_root: str | Path,
    task_specs: tuple[BenchmarkTaskSpec, ...] | None = None,
    arm_specs: tuple[BenchmarkArmSpec, ...] | None = None,
    discovery_split_name: str | None = None,
    test_split_name: str | None = None,
    session_holdout_fraction: float = 0.5,
    session_holdout_seed: int | None = None,
    test_max_batches: int | None = None,
    neighbor_k: int = 5,
    progress_callback: BenchmarkProgressCallback | None = None,
) -> tuple[RepresentationBenchmarkResult, ...]:
    dataset_view = resolve_runtime_dataset_view(
        experiment_config=experiment_config,
        data_config_path=data_config_path,
    )
    tasks = task_specs or default_benchmark_task_specs()
    arms = arm_specs or default_motif_arm_specs()
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    resolved_discovery_split = discovery_split_name or experiment_config.splits.discovery
    resolved_test_split = test_split_name or experiment_config.splits.test
    resolved_holdout_seed = (
        int(session_holdout_seed)
        if session_holdout_seed is not None
        else int(experiment_config.discovery.shuffle_seed)
    )
    representation_tasks = tuple(task for task in tasks if task.include_in_representation)
    results: list[RepresentationBenchmarkResult] = []
    total_arms = len(representation_tasks) * len(arms)
    completed_arms = 0

    for task_index, task in enumerate(representation_tasks, start=1):
        _maybe_emit_benchmark_progress(
            progress_callback,
            BenchmarkProgressEvent(
                benchmark_name="removed_feature_matrix",
                event_type="task_start",
                task_name=task.name,
                task_index=task_index,
                task_total=len(representation_tasks),
                arm_total=total_arms,
                message=f"removed feature task {task.name}",
            ),
        )
        if not task.include_in_representation:
            continue
        skip_status, skip_reason = _task_skip_status(
            task=task,
            dataset_view=dataset_view,
            discovery_split_name=resolved_discovery_split,
        )
        task_config = _task_config(experiment_config, task)
        task_output_root = output_root_path / task.name
        task_output_root.mkdir(parents=True, exist_ok=True)
        family_cache: dict[str, tuple[BenchmarkWindowCollection, BenchmarkWindowCollection, Any]] = {}

        for arm_index, arm in enumerate(arms, start=1):
            arm_root = task_output_root / arm.name
            arm_root.mkdir(parents=True, exist_ok=True)
            summary_json_path = arm_root / "summary.json"
            summary_csv_path = arm_root / "summary.csv"
            transform_summary_json_path = arm_root / "transform_summary.json"
            transform_summary_csv_path = arm_root / "transform_summary.csv"
            _maybe_emit_benchmark_progress(
                progress_callback,
                BenchmarkProgressEvent(
                    benchmark_name="removed_feature_matrix",
                    event_type="arm_start",
                    task_name=task.name,
                    task_index=task_index,
                    task_total=len(representation_tasks),
                    arm_name=arm.name,
                    arm_index=completed_arms + 1,
                    arm_total=total_arms,
                    message=f"{task.name} / {arm.name}",
                ),
            )

            if skip_status is not None:
                row = _representation_row(
                    experiment_config=experiment_config,
                    task=task,
                    arm=arm,
                    status=skip_status,
                    failure_reason=skip_reason,
                    discovery_fit_metrics=None,
                    discovery_shuffled_metrics=None,
                    discovery_heldout_metrics=None,
                    test_metrics=None,
                    geometry_summary=None,
                    summary_path=summary_json_path,
                    transform_summary_path=transform_summary_json_path,
                )
                write_single_row_summary(
                    row,
                    output_json_path=summary_json_path,
                    output_csv_path=summary_csv_path,
                    root_key="removed_feature_matrix",
                )
                transform_payload = {
                    "task_name": task.name,
                    "arm_name": arm.name,
                    "variant_name": experiment_config.experiment.variant_name,
                    "claim_safe": bool(arm.claim_safe),
                    "supervision_level": arm.supervision_level,
                    "status": skip_status,
                    "failure_reason": skip_reason,
                }
                write_single_row_summary(
                    transform_payload,
                    output_json_path=transform_summary_json_path,
                    output_csv_path=transform_summary_csv_path,
                    root_key="transform_summary",
                )
                transform_summary_json_path.write_text(json.dumps(transform_payload, indent=2), encoding="utf-8")
                results.append(
                    RepresentationBenchmarkResult(
                        task_name=task.name,
                        arm_name=arm.name,
                        status=skip_status,
                        summary=row,
                        summary_json_path=summary_json_path,
                        summary_csv_path=summary_csv_path,
                        transform_summary_json_path=transform_summary_json_path,
                        transform_summary_csv_path=transform_summary_csv_path,
                    )
                )
                completed_arms += 1
                _maybe_emit_benchmark_progress(
                    progress_callback,
                    BenchmarkProgressEvent(
                        benchmark_name="representation",
                        event_type="arm_end",
                        task_name=task.name,
                        task_index=task_index,
                        task_total=len(representation_tasks),
                        arm_name=arm.name,
                        arm_index=completed_arms,
                        arm_total=total_arms,
                        status=skip_status,
                        metrics=row,
                    ),
                )
                continue

            try:
                cached = family_cache.get("encoder")
                if cached is None:
                    _maybe_emit_benchmark_progress(
                        progress_callback,
                        BenchmarkProgressEvent(
                            benchmark_name="removed_feature_matrix",
                            event_type="arm_step",
                            task_name=task.name,
                            arm_name=arm.name,
                            step_name="discovery_extraction",
                            current=0,
                            total=None,
                            message="extract discovery windows",
                        ),
                    )
                    window_plan = prepare_discovery_collection(
                        experiment_config=task_config,
                        data_config_path=data_config_path,
                        split_name=resolved_discovery_split,
                        dataset_view=dataset_view,
                    )
                    discovery_collection = extract_benchmark_selected_windows(
                        experiment_config=task_config,
                        data_config_path=data_config_path,
                        feature_family="encoder",
                        checkpoint_path=checkpoint_path,
                        dataset_view=dataset_view,
                        window_plan=window_plan,
                        progress_callback=lambda current, total, task_name=task.name, arm_name=arm.name: _maybe_emit_benchmark_progress(
                            progress_callback,
                            BenchmarkProgressEvent(
                                benchmark_name="representation",
                                event_type="arm_step",
                                task_name=task_name,
                                arm_name=arm_name,
                                step_name="discovery_extraction",
                                current=current,
                                total=total,
                                message="extract discovery windows",
                            ),
                        ),
                    )
                    _ensure_binary_label_coverage(discovery_collection.coverage_summary)  # type: ignore[arg-type]
                    _maybe_emit_benchmark_progress(
                        progress_callback,
                        BenchmarkProgressEvent(
                            benchmark_name="removed_feature_matrix",
                            event_type="arm_step",
                            task_name=task.name,
                            arm_name=arm.name,
                            step_name="test_extraction",
                            current=0,
                            total=test_max_batches or task_config.evaluation.max_batches,
                            message="extract test split",
                        ),
                    )
                    test_collection = extract_benchmark_split_collection(
                        experiment_config=task_config,
                        data_config_path=data_config_path,
                        feature_family="encoder",
                        split_name=resolved_test_split,
                        target_label=task.target_label,
                        target_label_mode=task.target_label_mode,
                        target_label_match_value=task.target_label_match_value,
                        checkpoint_path=checkpoint_path,
                        dataset_view=dataset_view,
                        max_batches=test_max_batches or task_config.evaluation.max_batches,
                        progress_callback=lambda current, total, task_name=task.name, arm_name=arm.name: _maybe_emit_benchmark_progress(
                            progress_callback,
                            BenchmarkProgressEvent(
                                benchmark_name="representation",
                                event_type="arm_step",
                                task_name=task_name,
                                arm_name=arm_name,
                                step_name="test_extraction",
                                current=current,
                                total=total,
                                message="extract test split",
                            ),
                        ),
                    )
                    split = build_session_stratified_holdout_split(
                        labels=discovery_collection.labels,
                        session_ids=discovery_collection.window_session_ids,
                        subject_ids=discovery_collection.window_subject_ids,
                        holdout_fraction=session_holdout_fraction,
                        seed=resolved_holdout_seed,
                    )
                    family_cache["encoder"] = (discovery_collection, test_collection, split)
                    cached = family_cache["encoder"]
                else:
                    _maybe_emit_benchmark_progress(
                        progress_callback,
                        BenchmarkProgressEvent(
                            benchmark_name="removed_feature_matrix",
                            event_type="arm_step",
                            task_name=task.name,
                            arm_name=arm.name,
                            step_name="reuse_cached_features",
                            current=1,
                            total=1,
                            message="reuse encoder features",
                        ),
                    )

                discovery_collection, test_collection, split = cached
                if arm.use_pca:
                    _maybe_emit_benchmark_progress(
                        progress_callback,
                        BenchmarkProgressEvent(
                            benchmark_name="removed_feature_matrix",
                            event_type="arm_step",
                            task_name=task.name,
                            arm_name=arm.name,
                            step_name="pca",
                            current=0,
                            total=1,
                            message="fit/apply PCA",
                        ),
                    )
                    transformed_discovery, transformed_test, _, pca_summary = _fit_optional_pca(
                        arm=arm,
                        discovery_collection=discovery_collection,
                        test_collection=test_collection,
                        fit_indices=split.fit_indices,
                    )
                    _maybe_emit_benchmark_progress(
                        progress_callback,
                        BenchmarkProgressEvent(
                            benchmark_name="removed_feature_matrix",
                            event_type="arm_step",
                            task_name=task.name,
                            arm_name=arm.name,
                            step_name="pca",
                            current=1,
                            total=1,
                            message="fit/apply PCA",
                        ),
                    )
                else:
                    transformed_discovery = discovery_collection
                    transformed_test = test_collection
                    pca_summary = None

                whitening_summary = None
                test_whitening_summary = None
                if arm.geometry_mode == "whitened":
                    _maybe_emit_benchmark_progress(
                        progress_callback,
                        BenchmarkProgressEvent(
                            benchmark_name="removed_feature_matrix",
                            event_type="arm_step",
                            task_name=task.name,
                            arm_name=arm.name,
                            step_name="whitening",
                            current=0,
                            total=1,
                            message="apply whitening",
                        ),
                    )
                    transformed_discovery, whitening_summary, _ = _apply_session_whitening_to_collection(
                        collection=transformed_discovery,
                        fit_indices=split.fit_indices,
                    )
                    transformed_test, test_whitening_summary, _ = _build_whitened_test_collection(
                        collection=transformed_test,
                    )
                    _maybe_emit_benchmark_progress(
                        progress_callback,
                        BenchmarkProgressEvent(
                            benchmark_name="removed_feature_matrix",
                            event_type="arm_step",
                            task_name=task.name,
                            arm_name=arm.name,
                            step_name="whitening",
                            current=1,
                            total=1,
                            message="apply whitening",
                        ),
                    )

                fit_features = transformed_discovery.pooled_features.index_select(0, split.fit_indices)
                fit_labels = transformed_discovery.labels.index_select(0, split.fit_indices)
                _maybe_emit_benchmark_progress(
                    progress_callback,
                    BenchmarkProgressEvent(
                            benchmark_name="removed_feature_matrix",
                        event_type="arm_step",
                        task_name=task.name,
                        arm_name=arm.name,
                        step_name="probe_fit",
                        current=0,
                        total=1,
                        message="fit additive probe",
                    ),
                )
                discovery_fit = fit_additive_probe_features(
                    features=fit_features,
                    labels=fit_labels,
                    epochs=task_config.discovery.probe_epochs,
                    learning_rate=task_config.discovery.probe_learning_rate,
                    label_name=task.target_label,
                )
                _maybe_emit_benchmark_progress(
                    progress_callback,
                    BenchmarkProgressEvent(
                            benchmark_name="removed_feature_matrix",
                        event_type="arm_step",
                        task_name=task.name,
                        arm_name=arm.name,
                        step_name="probe_fit",
                        current=1,
                        total=1,
                        message="fit additive probe",
                    ),
                )
                _maybe_emit_benchmark_progress(
                    progress_callback,
                    BenchmarkProgressEvent(
                            benchmark_name="removed_feature_matrix",
                        event_type="arm_step",
                        task_name=task.name,
                        arm_name=arm.name,
                        step_name="shuffled_probe_fit",
                        current=0,
                        total=1,
                        message="fit shuffled probe",
                    ),
                )
                shuffled_fit = _fit_shuffled_probe_features(
                    features=fit_features,
                    labels=fit_labels,
                    epochs=task_config.discovery.probe_epochs,
                    learning_rate=task_config.discovery.probe_learning_rate,
                    seed=task_config.discovery.shuffle_seed,
                    label_name=task.target_label,
                )
                _maybe_emit_benchmark_progress(
                    progress_callback,
                    BenchmarkProgressEvent(
                            benchmark_name="removed_feature_matrix",
                        event_type="arm_step",
                        task_name=task.name,
                        arm_name=arm.name,
                        step_name="shuffled_probe_fit",
                        current=1,
                        total=1,
                        message="fit shuffled probe",
                    ),
                )
                discovery_heldout_metrics = _collection_subset_metrics(
                    state_dict=discovery_fit.state_dict,
                    collection=transformed_discovery,
                    indices=split.heldout_indices,
                )
                test_logits = _probe_logits_from_features(
                    state_dict=discovery_fit.state_dict,
                    features=transformed_test.pooled_features,
                )
                test_metrics = _probe_metrics_from_logits(
                    sample_logits=test_logits,
                    labels=transformed_test.labels,
                )
                geometry_summary = summarize_neighbor_geometry(
                    features=transformed_discovery.pooled_features,
                    attributes={
                        "label": tuple("positive" if float(value) > 0.0 else "negative" for value in transformed_discovery.labels.tolist()),
                        "session_id": transformed_discovery.window_session_ids,
                        "subject_id": transformed_discovery.window_subject_ids,
                    },
                    neighbor_k=neighbor_k,
                )
                _maybe_emit_benchmark_progress(
                    progress_callback,
                    BenchmarkProgressEvent(
                            benchmark_name="removed_feature_matrix",
                        event_type="arm_step",
                        task_name=task.name,
                        arm_name=arm.name,
                        step_name="geometry_summary",
                        current=1,
                        total=1,
                        message="neighbor geometry summary",
                    ),
                )
                status = "ok"
                failure_reason = None
            except Exception as exc:
                status = "degraded"
                failure_reason = str(exc)
                discovery_fit = None
                shuffled_fit = None
                discovery_heldout_metrics = None
                test_metrics = None
                geometry_summary = None
                pca_summary = None
                whitening_summary = None
                test_whitening_summary = None

            row = _representation_row(
                experiment_config=experiment_config,
                task=task,
                arm=arm,
                status=status,
                failure_reason=failure_reason,
                discovery_fit_metrics=(discovery_fit.metrics if discovery_fit is not None else None),
                discovery_shuffled_metrics=(shuffled_fit.metrics if shuffled_fit is not None else None),
                discovery_heldout_metrics=discovery_heldout_metrics,
                test_metrics=test_metrics,
                geometry_summary=geometry_summary,
                summary_path=summary_json_path,
                transform_summary_path=transform_summary_json_path,
            )
            write_single_row_summary(
                row,
                output_json_path=summary_json_path,
                output_csv_path=summary_csv_path,
                root_key="removed_feature_matrix",
            )
            transform_payload = {
                "task_name": task.name,
                "arm_name": arm.name,
                "variant_name": experiment_config.experiment.variant_name,
                "status": status,
                "failure_reason": failure_reason,
                "pca_summary": pca_summary,
                "whitening_summary": whitening_summary,
                "test_whitening_summary": test_whitening_summary,
                "row": _transform_summary_row(
                    experiment_config=experiment_config,
                    task=task,
                    arm=arm,
                    pca_summary=pca_summary,
                    whitening_summary=whitening_summary,
                    test_whitening_summary=test_whitening_summary,
                ),
            }
            write_single_row_summary(
                transform_payload["row"],
                output_json_path=transform_summary_json_path,
                output_csv_path=transform_summary_csv_path,
                root_key="transform_summary",
            )
            transform_summary_json_path.write_text(json.dumps(transform_payload, indent=2), encoding="utf-8")
            results.append(
                RepresentationBenchmarkResult(
                    task_name=task.name,
                    arm_name=arm.name,
                    status=status,
                    summary=row,
                    summary_json_path=summary_json_path,
                    summary_csv_path=summary_csv_path,
                    transform_summary_json_path=transform_summary_json_path,
                    transform_summary_csv_path=transform_summary_csv_path,
                )
            )
            completed_arms += 1
            _maybe_emit_benchmark_progress(
                progress_callback,
                BenchmarkProgressEvent(
                    benchmark_name="representation",
                    event_type="arm_end",
                    task_name=task.name,
                    task_index=task_index,
                    task_total=len(representation_tasks),
                    arm_name=arm.name,
                    arm_index=completed_arms,
                    arm_total=total_arms,
                    status=status,
                    metrics=row,
                ),
            )

        _maybe_emit_benchmark_progress(
            progress_callback,
            BenchmarkProgressEvent(
                benchmark_name="removed_feature_matrix",
                event_type="task_end",
                task_name=task.name,
                task_index=task_index,
                task_total=len(representation_tasks),
                arm_index=completed_arms,
                arm_total=total_arms,
            ),
        )
    return tuple(results)


def run_motif_benchmark_matrix(
    *,
    experiment_config: ExperimentConfig,
    data_config_path: str | Path,
    checkpoint_path: str | Path,
    output_root: str | Path,
    task_specs: tuple[BenchmarkTaskSpec, ...] | None = None,
    arm_specs: tuple[BenchmarkArmSpec, ...] | None = None,
    discovery_split_name: str | None = None,
    test_split_name: str | None = None,
    session_holdout_fraction: float = 0.5,
    session_holdout_seed: int | None = None,
    test_max_batches: int | None = None,
    debug_retain_intermediates: bool = False,
    progress_callback: BenchmarkProgressCallback | None = None,
) -> tuple[MotifBenchmarkResult, ...]:
    dataset_view = resolve_runtime_dataset_view(
        experiment_config=experiment_config,
        data_config_path=data_config_path,
    )
    tasks = task_specs or default_benchmark_task_specs()
    arms = arm_specs or default_motif_arm_specs()
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    resolved_discovery_split = discovery_split_name or experiment_config.splits.discovery
    resolved_test_split = test_split_name or experiment_config.splits.test
    resolved_holdout_seed = (
        int(session_holdout_seed)
        if session_holdout_seed is not None
        else int(experiment_config.discovery.shuffle_seed)
    )
    motif_tasks = tuple(task for task in tasks if task.include_in_motifs)
    results: list[MotifBenchmarkResult] = []
    total_arms = len(motif_tasks) * len(arms)
    completed_arms = 0

    for task_index, task in enumerate(motif_tasks, start=1):
        _maybe_emit_benchmark_progress(
            progress_callback,
            BenchmarkProgressEvent(
                benchmark_name="motif",
                event_type="task_start",
                task_name=task.name,
                task_index=task_index,
                task_total=len(motif_tasks),
                arm_total=total_arms,
                message=f"motif task {task.name}",
            ),
        )
        if not task.include_in_motifs:
            continue
        skip_status, skip_reason = _task_skip_status(
            task=task,
            dataset_view=dataset_view,
            discovery_split_name=resolved_discovery_split,
        )
        task_config = _task_config(experiment_config, task)
        task_output_root = output_root_path / task.name
        task_output_root.mkdir(parents=True, exist_ok=True)
        raw_cache: dict[str, tuple[BenchmarkWindowCollection, BenchmarkWindowCollection, Any, Path]] = {}

        try:
            plan = prepare_discovery_collection(
                experiment_config=task_config,
                data_config_path=data_config_path,
                split_name=resolved_discovery_split,
                dataset_view=dataset_view,
            )
        except Exception as exc:
            plan = None
            plan_error = str(exc)
        else:
            plan_error = None

        for arm_index, arm in enumerate(arms, start=1):
            arm_root = task_output_root / arm.name
            arm_root.mkdir(parents=True, exist_ok=True)
            summary_json_path = arm_root / "summary.json"
            summary_csv_path = arm_root / "summary.csv"
            cluster_summary_json_path = arm_root / "cluster_summary.json"
            cluster_summary_csv_path = arm_root / "cluster_summary.csv"
            discovery_artifact_path = arm_root / "discovery_artifact.json"
            transform_summary_json_path = arm_root / "transform_summary.json"
            transform_summary_csv_path = arm_root / "transform_summary.csv"
            _maybe_emit_benchmark_progress(
                progress_callback,
                BenchmarkProgressEvent(
                    benchmark_name="motif",
                    event_type="arm_start",
                    task_name=task.name,
                    task_index=task_index,
                    task_total=len(motif_tasks),
                    arm_name=arm.name,
                    arm_index=completed_arms + 1,
                    arm_total=total_arms,
                    message=f"{task.name} / {arm.name}",
                ),
            )

            if skip_status is not None:
                row = _motif_row(
                    experiment_config=experiment_config,
                    task=task,
                    arm=arm,
                    status=skip_status,
                    failure_reason=skip_reason,
                    fit_metrics=None,
                    shuffled_metrics=None,
                    test_metrics=None,
                    held_out_similarity_summary=None,
                    cluster_quality_summary=None,
                    candidate_count=0,
                    cluster_count=0,
                    cluster_report={"clusters": [], "cluster_count": 0},
                    candidate_selection_fallback_used=False,
                    candidate_selection_effective_min_score=None,
                    summary_path=summary_json_path,
                    cluster_summary_path=cluster_summary_json_path,
                    discovery_artifact_path=discovery_artifact_path,
                    transform_summary_path=transform_summary_json_path,
                )
                write_single_row_summary(
                    row,
                    output_json_path=summary_json_path,
                    output_csv_path=summary_csv_path,
                    root_key="motif_benchmark",
                )
                write_discovery_cluster_report_json({"cluster_count": 0, "clusters": []}, cluster_summary_json_path)
                write_discovery_cluster_report_csv({"cluster_count": 0, "clusters": []}, cluster_summary_csv_path)
                transform_payload = {
                    "task_name": task.name,
                    "arm_name": arm.name,
                    "variant_name": experiment_config.experiment.variant_name,
                    "claim_safe": bool(arm.claim_safe),
                    "supervision_level": arm.supervision_level,
                    "status": skip_status,
                    "failure_reason": skip_reason,
                }
                write_single_row_summary(
                    transform_payload,
                    output_json_path=transform_summary_json_path,
                    output_csv_path=transform_summary_csv_path,
                    root_key="transform_summary",
                )
                transform_summary_json_path.write_text(json.dumps(transform_payload, indent=2), encoding="utf-8")
                results.append(
                    MotifBenchmarkResult(
                        task_name=task.name,
                        arm_name=arm.name,
                        status=skip_status,
                        summary=row,
                        summary_json_path=summary_json_path,
                        summary_csv_path=summary_csv_path,
                        cluster_summary_json_path=cluster_summary_json_path,
                        cluster_summary_csv_path=cluster_summary_csv_path,
                        discovery_artifact_path=discovery_artifact_path,
                        transform_summary_json_path=transform_summary_json_path,
                        transform_summary_csv_path=transform_summary_csv_path,
                    )
                )
                completed_arms += 1
                _maybe_emit_benchmark_progress(
                    progress_callback,
                    BenchmarkProgressEvent(
                        benchmark_name="motif",
                        event_type="arm_end",
                        task_name=task.name,
                        task_index=task_index,
                        task_total=len(motif_tasks),
                        arm_name=arm.name,
                        arm_index=completed_arms,
                        arm_total=total_arms,
                        status=skip_status,
                        metrics=row,
                    ),
                )
                continue

            if plan is None:
                status = "degraded"
                failure_reason = plan_error
                fit_metrics = None
                shuffled_metrics = None
                test_metrics = None
                held_out_similarity_summary = None
                cluster_quality_summary = None
                candidate_count = 0
                cluster_count = 0
                cluster_report = {"cluster_count": 0, "clusters": []}
                artifact = DiscoveryArtifact(
                    dataset_id=task_config.dataset_id,
                    split_name=resolved_discovery_split,
                    checkpoint_path=str(checkpoint_path),
                    config_snapshot=task_config.to_dict(),
                    decoder_summary=DecoderSummary(
                        target_label=task.target_label,
                        epochs=task_config.discovery.probe_epochs,
                        learning_rate=task_config.discovery.probe_learning_rate,
                        metrics={},
                        probe_state=None,
                        metric_scope="fit_selected_windows",
                    ),
                    candidates=tuple(),
                    cluster_stats=_default_cluster_stats(),
                    cluster_quality_summary=_default_cluster_stats(),
                )
                candidate_selection_fallback_used = False
                candidate_selection_effective_min_score = None
                transform_payload = {
                    "task_name": task.name,
                    "arm_name": arm.name,
                    "status": status,
                    "failure_reason": failure_reason,
                }
            else:
                try:
                    cached = raw_cache.get("encoder")
                    if cached is None:
                        raw_shard_dir = task_output_root / "_tmp_feature_shards" / "encoder"
                        _maybe_emit_benchmark_progress(
                            progress_callback,
                            BenchmarkProgressEvent(
                                benchmark_name="motif",
                                event_type="arm_step",
                                task_name=task.name,
                                arm_name=arm.name,
                                step_name="discovery_extraction",
                                current=0,
                                total=None,
                                message="extract discovery windows",
                            ),
                        )
                        discovery_collection = extract_benchmark_selected_windows(
                            experiment_config=task_config,
                            data_config_path=data_config_path,
                            feature_family="encoder",
                            checkpoint_path=checkpoint_path,
                            dataset_view=dataset_view,
                            window_plan=plan,
                            shard_dir=raw_shard_dir,
                            progress_callback=lambda current, total, task_name=task.name, arm_name=arm.name: _maybe_emit_benchmark_progress(
                                progress_callback,
                                BenchmarkProgressEvent(
                                    benchmark_name="motif",
                                    event_type="arm_step",
                                    task_name=task_name,
                                    arm_name=arm_name,
                                    step_name="discovery_extraction",
                                    current=current,
                                    total=total,
                                    message="extract discovery windows",
                                ),
                            ),
                        )
                        _ensure_binary_label_coverage(discovery_collection.coverage_summary)  # type: ignore[arg-type]
                        _maybe_emit_benchmark_progress(
                            progress_callback,
                            BenchmarkProgressEvent(
                                benchmark_name="motif",
                                event_type="arm_step",
                                task_name=task.name,
                                arm_name=arm.name,
                                step_name="test_extraction",
                                current=0,
                                total=test_max_batches or task_config.evaluation.max_batches,
                                message="extract test split",
                            ),
                        )
                        test_collection = extract_benchmark_split_collection(
                            experiment_config=task_config,
                            data_config_path=data_config_path,
                            feature_family="encoder",
                            split_name=resolved_test_split,
                            target_label=task.target_label,
                            target_label_mode=task.target_label_mode,
                            target_label_match_value=task.target_label_match_value,
                            checkpoint_path=checkpoint_path,
                            dataset_view=dataset_view,
                            max_batches=test_max_batches or task_config.evaluation.max_batches,
                            progress_callback=lambda current, total, task_name=task.name, arm_name=arm.name: _maybe_emit_benchmark_progress(
                                progress_callback,
                                BenchmarkProgressEvent(
                                    benchmark_name="motif",
                                    event_type="arm_step",
                                    task_name=task_name,
                                    arm_name=arm_name,
                                    step_name="test_extraction",
                                    current=current,
                                    total=total,
                                    message="extract test split",
                                ),
                            ),
                        )
                        split = build_session_stratified_holdout_split(
                            labels=discovery_collection.labels,
                            session_ids=discovery_collection.window_session_ids,
                            subject_ids=discovery_collection.window_subject_ids,
                            holdout_fraction=session_holdout_fraction,
                            seed=resolved_holdout_seed,
                        )
                        raw_cache["encoder"] = (discovery_collection, test_collection, split, raw_shard_dir)
                        cached = raw_cache["encoder"]
                    else:
                        _maybe_emit_benchmark_progress(
                            progress_callback,
                            BenchmarkProgressEvent(
                                benchmark_name="motif",
                                event_type="arm_step",
                                task_name=task.name,
                                arm_name=arm.name,
                                step_name="reuse_cached_features",
                                current=1,
                                total=1,
                                message="reuse encoder features",
                            ),
                        )

                    discovery_collection, test_collection, split, _ = cached
                    if arm.use_pca:
                        _maybe_emit_benchmark_progress(
                            progress_callback,
                            BenchmarkProgressEvent(
                                benchmark_name="motif",
                                event_type="arm_step",
                                task_name=task.name,
                                arm_name=arm.name,
                                step_name="pca",
                                current=0,
                                total=1,
                                message="fit/apply PCA",
                            ),
                        )
                        transformed_discovery, transformed_test, global_transform, pca_summary = _fit_optional_pca(
                            arm=arm,
                            discovery_collection=discovery_collection,
                            test_collection=test_collection,
                            fit_indices=split.fit_indices,
                        )
                        _maybe_emit_benchmark_progress(
                            progress_callback,
                            BenchmarkProgressEvent(
                                benchmark_name="motif",
                                event_type="arm_step",
                                task_name=task.name,
                                arm_name=arm.name,
                                step_name="pca",
                                current=1,
                                total=1,
                                message="fit/apply PCA",
                            ),
                        )
                    else:
                        transformed_discovery = discovery_collection
                        transformed_test = test_collection
                        global_transform = None
                        pca_summary = None
                    whitening_summary = None
                    discovery_session_transforms = None
                    test_whitening_summary = None
                    token_normalization_summary = None
                    alignment_summary = None
                    test_alignment_summary = None
                    candidate_geometry_summary = None
                    if arm.geometry_mode == "whitened":
                        _maybe_emit_benchmark_progress(
                            progress_callback,
                            BenchmarkProgressEvent(
                                benchmark_name="motif",
                                event_type="arm_step",
                                task_name=task.name,
                                arm_name=arm.name,
                                step_name="whitening",
                                current=0,
                                total=1,
                                message="apply whitening",
                            ),
                        )
                        transformed_discovery, whitening_summary, discovery_session_transforms = _apply_session_whitening_to_collection(
                            collection=transformed_discovery,
                            fit_indices=split.fit_indices,
                        )
                        transformed_test, test_whitening_summary, _ = _build_whitened_test_collection(
                            collection=transformed_test,
                        )
                        _maybe_emit_benchmark_progress(
                            progress_callback,
                            BenchmarkProgressEvent(
                                benchmark_name="motif",
                                event_type="arm_step",
                                task_name=task.name,
                                arm_name=arm.name,
                                step_name="whitening",
                                current=1,
                                total=1,
                                message="apply whitening",
                            ),
                        )
                    elif arm.geometry_mode == "token_normalized":
                        transformed_discovery, token_normalization_summary = _normalize_collection_tokens(
                            transformed_discovery
                        )
                        transformed_test, _ = _normalize_collection_tokens(transformed_test)
                    elif arm.geometry_mode == "aligned_oracle":
                        transformed_discovery, alignment_summary, discovery_session_transforms = _apply_oracle_alignment_to_collection(
                            collection=transformed_discovery,
                        )
                        transformed_test, test_alignment_summary, _ = _apply_oracle_alignment_to_collection(
                            collection=transformed_test,
                        )

                    transformed_shard_dir = arm_root / "_tmp_transformed_shards"
                    _maybe_emit_benchmark_progress(
                        progress_callback,
                        BenchmarkProgressEvent(
                            benchmark_name="motif",
                            event_type="arm_step",
                            task_name=task.name,
                            arm_name=arm.name,
                            step_name="token_shards",
                            current=0,
                            total=len(discovery_collection.shard_paths),
                            message="write transformed token shards",
                        ),
                    )
                    transformed_shards = write_collection_token_shards(
                        collection=discovery_collection,
                        output_dir=transformed_shard_dir,
                        global_transform=global_transform,
                        session_transforms=discovery_session_transforms,
                        normalize_tokens=arm.geometry_mode == "token_normalized",
                        progress_callback=lambda current, total, task_name=task.name, arm_name=arm.name: _maybe_emit_benchmark_progress(
                            progress_callback,
                            BenchmarkProgressEvent(
                                benchmark_name="motif",
                                event_type="arm_step",
                                task_name=task_name,
                                arm_name=arm_name,
                                step_name="token_shards",
                                current=current,
                                total=total,
                                message="write transformed token shards",
                            ),
                        ),
                    )

                    _maybe_emit_benchmark_progress(
                        progress_callback,
                        BenchmarkProgressEvent(
                            benchmark_name="motif",
                            event_type="arm_step",
                            task_name=task.name,
                            arm_name=arm.name,
                            step_name="probe_fit",
                            current=0,
                            total=1,
                            message="fit additive probe",
                        ),
                    )
                    fit_probe = fit_additive_probe_features(
                        features=transformed_discovery.pooled_features,
                        labels=transformed_discovery.labels,
                        epochs=task_config.discovery.probe_epochs,
                        learning_rate=task_config.discovery.probe_learning_rate,
                        label_name=task.target_label,
                    )
                    _maybe_emit_benchmark_progress(
                        progress_callback,
                        BenchmarkProgressEvent(
                            benchmark_name="motif",
                            event_type="arm_step",
                            task_name=task.name,
                            arm_name=arm.name,
                            step_name="probe_fit",
                            current=1,
                            total=1,
                            message="fit additive probe",
                        ),
                    )
                    _maybe_emit_benchmark_progress(
                        progress_callback,
                        BenchmarkProgressEvent(
                            benchmark_name="motif",
                            event_type="arm_step",
                            task_name=task.name,
                            arm_name=arm.name,
                            step_name="shuffled_probe_fit",
                            current=0,
                            total=1,
                            message="fit shuffled probe",
                        ),
                    )
                    shuffled_probe = _fit_shuffled_probe_features(
                        features=transformed_discovery.pooled_features,
                        labels=transformed_discovery.labels,
                        epochs=task_config.discovery.probe_epochs,
                        learning_rate=task_config.discovery.probe_learning_rate,
                        seed=task_config.discovery.shuffle_seed,
                        label_name=task.target_label,
                    )
                    _maybe_emit_benchmark_progress(
                        progress_callback,
                        BenchmarkProgressEvent(
                            benchmark_name="motif",
                            event_type="arm_step",
                            task_name=task.name,
                            arm_name=arm.name,
                            step_name="shuffled_probe_fit",
                            current=1,
                            total=1,
                            message="fit shuffled probe",
                        ),
                    )

                    _maybe_emit_benchmark_progress(
                        progress_callback,
                        BenchmarkProgressEvent(
                            benchmark_name="motif",
                            event_type="arm_step",
                            task_name=task.name,
                            arm_name=arm.name,
                            step_name="candidate_selection",
                            current=0,
                            total=1,
                            message="select candidate tokens",
                        ),
                    )
                    candidates = select_candidate_tokens_from_shards(
                        shard_paths=transformed_shards,
                        probe_state_dict=fit_probe.state_dict,
                        top_k=task_config.discovery.top_k_candidates,
                        min_score=task_config.discovery.min_candidate_score,
                        candidate_session_balance_fraction=task_config.discovery.candidate_session_balance_fraction,
                    )
                    candidate_selection_fallback_used = False
                    candidate_selection_effective_min_score = float(task_config.discovery.min_candidate_score)
                    if not candidates:
                        candidates = select_candidate_tokens_from_shards(
                            shard_paths=transformed_shards,
                            probe_state_dict=fit_probe.state_dict,
                            top_k=task_config.discovery.top_k_candidates,
                            min_score=float("-inf"),
                            candidate_session_balance_fraction=task_config.discovery.candidate_session_balance_fraction,
                        )
                        candidate_selection_fallback_used = True
                        candidate_selection_effective_min_score = None
                    clustering_candidates = candidates
                    similarity_test_tokens = transformed_test.token_tensors
                    if candidates and arm.candidate_geometry_mode == "probe_weighted":
                        clustering_candidates, candidate_geometry_summary = _apply_probe_weight_to_candidates(
                            candidates,
                            state_dict=fit_probe.state_dict,
                        )
                        similarity_test_tokens = _apply_probe_weight_to_tokens(
                            transformed_test.token_tensors,
                            state_dict=fit_probe.state_dict,
                        )
                    elif arm.candidate_geometry_mode != "embedding":
                        raise ValueError(f"Unsupported candidate_geometry_mode: {arm.candidate_geometry_mode}")
                    _maybe_emit_benchmark_progress(
                        progress_callback,
                        BenchmarkProgressEvent(
                            benchmark_name="motif",
                            event_type="arm_step",
                            task_name=task.name,
                            arm_name=arm.name,
                            step_name="candidate_selection",
                            current=1,
                            total=1,
                            message="select candidate tokens",
                        ),
                    )

                    if not clustering_candidates:
                        clustered_candidates = tuple()
                        cluster_stats = _default_cluster_stats()
                        cluster_quality_summary = _default_cluster_stats()
                        failure_reason = "no_candidates_selected"
                    else:
                        _maybe_emit_benchmark_progress(
                            progress_callback,
                            BenchmarkProgressEvent(
                                benchmark_name="motif",
                                event_type="arm_step",
                                task_name=task.name,
                                arm_name=arm.name,
                                step_name="clustering",
                                current=0,
                                total=1,
                                message="cluster candidate tokens",
                            ),
                        )
                        clustered_candidates, cluster_stats = cluster_candidate_tokens(
                            candidates=clustering_candidates,
                            min_cluster_size=task_config.discovery.min_cluster_size,
                        )
                        _maybe_emit_benchmark_progress(
                            progress_callback,
                            BenchmarkProgressEvent(
                                benchmark_name="motif",
                                event_type="arm_step",
                                task_name=task.name,
                                arm_name=arm.name,
                                step_name="clustering",
                                current=1,
                                total=1,
                                message="cluster candidate tokens",
                            ),
                        )
                        if int(cluster_stats.get("cluster_count", 0)) <= 0:
                            cluster_quality_summary = cluster_stats.copy()
                            failure_reason = "no_non_noise_clusters"
                        else:
                            _maybe_emit_benchmark_progress(
                                progress_callback,
                                BenchmarkProgressEvent(
                                    benchmark_name="motif",
                                    event_type="arm_step",
                                    task_name=task.name,
                                    arm_name=arm.name,
                                    step_name="stability",
                                    current=0,
                                    total=1,
                                    message="estimate cluster stability",
                                ),
                            )
                            cluster_quality_summary = estimate_clustering_stability(
                                candidates=clustered_candidates,
                                cluster_stats=cluster_stats,
                                min_cluster_size=task_config.discovery.min_cluster_size,
                                stability_rounds=task_config.discovery.stability_rounds,
                                seed=task_config.seed,
                            )
                            _maybe_emit_benchmark_progress(
                                progress_callback,
                                BenchmarkProgressEvent(
                                    benchmark_name="motif",
                                    event_type="arm_step",
                                    task_name=task.name,
                                    arm_name=arm.name,
                                    step_name="stability",
                                    current=1,
                                    total=1,
                                    message="estimate cluster stability",
                                ),
                            )
                            failure_reason = None

                    artifact = DiscoveryArtifact(
                        dataset_id=task_config.dataset_id,
                        split_name=resolved_discovery_split,
                        checkpoint_path=str(checkpoint_path),
                        config_snapshot=task_config.to_dict(),
                        decoder_summary=DecoderSummary(
                            target_label=task.target_label,
                            epochs=task_config.discovery.probe_epochs,
                            learning_rate=task_config.discovery.probe_learning_rate,
                            metrics=fit_probe.metrics,
                            probe_state=fit_probe.state_dict,
                            metric_scope="fit_selected_windows",
                        ),
                        candidates=clustered_candidates,
                        cluster_stats=(cluster_stats if clustering_candidates else _default_cluster_stats()),
                        cluster_quality_summary=cluster_quality_summary,
                    )
                    cluster_report = build_discovery_cluster_report(artifact)
                    test_logits = _probe_logits_from_features(
                        state_dict=fit_probe.state_dict,
                        features=transformed_test.pooled_features,
                    )
                    test_metrics = _probe_metrics_from_logits(
                        sample_logits=test_logits,
                        labels=transformed_test.labels,
                    )
                    centroids = _candidate_centroids(clustered_candidates)
                    if centroids:
                        _maybe_emit_benchmark_progress(
                            progress_callback,
                            BenchmarkProgressEvent(
                                benchmark_name="motif",
                                event_type="arm_step",
                                task_name=task.name,
                                arm_name=arm.name,
                                step_name="held_out_similarity",
                                current=0,
                                total=1,
                                message="score held-out motif similarity",
                            ),
                        )
                        held_out_similarity_summary = _held_out_similarity_summary(
                            labels=transformed_test.labels,
                            window_session_ids=transformed_test.window_session_ids,
                            window_scores=_window_similarity_scores(
                                tokens=similarity_test_tokens,
                                token_mask=transformed_test.token_mask,
                                centroids=centroids,
                            ),
                        )
                        _maybe_emit_benchmark_progress(
                            progress_callback,
                            BenchmarkProgressEvent(
                                benchmark_name="motif",
                                event_type="arm_step",
                                task_name=task.name,
                                arm_name=arm.name,
                                step_name="held_out_similarity",
                                current=1,
                                total=1,
                                message="score held-out motif similarity",
                            ),
                        )
                    else:
                        held_out_similarity_summary = _empty_similarity_summary(
                            failure_reason or "no_non_noise_clusters_for_similarity_validation"
                        )
                    fit_metrics = fit_probe.metrics
                    shuffled_metrics = shuffled_probe.metrics
                    candidate_count = len(clustered_candidates)
                    cluster_count = int(artifact.cluster_stats.get("cluster_count", 0))
                    status = "ok" if failure_reason is None else "degraded"
                    transform_payload = {
                        "task_name": task.name,
                        "arm_name": arm.name,
                        "variant_name": experiment_config.experiment.variant_name,
                        "claim_safe": bool(arm.claim_safe),
                        "supervision_level": arm.supervision_level,
                        "status": status,
                        "failure_reason": failure_reason,
                        "pca_summary": pca_summary,
                        "whitening_summary": whitening_summary,
                        "test_whitening_summary": test_whitening_summary,
                        "token_normalization_summary": token_normalization_summary,
                        "alignment_summary": alignment_summary,
                        "test_alignment_summary": test_alignment_summary,
                        "candidate_geometry_summary": candidate_geometry_summary,
                        "row": _transform_summary_row(
                            experiment_config=experiment_config,
                            task=task,
                            arm=arm,
                            pca_summary=pca_summary,
                            whitening_summary=whitening_summary,
                            test_whitening_summary=test_whitening_summary,
                            token_normalization_summary=token_normalization_summary,
                            alignment_summary=alignment_summary,
                            candidate_geometry_summary=candidate_geometry_summary,
                        ),
                    }
                    if transformed_shard_dir.exists() and not debug_retain_intermediates:
                        shutil.rmtree(transformed_shard_dir)
                except Exception as exc:
                    status = "degraded"
                    failure_reason = str(exc)
                    fit_metrics = None
                    shuffled_metrics = None
                    test_metrics = None
                    held_out_similarity_summary = None
                    cluster_quality_summary = None
                    candidate_count = 0
                    cluster_count = 0
                    cluster_report = {"cluster_count": 0, "clusters": []}
                    artifact = DiscoveryArtifact(
                        dataset_id=task_config.dataset_id,
                        split_name=resolved_discovery_split,
                        checkpoint_path=str(checkpoint_path),
                        config_snapshot=task_config.to_dict(),
                        decoder_summary=DecoderSummary(
                            target_label=task.target_label,
                            epochs=task_config.discovery.probe_epochs,
                            learning_rate=task_config.discovery.probe_learning_rate,
                            metrics={},
                            probe_state=None,
                            metric_scope="fit_selected_windows",
                        ),
                        candidates=tuple(),
                        cluster_stats=_default_cluster_stats(),
                        cluster_quality_summary=_default_cluster_stats(),
                    )
                    candidate_selection_fallback_used = False
                    candidate_selection_effective_min_score = None
                    transform_payload = {
                        "task_name": task.name,
                        "arm_name": arm.name,
                        "variant_name": experiment_config.experiment.variant_name,
                        "claim_safe": bool(arm.claim_safe),
                        "supervision_level": arm.supervision_level,
                        "status": status,
                        "failure_reason": failure_reason,
                    }

            row = _motif_row(
                experiment_config=experiment_config,
                task=task,
                arm=arm,
                status=status,
                failure_reason=failure_reason,
                fit_metrics=fit_metrics,
                shuffled_metrics=shuffled_metrics,
                test_metrics=test_metrics,
                held_out_similarity_summary=held_out_similarity_summary,
                cluster_quality_summary=cluster_quality_summary,
                candidate_count=candidate_count,
                cluster_count=cluster_count,
                cluster_report=cluster_report,
                candidate_selection_fallback_used=candidate_selection_fallback_used,
                candidate_selection_effective_min_score=candidate_selection_effective_min_score,
                summary_path=summary_json_path,
                cluster_summary_path=cluster_summary_json_path,
                discovery_artifact_path=discovery_artifact_path,
                transform_summary_path=transform_summary_json_path,
            )
            write_discovery_artifact(artifact, discovery_artifact_path)
            write_discovery_cluster_report_json(cluster_report, cluster_summary_json_path)
            write_discovery_cluster_report_csv(cluster_report, cluster_summary_csv_path)
            write_single_row_summary(
                row,
                output_json_path=summary_json_path,
                output_csv_path=summary_csv_path,
                root_key="motif_benchmark",
            )
            write_single_row_summary(
                transform_payload.get("row", transform_payload),
                output_json_path=transform_summary_json_path,
                output_csv_path=transform_summary_csv_path,
                root_key="transform_summary",
            )
            transform_summary_json_path.write_text(json.dumps(transform_payload, indent=2), encoding="utf-8")
            results.append(
                MotifBenchmarkResult(
                    task_name=task.name,
                    arm_name=arm.name,
                    status=status,
                    summary=row,
                    summary_json_path=summary_json_path,
                    summary_csv_path=summary_csv_path,
                    cluster_summary_json_path=cluster_summary_json_path,
                    cluster_summary_csv_path=cluster_summary_csv_path,
                    discovery_artifact_path=discovery_artifact_path,
                    transform_summary_json_path=transform_summary_json_path,
                    transform_summary_csv_path=transform_summary_csv_path,
                )
            )
            completed_arms += 1
            _maybe_emit_benchmark_progress(
                progress_callback,
                BenchmarkProgressEvent(
                    benchmark_name="motif",
                    event_type="arm_end",
                    task_name=task.name,
                    task_index=task_index,
                    task_total=len(motif_tasks),
                    arm_name=arm.name,
                    arm_index=completed_arms,
                    arm_total=total_arms,
                    status=status,
                    metrics=row,
                ),
            )

        for _, _, _, raw_shard_dir in raw_cache.values():
            if raw_shard_dir.exists() and not debug_retain_intermediates:
                shutil.rmtree(raw_shard_dir)

        _maybe_emit_benchmark_progress(
            progress_callback,
            BenchmarkProgressEvent(
                benchmark_name="motif",
                event_type="task_end",
                task_name=task.name,
                task_index=task_index,
                task_total=len(motif_tasks),
                arm_index=completed_arms,
                arm_total=total_arms,
            ),
        )

    return tuple(results)
