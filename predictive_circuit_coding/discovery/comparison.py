from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import shutil
from typing import Any

import torch

from predictive_circuit_coding.decoding import (
    apply_session_linear_transforms_to_features,
    apply_session_linear_transforms_to_tokens,
    build_session_stratified_holdout_split,
    evaluate_additive_probe,
    evaluate_additive_probe_features,
    extract_frozen_tokens,
    fit_additive_probe,
    fit_additive_probe_features,
    fit_session_alignment_transforms,
    fit_session_whitening_transforms,
)
from predictive_circuit_coding.decoding.extract import DiscoveryWindowPlan, EncodedDiscoverySelection
from predictive_circuit_coding.decoding.probes import (
    fit_shuffled_probe_features,
    probe_logits_from_features,
    probe_metrics_from_logits,
)
from predictive_circuit_coding.decoding.scoring import (
    candidate_centroids,
    held_out_similarity_summary as summarize_held_out_similarity,
    pooled_features_from_tokens,
    select_candidate_tokens_from_shards,
    window_similarity_scores,
)
from predictive_circuit_coding.discovery.clustering import cluster_candidate_tokens
from predictive_circuit_coding.discovery.reporting import build_discovery_cluster_report
from predictive_circuit_coding.discovery.run import _ensure_binary_label_coverage, _ensure_min_positive_windows
from predictive_circuit_coding.discovery.stability import estimate_clustering_stability
from predictive_circuit_coding.training.config import ExperimentConfig
from predictive_circuit_coding.training.contracts import (
    DecoderSummary,
    DiscoveryArtifact,
    DiscoveryCoverageSummary,
)


@dataclass(frozen=True)
class RepresentationComparisonArmResult:
    arm_name: str
    artifact: DiscoveryArtifact
    cluster_report: dict[str, Any]
    validation_summary: dict[str, Any]
    transform_summary: dict[str, Any] | None


@dataclass(frozen=True)
class RepresentationComparisonRunResult:
    coverage_summary: DiscoveryCoverageSummary
    split_summary: dict[str, Any]
    arm_results: tuple[RepresentationComparisonArmResult, ...]

def _window_key(*, recording_id: str, session_id: str, window_start_s: float, window_end_s: float) -> tuple[str, str, float, float]:
    return (
        str(recording_id),
        str(session_id),
        round(float(window_start_s), 6),
        round(float(window_end_s), 6),
    )


def _subset_tuple(values: tuple[Any, ...], indices: torch.Tensor) -> tuple[Any, ...]:
    index_list = [int(index) for index in indices.detach().cpu().tolist()]
    return tuple(values[index] for index in index_list)


def _unavailable_similarity_summary(*, reason: str) -> dict[str, Any]:
    return {
        "window_roc_auc": None,
        "window_pr_auc": None,
        "positive_window_count": None,
        "negative_window_count": None,
        "per_session_roc_auc": {},
        "comparison_available": False,
        "failure_reason": str(reason),
    }


def _copy_or_transform_selected_shards(
    *,
    shard_paths: tuple[Path, ...],
    output_dir: Path,
    allowed_window_keys: set[tuple[str, str, float, float]],
    transforms: dict[str, Any] | None = None,
) -> tuple[Path, ...]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    shard_index = 0

    for shard_path in shard_paths:
        payload = torch.load(Path(shard_path), map_location="cpu", weights_only=False)
        row_count = int(payload["embeddings"].shape[0])
        keep_indices: list[int] = []
        for index in range(row_count):
            key = _window_key(
                recording_id=payload["recording_ids"][index],
                session_id=payload["session_ids"][index],
                window_start_s=float(payload["window_start_s"][index].item()),
                window_end_s=float(payload["window_end_s"][index].item()),
            )
            if key in allowed_window_keys:
                keep_indices.append(index)
        if not keep_indices:
            continue
        keep_tensor = torch.tensor(keep_indices, dtype=torch.long)
        embeddings = payload["embeddings"].index_select(0, keep_tensor).to(dtype=torch.float32)
        session_ids = tuple(str(payload["session_ids"][index]) for index in keep_indices)
        if transforms is not None:
            embeddings = apply_session_linear_transforms_to_features(
                features=embeddings,
                session_ids=session_ids,
                transforms=transforms,
            )
        output_payload = {
            "embeddings": embeddings,
            "recording_ids": [payload["recording_ids"][index] for index in keep_indices],
            "session_ids": [payload["session_ids"][index] for index in keep_indices],
            "subject_ids": [payload["subject_ids"][index] for index in keep_indices],
            "unit_ids": [payload["unit_ids"][index] for index in keep_indices],
            "unit_regions": [payload["unit_regions"][index] for index in keep_indices],
            "unit_depth_um": payload["unit_depth_um"].index_select(0, keep_tensor),
            "patch_index": payload["patch_index"].index_select(0, keep_tensor),
            "patch_start_s": payload["patch_start_s"].index_select(0, keep_tensor),
            "patch_end_s": payload["patch_end_s"].index_select(0, keep_tensor),
            "window_start_s": payload["window_start_s"].index_select(0, keep_tensor),
            "window_end_s": payload["window_end_s"].index_select(0, keep_tensor),
            "labels": payload["labels"].index_select(0, keep_tensor),
        }
        target = output_dir / f"token_shard_{shard_index:05d}.pt"
        torch.save(output_payload, target)
        output_paths.append(target)
        shard_index += 1

    return tuple(output_paths)


def _summarize_selected_shards(
    *,
    shard_paths: tuple[Path, ...],
) -> dict[str, Any]:
    shard_file_count = 0
    token_row_count = 0
    positive_token_row_count = 0
    negative_token_row_count = 0
    window_keys: set[tuple[str, str, float, float]] = set()
    positive_window_keys: set[tuple[str, str, float, float]] = set()
    session_token_counts: dict[str, int] = {}
    positive_session_token_counts: dict[str, int] = {}

    for shard_path in shard_paths:
        payload = torch.load(Path(shard_path), map_location="cpu", weights_only=False)
        embeddings = payload.get("embeddings")
        labels = payload.get("labels")
        if embeddings is None or labels is None:
            continue
        row_count = int(embeddings.shape[0])
        shard_file_count += 1
        token_row_count += row_count
        for index in range(row_count):
            session_id = str(payload["session_ids"][index])
            label = float(labels[index].item())
            key = _window_key(
                recording_id=payload["recording_ids"][index],
                session_id=session_id,
                window_start_s=float(payload["window_start_s"][index].item()),
                window_end_s=float(payload["window_end_s"][index].item()),
            )
            window_keys.add(key)
            session_token_counts[session_id] = session_token_counts.get(session_id, 0) + 1
            if label > 0.0:
                positive_token_row_count += 1
                positive_window_keys.add(key)
                positive_session_token_counts[session_id] = positive_session_token_counts.get(session_id, 0) + 1
            else:
                negative_token_row_count += 1

    return {
        "shard_file_count": shard_file_count,
        "token_row_count": token_row_count,
        "positive_token_row_count": positive_token_row_count,
        "negative_token_row_count": negative_token_row_count,
        "window_row_count": len(window_keys),
        "positive_window_row_count": len(positive_window_keys),
        "session_token_counts": session_token_counts,
        "positive_session_token_counts": positive_session_token_counts,
    }


def _standard_cross_session_validation_summary(
    *,
    experiment_config: ExperimentConfig,
    data_config_path: str | Path,
    checkpoint_path: str | Path,
    probe_state_dict: dict[str, Any],
    candidates: tuple[Any, ...],
    transform_mode: str,
) -> dict[str, Any]:
    discovery_collection = extract_frozen_tokens(
        experiment_config=experiment_config,
        data_config_path=data_config_path,
        checkpoint_path=checkpoint_path,
        split_name=experiment_config.splits.discovery,
        max_batches=experiment_config.evaluation.max_batches,
        include_records=False,
        sampling_strategy_override="sequential",
    )
    test_collection = extract_frozen_tokens(
        experiment_config=experiment_config,
        data_config_path=data_config_path,
        checkpoint_path=checkpoint_path,
        split_name=experiment_config.splits.test,
        max_batches=experiment_config.evaluation.max_batches,
        include_records=False,
        positive_records_only=False,
        sampling_strategy_override="sequential",
    )

    discovery_tokens = discovery_collection.tokens
    test_tokens = test_collection.tokens
    if transform_mode == "whitening_only":
        discovery_features = pooled_features_from_tokens(discovery_tokens, discovery_collection.token_mask)
        test_features = pooled_features_from_tokens(test_tokens, test_collection.token_mask)
        discovery_transforms, _ = fit_session_whitening_transforms(
            features=discovery_features,
            session_ids=discovery_collection.window_session_ids,
            fit_indices=torch.arange(discovery_features.shape[0], dtype=torch.long),
        )
        test_transforms, _ = fit_session_whitening_transforms(
            features=test_features,
            session_ids=test_collection.window_session_ids,
            fit_indices=torch.arange(test_features.shape[0], dtype=torch.long),
        )
        discovery_tokens = apply_session_linear_transforms_to_tokens(
            tokens=discovery_tokens,
            session_ids=discovery_collection.window_session_ids,
            transforms=discovery_transforms,
        )
        test_tokens = apply_session_linear_transforms_to_tokens(
            tokens=test_tokens,
            session_ids=test_collection.window_session_ids,
            transforms=test_transforms,
        )

    real_label_metrics = evaluate_additive_probe(
        state_dict=probe_state_dict,
        tokens=discovery_tokens,
        token_mask=discovery_collection.token_mask,
        labels=discovery_collection.labels,
    )
    shuffled_labels = discovery_collection.labels.clone()
    permutation = list(range(len(shuffled_labels)))
    random.Random(int(experiment_config.discovery.shuffle_seed)).shuffle(permutation)
    shuffled_labels = shuffled_labels[torch.tensor(permutation, dtype=torch.long)]
    shuffled_fit = fit_additive_probe(
        tokens=discovery_tokens,
        token_mask=discovery_collection.token_mask,
        labels=shuffled_labels,
        epochs=experiment_config.discovery.probe_epochs,
        learning_rate=experiment_config.discovery.probe_learning_rate,
        mini_batch_size=256,
        label_name=experiment_config.discovery.target_label,
    )
    held_out_test_metrics = evaluate_additive_probe(
        state_dict=probe_state_dict,
        tokens=test_tokens,
        token_mask=test_collection.token_mask,
        labels=test_collection.labels,
    )
    centroids = candidate_centroids(candidates)
    if centroids:
        window_scores = window_similarity_scores(
            tokens=test_tokens,
            token_mask=test_collection.token_mask,
            centroids=centroids,
        )
        held_out_similarity_summary = summarize_held_out_similarity(
            labels=test_collection.labels,
            window_session_ids=test_collection.window_session_ids,
            window_scores=window_scores,
        )
    else:
        held_out_similarity_summary = _unavailable_similarity_summary(
            reason="no_non_noise_clusters_for_similarity_validation",
        )
    return {
        "transform_mode": transform_mode,
        "real_label_metrics": real_label_metrics,
        "shuffled_label_metrics": shuffled_fit.metrics,
        "held_out_test_metrics": held_out_test_metrics,
        "held_out_similarity_summary": held_out_similarity_summary,
        "sampling_summary": {
            "evaluation_max_batches": int(experiment_config.evaluation.max_batches),
            "discovery_sampled_window_count": int(discovery_collection.labels.shape[0]),
            "test_sampled_window_count": int(test_collection.labels.shape[0]),
        },
    }


def run_representation_comparison_from_encoded(
    *,
    experiment_config: ExperimentConfig,
    data_config_path: str | Path,
    checkpoint_path: str | Path,
    split_name: str,
    window_plan: DiscoveryWindowPlan,
    encoded: EncodedDiscoverySelection,
    session_holdout_fraction: float,
    session_holdout_seed: int,
    run_standard_test_validation: bool,
    temporary_root: str | Path,
    arm_names: tuple[str, ...] | None = None,
) -> RepresentationComparisonRunResult:
    _ensure_binary_label_coverage(window_plan.coverage_summary)
    if experiment_config.discovery.sampling_strategy == "label_balanced":
        _ensure_min_positive_windows(
            window_plan.coverage_summary,
            min_positive_windows=experiment_config.discovery.min_positive_windows,
        )

    split = build_session_stratified_holdout_split(
        labels=encoded.labels,
        session_ids=encoded.window_session_ids,
        subject_ids=encoded.window_subject_ids,
        holdout_fraction=session_holdout_fraction,
        seed=session_holdout_seed,
    )
    fit_indices = split.fit_indices
    heldout_indices = split.heldout_indices
    valid_session_ids = set(split.valid_session_ids)
    fit_window_keys = {
        _window_key(
            recording_id=encoded.window_recording_ids[index],
            session_id=encoded.window_session_ids[index],
            window_start_s=encoded.window_start_s[index],
            window_end_s=encoded.window_end_s[index],
        )
        for index in fit_indices.tolist()
    }

    fit_features = encoded.pooled_features.index_select(0, fit_indices)
    fit_labels = encoded.labels.index_select(0, fit_indices)
    fit_session_ids = _subset_tuple(encoded.window_session_ids, fit_indices)
    heldout_features = encoded.pooled_features.index_select(0, heldout_indices)
    heldout_labels = encoded.labels.index_select(0, heldout_indices)
    heldout_tokens = encoded.token_tensors.index_select(0, heldout_indices)
    heldout_token_mask = encoded.token_mask.index_select(0, heldout_indices)
    heldout_session_ids = _subset_tuple(encoded.window_session_ids, heldout_indices)
    fit_positive_window_count = int((fit_labels > 0.0).sum().item())
    fit_negative_window_count = int((fit_labels <= 0.0).sum().item())
    heldout_positive_window_count = int((heldout_labels > 0.0).sum().item())
    heldout_negative_window_count = int((heldout_labels <= 0.0).sum().item())
    fit_session_window_counts: dict[str, int] = {}
    fit_positive_session_window_counts: dict[str, int] = {}
    for session_id, label in zip(fit_session_ids, fit_labels.detach().cpu().tolist(), strict=False):
        fit_session_window_counts[str(session_id)] = fit_session_window_counts.get(str(session_id), 0) + 1
        if float(label) > 0.0:
            fit_positive_session_window_counts[str(session_id)] = fit_positive_session_window_counts.get(str(session_id), 0) + 1

    temp_root = Path(temporary_root)
    temp_root.mkdir(parents=True, exist_ok=True)
    arm_results: list[RepresentationComparisonArmResult] = []
    arm_specs: list[tuple[str, str]] = [
        ("baseline", "baseline"),
        ("whitening_only", "whitening_only"),
        ("whitening_plus_held_out_alignment", "whitening_plus_held_out_alignment"),
    ]
    if arm_names is not None:
        requested = {str(name) for name in arm_names}
        arm_specs = [spec for spec in arm_specs if spec[0] in requested]

    for arm_name, transform_mode in arm_specs:
        if transform_mode == "baseline":
            transforms = None
            transform_summary = {
                "transform_type": "baseline",
                "session_count": len(split.valid_session_ids),
                "reference_session_id": split.reference_session_id,
                "sessions": list(split.session_rows),
            }
        elif transform_mode == "whitening_only":
            transforms, transform_summary = fit_session_whitening_transforms(
                features=fit_features,
                session_ids=fit_session_ids,
                fit_indices=torch.arange(fit_features.shape[0], dtype=torch.long),
            )
            transform_summary = {
                **transform_summary,
                "reference_session_id": split.reference_session_id,
                "sessions": list(split.session_rows),
            }
        else:
            transforms, transform_summary = fit_session_alignment_transforms(
                features=fit_features,
                labels=fit_labels,
                session_ids=fit_session_ids,
                fit_indices=torch.arange(fit_features.shape[0], dtype=torch.long),
                reference_session_id=split.reference_session_id,
            )
            transform_summary = {
                **transform_summary,
                "sessions": list(split.session_rows),
            }

        arm_fit_features = (
            fit_features
            if transforms is None
            else apply_session_linear_transforms_to_features(
                features=fit_features,
                session_ids=fit_session_ids,
                transforms=transforms,
            )
        )
        arm_heldout_features = (
            heldout_features
            if transforms is None
            else apply_session_linear_transforms_to_features(
                features=heldout_features,
                session_ids=heldout_session_ids,
                transforms=transforms,
            )
        )
        arm_heldout_tokens = (
            heldout_tokens
            if transforms is None
            else apply_session_linear_transforms_to_tokens(
                tokens=heldout_tokens,
                session_ids=heldout_session_ids,
                transforms=transforms,
            )
        )

        probe_fit = fit_additive_probe_features(
            features=arm_fit_features,
            labels=fit_labels,
            epochs=experiment_config.discovery.probe_epochs,
            learning_rate=experiment_config.discovery.probe_learning_rate,
            label_name=experiment_config.discovery.target_label,
        )
        shuffled_fit = fit_shuffled_probe_features(
            features=arm_fit_features,
            labels=fit_labels,
            epochs=experiment_config.discovery.probe_epochs,
            learning_rate=experiment_config.discovery.probe_learning_rate,
            seed=experiment_config.discovery.shuffle_seed,
            label_name=experiment_config.discovery.target_label,
        )

        arm_shard_dir = temp_root / arm_name
        transformed_shards = _copy_or_transform_selected_shards(
            shard_paths=encoded.shard_paths,
            output_dir=arm_shard_dir,
            allowed_window_keys=fit_window_keys,
            transforms=transforms,
        )
        shard_debug_summary = _summarize_selected_shards(shard_paths=transformed_shards)
        candidates = select_candidate_tokens_from_shards(
            shard_paths=transformed_shards,
            probe_state_dict=probe_fit.state_dict,
            top_k=experiment_config.discovery.top_k_candidates,
            min_score=experiment_config.discovery.min_candidate_score,
            candidate_session_balance_fraction=experiment_config.discovery.candidate_session_balance_fraction,
        )
        candidate_selection_summary = {
            "configured_min_score": float(experiment_config.discovery.min_candidate_score),
            "effective_min_score": float(experiment_config.discovery.min_candidate_score),
            "fallback_used": False,
            "top_k_candidates": int(experiment_config.discovery.top_k_candidates),
        }
        if not candidates:
            candidates = select_candidate_tokens_from_shards(
                shard_paths=transformed_shards,
                probe_state_dict=probe_fit.state_dict,
                top_k=experiment_config.discovery.top_k_candidates,
                min_score=float("-inf"),
                candidate_session_balance_fraction=experiment_config.discovery.candidate_session_balance_fraction,
            )
            candidate_selection_summary = {
                **candidate_selection_summary,
                "effective_min_score": None,
                "fallback_used": True,
                "fallback_reason": "configured_min_score_selected_no_candidates",
            }
        candidate_selection_summary = {
            **candidate_selection_summary,
            "precluster_candidate_count": len(candidates),
            "arm_shard_debug": {
                "allowed_fit_window_key_count": len(fit_window_keys),
                "fit_window_count": int(fit_indices.numel()),
                "fit_positive_window_count": fit_positive_window_count,
                "fit_negative_window_count": fit_negative_window_count,
                "heldout_window_count": int(heldout_indices.numel()),
                "heldout_positive_window_count": heldout_positive_window_count,
                "heldout_negative_window_count": heldout_negative_window_count,
                "fit_session_window_counts": fit_session_window_counts,
                "fit_positive_session_window_counts": fit_positive_session_window_counts,
                **shard_debug_summary,
            },
        }
        if transformed_shards:
            shutil.rmtree(arm_shard_dir)
        failure_reason = None
        if not candidates:
            clustered_candidates = tuple()
            cluster_stats = {
                "cluster_count": 0.0,
                "noise_count": 0.0,
                "non_noise_fraction": 0.0,
                "silhouette_score": None,
                "cluster_persistence_mean": None,
                "cluster_persistence_min": None,
                "cluster_persistence_max": None,
                "cluster_persistence_by_cluster": {},
            }
            cluster_quality_summary = cluster_stats.copy()
            failure_reason = "no_candidates_selected"
        else:
            clustered_candidates, cluster_stats = cluster_candidate_tokens(
                candidates=candidates,
                min_cluster_size=experiment_config.discovery.min_cluster_size,
            )
            if int(cluster_stats.get("cluster_count", 0)) <= 0:
                cluster_quality_summary = cluster_stats.copy()
                failure_reason = "no_non_noise_clusters"
            else:
                cluster_quality_summary = estimate_clustering_stability(
                    candidates=clustered_candidates,
                    cluster_stats=cluster_stats,
                    min_cluster_size=experiment_config.discovery.min_cluster_size,
                    stability_rounds=experiment_config.discovery.stability_rounds,
                    seed=experiment_config.seed,
                )
        artifact = DiscoveryArtifact(
            dataset_id=experiment_config.dataset_id,
            split_name=split_name,
            checkpoint_path=str(checkpoint_path),
            config_snapshot=experiment_config.to_dict(),
            decoder_summary=DecoderSummary(
                target_label=experiment_config.discovery.target_label,
                epochs=experiment_config.discovery.probe_epochs,
                learning_rate=experiment_config.discovery.probe_learning_rate,
                metrics=probe_fit.metrics,
                probe_state=probe_fit.state_dict,
                metric_scope="fit_within_session_fit_windows",
            ),
            candidates=clustered_candidates,
            cluster_stats=cluster_stats,
            cluster_quality_summary=cluster_quality_summary,
        )
        cluster_report = build_discovery_cluster_report(artifact)

        heldout_logits = probe_logits_from_features(
            state_dict=probe_fit.state_dict,
            features=arm_heldout_features,
        )
        primary_metrics = probe_metrics_from_logits(
            sample_logits=heldout_logits,
            labels=heldout_labels,
        )
        centroids = candidate_centroids(clustered_candidates)
        if centroids:
            heldout_similarity_summary = summarize_held_out_similarity(
                labels=heldout_labels,
                window_session_ids=heldout_session_ids,
                window_scores=window_similarity_scores(
                    tokens=arm_heldout_tokens,
                    token_mask=heldout_token_mask,
                    centroids=centroids,
                ),
            )
        else:
            heldout_similarity_summary = _unavailable_similarity_summary(
                reason=failure_reason or "no_non_noise_clusters_for_similarity_validation",
            )
        standard_validation_summary = None
        if run_standard_test_validation and transform_mode in {"baseline", "whitening_only"}:
            standard_validation_summary = _standard_cross_session_validation_summary(
                experiment_config=experiment_config,
                data_config_path=data_config_path,
                checkpoint_path=checkpoint_path,
                probe_state_dict=probe_fit.state_dict,
                candidates=clustered_candidates,
                transform_mode=transform_mode,
            )
        validation_summary = {
            "comparison_scope": "within_session_held_out",
            "arm_name": arm_name,
            "transform_mode": transform_mode,
            "fit_window_count": int(fit_indices.numel()),
            "heldout_window_count": int(heldout_indices.numel()),
            "valid_session_ids": list(split.valid_session_ids),
            "excluded_sessions": list(split.excluded_sessions),
            "reference_session_id": split.reference_session_id,
            "discovery_fit_metrics": probe_fit.metrics,
            "shuffled_fit_metrics": shuffled_fit.metrics,
            "primary_held_out_metrics": primary_metrics,
            "primary_held_out_similarity_summary": heldout_similarity_summary,
            "candidate_count": len(clustered_candidates),
            "cluster_count": int(cluster_stats.get("cluster_count", 0)),
            "cluster_quality_summary": cluster_quality_summary,
            "candidate_selection_summary": candidate_selection_summary,
            "comparison_status": "ok" if failure_reason is None else "degraded",
            "failure_reason": failure_reason,
            "standard_test_validation": standard_validation_summary,
        }
        arm_results.append(
            RepresentationComparisonArmResult(
                arm_name=arm_name,
                artifact=artifact,
                cluster_report=cluster_report,
                validation_summary=validation_summary,
                transform_summary=transform_summary,
            )
        )

    return RepresentationComparisonRunResult(
        coverage_summary=window_plan.coverage_summary,
        split_summary={
            "session_holdout_fraction": float(session_holdout_fraction),
            "session_holdout_seed": int(session_holdout_seed),
            "fit_window_count": int(fit_indices.numel()),
            "heldout_window_count": int(heldout_indices.numel()),
            "valid_session_ids": list(split.valid_session_ids),
            "excluded_sessions": list(split.excluded_sessions),
            "reference_session_id": split.reference_session_id,
            "session_rows": list(split.session_rows),
        },
        arm_results=tuple(arm_results),
    )
