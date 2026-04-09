from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import shutil
from typing import Any

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

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
from predictive_circuit_coding.decoding.scoring import select_candidate_tokens_from_shards
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


_SIMILARITY_CHUNK_SIZE = 512


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


def _probe_metrics_from_logits(*, sample_logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
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
        "positive_rate": float(labels.mean().item()),
        "probe_roc_auc": roc_auc,
        "probe_pr_auc": pr_auc,
    }


def _probe_logits_from_features(*, state_dict: dict[str, Any], features: torch.Tensor) -> torch.Tensor:
    weight = state_dict["linear.weight"].detach().cpu().reshape(-1).to(dtype=torch.float32)
    bias = float(state_dict["linear.bias"].detach().cpu().item())
    feature_tensor = features.detach().cpu().to(dtype=torch.float32)
    return (feature_tensor @ weight) + bias


def _candidate_centroids(candidates: tuple[Any, ...]) -> list[torch.Tensor]:
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
    positive_count = int((labels > 0.0).sum().item())
    negative_count = int((labels <= 0.0).sum().item())
    if positive_count <= 0 or negative_count <= 0:
        raise ValueError(
            "Held-out motif similarity validation requires both positive and negative windows in the held-out subset."
        )
    labels_np = labels.detach().cpu().numpy()
    scores_np = window_scores.detach().cpu().numpy()
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
        "positive_window_count": positive_count,
        "negative_window_count": negative_count,
        "per_session_roc_auc": per_session_roc_auc,
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


def _fit_shuffled_probe_features(
    *,
    features: torch.Tensor,
    labels: torch.Tensor,
    epochs: int,
    learning_rate: float,
    seed: int,
    label_name: str,
) -> Any:
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
        discovery_features = (
            discovery_tokens
            * discovery_collection.token_mask.unsqueeze(-1).to(dtype=discovery_tokens.dtype)
        ).sum(dim=1) / discovery_collection.token_mask.unsqueeze(-1).to(dtype=discovery_tokens.dtype).sum(dim=1).clamp_min(1.0)
        test_features = (
            test_tokens
            * test_collection.token_mask.unsqueeze(-1).to(dtype=test_tokens.dtype)
        ).sum(dim=1) / test_collection.token_mask.unsqueeze(-1).to(dtype=test_tokens.dtype).sum(dim=1).clamp_min(1.0)
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
    centroids = _candidate_centroids(candidates)
    window_scores = _window_similarity_scores(
        tokens=test_tokens,
        token_mask=test_collection.token_mask,
        centroids=centroids,
    )
    held_out_similarity_summary = _held_out_similarity_summary(
        labels=test_collection.labels,
        window_session_ids=test_collection.window_session_ids,
        window_scores=window_scores,
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
        shuffled_fit = _fit_shuffled_probe_features(
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
        candidates = select_candidate_tokens_from_shards(
            shard_paths=transformed_shards,
            probe_state_dict=probe_fit.state_dict,
            top_k=experiment_config.discovery.top_k_candidates,
            min_score=experiment_config.discovery.min_candidate_score,
            candidate_session_balance_fraction=experiment_config.discovery.candidate_session_balance_fraction,
        )
        if transformed_shards:
            shutil.rmtree(arm_shard_dir)
        if not candidates:
            raise ValueError(
                f"No candidate tokens were selected for comparison arm '{arm_name}'. "
                "Lower the candidate threshold, increase discovery coverage, or confirm positive labels exist."
            )
        clustered_candidates, cluster_stats = cluster_candidate_tokens(
            candidates=candidates,
            min_cluster_size=experiment_config.discovery.min_cluster_size,
        )
        if int(cluster_stats.get("cluster_count", 0)) <= 0:
            raise ValueError(
                f"Comparison arm '{arm_name}' selected candidate tokens but produced no non-noise clusters."
            )
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

        heldout_logits = _probe_logits_from_features(
            state_dict=probe_fit.state_dict,
            features=arm_heldout_features,
        )
        primary_metrics = _probe_metrics_from_logits(
            sample_logits=heldout_logits,
            labels=heldout_labels,
        )
        centroids = _candidate_centroids(clustered_candidates)
        heldout_similarity_summary = _held_out_similarity_summary(
            labels=heldout_labels,
            window_session_ids=heldout_session_ids,
            window_scores=_window_similarity_scores(
                tokens=arm_heldout_tokens,
                token_mask=heldout_token_mask,
                centroids=centroids,
            ),
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
