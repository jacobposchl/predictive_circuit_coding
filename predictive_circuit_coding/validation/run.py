from __future__ import annotations

import gc
import os
from pathlib import Path
import random
from typing import Any, Callable

import torch

from predictive_circuit_coding.data import resolve_runtime_dataset_view
from predictive_circuit_coding.decoding import evaluate_additive_probe, extract_frozen_tokens, fit_additive_probe
from predictive_circuit_coding.decoding.scoring import (
    candidate_centroids,
    held_out_similarity_summary as summarize_held_out_similarity,
    window_similarity_scores,
)
from predictive_circuit_coding.training.artifacts import load_training_checkpoint
from predictive_circuit_coding.training.config import ExperimentConfig
from predictive_circuit_coding.training.contracts import ValidationSummary
from predictive_circuit_coding.training.factories import build_model_from_config
from predictive_circuit_coding.training.runtime import resolve_device
from predictive_circuit_coding.validation.artifact_checks import (
    load_discovery_artifact,
    validate_discovery_artifact_identity,
)
from predictive_circuit_coding.windowing import (
    FixedWindowConfig,
    build_dataset_bundle,
    build_sequential_fixed_window_sampler,
)


def _ram_mb() -> str:
    try:
        import psutil

        rss = psutil.Process(os.getpid()).memory_info().rss / 1_048_576
        return f"[RAM {rss:.0f} MB]"
    except Exception:
        return ""


def _deserialize_probe_state(raw_state: dict[str, Any] | None) -> dict[str, torch.Tensor] | None:
    if not raw_state:
        return None
    return {
        str(key): torch.tensor(value, dtype=torch.float32)
        for key, value in raw_state.items()
    }


def _count_split_windows(
    *,
    experiment_config: ExperimentConfig,
    dataset_view,
    split_name: str,
) -> int:
    bundle = build_dataset_bundle(
        workspace=dataset_view.workspace,
        split_manifest=dataset_view.split_manifest,
        split=split_name,
        config_dir=dataset_view.config_dir,
        config_name_prefix=dataset_view.config_name_prefix,
        dataset_split=dataset_view.dataset_split,
    )
    sampler = build_sequential_fixed_window_sampler(
        bundle.dataset,
        window=FixedWindowConfig(
            window_length_s=experiment_config.data_runtime.context_duration_s,
            step_s=experiment_config.evaluation.sequential_step_s,
        ),
    )
    try:
        try:
            return int(len(sampler))
        except TypeError:
            return sum(1 for _ in sampler)
    finally:
        if hasattr(bundle.dataset, "_close_open_files"):
            bundle.dataset._close_open_files()


_REQUIRED_CANDIDATE_FIELDS = frozenset(
    {
        "candidate_id",
        "cluster_id",
        "recording_id",
        "session_id",
        "subject_id",
        "unit_id",
        "unit_region",
        "unit_depth_um",
        "patch_index",
        "patch_start_s",
        "patch_end_s",
        "window_start_s",
        "window_end_s",
        "label",
        "score",
        "embedding",
    }
)


def _validate_candidate_schema(
    *,
    candidates: Any,
    expected_embedding_dim: int,
) -> list[dict]:
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Validation cannot run because the discovery artifact contains no candidate tokens.")
    issues: list[str] = []
    non_noise_cluster_count = 0
    parsed_candidates: list[dict] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            issues.append(f"candidate[{index}] is not an object")
            continue
        candidate_id = str(candidate.get("candidate_id", f"index_{index}"))
        missing = sorted(_REQUIRED_CANDIDATE_FIELDS - set(candidate))
        if missing:
            issues.append(f"{candidate_id} missing fields: {', '.join(missing)}")
            continue
        try:
            cluster_id = int(candidate["cluster_id"])
        except (TypeError, ValueError):
            issues.append(f"{candidate_id} has invalid cluster_id")
            continue
        if cluster_id != -1:
            non_noise_cluster_count += 1
        embedding = candidate.get("embedding")
        if not isinstance(embedding, (list, tuple)):
            issues.append(f"{candidate_id} embedding must be a list")
            continue
        if len(embedding) != expected_embedding_dim:
            issues.append(
                f"{candidate_id} embedding dimension {len(embedding)} does not match model.d_model {expected_embedding_dim}"
            )
            continue
        try:
            tuple(float(value) for value in embedding)
        except (TypeError, ValueError):
            issues.append(f"{candidate_id} embedding contains non-numeric values")
            continue
        parsed_candidates.append(candidate)
    if issues:
        raise ValueError("Discovery artifact candidate schema is invalid: " + "; ".join(issues))
    if non_noise_cluster_count <= 0:
        raise ValueError("Discovery artifact does not contain any non-noise candidate clusters to validate.")
    return parsed_candidates


def _provenance_issues(
    *,
    candidates: list[dict],
    dataset_view,
    experiment_config: ExperimentConfig,
    split_name: str,
) -> tuple[str, ...]:
    issues: list[str] = []
    dataset_id = str(experiment_config.dataset_id)

    def _normalize_identity(value: str) -> str:
        prefix = f"{dataset_id}/"
        return value[len(prefix) :] if value.startswith(prefix) else value

    catalog_by_recording = {
        str(record.recording_id): record
        for record in dataset_view.session_catalog.records
    }
    split_by_recording = {
        str(assignment.recording_id): str(assignment.split)
        for assignment in dataset_view.split_manifest.assignments
    }
    expected_patch_count = int(experiment_config.data_runtime.patches_per_window)
    expected_patch_duration = float(experiment_config.data_runtime.patch_bins * experiment_config.data_runtime.bin_width_s)
    expected_window_duration = float(experiment_config.data_runtime.context_duration_s)
    expected_embedding_dim = int(experiment_config.model.d_model)
    tolerance = 1.0e-4

    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id", "unknown"))
        recording_id = str(candidate.get("recording_id", "") or "")
        session_id = _normalize_identity(str(candidate.get("session_id", "") or ""))
        subject_id = _normalize_identity(str(candidate.get("subject_id", "") or ""))
        unit_id = str(candidate.get("unit_id", "") or "")

        for key, value in (
            ("recording_id", recording_id),
            ("session_id", session_id),
            ("subject_id", subject_id),
            ("unit_id", unit_id),
        ):
            if not value:
                issues.append(f"missing_{key}:{candidate_id}")

        record = catalog_by_recording.get(recording_id)
        if not record:
            issues.append(f"unknown_recording_id:{candidate_id}")
        else:
            if session_id and _normalize_identity(str(record.session_id)) != session_id:
                issues.append(f"session_recording_mismatch:{candidate_id}")
            if subject_id and _normalize_identity(str(record.subject_id)) != subject_id:
                issues.append(f"subject_recording_mismatch:{candidate_id}")

        if recording_id and split_by_recording.get(recording_id) not in {None, split_name}:
            issues.append(f"candidate_outside_split:{candidate_id}")

        patch_start = float(candidate["patch_start_s"])
        patch_end = float(candidate["patch_end_s"])
        window_start = float(candidate["window_start_s"])
        window_end = float(candidate["window_end_s"])
        if patch_end <= patch_start:
            issues.append(f"bad_patch_interval:{candidate_id}")
        if window_end <= window_start:
            issues.append(f"bad_window_interval:{candidate_id}")
        if patch_start < (window_start - tolerance) or patch_end > (window_end + tolerance):
            issues.append(f"patch_outside_window:{candidate_id}")
        if abs((patch_end - patch_start) - expected_patch_duration) > tolerance:
            issues.append(f"unexpected_patch_duration:{candidate_id}")
        if abs((window_end - window_start) - expected_window_duration) > tolerance:
            issues.append(f"unexpected_window_duration:{candidate_id}")

        patch_index = int(candidate["patch_index"])
        if patch_index < 0 or patch_index >= expected_patch_count:
            issues.append(f"bad_patch_index:{candidate_id}")

        unit_depth = float(candidate["unit_depth_um"])
        if unit_depth < 0.0:
            issues.append(f"negative_unit_depth:{candidate_id}")

        embedding = tuple(candidate.get("embedding", ()))
        if len(embedding) != expected_embedding_dim:
            issues.append(f"bad_embedding_dim:{candidate_id}")

    return tuple(issues)


def _baseline_sensitivity_summary(
    *,
    baseline_type: str,
    probe_state: dict[str, torch.Tensor],
    test_collection,
    centroids: list[torch.Tensor],
    held_out_test_metrics: dict[str, float],
    held_out_similarity_summary: dict[str, Any],
) -> dict[str, Any]:
    zero_tokens = torch.zeros_like(test_collection.tokens)
    zero_probe_metrics = evaluate_additive_probe(
        state_dict=probe_state,
        tokens=zero_tokens,
        token_mask=test_collection.token_mask,
        labels=test_collection.labels,
    )
    zero_window_scores = window_similarity_scores(
        tokens=zero_tokens,
        token_mask=test_collection.token_mask,
        centroids=centroids,
        require_centroids=True,
    )
    zero_similarity_summary = summarize_held_out_similarity(
        labels=test_collection.labels,
        window_session_ids=test_collection.window_session_ids,
        window_scores=zero_window_scores,
        missing_class_error_message=(
            "Held-out motif similarity validation requires both positive and negative windows on the test split."
        ),
    )
    return {
        "evaluated_baseline_type": baseline_type,
        "comparison_available": True,
        "null_feature_probe_metrics": zero_probe_metrics,
        "null_feature_similarity_summary": zero_similarity_summary,
        "held_out_probe_accuracy_delta_vs_null": (
            float(held_out_test_metrics.get("probe_accuracy", 0.0))
            - float(zero_probe_metrics.get("probe_accuracy", 0.0))
        ),
        "held_out_similarity_roc_auc_delta_vs_null": (
            float(held_out_similarity_summary.get("window_roc_auc", 0.0))
            - float(zero_similarity_summary.get("window_roc_auc", 0.0))
        ),
    }


def validate_discovery_artifact(
    *,
    experiment_config: ExperimentConfig,
    data_config_path: str | Path,
    checkpoint_path: str | Path,
    discovery_artifact_path: str | Path,
    dataset_view=None,
    progress_callback: Callable[[int, int | None], None] | None = None,
    log_sink: Callable[[str], None] | None = None,
) -> ValidationSummary:
    def _log(message: str) -> None:
        if log_sink is not None:
            log_sink(message)

    dataset_view = dataset_view or resolve_runtime_dataset_view(
        experiment_config=experiment_config,
        data_config_path=data_config_path,
    )
    _log(f"validate: start {_ram_mb()}")
    artifact = load_discovery_artifact(discovery_artifact_path)
    validate_discovery_artifact_identity(
        artifact=artifact,
        checkpoint_path=checkpoint_path,
        dataset_id=experiment_config.dataset_id,
        split_name=experiment_config.splits.discovery,
        target_label=experiment_config.discovery.target_label,
        require_fields=True,
    )
    candidates = _validate_candidate_schema(
        candidates=artifact.get("candidates"),
        expected_embedding_dim=int(experiment_config.model.d_model),
    )
    artifact_probe_state = _deserialize_probe_state(artifact.get("decoder_summary", {}).get("probe_state"))

    device = resolve_device(experiment_config.execution.device)
    shared_model = build_model_from_config(experiment_config).to(device)
    checkpoint = load_training_checkpoint(checkpoint_path, map_location=device)
    shared_model.load_state_dict(checkpoint["model_state"])
    del checkpoint
    shared_model.eval()
    _log(f"validate: model loaded {_ram_mb()}")

    discovery_collection = extract_frozen_tokens(
        experiment_config=experiment_config,
        data_config_path=data_config_path,
        checkpoint_path=checkpoint_path,
        split_name=experiment_config.splits.discovery,
        max_batches=experiment_config.evaluation.max_batches,
        dataset_view=dataset_view,
        include_records=False,
        sampling_strategy_override="sequential",
        model=shared_model,
    )
    _log(
        f"validate: discovery extracted  windows={discovery_collection.tokens.shape[0]}"
        f"  seq_len={discovery_collection.tokens.shape[1]}"
        f"  tokens_MB={discovery_collection.tokens.nbytes / 1_048_576:.0f}"
        f"  {_ram_mb()}"
    )
    if artifact_probe_state is None:
        real_probe_fit = fit_additive_probe(
            tokens=discovery_collection.tokens,
            token_mask=discovery_collection.token_mask,
            labels=discovery_collection.labels,
            epochs=experiment_config.discovery.probe_epochs,
            learning_rate=experiment_config.discovery.probe_learning_rate,
            mini_batch_size=256,
            label_name=experiment_config.discovery.target_label,
            seed=experiment_config.discovery.shuffle_seed,
        )
        artifact_probe_state = real_probe_fit.state_dict
        real_label_metrics = real_probe_fit.metrics
        _log(f"validate: real probe trained {_ram_mb()}")
    else:
        real_label_metrics = evaluate_additive_probe(
            state_dict=artifact_probe_state,
            tokens=discovery_collection.tokens,
            token_mask=discovery_collection.token_mask,
            labels=discovery_collection.labels,
        )
        _log(f"validate: real probe re-evaluated {_ram_mb()}")

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
        mini_batch_size=256,
        label_name=experiment_config.discovery.target_label,
        seed=experiment_config.discovery.shuffle_seed + 1,
    )
    _log(f"validate: shuffled probe trained {_ram_mb()}")
    discovery_sampled_window_count = int(discovery_collection.labels.shape[0])
    del discovery_collection
    del shuffled_labels
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _log(f"validate: discovery freed {_ram_mb()}")

    test_collection = extract_frozen_tokens(
        experiment_config=experiment_config,
        data_config_path=data_config_path,
        checkpoint_path=checkpoint_path,
        split_name=experiment_config.splits.test,
        max_batches=experiment_config.evaluation.max_batches,
        dataset_view=dataset_view,
        include_token_tensors=True,
        include_records=False,
        positive_records_only=False,
        sampling_strategy_override="sequential",
        progress_callback=progress_callback,
        model=shared_model,
    )
    _log(
        f"validate: test extracted  windows={test_collection.tokens.shape[0]}"
        f"  seq_len={test_collection.tokens.shape[1]}"
        f"  tokens_MB={test_collection.tokens.nbytes / 1_048_576:.0f}"
        f"  {_ram_mb()}"
    )

    del shared_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _log(f"validate: model freed {_ram_mb()}")

    held_out_test_metrics = evaluate_additive_probe(
        state_dict=artifact_probe_state,
        tokens=test_collection.tokens,
        token_mask=test_collection.token_mask,
        labels=test_collection.labels,
    )
    centroids = candidate_centroids(candidates, require_non_noise=True)
    window_scores = window_similarity_scores(
        tokens=test_collection.tokens,
        token_mask=test_collection.token_mask,
        centroids=centroids,
        require_centroids=True,
    )
    held_out_similarity_summary = summarize_held_out_similarity(
        labels=test_collection.labels,
        window_session_ids=test_collection.window_session_ids,
        window_scores=window_scores,
        missing_class_error_message=(
            "Held-out motif similarity validation requires both positive and negative windows on the test split."
        ),
    )

    checkpoint_state = load_training_checkpoint(checkpoint_path, map_location="cpu")
    metadata = checkpoint_state.get("metadata", {})
    del checkpoint_state
    cluster_count = len(
        {
            int(candidate["cluster_id"])
            for candidate in candidates
            if int(candidate["cluster_id"]) != -1
        }
    )
    baseline_sensitivity_summary = _baseline_sensitivity_summary(
        baseline_type=metadata.get(
            "continuation_baseline_type",
            experiment_config.objective.continuation_baseline_type,
        ),
        probe_state=artifact_probe_state,
        test_collection=test_collection,
        centroids=centroids,
        held_out_test_metrics=held_out_test_metrics,
        held_out_similarity_summary=held_out_similarity_summary,
    )
    test_sampled_window_count = int(test_collection.labels.shape[0])
    discovery_full_split_window_count = _count_split_windows(
        experiment_config=experiment_config,
        dataset_view=dataset_view,
        split_name=experiment_config.splits.discovery,
    )
    test_full_split_window_count = _count_split_windows(
        experiment_config=experiment_config,
        dataset_view=dataset_view,
        split_name=experiment_config.splits.test,
    )

    return ValidationSummary(
        dataset_id=experiment_config.dataset_id,
        checkpoint_path=str(checkpoint_path),
        discovery_artifact_path=str(discovery_artifact_path),
        real_label_metrics=real_label_metrics,
        shuffled_label_metrics=shuffled_fit.metrics,
        held_out_test_metrics=held_out_test_metrics,
        held_out_similarity_summary=held_out_similarity_summary,
        baseline_sensitivity_summary=baseline_sensitivity_summary,
        candidate_count=len(candidates),
        cluster_count=cluster_count,
        cluster_quality_summary=artifact.get("cluster_quality_summary", {}),
        provenance_issues=_provenance_issues(
            candidates=candidates,
            dataset_view=dataset_view,
            experiment_config=experiment_config,
            split_name=str(artifact.get("split_name", experiment_config.splits.discovery)),
        ),
        sampling_summary={
            "evaluation_max_batches": int(experiment_config.evaluation.max_batches),
            "discovery_sampled_window_count": discovery_sampled_window_count,
            "discovery_full_split_window_count": discovery_full_split_window_count,
            "discovery_sampling_is_capped": discovery_sampled_window_count < discovery_full_split_window_count,
            "test_sampled_window_count": test_sampled_window_count,
            "test_full_split_window_count": test_full_split_window_count,
            "test_sampling_is_capped": test_sampled_window_count < test_full_split_window_count,
        },
    )
