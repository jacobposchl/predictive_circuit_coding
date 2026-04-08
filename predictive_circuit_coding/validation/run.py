from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import random
from typing import Any, Callable

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from predictive_circuit_coding.data import resolve_runtime_dataset_view
from predictive_circuit_coding.decoding import evaluate_additive_probe, extract_frozen_tokens, fit_additive_probe
from predictive_circuit_coding.training.artifacts import load_training_checkpoint
from predictive_circuit_coding.training.config import ExperimentConfig
from predictive_circuit_coding.training.contracts import ValidationSummary
from predictive_circuit_coding.training.factories import build_model_from_config
from predictive_circuit_coding.training.runtime import resolve_device
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


def _load_discovery_artifact(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _deserialize_probe_state(raw_state: dict[str, Any] | None) -> dict[str, torch.Tensor] | None:
    if not raw_state:
        return None
    return {
        str(key): torch.tensor(value, dtype=torch.float32)
        for key, value in raw_state.items()
    }


def _candidate_centroids(candidates: list[dict]) -> list[torch.Tensor]:
    grouped: dict[int, list[torch.Tensor]] = {}
    for candidate in candidates:
        cluster_id = int(candidate["cluster_id"])
        if cluster_id == -1:
            continue
        grouped.setdefault(cluster_id, []).append(torch.tensor(candidate["embedding"], dtype=torch.float32))
    return [torch.stack(values, dim=0).mean(dim=0) for _, values in sorted(grouped.items())]


_SIMILARITY_CHUNK_SIZE = 512


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
            "Held-out motif similarity validation requires both positive and negative windows on the test split. "
            f"Found positive_windows={positive_count}, negative_windows={negative_count}."
        )
    labels_np = labels.detach().cpu().numpy()
    scores_np = window_scores.detach().cpu().numpy()
    per_session_roc_auc: dict[str, float] = {}
    for session_id in sorted(set(window_session_ids)):
        indices = [index for index, value in enumerate(window_session_ids) if value == session_id]
        session_labels = labels_np[indices]
        if len(set(int(value) for value in session_labels.tolist())) < 2:
            continue
        per_session_roc_auc[session_id] = float(roc_auc_score(session_labels, scores_np[indices]))
    return {
        "window_roc_auc": float(roc_auc_score(labels_np, scores_np)),
        "window_pr_auc": float(average_precision_score(labels_np, scores_np)),
        "positive_window_count": positive_count,
        "negative_window_count": negative_count,
        "per_session_roc_auc": per_session_roc_auc,
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


def _validate_artifact_provenance(
    *,
    artifact: dict[str, Any],
    checkpoint_path: str | Path,
    target_label: str,
) -> None:
    artifact_checkpoint = artifact.get("checkpoint_path")
    if artifact_checkpoint and Path(artifact_checkpoint).resolve() != Path(checkpoint_path).resolve():
        raise ValueError(
            "Discovery artifact checkpoint_path does not match the checkpoint selected for validation. "
            f"artifact={Path(artifact_checkpoint).resolve()}, checkpoint={Path(checkpoint_path).resolve()}."
        )
    artifact_target_label = artifact.get("decoder_summary", {}).get("target_label")
    if artifact_target_label and str(artifact_target_label) != str(target_label):
        raise ValueError(
            "Discovery artifact target label does not match the validation config target label. "
            f"artifact='{artifact_target_label}', config='{target_label}'."
        )


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
    zero_window_scores = _window_similarity_scores(
        tokens=zero_tokens,
        token_mask=test_collection.token_mask,
        centroids=centroids,
    )
    zero_similarity_summary = _held_out_similarity_summary(
        labels=test_collection.labels,
        window_session_ids=test_collection.window_session_ids,
        window_scores=zero_window_scores,
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
) -> ValidationSummary:
    dataset_view = dataset_view or resolve_runtime_dataset_view(
        experiment_config=experiment_config,
        data_config_path=data_config_path,
    )
    print(f"validate: start {_ram_mb()}")
    artifact = _load_discovery_artifact(discovery_artifact_path)
    if not artifact.get("candidates"):
        raise ValueError("Validation cannot run because the discovery artifact contains no candidate tokens.")
    _validate_artifact_provenance(
        artifact=artifact,
        checkpoint_path=checkpoint_path,
        target_label=experiment_config.discovery.target_label,
    )
    artifact_probe_state = _deserialize_probe_state(artifact.get("decoder_summary", {}).get("probe_state"))

    device = resolve_device(experiment_config.execution.device)
    shared_model = build_model_from_config(experiment_config).to(device)
    checkpoint = load_training_checkpoint(checkpoint_path, map_location=device)
    shared_model.load_state_dict(checkpoint["model_state"])
    del checkpoint
    shared_model.eval()
    print(f"validate: model loaded {_ram_mb()}")

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
    print(
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
        )
        artifact_probe_state = real_probe_fit.state_dict
        real_label_metrics = real_probe_fit.metrics
        print(f"validate: real probe trained {_ram_mb()}")
    else:
        real_label_metrics = evaluate_additive_probe(
            state_dict=artifact_probe_state,
            tokens=discovery_collection.tokens,
            token_mask=discovery_collection.token_mask,
            labels=discovery_collection.labels,
        )
        print(f"validate: real probe re-evaluated {_ram_mb()}")

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
    )
    print(f"validate: shuffled probe trained {_ram_mb()}")
    discovery_sampled_window_count = int(discovery_collection.labels.shape[0])
    del discovery_collection
    del shuffled_labels
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"validate: discovery freed {_ram_mb()}")

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
    print(
        f"validate: test extracted  windows={test_collection.tokens.shape[0]}"
        f"  seq_len={test_collection.tokens.shape[1]}"
        f"  tokens_MB={test_collection.tokens.nbytes / 1_048_576:.0f}"
        f"  {_ram_mb()}"
    )

    del shared_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"validate: model freed {_ram_mb()}")

    held_out_test_metrics = evaluate_additive_probe(
        state_dict=artifact_probe_state,
        tokens=test_collection.tokens,
        token_mask=test_collection.token_mask,
        labels=test_collection.labels,
    )
    centroids = _candidate_centroids(artifact["candidates"])
    window_scores = _window_similarity_scores(
        tokens=test_collection.tokens,
        token_mask=test_collection.token_mask,
        centroids=centroids,
    )
    held_out_similarity_summary = _held_out_similarity_summary(
        labels=test_collection.labels,
        window_session_ids=test_collection.window_session_ids,
        window_scores=window_scores,
    )

    checkpoint_state = load_training_checkpoint(checkpoint_path, map_location="cpu")
    metadata = checkpoint_state.get("metadata", {})
    del checkpoint_state
    cluster_count = len(
        {
            int(candidate["cluster_id"])
            for candidate in artifact["candidates"]
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
        candidate_count=len(artifact["candidates"]),
        cluster_count=cluster_count,
        cluster_quality_summary=artifact.get("cluster_quality_summary", {}),
        provenance_issues=_provenance_issues(
            candidates=artifact["candidates"],
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
