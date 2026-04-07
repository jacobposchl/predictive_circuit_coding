from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import random
from typing import Any, Callable

import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def _ram_mb() -> str:
    """Return current process RSS in MB, or empty string if psutil unavailable."""
    try:
        import psutil
        rss = psutil.Process(os.getpid()).memory_info().rss / 1_048_576
        return f"[RAM {rss:.0f} MB]"
    except Exception:
        return ""

from predictive_circuit_coding.decoding import evaluate_additive_probe, extract_frozen_tokens, fit_additive_probe
from predictive_circuit_coding.training.artifacts import load_training_checkpoint
from predictive_circuit_coding.training.config import ExperimentConfig
from predictive_circuit_coding.training.contracts import ValidationSummary
from predictive_circuit_coding.training.factories import build_model_from_config
from predictive_circuit_coding.training.runtime import resolve_device


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


def _cosine_similarity(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    lhs = lhs / lhs.norm().clamp_min(1.0e-8)
    rhs = rhs / rhs.norm().clamp_min(1.0e-8)
    return float(torch.dot(lhs, rhs).item())


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
    dataset_view=None,
    progress_callback: Callable[[int, int | None], None] | None = None,
) -> ValidationSummary:
    print(f"validate: start {_ram_mb()}")
    artifact = _load_discovery_artifact(discovery_artifact_path)
    if not artifact.get("candidates"):
        raise ValueError(
            "Validation cannot run because the discovery artifact contains no candidate tokens."
        )
    artifact_decoder_metrics = dict(artifact.get("decoder_summary", {}).get("metrics", {}))
    artifact_probe_state = _deserialize_probe_state(artifact.get("decoder_summary", {}).get("probe_state"))

    # Load the model once and reuse it for both extractions, avoiding a second
    # checkpoint load (model + optimizer state) while the first is still live.
    device = resolve_device(experiment_config.execution.device)
    shared_model = build_model_from_config(experiment_config).to(device)
    checkpoint = load_training_checkpoint(checkpoint_path, map_location=device)
    shared_model.load_state_dict(checkpoint["model_state"])
    del checkpoint
    shared_model.eval()
    print(f"validate: model loaded {_ram_mb()}")

    # Cap the discovery re-extraction to evaluation.max_batches — the shuffle control is a
    # relative statistical check and does not require the full discovery pass.
    # Override to "sequential" so max_batches is actually respected: "label_balanced" ignores
    # max_batches and scans the entire split, which can exhaust CPU RAM on large datasets.
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
        print(f"validate: real probe trained {_ram_mb()}")
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

    # Model no longer needed — free GPU memory before CPU-only metrics work.
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

    # Load only metadata from checkpoint — map to CPU so no GPU memory is used.
    checkpoint_state = load_training_checkpoint(checkpoint_path, map_location="cpu")
    metadata = checkpoint_state.get("metadata", {})
    del checkpoint_state
    cluster_count = len({int(candidate["cluster_id"]) for candidate in artifact["candidates"] if int(candidate["cluster_id"]) != -1})
    return ValidationSummary(
        dataset_id=experiment_config.dataset_id,
        checkpoint_path=str(checkpoint_path),
        discovery_artifact_path=str(discovery_artifact_path),
        real_label_metrics=artifact_decoder_metrics,
        shuffled_label_metrics=shuffled_fit.metrics,
        held_out_test_metrics=held_out_test_metrics,
        held_out_similarity_summary=held_out_similarity_summary,
        baseline_sensitivity_summary={
            "evaluated_baseline_type": metadata.get("continuation_baseline_type", experiment_config.objective.continuation_baseline_type),
            "comparison_available": False,
        },
        candidate_count=len(artifact["candidates"]),
        cluster_count=cluster_count,
        cluster_quality_summary=artifact.get("cluster_quality_summary", {}),
        provenance_issues=_provenance_issues(artifact["candidates"]),
    )
