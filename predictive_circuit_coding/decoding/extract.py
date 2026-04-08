from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from pathlib import Path
import random
import shutil
from typing import Callable

import torch

from predictive_circuit_coding.data import resolve_runtime_dataset_view
from predictive_circuit_coding.decoding.labels import extract_binary_label_from_annotations, extract_binary_labels
from predictive_circuit_coding.training.artifacts import load_training_checkpoint
from predictive_circuit_coding.training.config import ExperimentConfig
from predictive_circuit_coding.training.contracts import DiscoveryCoverageSummary, FrozenTokenRecord
from predictive_circuit_coding.training.factories import build_model_from_config, build_tokenizer_from_config
from predictive_circuit_coding.training.runtime import iter_sampler_batches, resolve_device
from predictive_circuit_coding.tokenization import extract_sample_event_annotations, extract_sample_recording_metadata
from predictive_circuit_coding.windowing import (
    FixedWindowConfig,
    build_dataset_bundle,
    build_sequential_fixed_window_sampler,
)


@dataclass(frozen=True)
class FrozenTokenCollection:
    tokens: torch.Tensor
    token_mask: torch.Tensor
    labels: torch.Tensor
    records: tuple[FrozenTokenRecord, ...]
    coverage_summary: DiscoveryCoverageSummary
    window_session_ids: tuple[str, ...]


@dataclass(frozen=True)
class DiscoveryWindowPlanRecord:
    recording_id: str
    session_id: str
    subject_id: str
    window_start_s: float
    window_end_s: float
    label: float


@dataclass(frozen=True)
class DiscoveryWindowPlan:
    split_name: str
    target_label: str
    windows: tuple[DiscoveryWindowPlanRecord, ...]
    selected_indices: torch.Tensor
    coverage_summary: DiscoveryCoverageSummary


@dataclass(frozen=True)
class EncodedDiscoverySelection:
    pooled_features: torch.Tensor
    labels: torch.Tensor
    window_session_ids: tuple[str, ...]
    window_subject_ids: tuple[str, ...]
    shard_paths: tuple[Path, ...]
    coverage_summary: DiscoveryCoverageSummary
    encoder_device: str


ProgressCallback = Callable[[int, int | None], None]


def _session_stratified_subsample(
    *,
    indices: torch.Tensor,
    session_ids: tuple[str, ...],
    target_count: int,
    seed: int,
) -> torch.Tensor:
    if target_count >= len(indices):
        return indices
    grouped_indices: dict[str, list[int]] = {}
    for index in indices.tolist():
        grouped_indices.setdefault(session_ids[index], []).append(int(index))
    ordered_sessions = sorted(grouped_indices)
    rng = random.Random(seed)
    for session_id in ordered_sessions:
        rng.shuffle(grouped_indices[session_id])
    selected: list[int] = []
    while len(selected) < target_count:
        made_progress = False
        for session_id in ordered_sessions:
            session_pool = grouped_indices[session_id]
            if not session_pool:
                continue
            selected.append(session_pool.pop())
            made_progress = True
            if len(selected) >= target_count:
                break
        if not made_progress:
            break
    selected.sort()
    return torch.tensor(selected, dtype=torch.long)


def _select_label_balanced_indices(
    *,
    labels: torch.Tensor,
    session_ids: tuple[str, ...],
    seed: int,
    max_selected_windows: int,
    negative_to_positive_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positive_indices = torch.nonzero(labels > 0.0, as_tuple=False).flatten()
    negative_indices = torch.nonzero(labels <= 0.0, as_tuple=False).flatten()
    if len(positive_indices) == 0 or len(negative_indices) == 0:
        return torch.empty(0, dtype=torch.long), positive_indices, negative_indices
    budget = max(1, int(max_selected_windows))
    target_positive = 0
    target_negative = 0
    ratio = max(0.0, float(negative_to_positive_ratio))
    for positive_target in range(min(len(positive_indices), budget), 0, -1):
        negative_target = min(len(negative_indices), int(math.floor(float(positive_target) * ratio)))
        if positive_target + negative_target <= budget:
            target_positive = positive_target
            target_negative = negative_target
            break
    if target_positive <= 0:
        return torch.empty(0, dtype=torch.long), positive_indices, negative_indices
    selected_positive = _session_stratified_subsample(
        indices=positive_indices,
        session_ids=session_ids,
        target_count=target_positive,
        seed=seed,
    )
    selected_negative = _session_stratified_subsample(
        indices=negative_indices,
        session_ids=session_ids,
        target_count=target_negative,
        seed=seed + 1,
    )
    selected_indices = torch.cat((selected_positive, selected_negative), dim=0).sort().values
    return selected_indices, selected_positive, selected_negative


def _build_coverage_summary(
    *,
    split_name: str,
    target_label: str,
    labels: torch.Tensor,
    session_ids: tuple[str, ...],
    selected_indices: torch.Tensor | None = None,
    sampling_strategy: str,
    scan_max_batches: int | None,
) -> DiscoveryCoverageSummary:
    positive_indices = torch.nonzero(labels > 0.0, as_tuple=False).flatten()
    negative_indices = torch.nonzero(labels <= 0.0, as_tuple=False).flatten()
    selected_labels = labels if selected_indices is None else labels[selected_indices]
    sessions_with_positive_windows = tuple(sorted({session_ids[index] for index in positive_indices.tolist()}))
    return DiscoveryCoverageSummary(
        split_name=split_name,
        target_label=target_label,
        total_scanned_windows=int(labels.numel()),
        positive_window_count=int(len(positive_indices)),
        negative_window_count=int(len(negative_indices)),
        selected_positive_count=int((selected_labels > 0.0).sum().item()),
        selected_negative_count=int((selected_labels <= 0.0).sum().item()),
        sessions_with_positive_windows=sessions_with_positive_windows,
        sampling_strategy=sampling_strategy,
        scan_max_batches=scan_max_batches,
        selected_window_count=int(selected_labels.numel()),
    )


def _maybe_update_progress(progress_callback: ProgressCallback | None, current: int, total: int | None) -> None:
    if progress_callback is not None:
        progress_callback(current, total)


def _pooled_features_from_tokens(tokens: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    mask = token_mask.to(dtype=tokens.dtype).unsqueeze(-1)
    token_counts = mask.sum(dim=1).clamp_min(1.0)
    return (tokens * mask).sum(dim=1) / token_counts


def build_discovery_window_plan(
    *,
    experiment_config: ExperimentConfig,
    data_config_path: str | Path,
    split_name: str,
    dataset_view=None,
    progress_callback: ProgressCallback | None = None,
) -> DiscoveryWindowPlan:
    dataset_view = dataset_view or resolve_runtime_dataset_view(
        experiment_config=experiment_config,
        data_config_path=data_config_path,
    )
    workspace = dataset_view.workspace
    split_manifest = dataset_view.split_manifest
    bundle = build_dataset_bundle(
        workspace=workspace,
        split_manifest=split_manifest,
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
    sampling_strategy = experiment_config.discovery.sampling_strategy
    if sampling_strategy == "sequential":
        scan_max_batches = (
            experiment_config.discovery.search_max_batches
            if experiment_config.discovery.search_max_batches is not None
            else experiment_config.discovery.max_batches
        )
    else:
        scan_max_batches = experiment_config.discovery.search_max_batches
    scan_max_windows = (
        None
        if scan_max_batches is None
        else int(scan_max_batches) * int(experiment_config.optimization.batch_size)
    )
    planned_windows: list[DiscoveryWindowPlanRecord] = []
    for item in sampler:
        if scan_max_windows is not None and len(planned_windows) >= scan_max_windows:
            break
        sample = bundle.dataset.get(item.recording_id, item.start, item.end)
        recording_id, session_id, subject_id = extract_sample_recording_metadata(sample)
        annotations = extract_sample_event_annotations(
            sample,
            experiment_config.data_runtime,
            window_start_s=float(item.start),
            window_end_s=float(item.end),
        )
        label = extract_binary_label_from_annotations(
            annotations,
            target_label=experiment_config.discovery.target_label,
            target_label_mode=experiment_config.discovery.target_label_mode,
            target_label_match_value=experiment_config.discovery.target_label_match_value,
            window_duration_s=float(item.end) - float(item.start),
        )
        planned_windows.append(
            DiscoveryWindowPlanRecord(
                recording_id=recording_id or str(item.recording_id),
                session_id=session_id,
                subject_id=subject_id,
                window_start_s=float(item.start),
                window_end_s=float(item.end),
                label=float(label),
            )
        )
        _maybe_update_progress(progress_callback, len(planned_windows), scan_max_windows)
    if hasattr(bundle.dataset, "_close_open_files"):
        bundle.dataset._close_open_files()
    labels = torch.tensor([window.label for window in planned_windows], dtype=torch.float32)
    session_ids = tuple(window.session_id for window in planned_windows)
    if sampling_strategy == "label_balanced":
        selected_indices, _, _ = _select_label_balanced_indices(
            labels=labels,
            session_ids=session_ids,
            seed=experiment_config.seed,
            max_selected_windows=int(experiment_config.discovery.max_batches) * int(experiment_config.optimization.batch_size),
            negative_to_positive_ratio=experiment_config.discovery.negative_to_positive_ratio,
        )
    else:
        selected_indices = torch.arange(labels.numel(), dtype=torch.long)
    coverage_summary = _build_coverage_summary(
        split_name=split_name,
        target_label=experiment_config.discovery.target_label,
        labels=labels,
        session_ids=session_ids,
        selected_indices=selected_indices,
        sampling_strategy=sampling_strategy,
        scan_max_batches=scan_max_batches,
    )
    return DiscoveryWindowPlan(
        split_name=split_name,
        target_label=experiment_config.discovery.target_label,
        windows=tuple(planned_windows),
        selected_indices=selected_indices,
        coverage_summary=coverage_summary,
    )


def _write_token_shard(
    *,
    shard_dir: Path,
    shard_index: int,
    batch,
    tokens: torch.Tensor,
    labels: torch.Tensor,
) -> Path | None:
    flat_tokens = tokens.reshape(tokens.shape[0], -1, tokens.shape[-1])
    flat_mask = batch.patch_mask.reshape(batch.patch_mask.shape[0], -1).detach().cpu()

    embeddings: list[torch.Tensor] = []
    recording_ids: list[str] = []
    session_ids: list[str] = []
    subject_ids: list[str] = []
    unit_ids: list[str] = []
    unit_regions: list[str] = []
    unit_depth_um: list[float] = []
    patch_index: list[int] = []
    patch_start_s: list[float] = []
    patch_end_s: list[float] = []
    window_start_s: list[float] = []
    window_end_s: list[float] = []
    window_labels: list[int] = []

    for batch_index in range(tokens.shape[0]):
        sample_tokens = flat_tokens[batch_index]
        sample_mask = flat_mask[batch_index]
        sample_label = int(labels[batch_index].item() > 0.0)
        flat_position = 0
        for unit_index, unit_id in enumerate(batch.provenance.unit_ids[batch_index]):
            for patch_idx in range(tokens.shape[2]):
                if not bool(batch.patch_mask[batch_index, unit_index, patch_idx].item()):
                    flat_position += 1
                    continue
                if bool(sample_mask[flat_position].item()):
                    embeddings.append(sample_tokens[flat_position].detach().cpu())
                    recording_ids.append(batch.provenance.recording_ids[batch_index])
                    session_ids.append(batch.provenance.session_ids[batch_index])
                    subject_ids.append(batch.provenance.subject_ids[batch_index])
                    unit_ids.append(unit_id)
                    unit_regions.append(batch.provenance.unit_regions[batch_index][unit_index])
                    unit_depth_um.append(float(batch.provenance.unit_depth_um[batch_index, unit_index].item()))
                    patch_index.append(int(patch_idx))
                    patch_start_s.append(float(batch.provenance.patch_start_s[batch_index, patch_idx].item()))
                    patch_end_s.append(float(batch.provenance.patch_end_s[batch_index, patch_idx].item()))
                    window_start_s.append(float(batch.provenance.window_start_s[batch_index].item()))
                    window_end_s.append(float(batch.provenance.window_end_s[batch_index].item()))
                    window_labels.append(sample_label)
                flat_position += 1

    if not embeddings:
        return None

    shard_path = shard_dir / f"token_shard_{shard_index:05d}.pt"
    torch.save(
        {
            "embeddings": torch.stack(embeddings, dim=0),
            "recording_ids": recording_ids,
            "session_ids": session_ids,
            "subject_ids": subject_ids,
            "unit_ids": unit_ids,
            "unit_regions": unit_regions,
            "unit_depth_um": torch.tensor(unit_depth_um, dtype=torch.float32),
            "patch_index": torch.tensor(patch_index, dtype=torch.long),
            "patch_start_s": torch.tensor(patch_start_s, dtype=torch.float32),
            "patch_end_s": torch.tensor(patch_end_s, dtype=torch.float32),
            "window_start_s": torch.tensor(window_start_s, dtype=torch.float32),
            "window_end_s": torch.tensor(window_end_s, dtype=torch.float32),
            "labels": torch.tensor(window_labels, dtype=torch.long),
        },
        shard_path,
    )
    return shard_path


def extract_selected_discovery_windows(
    *,
    experiment_config: ExperimentConfig,
    data_config_path: str | Path,
    checkpoint_path: str | Path,
    window_plan: DiscoveryWindowPlan,
    dataset_view=None,
    shard_dir: str | Path,
    progress_callback: ProgressCallback | None = None,
) -> EncodedDiscoverySelection:
    dataset_view = dataset_view or resolve_runtime_dataset_view(
        experiment_config=experiment_config,
        data_config_path=data_config_path,
    )
    workspace = dataset_view.workspace
    split_manifest = dataset_view.split_manifest
    tokenizer = build_tokenizer_from_config(experiment_config)
    device = resolve_device(experiment_config.execution.device)
    model = build_model_from_config(experiment_config).to(device)
    checkpoint = load_training_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    del checkpoint
    model.eval()

    bundle = build_dataset_bundle(
        workspace=workspace,
        split_manifest=split_manifest,
        split=window_plan.split_name,
        config_dir=dataset_view.config_dir,
        config_name_prefix=dataset_view.config_name_prefix,
        dataset_split=dataset_view.dataset_split,
    )
    selected_windows = [window_plan.windows[index] for index in window_plan.selected_indices.tolist()]
    total_selected = len(selected_windows)
    shard_root = Path(shard_dir)
    if shard_root.exists():
        shutil.rmtree(shard_root)
    shard_root.mkdir(parents=True, exist_ok=True)

    pooled_feature_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    window_session_ids: list[str] = []
    window_subject_ids: list[str] = []
    shard_paths: list[Path] = []
    shard_index = 0
    processed_windows = 0
    scaler_enabled = bool(experiment_config.execution.mixed_precision and device.type == "cuda")

    with torch.no_grad():
        for start in range(0, total_selected, experiment_config.optimization.batch_size):
            window_batch = selected_windows[start : start + experiment_config.optimization.batch_size]
            samples = [
                bundle.dataset.get(window.recording_id, window.window_start_s, window.window_end_s)
                for window in window_batch
            ]
            batch = tokenizer(samples)
            labels = torch.tensor([window.label for window in window_batch], dtype=torch.float32)
            device_batch = batch.to(device)
            autocast_context = (
                torch.autocast(device_type=device.type, dtype=torch.float16) if scaler_enabled else nullcontext()
            )
            with autocast_context:
                output = model(device_batch)
            flat_tokens = output.tokens.detach().cpu().reshape(output.tokens.shape[0], -1, output.tokens.shape[-1])
            flat_mask = output.patch_mask.detach().cpu().reshape(output.patch_mask.shape[0], -1)
            pooled_feature_chunks.append(_pooled_features_from_tokens(flat_tokens, flat_mask))
            label_chunks.append(labels)
            window_session_ids.extend(window.session_id for window in window_batch)
            window_subject_ids.extend(window.subject_id for window in window_batch)
            shard_path = _write_token_shard(
                shard_dir=shard_root,
                shard_index=shard_index,
                batch=batch,
                tokens=output.tokens.detach().cpu(),
                labels=labels,
            )
            if shard_path is not None:
                shard_paths.append(shard_path)
                shard_index += 1
            processed_windows += len(window_batch)
            _maybe_update_progress(progress_callback, processed_windows, total_selected)

    if hasattr(bundle.dataset, "_close_open_files"):
        bundle.dataset._close_open_files()

    return EncodedDiscoverySelection(
        pooled_features=torch.cat(pooled_feature_chunks, dim=0) if pooled_feature_chunks else torch.empty((0, 0), dtype=torch.float32),
        labels=torch.cat(label_chunks, dim=0) if label_chunks else torch.empty((0,), dtype=torch.float32),
        window_session_ids=tuple(window_session_ids),
        window_subject_ids=tuple(window_subject_ids),
        shard_paths=tuple(shard_paths),
        coverage_summary=window_plan.coverage_summary,
        encoder_device=str(device),
    )


def extract_frozen_tokens(
    *,
    experiment_config: ExperimentConfig,
    data_config_path: str | Path,
    checkpoint_path: str | Path,
    split_name: str,
    max_batches: int | None = None,
    dataset_view=None,
    include_token_tensors: bool = True,
    include_records: bool = True,
    positive_records_only: bool = False,
    sampling_strategy_override: str | None = None,
    progress_callback: ProgressCallback | None = None,
    model: torch.nn.Module | None = None,
) -> FrozenTokenCollection:
    dataset_view = dataset_view or resolve_runtime_dataset_view(
        experiment_config=experiment_config,
        data_config_path=data_config_path,
    )
    workspace = dataset_view.workspace
    split_manifest = dataset_view.split_manifest
    tokenizer = build_tokenizer_from_config(experiment_config)
    device = resolve_device(experiment_config.execution.device)
    if model is None:
        model = build_model_from_config(experiment_config).to(device)
        checkpoint = load_training_checkpoint(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        del checkpoint
        model.eval()

    bundle = build_dataset_bundle(
        workspace=workspace,
        split_manifest=split_manifest,
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

    token_chunks: list[torch.Tensor] = []
    mask_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    sample_record_groups: list[tuple[FrozenTokenRecord, ...]] = []
    sample_session_ids: list[str] = []
    sampling_strategy = sampling_strategy_override or experiment_config.discovery.sampling_strategy
    if sampling_strategy == "sequential":
        sampler_max_batches = (
            max_batches
            or experiment_config.discovery.search_max_batches
            or experiment_config.discovery.max_batches
        )
    else:
        sampler_max_batches = max_batches or experiment_config.discovery.search_max_batches

    with torch.no_grad():
        processed_batches = 0
        total_batches = sampler_max_batches
        for batch in iter_sampler_batches(
            dataset=bundle.dataset,
            sampler=sampler,
            collator=tokenizer,
            batch_size=experiment_config.optimization.batch_size,
            max_batches=sampler_max_batches,
        ):
            labels = extract_binary_labels(
                batch,
                target_label=experiment_config.discovery.target_label,
                target_label_mode=experiment_config.discovery.target_label_mode,
                target_label_match_value=experiment_config.discovery.target_label_match_value,
            )
            device_batch = batch.to(device)
            output = model(device_batch)
            batch_labels = labels.detach().cpu()
            tokens = output.tokens.detach().cpu() if (include_token_tensors or include_records) else None
            mask = output.patch_mask.detach().cpu() if (include_token_tensors or include_records) else None

            if include_token_tensors:
                assert tokens is not None
                assert mask is not None
                flat_tokens = tokens.reshape(tokens.shape[0], -1, tokens.shape[-1])
                flat_mask = mask.reshape(mask.shape[0], -1)
                token_chunks.append(flat_tokens)
                mask_chunks.append(flat_mask)
            label_chunks.append(batch_labels)
            sample_session_ids.extend(str(session_id) for session_id in batch.provenance.session_ids)

            if include_records:
                assert tokens is not None
                for batch_index in range(tokens.shape[0]):
                    sample_label = int(batch_labels[batch_index].item())
                    if positive_records_only and sample_label <= 0:
                        sample_record_groups.append(tuple())
                        continue
                    sample_records: list[FrozenTokenRecord] = []
                    for unit_index, unit_id in enumerate(batch.provenance.unit_ids[batch_index]):
                        for patch_index in range(tokens.shape[2]):
                            if not bool(batch.patch_mask[batch_index, unit_index, patch_index].item()):
                                continue
                            sample_records.append(
                                FrozenTokenRecord(
                                    recording_id=batch.provenance.recording_ids[batch_index],
                                    session_id=batch.provenance.session_ids[batch_index],
                                    subject_id=batch.provenance.subject_ids[batch_index],
                                    unit_id=unit_id,
                                    unit_region=batch.provenance.unit_regions[batch_index][unit_index],
                                    unit_depth_um=float(batch.provenance.unit_depth_um[batch_index, unit_index].item()),
                                    patch_index=int(patch_index),
                                    patch_start_s=float(batch.provenance.patch_start_s[batch_index, patch_index].item()),
                                    patch_end_s=float(batch.provenance.patch_end_s[batch_index, patch_index].item()),
                                    window_start_s=float(batch.provenance.window_start_s[batch_index].item()),
                                    window_end_s=float(batch.provenance.window_end_s[batch_index].item()),
                                    label=sample_label,
                                    score=0.0,
                                    embedding=tuple(float(value) for value in tokens[batch_index, unit_index, patch_index].tolist()),
                                )
                            )
                    sample_record_groups.append(tuple(sample_records))
            processed_batches += 1
            _maybe_update_progress(progress_callback, processed_batches, total_batches)
    if hasattr(bundle.dataset, "_close_open_files"):
        bundle.dataset._close_open_files()
    if not label_chunks:
        raise ValueError(
            f"No windows were sampled from split '{split_name}'. Check that the split contains prepared sessions "
            "and that the configured context window fits within the sampled recording intervals."
        )
    labels = torch.cat(label_chunks, dim=0)
    if include_token_tensors:
        max_seq_len = max(chunk.shape[1] for chunk in token_chunks)
        total_windows = sum(chunk.shape[0] for chunk in token_chunks)
        token_dim = token_chunks[0].shape[2]
        tokens = torch.zeros((total_windows, max_seq_len, token_dim), dtype=torch.float32)
        token_mask = torch.zeros((total_windows, max_seq_len), dtype=torch.bool)
        offset = 0
        for i in range(len(token_chunks)):
            tok_chunk = token_chunks[i]
            mask_chunk = mask_chunks[i]
            n, seq = tok_chunk.shape[0], tok_chunk.shape[1]
            tokens[offset:offset + n, :seq] = tok_chunk
            token_mask[offset:offset + n, :seq] = mask_chunk
            offset += n
            token_chunks[i] = None  # type: ignore[assignment]
            mask_chunks[i] = None  # type: ignore[assignment]
            del tok_chunk, mask_chunk
        del token_chunks, mask_chunks
    else:
        tokens = torch.empty((0, 0, 0), dtype=torch.float32)
        token_mask = torch.empty((0, 0), dtype=torch.bool)
    session_ids = tuple(sample_session_ids)
    if len(session_ids) != int(labels.shape[0]):
        raise ValueError("Internal error: sampled session provenance does not align with collected discovery labels.")
    if sampling_strategy == "label_balanced":
        positive_window_count = int((labels > 0.0).sum().item())
        if positive_window_count < int(experiment_config.discovery.min_positive_windows):
            raise ValueError(
                "Discovery label-balanced extraction did not find enough positive windows for the requested target "
                f"label '{experiment_config.discovery.target_label}'. Required min_positive_windows="
                f"{experiment_config.discovery.min_positive_windows}, found={positive_window_count}, "
                f"scanned_windows={int(labels.numel())}."
            )
        selected_indices, _, _ = _select_label_balanced_indices(
            labels=labels,
            session_ids=session_ids,
            seed=experiment_config.seed,
            max_selected_windows=int(experiment_config.discovery.max_batches) * int(experiment_config.optimization.batch_size),
            negative_to_positive_ratio=experiment_config.discovery.negative_to_positive_ratio,
        )
        coverage_summary = _build_coverage_summary(
            split_name=split_name,
            target_label=experiment_config.discovery.target_label,
            labels=labels,
            session_ids=session_ids,
            selected_indices=selected_indices,
            sampling_strategy=sampling_strategy,
            scan_max_batches=sampler_max_batches,
        )
        if include_token_tensors:
            tokens = tokens[selected_indices]
            token_mask = token_mask[selected_indices]
        labels = labels[selected_indices]
        selected_records: list[FrozenTokenRecord] = []
        if include_records:
            for index in selected_indices.tolist():
                selected_records.extend(sample_record_groups[index])
        records = tuple(selected_records)
    else:
        selected_indices = torch.arange(labels.numel(), dtype=torch.long)
        coverage_summary = _build_coverage_summary(
            split_name=split_name,
            target_label=experiment_config.discovery.target_label,
            labels=labels,
            session_ids=session_ids,
            selected_indices=selected_indices,
            sampling_strategy=sampling_strategy,
            scan_max_batches=sampler_max_batches,
        )
        records = tuple(record for group in sample_record_groups for record in group) if include_records else tuple()
    return FrozenTokenCollection(
        tokens=tokens,
        token_mask=token_mask,
        labels=labels,
        records=records,
        coverage_summary=coverage_summary,
        window_session_ids=tuple(session_ids[index] for index in selected_indices.tolist()) if sampling_strategy == "label_balanced" else session_ids,
    )
