from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import torch

from predictive_circuit_coding.data import resolve_runtime_dataset_view
from predictive_circuit_coding.decoding.labels import extract_binary_labels
from predictive_circuit_coding.training.artifacts import load_training_checkpoint
from predictive_circuit_coding.training.config import ExperimentConfig
from predictive_circuit_coding.training.contracts import DiscoveryCoverageSummary, FrozenTokenRecord
from predictive_circuit_coding.training.factories import build_model_from_config, build_tokenizer_from_config
from predictive_circuit_coding.training.runtime import iter_sampler_batches, resolve_device
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positive_indices = torch.nonzero(labels > 0.0, as_tuple=False).flatten()
    negative_indices = torch.nonzero(labels <= 0.0, as_tuple=False).flatten()
    if len(positive_indices) == 0 or len(negative_indices) == 0:
        return torch.empty(0, dtype=torch.long), positive_indices, negative_indices
    target_count = min(len(positive_indices), len(negative_indices))
    selected_positive = _session_stratified_subsample(
        indices=positive_indices,
        session_ids=session_ids,
        target_count=target_count,
        seed=seed,
    )
    selected_negative = _session_stratified_subsample(
        indices=negative_indices,
        session_ids=session_ids,
        target_count=target_count,
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
) -> FrozenTokenCollection:
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
    sampler_max_batches = (
        None
        if sampling_strategy == "label_balanced"
        else (max_batches or experiment_config.discovery.max_batches)
    )

    with torch.no_grad():
        for batch in iter_sampler_batches(
            dataset=bundle.dataset,
            sampler=sampler,
            collator=tokenizer,
            batch_size=experiment_config.optimization.batch_size,
            max_batches=sampler_max_batches,
        ):
            labels = extract_binary_labels(batch, target_label=experiment_config.discovery.target_label)
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
    if hasattr(bundle.dataset, "_close_open_files"):
        bundle.dataset._close_open_files()
    if not label_chunks:
        raise ValueError(
            f"No windows were sampled from split '{split_name}'. Check that the split contains prepared sessions "
            "and that the configured context window fits within the sampled recording intervals."
        )
    labels = torch.cat(label_chunks, dim=0)
    if include_token_tensors:
        tokens = torch.cat(token_chunks, dim=0)
        token_mask = torch.cat(mask_chunks, dim=0)
    else:
        tokens = torch.empty((0, 0, 0), dtype=torch.float32)
        token_mask = torch.empty((0, 0), dtype=torch.bool)
    session_ids = tuple(sample_session_ids)
    if len(session_ids) != int(labels.shape[0]):
        raise ValueError("Internal error: sampled session provenance does not align with collected discovery labels.")
    if sampling_strategy == "label_balanced":
        selected_indices, _, _ = _select_label_balanced_indices(
            labels=labels,
            session_ids=session_ids,
            seed=experiment_config.seed,
        )
        coverage_summary = _build_coverage_summary(
            split_name=split_name,
            target_label=experiment_config.discovery.target_label,
            labels=labels,
            session_ids=session_ids,
            selected_indices=selected_indices,
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
        coverage_summary = _build_coverage_summary(
            split_name=split_name,
            target_label=experiment_config.discovery.target_label,
            labels=labels,
            session_ids=session_ids,
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
