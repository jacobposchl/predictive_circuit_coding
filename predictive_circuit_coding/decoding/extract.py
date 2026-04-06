from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import torch

from predictive_circuit_coding.data import resolve_runtime_dataset_view
from predictive_circuit_coding.decoding.labels import extract_binary_labels
from predictive_circuit_coding.training.artifacts import load_training_checkpoint
from predictive_circuit_coding.training.config import ExperimentConfig
from predictive_circuit_coding.training.contracts import FrozenTokenRecord
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


def _select_label_balanced_indices(
    *,
    labels: torch.Tensor,
    negative_to_positive_ratio: float,
) -> torch.Tensor:
    positive_indices = torch.nonzero(labels > 0.0, as_tuple=False).flatten()
    if len(positive_indices) == 0:
        return positive_indices
    negative_indices = torch.nonzero(labels <= 0.0, as_tuple=False).flatten()
    negative_budget = int(math.ceil(len(positive_indices) * max(negative_to_positive_ratio, 0.0)))
    if negative_budget > 0:
        negative_indices = negative_indices[:negative_budget]
        return torch.cat((positive_indices, negative_indices), dim=0)
    return positive_indices


def extract_frozen_tokens(
    *,
    experiment_config: ExperimentConfig,
    data_config_path: str | Path,
    checkpoint_path: str | Path,
    split_name: str,
    max_batches: int | None = None,
    dataset_view=None,
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
    positive_window_count = 0
    search_max_batches = (
        experiment_config.discovery.search_max_batches
        if experiment_config.discovery.sampling_strategy == "label_balanced"
        and experiment_config.discovery.search_max_batches is not None
        else (max_batches or experiment_config.discovery.max_batches)
    )

    with torch.no_grad():
        for batch in iter_sampler_batches(
            dataset=bundle.dataset,
            sampler=sampler,
            collator=tokenizer,
            batch_size=experiment_config.optimization.batch_size,
            max_batches=search_max_batches,
        ):
            labels = extract_binary_labels(batch, target_label=experiment_config.discovery.target_label)
            device_batch = batch.to(device)
            output = model(device_batch)
            tokens = output.tokens.detach().cpu()
            mask = output.patch_mask.detach().cpu()
            batch_labels = labels.detach().cpu()
            positive_window_count += int((batch_labels > 0.0).sum().item())

            flat_tokens = tokens.reshape(tokens.shape[0], -1, tokens.shape[-1])
            flat_mask = mask.reshape(mask.shape[0], -1)
            token_chunks.append(flat_tokens)
            mask_chunks.append(flat_mask)
            label_chunks.append(batch_labels)

            for batch_index in range(tokens.shape[0]):
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
                                label=int(batch_labels[batch_index].item()),
                                score=0.0,
                                embedding=tuple(float(value) for value in tokens[batch_index, unit_index, patch_index].tolist()),
                            )
                        )
                sample_record_groups.append(tuple(sample_records))
            if (
                experiment_config.discovery.sampling_strategy == "label_balanced"
                and positive_window_count >= experiment_config.discovery.min_positive_windows
            ):
                break
    if hasattr(bundle.dataset, "_close_open_files"):
        bundle.dataset._close_open_files()
    if not token_chunks or not sample_record_groups:
        raise ValueError(
            f"No windows were sampled from split '{split_name}'. Check that the split contains prepared sessions "
            "and that the configured context window fits within the sampled recording intervals."
        )
    labels = torch.cat(label_chunks, dim=0)
    tokens = torch.cat(token_chunks, dim=0)
    token_mask = torch.cat(mask_chunks, dim=0)
    if experiment_config.discovery.sampling_strategy == "label_balanced":
        selected_indices = _select_label_balanced_indices(
            labels=labels,
            negative_to_positive_ratio=experiment_config.discovery.negative_to_positive_ratio,
        )
        if len(selected_indices) == 0:
            raise ValueError(
                f"Cannot collect discovery windows because no positive '{experiment_config.discovery.target_label}' "
                "labels were found in the sampled windows."
            )
        tokens = tokens[selected_indices]
        token_mask = token_mask[selected_indices]
        labels = labels[selected_indices]
        selected_records: list[FrozenTokenRecord] = []
        for index in selected_indices.tolist():
            selected_records.extend(sample_record_groups[index])
        records = tuple(selected_records)
    else:
        records = tuple(record for group in sample_record_groups for record in group)
    return FrozenTokenCollection(
        tokens=tokens,
        token_mask=token_mask,
        labels=labels,
        records=records,
    )
