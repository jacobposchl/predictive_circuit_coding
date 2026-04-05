from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from predictive_circuit_coding.data import build_workspace, load_preparation_config, load_split_manifest
from predictive_circuit_coding.decoding.labels import extract_stimulus_change_labels
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


def extract_frozen_tokens(
    *,
    experiment_config: ExperimentConfig,
    data_config_path: str | Path,
    checkpoint_path: str | Path,
    split_name: str,
    max_batches: int | None = None,
) -> FrozenTokenCollection:
    prep_config = load_preparation_config(data_config_path)
    workspace = build_workspace(prep_config)
    split_manifest = load_split_manifest(workspace.split_manifest_path)
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
    records: list[FrozenTokenRecord] = []

    with torch.no_grad():
        for batch in iter_sampler_batches(
            dataset=bundle.dataset,
            sampler=sampler,
            collator=tokenizer,
            batch_size=experiment_config.optimization.batch_size,
            max_batches=max_batches or experiment_config.discovery.max_batches,
        ):
            labels = extract_stimulus_change_labels(batch)
            device_batch = batch.to(device)
            output = model(device_batch)
            tokens = output.tokens.detach().cpu()
            mask = output.patch_mask.detach().cpu()
            batch_labels = labels.detach().cpu()

            flat_tokens = tokens.reshape(tokens.shape[0], -1, tokens.shape[-1])
            flat_mask = mask.reshape(mask.shape[0], -1)
            token_chunks.append(flat_tokens)
            mask_chunks.append(flat_mask)
            label_chunks.append(batch_labels)

            for batch_index in range(tokens.shape[0]):
                for unit_index, unit_id in enumerate(batch.provenance.unit_ids[batch_index]):
                    for patch_index in range(tokens.shape[2]):
                        if not bool(batch.patch_mask[batch_index, unit_index, patch_index].item()):
                            continue
                        records.append(
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
    if hasattr(bundle.dataset, "_close_open_files"):
        bundle.dataset._close_open_files()
    if not token_chunks or not records:
        raise ValueError(
            f"No windows were sampled from split '{split_name}'. Check that the split contains prepared sessions "
            "and that the configured context window fits within the sampled recording intervals."
        )
    return FrozenTokenCollection(
        tokens=torch.cat(token_chunks, dim=0),
        token_mask=torch.cat(mask_chunks, dim=0),
        labels=torch.cat(label_chunks, dim=0),
        records=tuple(records),
    )
