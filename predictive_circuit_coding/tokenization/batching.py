from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from predictive_circuit_coding.training.config import DataRuntimeConfig
from predictive_circuit_coding.training.contracts import (
    PopulationWindowBatch,
    TokenProvenanceBatch,
)


def _normalize_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _normalize_annotation_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default) if obj is not None else default


def extract_sample_event_annotations(sample, config: DataRuntimeConfig, *, window_start_s: float, window_end_s: float) -> dict[str, Any]:
    annotations: dict[str, Any] = {}
    sample_window_start_s = float(np.asarray(sample.domain.start, dtype=np.float64)[0])
    sample_window_end_s = float(np.asarray(sample.domain.end, dtype=np.float64)[0])
    field_specs = (
        ("trials", config.include_trials),
        ("stimulus_presentations", config.include_stimulus_presentations),
        ("optotagging", config.include_optotagging),
    )
    for field_name, enabled in field_specs:
        if not enabled:
            continue
        interval = getattr(sample, field_name, None)
        starts = _safe_attr(interval, "start")
        ends = _safe_attr(interval, "end")
        if starts is None or ends is None:
            continue
        starts_np = np.asarray(starts, dtype=np.float64)
        ends_np = np.asarray(ends, dtype=np.float64)
        mask = (ends_np > sample_window_start_s) & (starts_np < sample_window_end_s)
        if not np.any(mask):
            continue
        payload: dict[str, Any] = {
            "start_s": tuple((starts_np[mask] - sample_window_start_s).tolist()),
            "end_s": tuple((ends_np[mask] - sample_window_start_s).tolist()),
        }
        interval_keys = list(interval.keys()) if hasattr(interval, "keys") else []
        for extra in interval_keys:
            if extra in {"start", "end"}:
                continue
            values = _safe_attr(interval, extra)
            if values is None:
                continue
            selected = np.asarray(values, dtype=object)[mask].tolist()
            if extra == "timestamps":
                payload["timestamps_s"] = tuple(float(value) - sample_window_start_s for value in selected)
                continue
            payload[extra] = tuple(_normalize_annotation_scalar(value) for value in selected)
        annotations[field_name] = payload
    return annotations


def extract_sample_recording_metadata(sample) -> tuple[str, str, str]:
    brainset_id = _normalize_string(_safe_attr(_safe_attr(sample, "brainset"), "id"))
    session_id = _normalize_string(_safe_attr(_safe_attr(sample, "session"), "id"))
    subject_id = _normalize_string(_safe_attr(_safe_attr(sample, "subject"), "id"))
    if brainset_id and session_id.startswith(f"{brainset_id}/"):
        recording_id = session_id
    elif brainset_id:
        recording_id = f"{brainset_id}/{session_id}"
    else:
        recording_id = session_id
    return recording_id, session_id, subject_id


def _extract_unit_metadata(sample) -> tuple[list[str], list[str], np.ndarray]:
    units = _safe_attr(sample, "units")
    unit_ids = [_normalize_string(value) for value in np.asarray(_safe_attr(units, "id", []), dtype=object).tolist()]
    regions_source = (
        _safe_attr(units, "brain_region"),
        _safe_attr(units, "structure_acronym"),
        _safe_attr(units, "ecephys_structure_acronym"),
        _safe_attr(units, "location"),
    )
    regions: list[str] = []
    for values in regions_source:
        if values is None:
            continue
        regions = [_normalize_string(value) for value in np.asarray(values, dtype=object).tolist()]
        break
    if not regions:
        regions = [""] * len(unit_ids)

    depths_source = (
        _safe_attr(units, "probe_depth_um"),
        _safe_attr(units, "probe_vertical_position"),
    )
    depths = np.asarray([], dtype=np.float32)
    for values in depths_source:
        if values is None:
            continue
        depths = np.asarray(values, dtype=np.float32)
        break
    if depths.size == 0:
        depths = np.zeros((len(unit_ids),), dtype=np.float32)
    return unit_ids, regions, depths


def _bin_spike_counts(sample, config: DataRuntimeConfig) -> torch.Tensor:
    window_start_s = float(np.asarray(sample.domain.start, dtype=np.float64)[0])
    spikes = sample.spikes
    timestamps = np.asarray(spikes.timestamps, dtype=np.float64) - window_start_s
    unit_index = np.asarray(spikes.unit_index, dtype=np.int64)
    counts = torch.zeros((len(sample.units.id), config.context_bins), dtype=torch.float32)
    bin_indices = np.floor(timestamps / config.bin_width_s).astype(np.int64)
    valid = (bin_indices >= 0) & (bin_indices < config.context_bins) & (unit_index >= 0) & (unit_index < counts.shape[0])
    if np.any(valid):
        bin_tensor = torch.from_numpy(bin_indices[valid])
        unit_tensor = torch.from_numpy(unit_index[valid])
        ones = torch.ones((int(valid.sum()),), dtype=torch.float32)
        counts.index_put_((unit_tensor, bin_tensor), ones, accumulate=True)
    return counts


def _select_units(
    counts: torch.Tensor,
    unit_ids: list[str],
    unit_regions: list[str],
    unit_depth_um: np.ndarray,
    config: DataRuntimeConfig,
) -> tuple[torch.Tensor, list[str], list[str], np.ndarray]:
    keep = torch.ones((counts.shape[0],), dtype=torch.bool)
    if config.min_unit_spikes > 0:
        keep &= counts.sum(dim=-1) >= float(config.min_unit_spikes)
    keep_indices = torch.nonzero(keep, as_tuple=False).flatten()
    if len(keep_indices) == 0 and counts.shape[0] > 0:
        keep_indices = torch.tensor([int(torch.argmax(counts.sum(dim=-1)).item())], dtype=torch.long)
    if config.max_units is not None and len(keep_indices) > config.max_units:
        selected_counts = counts[keep_indices].sum(dim=-1)
        order = torch.argsort(selected_counts, descending=True)[: config.max_units]
        keep_indices = keep_indices[order]
    counts = counts[keep_indices]
    indices = keep_indices.tolist()
    return (
        counts,
        [unit_ids[index] for index in indices],
        [unit_regions[index] for index in indices],
        np.asarray([unit_depth_um[index] for index in indices], dtype=np.float32),
    )


@dataclass(frozen=True)
class WindowBinner:
    config: DataRuntimeConfig

    def __call__(self, sample) -> torch.Tensor:
        return _bin_spike_counts(sample, self.config)


@dataclass(frozen=True)
class TemporalPatchBuilder:
    config: DataRuntimeConfig

    def __call__(self, counts: torch.Tensor) -> torch.Tensor:
        if counts.shape[-1] != self.config.context_bins:
            raise ValueError(
                f"Expected counts with {self.config.context_bins} context bins, received {counts.shape[-1]}"
            )
        return counts.reshape(counts.shape[0], self.config.patches_per_window, self.config.patch_bins)


class PopulationWindowBatchCollator:
    def __init__(self, config: DataRuntimeConfig):
        self.config = config
        self.binner = WindowBinner(config)
        self.patch_builder = TemporalPatchBuilder(config)

    def __call__(self, samples: list[Any]) -> PopulationWindowBatch:
        if not samples:
            raise ValueError("PopulationWindowBatchCollator requires at least one sample")

        per_sample_counts: list[torch.Tensor] = []
        recording_ids: list[str] = []
        session_ids: list[str] = []
        subject_ids: list[str] = []
        unit_ids_payload: list[tuple[str, ...]] = []
        unit_regions_payload: list[tuple[str, ...]] = []
        unit_depths_payload: list[np.ndarray] = []
        window_start_values: list[float] = []
        window_end_values: list[float] = []
        event_annotations: list[dict[str, Any]] = []

        max_units = 0
        for sample in samples:
            counts = self.binner(sample)
            unit_ids, unit_regions, unit_depth_um = _extract_unit_metadata(sample)
            counts, unit_ids, unit_regions, unit_depth_um = _select_units(
                counts,
                unit_ids=unit_ids,
                unit_regions=unit_regions,
                unit_depth_um=unit_depth_um,
                config=self.config,
            )
            max_units = max(max_units, counts.shape[0])
            per_sample_counts.append(counts)

            recording_id, session_id, subject_id = extract_sample_recording_metadata(sample)
            recording_ids.append(recording_id)
            session_ids.append(session_id)
            subject_ids.append(subject_id)
            unit_ids_payload.append(tuple(unit_ids))
            unit_regions_payload.append(tuple(unit_regions))
            unit_depths_payload.append(unit_depth_um)

            window_start_s = float(np.asarray(sample.domain.start, dtype=np.float64)[0])
            window_end_s = float(np.asarray(sample.domain.end, dtype=np.float64)[-1])
            window_start_values.append(window_start_s)
            window_end_values.append(window_end_s)
            event_annotations.append(
                extract_sample_event_annotations(
                    sample,
                    self.config,
                    window_start_s=window_start_s,
                    window_end_s=window_end_s,
                )
            )
        if max_units == 0:
            raise ValueError("PopulationWindowBatchCollator could not retain any units from the provided samples")

        batch_size = len(samples)
        counts_batch = torch.zeros((batch_size, max_units, self.config.context_bins), dtype=torch.float32)
        patch_counts_batch = torch.zeros(
            (batch_size, max_units, self.config.patches_per_window, self.config.patch_bins),
            dtype=torch.float32,
        )
        unit_mask = torch.zeros((batch_size, max_units), dtype=torch.bool)
        unit_depth_batch = torch.zeros((batch_size, max_units), dtype=torch.float32)

        for batch_index, counts in enumerate(per_sample_counts):
            num_units = counts.shape[0]
            if num_units == 0:
                continue
            counts_batch[batch_index, :num_units] = counts
            patch_counts_batch[batch_index, :num_units] = self.patch_builder(counts)
            unit_mask[batch_index, :num_units] = True
            unit_depth_batch[batch_index, :num_units] = torch.from_numpy(unit_depths_payload[batch_index])

        patch_mask = unit_mask.unsqueeze(-1).expand(-1, -1, self.config.patches_per_window)

        patch_offsets = torch.arange(self.config.patches_per_window, dtype=torch.float32) * (
            self.config.patch_bins * self.config.bin_width_s
        )
        patch_start_s = torch.stack(
            [torch.full_like(patch_offsets, fill_value=float(window_start)) + patch_offsets for window_start in window_start_values],
            dim=0,
        )
        patch_end_s = patch_start_s + (self.config.patch_bins * self.config.bin_width_s)

        provenance = TokenProvenanceBatch(
            recording_ids=tuple(recording_ids),
            session_ids=tuple(session_ids),
            subject_ids=tuple(subject_ids),
            unit_ids=tuple(unit_ids_payload),
            unit_regions=tuple(unit_regions_payload),
            unit_depth_um=unit_depth_batch,
            patch_start_s=patch_start_s,
            patch_end_s=patch_end_s,
            window_start_s=torch.tensor(window_start_values, dtype=torch.float32),
            window_end_s=torch.tensor(window_end_values, dtype=torch.float32),
            event_annotations=tuple(event_annotations),
        )
        return PopulationWindowBatch(
            counts=counts_batch,
            patch_counts=patch_counts_batch,
            unit_mask=unit_mask,
            patch_mask=patch_mask,
            bin_width_s=self.config.bin_width_s,
            provenance=provenance,
        )


def build_population_window_collator(config: DataRuntimeConfig) -> PopulationWindowBatchCollator:
    return PopulationWindowBatchCollator(config)
