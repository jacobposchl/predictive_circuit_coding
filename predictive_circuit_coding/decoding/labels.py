from __future__ import annotations

from typing import Any

import torch

from predictive_circuit_coding.training.contracts import PopulationWindowBatch


LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    # Allen sampled windows can surface change labels either on stimulus presentations
    # or on trials, depending on which interval payload survives into the sampled view.
    "stimulus_change": ("stimulus_presentations.is_change", "trials.is_change"),
}


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _resolve_target_label_paths(target_label: str) -> tuple[tuple[str, ...], ...]:
    candidates = LABEL_ALIASES.get(target_label, (target_label,))
    paths = tuple(tuple(part for part in candidate.split(".") if part) for candidate in candidates)
    if not paths or any(not path for path in paths):
        raise ValueError("target_label must not be empty")
    return paths


def _lookup_nested_value(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for part in path:
        if not isinstance(current, dict):
            return ()
        current = current.get(part)
        if current is None:
            return ()
    return current


def extract_binary_labels(batch: PopulationWindowBatch, *, target_label: str) -> torch.Tensor:
    paths = _resolve_target_label_paths(target_label)
    labels = []
    for annotation in batch.provenance.event_annotations:
        label = 0.0
        for path in paths:
            values = _lookup_nested_value(annotation, path)
            if isinstance(values, (tuple, list)):
                if any(_coerce_bool(value) for value in values):
                    label = 1.0
                    break
            elif _coerce_bool(values):
                label = 1.0
                break
        labels.append(label)
    return torch.tensor(labels, dtype=torch.float32)


def extract_stimulus_change_labels(batch: PopulationWindowBatch) -> torch.Tensor:
    return extract_binary_labels(batch, target_label="stimulus_change")
