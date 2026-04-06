from __future__ import annotations

from typing import Any

import torch

from predictive_circuit_coding.training.contracts import PopulationWindowBatch


LABEL_ALIASES: dict[str, str] = {
    "stimulus_change": "stimulus_presentations.is_change",
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


def _resolve_target_label_path(target_label: str) -> tuple[str, ...]:
    normalized = LABEL_ALIASES.get(target_label, target_label)
    parts = tuple(part for part in normalized.split(".") if part)
    if not parts:
        raise ValueError("target_label must not be empty")
    return parts


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
    path = _resolve_target_label_path(target_label)
    labels = []
    for annotation in batch.provenance.event_annotations:
        values = _lookup_nested_value(annotation, path)
        if isinstance(values, (tuple, list)):
            labels.append(1.0 if any(_coerce_bool(value) for value in values) else 0.0)
        else:
            labels.append(1.0 if _coerce_bool(values) else 0.0)
    return torch.tensor(labels, dtype=torch.float32)


def extract_stimulus_change_labels(batch: PopulationWindowBatch) -> torch.Tensor:
    return extract_binary_labels(batch, target_label="stimulus_change")
