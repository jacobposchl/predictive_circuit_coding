from __future__ import annotations

from typing import Any

import torch

from predictive_circuit_coding.training.contracts import PopulationWindowBatch


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def extract_stimulus_change_labels(batch: PopulationWindowBatch) -> torch.Tensor:
    labels = []
    for annotation in batch.provenance.event_annotations:
        presentations = annotation.get("stimulus_presentations", {})
        values = presentations.get("is_change", ())
        labels.append(1.0 if any(_coerce_bool(value) for value in values) else 0.0)
    return torch.tensor(labels, dtype=torch.float32)
