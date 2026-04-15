from __future__ import annotations

from collections.abc import Iterator

import torch

from predictive_circuit_coding.objectives import CombinedObjective
from predictive_circuit_coding.training.contracts import PopulationWindowBatch, TrainingStepOutput


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def iter_sampler_batches(
    *,
    dataset,
    sampler,
    collator,
    batch_size: int,
    max_batches: int | None,
) -> Iterator:
    pending = []
    yielded = 0
    for item in sampler:
        pending.append(dataset.get(item.recording_id, item.start, item.end))
        if len(pending) == batch_size:
            yield collator(pending)
            yielded += 1
            pending = []
            if max_batches is not None and yielded >= max_batches:
                return
    if pending and (max_batches is None or yielded < max_batches):
        yield collator(pending)


def run_training_step(model, objective: CombinedObjective, batch: PopulationWindowBatch) -> TrainingStepOutput:
    forward_output = model(batch)
    objective_output = objective(batch, forward_output)
    return TrainingStepOutput(
        loss=objective_output.loss,
        losses=objective_output.losses,
        metrics=objective_output.metrics,
        batch_size=batch.batch_size,
        token_count=batch.token_count,
    )
