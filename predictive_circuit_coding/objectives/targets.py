from __future__ import annotations

from dataclasses import dataclass

import torch

from predictive_circuit_coding.training.config import ObjectiveConfig
from predictive_circuit_coding.training.contracts import PopulationWindowBatch


@dataclass(frozen=True)
class PredictiveTargets:
    target: torch.Tensor
    baseline: torch.Tensor
    future_counts: torch.Tensor
    valid_mask: torch.Tensor


class ContinuationBaselineBuilder:
    def __init__(self, *, baseline_type: str):
        self.baseline_type = baseline_type

    def __call__(self, patch_counts: torch.Tensor) -> torch.Tensor:
        if self.baseline_type == "previous_patch":
            return patch_counts.clone()
        if self.baseline_type == "zeros":
            return torch.zeros_like(patch_counts)
        raise ValueError(f"Unsupported continuation baseline type: {self.baseline_type}")


class CountTargetBuilder:
    def __init__(self, config: ObjectiveConfig):
        self.config = config
        self.baseline_builder = ContinuationBaselineBuilder(
            baseline_type=config.continuation_baseline_type,
        )

    def __call__(self, batch: PopulationWindowBatch) -> PredictiveTargets:
        future_counts = torch.zeros_like(batch.patch_counts)
        future_counts[:, :, :-1] = batch.patch_counts[:, :, 1:]
        valid_mask = batch.patch_mask.clone()
        if self.config.exclude_final_prediction_patch:
            valid_mask = valid_mask.clone()
            valid_mask[:, :, -1] = False
        baseline = self.baseline_builder(batch.patch_counts)
        if self.config.predictive_target_type == "delta":
            target = future_counts - baseline
        else:
            target = future_counts
        return PredictiveTargets(
            target=target,
            baseline=baseline,
            future_counts=future_counts,
            valid_mask=valid_mask,
        )
