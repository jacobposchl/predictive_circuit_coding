from __future__ import annotations

from dataclasses import dataclass

import torch

from predictive_circuit_coding.objectives.targets import CountTargetBuilder, PredictiveTargets
from predictive_circuit_coding.training.config import ObjectiveConfig
from predictive_circuit_coding.training.contracts import ModelForwardOutput, PopulationWindowBatch


@dataclass(frozen=True)
class ObjectiveOutput:
    loss: torch.Tensor
    losses: dict[str, float]
    metrics: dict[str, float]


def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = mask.unsqueeze(-1).to(dtype=prediction.dtype)
    squared_error = ((prediction - target) ** 2) * expanded_mask
    denominator = expanded_mask.sum().clamp_min(1.0) * prediction.shape[-1]
    return squared_error.sum() / denominator


class PredictiveObjective:
    def __init__(self, config: ObjectiveConfig):
        self.config = config
        self.target_builder = CountTargetBuilder(config)

    def evaluate(self, batch: PopulationWindowBatch, forward_output: ModelForwardOutput) -> tuple[torch.Tensor, PredictiveTargets, dict[str, float]]:
        targets = self.target_builder(batch)
        predictive_loss = _masked_mse(
            forward_output.predictive_outputs,
            targets.target,
            targets.valid_mask,
        )
        if self.config.predictive_target_type == "delta":
            raw_prediction = targets.baseline + forward_output.predictive_outputs
        else:
            raw_prediction = forward_output.predictive_outputs
        baseline_mse = _masked_mse(
            targets.baseline,
            targets.future_counts,
            targets.valid_mask,
        )
        model_raw_mse = _masked_mse(
            raw_prediction,
            targets.future_counts,
            targets.valid_mask,
        )
        metrics = {
            "predictive_loss": float(predictive_loss.detach().item()),
            "predictive_baseline_mse": float(baseline_mse.detach().item()),
            "predictive_raw_mse": float(model_raw_mse.detach().item()),
            "predictive_improvement": float((baseline_mse - model_raw_mse).detach().item()),
        }
        return predictive_loss, targets, metrics


class ReconstructionObjective:
    def __init__(self, config: ObjectiveConfig):
        self.config = config

    def _normalize_window_zscore(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expanded_mask = mask.unsqueeze(-1).to(dtype=target.dtype)
        valid_counts = expanded_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        mean = (target * expanded_mask).sum(dim=2, keepdim=True) / valid_counts
        centered = (target - mean) * expanded_mask
        variance = (centered * centered).sum(dim=2, keepdim=True) / valid_counts
        std = variance.sqrt().clamp_min(1.0e-6)
        return (prediction - mean) / std, (target - mean) / std

    def evaluate(self, batch: PopulationWindowBatch, forward_output: ModelForwardOutput) -> tuple[torch.Tensor, dict[str, float]]:
        prediction = forward_output.reconstruction_outputs
        target = batch.patch_counts
        if self.config.reconstruction_target_mode == "window_zscore":
            prediction, target = self._normalize_window_zscore(prediction, target, batch.patch_mask)
        reconstruction_loss = _masked_mse(
            prediction,
            target,
            batch.patch_mask,
        )
        return reconstruction_loss, {"reconstruction_loss": float(reconstruction_loss.detach().item())}


class CombinedObjective:
    def __init__(self, config: ObjectiveConfig):
        self.config = config
        self.predictive = PredictiveObjective(config)
        self.reconstruction = ReconstructionObjective(config)

    def __call__(self, batch: PopulationWindowBatch, forward_output: ModelForwardOutput) -> ObjectiveOutput:
        predictive_loss, _, predictive_metrics = self.predictive.evaluate(batch, forward_output)
        reconstruction_loss, reconstruction_metrics = self.reconstruction.evaluate(batch, forward_output)
        total_loss = predictive_loss + (self.config.reconstruction_weight * reconstruction_loss)
        losses = {
            "total_loss": float(total_loss.detach().item()),
            "predictive_loss": float(predictive_loss.detach().item()),
            "reconstruction_loss": float(reconstruction_loss.detach().item()),
        }
        metrics = dict(predictive_metrics)
        metrics.update(reconstruction_metrics)
        return ObjectiveOutput(loss=total_loss, losses=losses, metrics=metrics)
