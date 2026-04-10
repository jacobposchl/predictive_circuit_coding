from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from predictive_circuit_coding.objectives.region_targets import RegionRateTargets
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

    def evaluate(self, batch: PopulationWindowBatch, forward_output: ModelForwardOutput) -> tuple[torch.Tensor, dict[str, float]]:
        reconstruction_loss = _masked_mse(
            forward_output.reconstruction_outputs,
            batch.patch_counts,
            batch.patch_mask,
        )
        return reconstruction_loss, {"reconstruction_loss": float(reconstruction_loss.detach().item())}


class CrossSessionRegionLoss:
    def evaluate(
        self,
        *,
        predicted_region_rates: torch.Tensor,
        region_targets: RegionRateTargets,
        region_loss_weight: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        target = region_targets.region_rate_future.to(device=predicted_region_rates.device, dtype=predicted_region_rates.dtype)
        valid_mask = (
            region_targets.valid_patch_mask.to(device=predicted_region_rates.device)
            & region_targets.is_augmented.to(device=predicted_region_rates.device).unsqueeze(-1).unsqueeze(-1)
        )
        if not bool(valid_mask.any().item()):
            zero = predicted_region_rates.sum() * 0.0
            metrics = dict(region_targets.diagnostics)
            metrics.update(
                {
                    "cross_session_region_loss": 0.0,
                    "cross_session_region_loss_weighted": 0.0,
                    "cross_session_region_loss_weight": float(region_loss_weight),
                }
            )
            return zero, metrics

        expanded_mask = valid_mask.to(dtype=predicted_region_rates.dtype)
        squared_error = ((predicted_region_rates - target) ** 2) * expanded_mask
        denominator = expanded_mask.sum().clamp_min(1.0)
        raw_loss = squared_error.sum() / denominator
        weighted_loss = raw_loss * float(region_loss_weight)
        metrics = dict(region_targets.diagnostics)
        metrics.update(
            {
                "cross_session_region_loss": float(raw_loss.detach().item()),
                "cross_session_region_loss_weighted": float(weighted_loss.detach().item()),
                "cross_session_region_loss_weight": float(region_loss_weight),
            }
        )
        return weighted_loss, metrics


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
