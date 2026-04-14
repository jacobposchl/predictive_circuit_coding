from __future__ import annotations

from predictive_circuit_coding.models import PredictiveCircuitModel
from predictive_circuit_coding.objectives import CombinedObjective
from predictive_circuit_coding.tokenization import build_population_window_collator
from predictive_circuit_coding.training.config import ExperimentConfig


def build_tokenizer_from_config(config: ExperimentConfig):
    return build_population_window_collator(
        config.data_runtime,
        count_normalization=config.count_normalization,
    )


def build_model_from_config(config: ExperimentConfig) -> PredictiveCircuitModel:
    return PredictiveCircuitModel(
        model_config=config.model,
        patch_bins=config.data_runtime.patch_bins,
        num_patches=config.data_runtime.patches_per_window,
    )


def build_objective_from_config(config: ExperimentConfig) -> CombinedObjective:
    return CombinedObjective(config.objective)
