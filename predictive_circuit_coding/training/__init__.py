from __future__ import annotations

from importlib import import_module

_CONFIG_EXPORTS = {
    "ArtifactConfig",
    "CountNormalizationConfig",
    "DataRuntimeConfig",
    "DatasetSelectionConfig",
    "DiscoveryConfig",
    "EvaluationConfig",
    "ExecutionConfig",
    "ExperimentIdentityConfig",
    "ExperimentConfig",
    "ModelConfig",
    "ObjectiveConfig",
    "OptimizationConfig",
    "RuntimeSubsetConfig",
    "SplitConfig",
    "TrainingRuntimeConfig",
    "load_experiment_config",
    "validate_experiment_config",
}

_ARTIFACT_EXPORTS = {
    "load_training_checkpoint",
    "save_training_checkpoint",
    "write_checkpoint_metadata",
    "write_evaluation_summary",
    "write_run_manifest",
    "write_training_summary",
    "write_validation_summary",
    "write_validation_summary_csv",
}

_CONTRACT_EXPORTS = {
    "CandidateTokenRecord",
    "CheckpointMetadata",
    "DecoderSummary",
    "DiscoveryArtifact",
    "EvaluationSummary",
    "FrozenTokenRecord",
    "ModelForwardOutput",
    "PopulationWindowBatch",
    "TokenProvenanceBatch",
    "TrainingCheckpoint",
    "TrainingStepOutput",
    "TrainingSummary",
    "ValidationSummary",
}

__all__ = sorted(
    _CONFIG_EXPORTS
    | _ARTIFACT_EXPORTS
    | _CONTRACT_EXPORTS
    | {
        "build_model_from_config",
        "build_objective_from_config",
        "build_tokenizer_from_config",
        "run_training_step",
        "train_model",
    }
)


def __getattr__(name: str):
    if name in _CONFIG_EXPORTS:
        module = import_module("predictive_circuit_coding.training.config")
        return getattr(module, name)
    if name in _ARTIFACT_EXPORTS:
        module = import_module("predictive_circuit_coding.training.artifacts")
        return getattr(module, name)
    if name in _CONTRACT_EXPORTS:
        module = import_module("predictive_circuit_coding.training.contracts")
        return getattr(module, name)
    if name == "build_tokenizer_from_config":
        module = import_module("predictive_circuit_coding.training.factories")
        return getattr(module, name)
    if name == "build_model_from_config":
        module = import_module("predictive_circuit_coding.training.factories")
        return getattr(module, name)
    if name == "build_objective_from_config":
        module = import_module("predictive_circuit_coding.training.factories")
        return getattr(module, name)
    if name == "run_training_step":
        module = import_module("predictive_circuit_coding.training.step")
        return getattr(module, name)
    if name == "train_model":
        module = import_module("predictive_circuit_coding.training.loop")
        return getattr(module, name)
    raise AttributeError(f"module 'predictive_circuit_coding.training' has no attribute {name!r}")
