from predictive_circuit_coding.training.artifacts import (
    load_training_checkpoint,
    save_training_checkpoint,
    write_checkpoint_metadata,
    write_evaluation_summary,
    write_run_manifest,
    write_training_summary,
    write_validation_summary,
    write_validation_summary_csv,
)
from predictive_circuit_coding.training.config import (
    ArtifactConfig,
    DataRuntimeConfig,
    DiscoveryConfig,
    EvaluationConfig,
    ExecutionConfig,
    ExperimentConfig,
    ModelConfig,
    ObjectiveConfig,
    OptimizationConfig,
    SplitConfig,
    TrainingRuntimeConfig,
    load_experiment_config,
    validate_experiment_config,
)
from predictive_circuit_coding.training.contracts import (
    CandidateTokenRecord,
    CheckpointMetadata,
    DecoderSummary,
    DiscoveryArtifact,
    EvaluationSummary,
    FrozenTokenRecord,
    ModelForwardOutput,
    PopulationWindowBatch,
    TokenProvenanceBatch,
    TrainingCheckpoint,
    TrainingStepOutput,
    TrainingSummary,
    ValidationSummary,
)


def build_tokenizer_from_config(config):
    from predictive_circuit_coding.training.factories import build_tokenizer_from_config as _build

    return _build(config)


def build_model_from_config(config):
    from predictive_circuit_coding.training.factories import build_model_from_config as _build

    return _build(config)


def build_objective_from_config(config):
    from predictive_circuit_coding.training.factories import build_objective_from_config as _build

    return _build(config)


def run_training_step(model, objective, batch):
    from predictive_circuit_coding.training.step import run_training_step as _run

    return _run(model, objective, batch)


def train_model(*, experiment_config, data_config_path, train_split, valid_split):
    from predictive_circuit_coding.training.loop import train_model as _train

    return _train(
        experiment_config=experiment_config,
        data_config_path=data_config_path,
        train_split=train_split,
        valid_split=valid_split,
    )

__all__ = [
    "ArtifactConfig",
    "CandidateTokenRecord",
    "CheckpointMetadata",
    "DataRuntimeConfig",
    "DecoderSummary",
    "DiscoveryArtifact",
    "DiscoveryConfig",
    "EvaluationConfig",
    "EvaluationSummary",
    "ExecutionConfig",
    "ExperimentConfig",
    "FrozenTokenRecord",
    "ModelConfig",
    "ModelForwardOutput",
    "ObjectiveConfig",
    "OptimizationConfig",
    "PopulationWindowBatch",
    "SplitConfig",
    "TokenProvenanceBatch",
    "TrainingCheckpoint",
    "TrainingRuntimeConfig",
    "TrainingStepOutput",
    "TrainingSummary",
    "ValidationSummary",
    "build_model_from_config",
    "build_objective_from_config",
    "build_tokenizer_from_config",
    "load_training_checkpoint",
    "load_experiment_config",
    "run_training_step",
    "save_training_checkpoint",
    "train_model",
    "validate_experiment_config",
    "write_checkpoint_metadata",
    "write_evaluation_summary",
    "write_run_manifest",
    "write_training_summary",
    "write_validation_summary",
    "write_validation_summary_csv",
]
