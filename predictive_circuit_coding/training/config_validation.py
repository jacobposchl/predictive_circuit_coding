from __future__ import annotations

import torch

from predictive_circuit_coding.training.config_schema import (
    CountNormalizationConfig,
    DataRuntimeConfig,
    DatasetSelectionConfig,
    DiscoveryConfig,
    EvaluationConfig,
    ExecutionConfig,
    ExperimentConfig,
    ExperimentIdentityConfig,
    ModelConfig,
    ObjectiveConfig,
    OptimizationConfig,
    RuntimeSubsetConfig,
    TrainingRuntimeConfig,
)

def _validate_experiment_root(config: ExperimentConfig) -> None:
    if not config.dataset_id.strip():
        raise ValueError("dataset_id must not be blank")
    if not config.split_name.strip():
        raise ValueError("split_name must not be blank")


def _validate_data_runtime_config(config: DataRuntimeConfig) -> None:
    if config.bin_width_ms <= 0:
        raise ValueError("data_runtime.bin_width_ms must be > 0")
    if config.context_bins <= 0:
        raise ValueError("data_runtime.context_bins must be > 0")
    if config.patch_bins <= 0:
        raise ValueError("data_runtime.patch_bins must be > 0")
    if config.min_unit_spikes < 0:
        raise ValueError("data_runtime.min_unit_spikes must be >= 0")
    if config.max_units is not None and config.max_units < 1:
        raise ValueError("data_runtime.max_units must be >= 1 when provided")
    if config.context_bins % config.patch_bins != 0:
        raise ValueError("data_runtime.context_bins must be divisible by data_runtime.patch_bins")
    if config.padding_strategy != "mask":
        raise ValueError("Only mask padding_strategy is currently supported")


def _validate_model_config(config: ModelConfig) -> None:
    if config.d_model <= 0:
        raise ValueError("model.d_model must be > 0")
    if config.num_heads <= 0:
        raise ValueError("model.num_heads must be > 0")
    if config.d_model % config.num_heads != 0:
        raise ValueError("model.d_model must be divisible by model.num_heads")
    if config.temporal_layers < 1:
        raise ValueError("model.temporal_layers must be >= 1")
    if config.spatial_layers < 1:
        raise ValueError("model.spatial_layers must be >= 1")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")
    if config.mlp_ratio <= 0:
        raise ValueError("model.mlp_ratio must be > 0")
    if config.norm_eps <= 0:
        raise ValueError("model.norm_eps must be > 0")
    if config.population_token_mode not in {"none", "per_patch_cls"}:
        raise ValueError("model.population_token_mode must be 'none' or 'per_patch_cls'")


def _validate_experiment_identity_config(config: ExperimentIdentityConfig) -> None:
    if not config.variant_name.strip():
        raise ValueError("experiment.variant_name must not be blank")


def _validate_count_normalization_config(config: CountNormalizationConfig) -> None:
    if config.mode not in {"none", "log1p_train_zscore"}:
        raise ValueError("count_normalization.mode must be 'none' or 'log1p_train_zscore'")
    if config.mode == "log1p_train_zscore" and config.stats_path is None:
        raise ValueError("count_normalization.stats_path is required for log1p_train_zscore")


def _validate_objective_config(config: ObjectiveConfig) -> None:
    if config.predictive_target_type not in {"delta", "next_patch_counts"}:
        raise ValueError("objective.predictive_target_type must be 'delta' or 'next_patch_counts'")
    if config.continuation_baseline_type not in {"previous_patch", "zeros"}:
        raise ValueError("objective.continuation_baseline_type must be 'previous_patch' or 'zeros'")
    if config.predictive_loss != "mse":
        raise ValueError("Only mse predictive_loss is currently supported")
    if config.reconstruction_loss != "mse":
        raise ValueError("Only mse reconstruction_loss is currently supported")
    if config.reconstruction_target_mode not in {"raw", "window_zscore"}:
        raise ValueError("objective.reconstruction_target_mode must be 'raw' or 'window_zscore'")
    if config.reconstruction_weight < 0:
        raise ValueError("objective.reconstruction_weight must be >= 0")


def _validate_discovery_feature_config(discovery: DiscoveryConfig, model: ModelConfig) -> None:
    if discovery.pooled_feature_mode not in {"mean_tokens", "cls_tokens"}:
        raise ValueError("discovery.pooled_feature_mode must be 'mean_tokens' or 'cls_tokens'")
    if discovery.pooled_feature_mode == "cls_tokens" and model.population_token_mode != "per_patch_cls":
        raise ValueError("discovery.pooled_feature_mode='cls_tokens' requires model.population_token_mode='per_patch_cls'")


def _validate_optimization_config(config: OptimizationConfig) -> None:
    if config.learning_rate <= 0:
        raise ValueError("optimization.learning_rate must be > 0")
    if config.weight_decay < 0:
        raise ValueError("optimization.weight_decay must be >= 0")
    if config.grad_clip_norm is not None and config.grad_clip_norm <= 0:
        raise ValueError("optimization.grad_clip_norm must be > 0 when provided")
    if config.batch_size < 1:
        raise ValueError("optimization.batch_size must be >= 1")
    if config.scheduler_type not in {"none", "cosine"}:
        raise ValueError("optimization.scheduler_type must be 'none' or 'cosine'")
    if config.scheduler_warmup_steps < 0:
        raise ValueError("optimization.scheduler_warmup_steps must be >= 0")


def _validate_training_runtime_config(config: TrainingRuntimeConfig) -> None:
    if config.num_epochs < 1:
        raise ValueError("training.num_epochs must be >= 1")
    if config.train_steps_per_epoch < 1:
        raise ValueError("training.train_steps_per_epoch must be >= 1")
    if config.validation_steps < 1:
        raise ValueError("training.validation_steps must be >= 1")
    if config.checkpoint_every_epochs < 1:
        raise ValueError("training.checkpoint_every_epochs must be >= 1")
    if config.evaluate_every_epochs < 1:
        raise ValueError("training.evaluate_every_epochs must be >= 1")
    if config.log_every_steps < 1:
        raise ValueError("training.log_every_steps must be >= 1")
    if config.dataloader_workers != 0:
        raise ValueError(
            "training.dataloader_workers is not implemented by the current in-process sampler path; set it to 0."
        )


def _validate_dataset_selection_config(config: DatasetSelectionConfig) -> None:
    if config.split_primary_axis not in {None, "subject", "session"}:
        raise ValueError("dataset_selection.split_primary_axis must be 'subject', 'session', or null")
    for name, value in (
        ("dataset_selection.train_fraction", config.train_fraction),
        ("dataset_selection.valid_fraction", config.valid_fraction),
        ("dataset_selection.discovery_fraction", config.discovery_fraction),
        ("dataset_selection.test_fraction", config.test_fraction),
    ):
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if any(
        value is not None
        for value in (
            config.train_fraction,
            config.valid_fraction,
            config.discovery_fraction,
            config.test_fraction,
        )
    ):
        total = sum(
            value or 0.0
            for value in (
                config.train_fraction,
                config.valid_fraction,
                config.discovery_fraction,
                config.test_fraction,
            )
        )
        if abs(total - 1.0) > 1.0e-6:
            raise ValueError("dataset_selection split fractions must sum to 1.0 when explicitly provided")


def _validate_execution_config(config: ExecutionConfig) -> None:
    if config.device == "auto":
        return
    try:
        resolved_device = torch.device(config.device)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            "execution.device must be 'auto', 'cpu', 'cuda', or a valid torch device string such as 'cuda:0'"
        ) from exc
    if resolved_device.type not in {"cpu", "cuda"}:
        raise ValueError(
            "execution.device must resolve to a CPU or CUDA device; unsupported device "
            f"type '{resolved_device.type}'"
        )


def _validate_evaluation_config(config: EvaluationConfig) -> None:
    if config.max_batches < 1:
        raise ValueError("evaluation.max_batches must be >= 1")
    if config.sequential_step_s is not None and config.sequential_step_s <= 0:
        raise ValueError("evaluation.sequential_step_s must be > 0 when provided")


def _validate_runtime_subset_config(config: RuntimeSubsetConfig | None) -> None:
    if config is None:
        return
    if not config.split_manifest_path.is_file():
        raise FileNotFoundError(f"runtime_subset.split_manifest_path not found: {config.split_manifest_path}")
    if not config.session_catalog_path.is_file():
        raise FileNotFoundError(f"runtime_subset.session_catalog_path not found: {config.session_catalog_path}")


def _validate_discovery_config(config: DiscoveryConfig) -> None:
    if not config.target_label.strip():
        raise ValueError("discovery.target_label must not be empty")
    if config.target_label_match_value is not None and not config.target_label_match_value.strip():
        raise ValueError("discovery.target_label_match_value must not be blank when provided")
    if config.target_label_mode not in {"auto", "overlap", "onset_within_window", "centered_onset"}:
        raise ValueError(
            "discovery.target_label_mode must be one of 'auto', 'overlap', 'onset_within_window', or "
            "'centered_onset'"
        )
    if config.max_batches < 1:
        raise ValueError("discovery.max_batches must be >= 1")
    if config.sampling_strategy not in {"sequential", "label_balanced"}:
        raise ValueError("discovery.sampling_strategy must be 'sequential' or 'label_balanced'")
    if config.min_positive_windows < 1:
        raise ValueError("discovery.min_positive_windows must be >= 1")
    if config.negative_to_positive_ratio < 0.0:
        raise ValueError("discovery.negative_to_positive_ratio must be >= 0")
    if config.search_max_batches is not None and config.search_max_batches < 1:
        raise ValueError("discovery.search_max_batches must be >= 1 when provided")
    if config.probe_epochs < 1:
        raise ValueError("discovery.probe_epochs must be >= 1")
    if config.probe_learning_rate <= 0:
        raise ValueError("discovery.probe_learning_rate must be > 0")
    if config.top_k_candidates < 1:
        raise ValueError("discovery.top_k_candidates must be >= 1")
    if not 0.0 < config.candidate_session_balance_fraction <= 1.0:
        raise ValueError("discovery.candidate_session_balance_fraction must be in (0, 1]")
    if config.min_cluster_size < 2:
        raise ValueError("discovery.min_cluster_size must be >= 2")
    if config.stability_rounds < 1:
        raise ValueError("discovery.stability_rounds must be >= 1")


def validate_experiment_config(config: ExperimentConfig) -> None:
    _validate_experiment_root(config)
    _validate_data_runtime_config(config.data_runtime)
    _validate_model_config(config.model)
    _validate_experiment_identity_config(config.experiment)
    _validate_count_normalization_config(config.count_normalization)
    _validate_objective_config(config.objective)
    _validate_discovery_feature_config(config.discovery, config.model)
    _validate_optimization_config(config.optimization)
    _validate_training_runtime_config(config.training)
    _validate_dataset_selection_config(config.dataset_selection)
    _validate_execution_config(config.execution)
    _validate_evaluation_config(config.evaluation)
    _validate_runtime_subset_config(config.runtime_subset)
    _validate_discovery_config(config.discovery)
