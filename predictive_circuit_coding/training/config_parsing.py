from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from predictive_circuit_coding.training.config_schema import (
    ArtifactConfig,
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
    SplitConfig,
    TrainingRuntimeConfig,
)
from predictive_circuit_coding.training.config_validation import validate_experiment_config

_PIPELINE_ONLY_KEYS = frozenset({"paths", "stages", "pipeline", "tasks", "arms", "notebook_ui"})
_ALLOWED_TOP_LEVEL_KEYS = frozenset(
    {
        "extends",
        "dataset_id",
        "split_name",
        "seed",
        "experiment",
        "data_runtime",
        "count_normalization",
        "model",
        "objective",
        "optimization",
        "artifacts",
        "splits",
        "dataset_selection",
        "runtime_subset",
        "training",
        "execution",
        "evaluation",
        "discovery",
    }
)
_ALLOWED_EXPERIMENT_KEYS = frozenset({"variant_name"})
_ALLOWED_DATA_RUNTIME_KEYS = frozenset(
    {
        "bin_width_ms",
        "context_bins",
        "patch_bins",
        "min_unit_spikes",
        "max_units",
        "padding_strategy",
        "include_trials",
        "include_stimulus_presentations",
        "include_optotagging",
    }
)
_ALLOWED_COUNT_NORMALIZATION_KEYS = frozenset({"mode", "stats_path"})
_ALLOWED_MODEL_KEYS = frozenset(
    {
        "d_model",
        "num_heads",
        "temporal_layers",
        "spatial_layers",
        "dropout",
        "mlp_ratio",
        "l2_normalize_tokens",
        "norm_eps",
        "population_token_mode",
    }
)
_ALLOWED_OBJECTIVE_KEYS = frozenset(
    {
        "predictive_target_type",
        "continuation_baseline_type",
        "predictive_loss",
        "reconstruction_loss",
        "reconstruction_weight",
        "exclude_final_prediction_patch",
        "reconstruction_target_mode",
    }
)
_ALLOWED_OPTIMIZATION_KEYS = frozenset(
    {
        "learning_rate",
        "weight_decay",
        "grad_clip_norm",
        "batch_size",
        "scheduler_type",
        "scheduler_warmup_steps",
    }
)
_ALLOWED_ARTIFACT_KEYS = frozenset(
    {"checkpoint_dir", "summary_path", "checkpoint_prefix", "save_config_snapshot"}
)
_ALLOWED_SPLIT_KEYS = frozenset({"train", "valid", "discovery", "test"})
_ALLOWED_DATASET_SELECTION_KEYS = frozenset(
    {
        "output_name",
        "session_ids",
        "subject_ids",
        "exclude_session_ids",
        "exclude_subject_ids",
        "session_ids_file",
        "subject_ids_file",
        "exclude_session_ids_file",
        "exclude_subject_ids_file",
        "experience_levels",
        "session_types",
        "image_sets",
        "session_numbers",
        "project_codes",
        "brain_regions_any",
        "min_n_units",
        "max_n_units",
        "min_trial_count",
        "max_trial_count",
        "min_duration_s",
        "max_duration_s",
        "split_seed",
        "split_primary_axis",
        "train_fraction",
        "valid_fraction",
        "discovery_fraction",
        "test_fraction",
    }
)
_ALLOWED_RUNTIME_SUBSET_KEYS = frozenset(
    {"split_manifest_path", "session_catalog_path", "config_dir", "config_name_prefix"}
)
_ALLOWED_TRAINING_KEYS = frozenset(
    {
        "num_epochs",
        "train_steps_per_epoch",
        "validation_steps",
        "checkpoint_every_epochs",
        "evaluate_every_epochs",
        "resume_checkpoint",
        "dataloader_workers",
        "train_window_seed",
        "log_every_steps",
    }
)
_ALLOWED_EXECUTION_KEYS = frozenset({"device", "mixed_precision"})
_ALLOWED_EVALUATION_KEYS = frozenset({"max_batches", "sequential_step_s"})
_ALLOWED_DISCOVERY_KEYS = frozenset(
    {
        "target_label",
        "target_label_mode",
        "target_label_match_value",
        "max_batches",
        "sampling_strategy",
        "min_positive_windows",
        "negative_to_positive_ratio",
        "search_max_batches",
        "probe_epochs",
        "probe_learning_rate",
        "top_k_candidates",
        "candidate_session_balance_fraction",
        "min_candidate_score",
        "min_cluster_size",
        "stability_rounds",
        "shuffle_seed",
        "pooled_feature_mode",
    }
)


def _type_label(value: Any) -> str:
    return type(value).__name__


def _ensure_mapping(section_name: str, value: Any, *, required: bool = False) -> dict[str, Any]:
    if value is None:
        if required:
            raise ValueError(f"Missing required '{section_name}' section")
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Section '{section_name}' must be a mapping, got {_type_label(value)}")
    return dict(value)


def _reject_unexpected_keys(section_name: str, payload: Mapping[str, Any], allowed_keys: frozenset[str]) -> None:
    unknown = sorted(set(payload.keys()) - allowed_keys)
    if not unknown:
        return
    message = f"Unsupported {section_name} keys: {', '.join(unknown)}"
    if section_name == "experiment config" and any(key in _PIPELINE_ONLY_KEYS for key in unknown):
        message += ". This looks like a workflow/pipeline config, not a training experiment config."
    raise ValueError(message)


def _read_required_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string, got {_type_label(value)}")
    return value


def _read_string(value: Any, *, field_name: str, default: str) -> str:
    if value is None:
        return default
    return _read_required_string(value, field_name=field_name)


def _read_optional_string(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    parsed = _read_required_string(value, field_name=field_name)
    return None if parsed == "" else parsed


def _read_bool(value: Any, *, field_name: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean, got {_type_label(value)}")
    return value


def _read_int(value: Any, *, field_name: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer, got {_type_label(value)}")
    return value


def _read_optional_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _read_int(value, field_name=field_name, default=0)


def _read_float(value: Any, *, field_name: str, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number, got {_type_label(value)}")
    return float(value)


def _read_optional_float(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    return _read_float(value, field_name=field_name, default=0.0)


def _read_string_sequence(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of strings, got {_type_label(value)}")
    parsed: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{field_name}[{index}] must be a string, got {_type_label(item)}")
        parsed.append(item)
    return tuple(parsed)


def _read_int_sequence(value: Any, *, field_name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of integers, got {_type_label(value)}")
    parsed: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{field_name}[{index}] must be an integer, got {_type_label(item)}")
        parsed.append(item)
    return tuple(parsed)


def _resolve_path(base_dir: Path, value: Any, *, field_name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{field_name} must be a path string, got {_type_label(value)}")
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _resolve_optional_path(base_dir: Path, value: Any, *, field_name: str) -> Path | None:
    if value in (None, ""):
        return None
    return _resolve_path(base_dir, value, field_name=field_name)


def _deep_merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_config(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _load_raw_experiment_payload(path: Path, *, seen_paths: set[Path] | None = None) -> dict[str, Any]:
    resolved_path = path.resolve()
    seen = set() if seen_paths is None else set(seen_paths)
    if resolved_path in seen:
        raise ValueError(f"Config extends cycle detected at {resolved_path}")
    seen.add(resolved_path)

    with resolved_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        raise ValueError(f"Experiment config {resolved_path} is empty")
    if not isinstance(raw, Mapping):
        raise ValueError(f"Experiment config {resolved_path} must contain a mapping")

    payload = dict(raw)
    extends_value = payload.get("extends")
    if extends_value in (None, ""):
        return payload

    base_path = _resolve_path(resolved_path.parent, extends_value, field_name="extends")
    base_raw = _load_raw_experiment_payload(base_path, seen_paths=seen)
    return _deep_merge_config(base_raw, payload)


def _parse_experiment_sections(raw: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    _reject_unexpected_keys("experiment config", raw, _ALLOWED_TOP_LEVEL_KEYS)

    def section(name: str, allowed_keys: frozenset[str], *, required: bool = False) -> dict[str, Any]:
        payload = _ensure_mapping(name, raw.get(name), required=required)
        _reject_unexpected_keys(f"{name} section", payload, allowed_keys)
        return payload

    return {
        "experiment": section("experiment", _ALLOWED_EXPERIMENT_KEYS),
        "data_runtime": section("data_runtime", _ALLOWED_DATA_RUNTIME_KEYS, required=True),
        "count_normalization": section("count_normalization", _ALLOWED_COUNT_NORMALIZATION_KEYS),
        "model": section("model", _ALLOWED_MODEL_KEYS, required=True),
        "objective": section("objective", _ALLOWED_OBJECTIVE_KEYS, required=True),
        "optimization": section("optimization", _ALLOWED_OPTIMIZATION_KEYS, required=True),
        "artifacts": section("artifacts", _ALLOWED_ARTIFACT_KEYS, required=True),
        "splits": section("splits", _ALLOWED_SPLIT_KEYS),
        "dataset_selection": section("dataset_selection", _ALLOWED_DATASET_SELECTION_KEYS),
        "runtime_subset": section("runtime_subset", _ALLOWED_RUNTIME_SUBSET_KEYS),
        "training": section("training", _ALLOWED_TRAINING_KEYS),
        "execution": section("execution", _ALLOWED_EXECUTION_KEYS),
        "evaluation": section("evaluation", _ALLOWED_EVALUATION_KEYS),
        "discovery": section("discovery", _ALLOWED_DISCOVERY_KEYS),
    }


def _parse_experiment_identity_config(raw: Mapping[str, Any]) -> ExperimentIdentityConfig:
    return ExperimentIdentityConfig(
        variant_name=_read_string(
            raw.get("variant_name"),
            field_name="experiment.variant_name",
            default="refined_core",
        ),
    )


def _parse_data_runtime_config(raw: Mapping[str, Any]) -> DataRuntimeConfig:
    return DataRuntimeConfig(
        bin_width_ms=_read_float(raw.get("bin_width_ms"), field_name="data_runtime.bin_width_ms", default=20.0),
        context_bins=_read_int(raw.get("context_bins"), field_name="data_runtime.context_bins", default=500),
        patch_bins=_read_int(raw.get("patch_bins"), field_name="data_runtime.patch_bins", default=50),
        min_unit_spikes=_read_int(
            raw.get("min_unit_spikes"),
            field_name="data_runtime.min_unit_spikes",
            default=0,
        ),
        max_units=_read_optional_int(raw.get("max_units"), field_name="data_runtime.max_units"),
        padding_strategy=_read_string(
            raw.get("padding_strategy"),
            field_name="data_runtime.padding_strategy",
            default="mask",
        ),
        include_trials=_read_bool(
            raw.get("include_trials"),
            field_name="data_runtime.include_trials",
            default=True,
        ),
        include_stimulus_presentations=_read_bool(
            raw.get("include_stimulus_presentations"),
            field_name="data_runtime.include_stimulus_presentations",
            default=True,
        ),
        include_optotagging=_read_bool(
            raw.get("include_optotagging"),
            field_name="data_runtime.include_optotagging",
            default=True,
        ),
    )


def _parse_count_normalization_config(raw: Mapping[str, Any], config_dir: Path) -> CountNormalizationConfig:
    return CountNormalizationConfig(
        mode=_read_string(
            raw.get("mode"),
            field_name="count_normalization.mode",
            default="none",
        ),
        stats_path=_resolve_optional_path(
            config_dir,
            raw.get("stats_path"),
            field_name="count_normalization.stats_path",
        ),
    )


def _parse_model_config(raw: Mapping[str, Any]) -> ModelConfig:
    return ModelConfig(
        d_model=_read_int(raw.get("d_model"), field_name="model.d_model", default=256),
        num_heads=_read_int(raw.get("num_heads"), field_name="model.num_heads", default=8),
        temporal_layers=_read_int(raw.get("temporal_layers"), field_name="model.temporal_layers", default=2),
        spatial_layers=_read_int(raw.get("spatial_layers"), field_name="model.spatial_layers", default=2),
        dropout=_read_float(raw.get("dropout"), field_name="model.dropout", default=0.1),
        mlp_ratio=_read_float(raw.get("mlp_ratio"), field_name="model.mlp_ratio", default=4.0),
        l2_normalize_tokens=_read_bool(
            raw.get("l2_normalize_tokens"),
            field_name="model.l2_normalize_tokens",
            default=True,
        ),
        norm_eps=_read_float(raw.get("norm_eps"), field_name="model.norm_eps", default=1.0e-5),
        population_token_mode=_read_string(
            raw.get("population_token_mode"),
            field_name="model.population_token_mode",
            default="none",
        ),
    )


def _parse_objective_config(raw: Mapping[str, Any]) -> ObjectiveConfig:
    return ObjectiveConfig(
        predictive_target_type=_read_string(
            raw.get("predictive_target_type"),
            field_name="objective.predictive_target_type",
            default="delta",
        ),
        continuation_baseline_type=_read_string(
            raw.get("continuation_baseline_type"),
            field_name="objective.continuation_baseline_type",
            default="previous_patch",
        ),
        predictive_loss=_read_string(
            raw.get("predictive_loss"),
            field_name="objective.predictive_loss",
            default="mse",
        ),
        reconstruction_loss=_read_string(
            raw.get("reconstruction_loss"),
            field_name="objective.reconstruction_loss",
            default="mse",
        ),
        reconstruction_weight=_read_float(
            raw.get("reconstruction_weight"),
            field_name="objective.reconstruction_weight",
            default=0.2,
        ),
        reconstruction_target_mode=_read_string(
            raw.get("reconstruction_target_mode"),
            field_name="objective.reconstruction_target_mode",
            default="raw",
        ),
        exclude_final_prediction_patch=_read_bool(
            raw.get("exclude_final_prediction_patch"),
            field_name="objective.exclude_final_prediction_patch",
            default=True,
        ),
    )


def _parse_optimization_config(raw: Mapping[str, Any]) -> OptimizationConfig:
    return OptimizationConfig(
        learning_rate=_read_float(
            raw.get("learning_rate"),
            field_name="optimization.learning_rate",
            default=1.0e-4,
        ),
        weight_decay=_read_float(
            raw.get("weight_decay"),
            field_name="optimization.weight_decay",
            default=1.0e-4,
        ),
        grad_clip_norm=_read_optional_float(
            raw.get("grad_clip_norm"),
            field_name="optimization.grad_clip_norm",
        ),
        batch_size=_read_int(raw.get("batch_size"), field_name="optimization.batch_size", default=4),
        scheduler_type=_read_string(
            raw.get("scheduler_type"),
            field_name="optimization.scheduler_type",
            default="none",
        ),
        scheduler_warmup_steps=_read_int(
            raw.get("scheduler_warmup_steps"),
            field_name="optimization.scheduler_warmup_steps",
            default=0,
        ),
    )


def _parse_artifact_config(raw: Mapping[str, Any], config_dir: Path) -> ArtifactConfig:
    return ArtifactConfig(
        checkpoint_dir=_resolve_path(
            config_dir,
            raw.get("checkpoint_dir", "artifacts/checkpoints"),
            field_name="artifacts.checkpoint_dir",
        ),
        summary_path=_resolve_path(
            config_dir,
            raw.get("summary_path", "artifacts/training_summary.json"),
            field_name="artifacts.summary_path",
        ),
        checkpoint_prefix=_read_string(
            raw.get("checkpoint_prefix"),
            field_name="artifacts.checkpoint_prefix",
            default="pcc",
        ),
        save_config_snapshot=_read_bool(
            raw.get("save_config_snapshot"),
            field_name="artifacts.save_config_snapshot",
            default=True,
        ),
    )


def _parse_split_config(raw: Mapping[str, Any]) -> SplitConfig:
    return SplitConfig(
        train=_read_string(raw.get("train"), field_name="splits.train", default="train"),
        valid=_read_string(raw.get("valid"), field_name="splits.valid", default="valid"),
        discovery=_read_string(raw.get("discovery"), field_name="splits.discovery", default="discovery"),
        test=_read_string(raw.get("test"), field_name="splits.test", default="test"),
    )


def _parse_dataset_selection_config(raw: Mapping[str, Any], config_dir: Path) -> DatasetSelectionConfig:
    return DatasetSelectionConfig(
        output_name=_read_string(
            raw.get("output_name"),
            field_name="dataset_selection.output_name",
            default="runtime_selection",
        ),
        session_ids=_read_string_sequence(
            raw.get("session_ids"),
            field_name="dataset_selection.session_ids",
        ),
        subject_ids=_read_string_sequence(
            raw.get("subject_ids"),
            field_name="dataset_selection.subject_ids",
        ),
        exclude_session_ids=_read_string_sequence(
            raw.get("exclude_session_ids"),
            field_name="dataset_selection.exclude_session_ids",
        ),
        exclude_subject_ids=_read_string_sequence(
            raw.get("exclude_subject_ids"),
            field_name="dataset_selection.exclude_subject_ids",
        ),
        session_ids_file=_resolve_optional_path(
            config_dir,
            raw.get("session_ids_file"),
            field_name="dataset_selection.session_ids_file",
        ),
        subject_ids_file=_resolve_optional_path(
            config_dir,
            raw.get("subject_ids_file"),
            field_name="dataset_selection.subject_ids_file",
        ),
        exclude_session_ids_file=_resolve_optional_path(
            config_dir,
            raw.get("exclude_session_ids_file"),
            field_name="dataset_selection.exclude_session_ids_file",
        ),
        exclude_subject_ids_file=_resolve_optional_path(
            config_dir,
            raw.get("exclude_subject_ids_file"),
            field_name="dataset_selection.exclude_subject_ids_file",
        ),
        experience_levels=_read_string_sequence(
            raw.get("experience_levels"),
            field_name="dataset_selection.experience_levels",
        ),
        session_types=_read_string_sequence(
            raw.get("session_types"),
            field_name="dataset_selection.session_types",
        ),
        image_sets=_read_string_sequence(
            raw.get("image_sets"),
            field_name="dataset_selection.image_sets",
        ),
        session_numbers=_read_int_sequence(
            raw.get("session_numbers"),
            field_name="dataset_selection.session_numbers",
        ),
        project_codes=_read_string_sequence(
            raw.get("project_codes"),
            field_name="dataset_selection.project_codes",
        ),
        brain_regions_any=_read_string_sequence(
            raw.get("brain_regions_any"),
            field_name="dataset_selection.brain_regions_any",
        ),
        min_n_units=_read_optional_int(raw.get("min_n_units"), field_name="dataset_selection.min_n_units"),
        max_n_units=_read_optional_int(raw.get("max_n_units"), field_name="dataset_selection.max_n_units"),
        min_trial_count=_read_optional_int(
            raw.get("min_trial_count"),
            field_name="dataset_selection.min_trial_count",
        ),
        max_trial_count=_read_optional_int(
            raw.get("max_trial_count"),
            field_name="dataset_selection.max_trial_count",
        ),
        min_duration_s=_read_optional_float(
            raw.get("min_duration_s"),
            field_name="dataset_selection.min_duration_s",
        ),
        max_duration_s=_read_optional_float(
            raw.get("max_duration_s"),
            field_name="dataset_selection.max_duration_s",
        ),
        split_seed=_read_optional_int(raw.get("split_seed"), field_name="dataset_selection.split_seed"),
        split_primary_axis=_read_optional_string(
            raw.get("split_primary_axis"),
            field_name="dataset_selection.split_primary_axis",
        ),
        train_fraction=_read_optional_float(
            raw.get("train_fraction"),
            field_name="dataset_selection.train_fraction",
        ),
        valid_fraction=_read_optional_float(
            raw.get("valid_fraction"),
            field_name="dataset_selection.valid_fraction",
        ),
        discovery_fraction=_read_optional_float(
            raw.get("discovery_fraction"),
            field_name="dataset_selection.discovery_fraction",
        ),
        test_fraction=_read_optional_float(
            raw.get("test_fraction"),
            field_name="dataset_selection.test_fraction",
        ),
    )


def _parse_runtime_subset_config(raw: Mapping[str, Any], config_dir: Path) -> RuntimeSubsetConfig | None:
    if not raw:
        return None
    return RuntimeSubsetConfig(
        split_manifest_path=_resolve_path(
            config_dir,
            raw["split_manifest_path"],
            field_name="runtime_subset.split_manifest_path",
        ),
        session_catalog_path=_resolve_path(
            config_dir,
            raw["session_catalog_path"],
            field_name="runtime_subset.session_catalog_path",
        ),
        config_dir=_resolve_path(
            config_dir,
            raw["config_dir"],
            field_name="runtime_subset.config_dir",
        ),
        config_name_prefix=_read_string(
            raw.get("config_name_prefix"),
            field_name="runtime_subset.config_name_prefix",
            default="torch_brain_runtime",
        ),
    )


def _parse_training_runtime_config(raw: Mapping[str, Any], config_dir: Path) -> TrainingRuntimeConfig:
    return TrainingRuntimeConfig(
        num_epochs=_read_int(raw.get("num_epochs"), field_name="training.num_epochs", default=1),
        train_steps_per_epoch=_read_int(
            raw.get("train_steps_per_epoch"),
            field_name="training.train_steps_per_epoch",
            default=8,
        ),
        validation_steps=_read_int(
            raw.get("validation_steps"),
            field_name="training.validation_steps",
            default=2,
        ),
        checkpoint_every_epochs=_read_int(
            raw.get("checkpoint_every_epochs"),
            field_name="training.checkpoint_every_epochs",
            default=1,
        ),
        evaluate_every_epochs=_read_int(
            raw.get("evaluate_every_epochs"),
            field_name="training.evaluate_every_epochs",
            default=1,
        ),
        resume_checkpoint=_resolve_optional_path(
            config_dir,
            raw.get("resume_checkpoint"),
            field_name="training.resume_checkpoint",
        ),
        dataloader_workers=_read_int(
            raw.get("dataloader_workers"),
            field_name="training.dataloader_workers",
            default=0,
        ),
        train_window_seed=_read_int(
            raw.get("train_window_seed"),
            field_name="training.train_window_seed",
            default=0,
        ),
        log_every_steps=_read_int(
            raw.get("log_every_steps"),
            field_name="training.log_every_steps",
            default=1,
        ),
    )


def _parse_execution_config(raw: Mapping[str, Any]) -> ExecutionConfig:
    return ExecutionConfig(
        device=_read_string(raw.get("device"), field_name="execution.device", default="cpu"),
        mixed_precision=_read_bool(
            raw.get("mixed_precision"),
            field_name="execution.mixed_precision",
            default=False,
        ),
    )


def _parse_evaluation_config(raw: Mapping[str, Any]) -> EvaluationConfig:
    return EvaluationConfig(
        max_batches=_read_int(raw.get("max_batches"), field_name="evaluation.max_batches", default=4),
        sequential_step_s=_read_optional_float(
            raw.get("sequential_step_s"),
            field_name="evaluation.sequential_step_s",
        ),
    )


def _parse_discovery_config(raw: Mapping[str, Any]) -> DiscoveryConfig:
    return DiscoveryConfig(
        target_label=_read_string(
            raw.get("target_label"),
            field_name="discovery.target_label",
            default="stimulus_change",
        ),
        target_label_mode=_read_string(
            raw.get("target_label_mode"),
            field_name="discovery.target_label_mode",
            default="auto",
        ),
        target_label_match_value=_read_optional_string(
            raw.get("target_label_match_value"),
            field_name="discovery.target_label_match_value",
        ),
        max_batches=_read_int(raw.get("max_batches"), field_name="discovery.max_batches", default=6),
        sampling_strategy=_read_string(
            raw.get("sampling_strategy"),
            field_name="discovery.sampling_strategy",
            default="sequential",
        ),
        min_positive_windows=_read_int(
            raw.get("min_positive_windows"),
            field_name="discovery.min_positive_windows",
            default=1,
        ),
        negative_to_positive_ratio=_read_float(
            raw.get("negative_to_positive_ratio"),
            field_name="discovery.negative_to_positive_ratio",
            default=1.0,
        ),
        search_max_batches=_read_optional_int(
            raw.get("search_max_batches"),
            field_name="discovery.search_max_batches",
        ),
        probe_epochs=_read_int(raw.get("probe_epochs"), field_name="discovery.probe_epochs", default=25),
        probe_learning_rate=_read_float(
            raw.get("probe_learning_rate"),
            field_name="discovery.probe_learning_rate",
            default=1.0e-2,
        ),
        top_k_candidates=_read_int(
            raw.get("top_k_candidates"),
            field_name="discovery.top_k_candidates",
            default=32,
        ),
        candidate_session_balance_fraction=_read_float(
            raw.get("candidate_session_balance_fraction"),
            field_name="discovery.candidate_session_balance_fraction",
            default=0.2,
        ),
        min_candidate_score=_read_float(
            raw.get("min_candidate_score"),
            field_name="discovery.min_candidate_score",
            default=0.0,
        ),
        min_cluster_size=_read_int(
            raw.get("min_cluster_size"),
            field_name="discovery.min_cluster_size",
            default=2,
        ),
        stability_rounds=_read_int(
            raw.get("stability_rounds"),
            field_name="discovery.stability_rounds",
            default=4,
        ),
        shuffle_seed=_read_int(raw.get("shuffle_seed"), field_name="discovery.shuffle_seed", default=17),
        pooled_feature_mode=_read_string(
            raw.get("pooled_feature_mode"),
            field_name="discovery.pooled_feature_mode",
            default="mean_tokens",
        ),
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    raw = _load_raw_experiment_payload(config_path)
    sections = _parse_experiment_sections(raw)
    config_dir = config_path.parent

    cfg = ExperimentConfig(
        dataset_id=_read_required_string(raw.get("dataset_id"), field_name="dataset_id"),
        split_name=_read_string(raw.get("split_name"), field_name="split_name", default="train"),
        seed=_read_int(raw.get("seed"), field_name="seed", default=0),
        experiment=_parse_experiment_identity_config(sections["experiment"]),
        data_runtime=_parse_data_runtime_config(sections["data_runtime"]),
        count_normalization=_parse_count_normalization_config(sections["count_normalization"], config_dir),
        model=_parse_model_config(sections["model"]),
        objective=_parse_objective_config(sections["objective"]),
        optimization=_parse_optimization_config(sections["optimization"]),
        artifacts=_parse_artifact_config(sections["artifacts"], config_dir),
        splits=_parse_split_config(sections["splits"]),
        dataset_selection=_parse_dataset_selection_config(sections["dataset_selection"], config_dir),
        runtime_subset=_parse_runtime_subset_config(sections["runtime_subset"], config_dir),
        training=_parse_training_runtime_config(sections["training"], config_dir),
        execution=_parse_execution_config(sections["execution"]),
        evaluation=_parse_evaluation_config(sections["evaluation"]),
        discovery=_parse_discovery_config(sections["discovery"]),
        config_path=config_path,
    )
    validate_experiment_config(cfg)
    return cfg
