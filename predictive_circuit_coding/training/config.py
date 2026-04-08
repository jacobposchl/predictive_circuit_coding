from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataRuntimeConfig:
    bin_width_ms: float
    context_bins: int
    patch_bins: int
    min_unit_spikes: int
    max_units: int | None
    padding_strategy: str
    include_trials: bool
    include_stimulus_presentations: bool
    include_optotagging: bool

    @property
    def bin_width_s(self) -> float:
        return float(self.bin_width_ms) / 1000.0

    @property
    def context_duration_s(self) -> float:
        return float(self.context_bins) * self.bin_width_s

    @property
    def patches_per_window(self) -> int:
        return int(self.context_bins // self.patch_bins)


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    num_heads: int
    temporal_layers: int
    spatial_layers: int
    dropout: float
    mlp_ratio: float
    l2_normalize_tokens: bool
    norm_eps: float


@dataclass(frozen=True)
class ObjectiveConfig:
    predictive_target_type: str
    continuation_baseline_type: str
    predictive_loss: str
    reconstruction_loss: str
    reconstruction_weight: float
    exclude_final_prediction_patch: bool


@dataclass(frozen=True)
class OptimizationConfig:
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float | None
    batch_size: int
    scheduler_type: str = "none"
    scheduler_warmup_steps: int = 0


@dataclass(frozen=True)
class ArtifactConfig:
    checkpoint_dir: Path
    summary_path: Path
    checkpoint_prefix: str
    save_config_snapshot: bool


@dataclass(frozen=True)
class SplitConfig:
    train: str = "train"
    valid: str = "valid"
    discovery: str = "discovery"
    test: str = "test"


@dataclass(frozen=True)
class RuntimeSubsetConfig:
    split_manifest_path: Path
    session_catalog_path: Path
    config_dir: Path
    config_name_prefix: str = "torch_brain_runtime"


@dataclass(frozen=True)
class DatasetSelectionConfig:
    output_name: str = "runtime_selection"
    session_ids: tuple[str, ...] = ()
    subject_ids: tuple[str, ...] = ()
    exclude_session_ids: tuple[str, ...] = ()
    exclude_subject_ids: tuple[str, ...] = ()
    session_ids_file: Path | None = None
    subject_ids_file: Path | None = None
    exclude_session_ids_file: Path | None = None
    exclude_subject_ids_file: Path | None = None
    experience_levels: tuple[str, ...] = ()
    session_types: tuple[str, ...] = ()
    image_sets: tuple[str, ...] = ()
    session_numbers: tuple[int, ...] = ()
    project_codes: tuple[str, ...] = ()
    brain_regions_any: tuple[str, ...] = ()
    min_n_units: int | None = None
    max_n_units: int | None = None
    min_trial_count: int | None = None
    max_trial_count: int | None = None
    min_duration_s: float | None = None
    max_duration_s: float | None = None
    split_seed: int | None = None
    split_primary_axis: str | None = None
    train_fraction: float | None = None
    valid_fraction: float | None = None
    discovery_fraction: float | None = None
    test_fraction: float | None = None

    @property
    def is_active(self) -> bool:
        return any(
            (
                self.session_ids,
                self.subject_ids,
                self.exclude_session_ids,
                self.exclude_subject_ids,
                self.session_ids_file is not None,
                self.subject_ids_file is not None,
                self.exclude_session_ids_file is not None,
                self.exclude_subject_ids_file is not None,
                self.experience_levels,
                self.session_types,
                self.image_sets,
                self.session_numbers,
                self.project_codes,
                self.brain_regions_any,
                self.min_n_units is not None,
                self.max_n_units is not None,
                self.min_trial_count is not None,
                self.max_trial_count is not None,
                self.min_duration_s is not None,
                self.max_duration_s is not None,
            )
        )


@dataclass(frozen=True)
class TrainingRuntimeConfig:
    num_epochs: int = 1
    train_steps_per_epoch: int = 8
    validation_steps: int = 2
    checkpoint_every_epochs: int = 1
    evaluate_every_epochs: int = 1
    resume_checkpoint: Path | None = None
    dataloader_workers: int = 0
    train_window_seed: int = 0
    log_every_steps: int = 1


@dataclass(frozen=True)
class ExecutionConfig:
    device: str = "cpu"
    mixed_precision: bool = False


@dataclass(frozen=True)
class EvaluationConfig:
    max_batches: int = 4
    sequential_step_s: float | None = None


@dataclass(frozen=True)
class DiscoveryConfig:
    target_label: str = "stimulus_change"
    target_label_mode: str = "auto"
    target_label_match_value: str | None = None
    max_batches: int = 6
    sampling_strategy: str = "sequential"
    min_positive_windows: int = 1
    negative_to_positive_ratio: float = 1.0
    search_max_batches: int | None = None
    probe_epochs: int = 25
    probe_learning_rate: float = 1.0e-2
    top_k_candidates: int = 32
    candidate_session_balance_fraction: float = 0.2
    min_candidate_score: float = 0.0
    min_cluster_size: int = 2
    stability_rounds: int = 4
    shuffle_seed: int = 17


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_id: str
    split_name: str
    seed: int
    data_runtime: DataRuntimeConfig
    model: ModelConfig
    objective: ObjectiveConfig
    optimization: OptimizationConfig
    artifacts: ArtifactConfig
    splits: SplitConfig = field(default_factory=SplitConfig)
    dataset_selection: DatasetSelectionConfig = field(default_factory=DatasetSelectionConfig)
    runtime_subset: RuntimeSubsetConfig | None = None
    training: TrainingRuntimeConfig = field(default_factory=TrainingRuntimeConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    config_path: Path = Path("experiment.yaml")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["config_path"] = str(self.config_path)
        payload["artifacts"]["checkpoint_dir"] = str(self.artifacts.checkpoint_dir)
        payload["artifacts"]["summary_path"] = str(self.artifacts.summary_path)
        for key in (
            "session_ids_file",
            "subject_ids_file",
            "exclude_session_ids_file",
            "exclude_subject_ids_file",
        ):
            value = payload["dataset_selection"].get(key)
            if value is not None:
                payload["dataset_selection"][key] = str(value)
        if self.training.resume_checkpoint is not None:
            payload["training"]["resume_checkpoint"] = str(self.training.resume_checkpoint)
        if self.runtime_subset is not None:
            payload["runtime_subset"] = {
                "split_manifest_path": str(self.runtime_subset.split_manifest_path),
                "session_catalog_path": str(self.runtime_subset.session_catalog_path),
                "config_dir": str(self.runtime_subset.config_dir),
                "config_name_prefix": self.runtime_subset.config_name_prefix,
            }
        return payload


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _resolve_optional_path(base_dir: Path, value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    return _resolve_path(base_dir, str(value))


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    config_dir = config_path.parent
    runtime = raw["data_runtime"]
    model = raw["model"]
    objective = raw["objective"]
    optimization = raw["optimization"]
    artifacts = raw["artifacts"]
    split_raw = raw.get("splits", {})
    selection_raw = raw.get("dataset_selection", {})
    runtime_subset_raw = raw.get("runtime_subset", {})
    training_raw = raw.get("training", {})
    execution_raw = raw.get("execution", {})
    evaluation_raw = raw.get("evaluation", {})
    discovery_raw = raw.get("discovery", {})

    cfg = ExperimentConfig(
        dataset_id=str(raw["dataset_id"]),
        split_name=str(raw.get("split_name", "train")),
        seed=int(raw.get("seed", 0)),
        data_runtime=DataRuntimeConfig(
            bin_width_ms=float(runtime.get("bin_width_ms", 20.0)),
            context_bins=int(runtime.get("context_bins", 500)),
            patch_bins=int(runtime.get("patch_bins", 50)),
            min_unit_spikes=int(runtime.get("min_unit_spikes", 0)),
            max_units=int(runtime["max_units"]) if runtime.get("max_units") is not None else None,
            padding_strategy=str(runtime.get("padding_strategy", "mask")),
            include_trials=bool(runtime.get("include_trials", True)),
            include_stimulus_presentations=bool(runtime.get("include_stimulus_presentations", True)),
            include_optotagging=bool(runtime.get("include_optotagging", True)),
        ),
        model=ModelConfig(
            d_model=int(model.get("d_model", 256)),
            num_heads=int(model.get("num_heads", 8)),
            temporal_layers=int(model.get("temporal_layers", 2)),
            spatial_layers=int(model.get("spatial_layers", 2)),
            dropout=float(model.get("dropout", 0.1)),
            mlp_ratio=float(model.get("mlp_ratio", 4.0)),
            l2_normalize_tokens=bool(model.get("l2_normalize_tokens", True)),
            norm_eps=float(model.get("norm_eps", 1.0e-5)),
        ),
        objective=ObjectiveConfig(
            predictive_target_type=str(objective.get("predictive_target_type", "delta")),
            continuation_baseline_type=str(objective.get("continuation_baseline_type", "previous_patch")),
            predictive_loss=str(objective.get("predictive_loss", "mse")),
            reconstruction_loss=str(objective.get("reconstruction_loss", "mse")),
            reconstruction_weight=float(objective.get("reconstruction_weight", 0.2)),
            exclude_final_prediction_patch=bool(objective.get("exclude_final_prediction_patch", True)),
        ),
        optimization=OptimizationConfig(
            learning_rate=float(optimization.get("learning_rate", 1.0e-4)),
            weight_decay=float(optimization.get("weight_decay", 1.0e-4)),
            grad_clip_norm=(
                float(optimization["grad_clip_norm"])
                if optimization.get("grad_clip_norm") is not None
                else None
            ),
            batch_size=int(optimization.get("batch_size", 4)),
            scheduler_type=str(optimization.get("scheduler_type", "none")),
            scheduler_warmup_steps=int(optimization.get("scheduler_warmup_steps", 0)),
        ),
        artifacts=ArtifactConfig(
            checkpoint_dir=_resolve_path(config_dir, str(artifacts.get("checkpoint_dir", "artifacts/checkpoints"))),
            summary_path=_resolve_path(config_dir, str(artifacts.get("summary_path", "artifacts/training_summary.json"))),
            checkpoint_prefix=str(artifacts.get("checkpoint_prefix", "pcc")),
            save_config_snapshot=bool(artifacts.get("save_config_snapshot", True)),
        ),
        splits=SplitConfig(
            train=str(split_raw.get("train", "train")),
            valid=str(split_raw.get("valid", "valid")),
            discovery=str(split_raw.get("discovery", "discovery")),
            test=str(split_raw.get("test", "test")),
        ),
        dataset_selection=DatasetSelectionConfig(
            output_name=str(selection_raw.get("output_name", "runtime_selection")),
            session_ids=tuple(str(value) for value in selection_raw.get("session_ids", []) or ()),
            subject_ids=tuple(str(value) for value in selection_raw.get("subject_ids", []) or ()),
            exclude_session_ids=tuple(str(value) for value in selection_raw.get("exclude_session_ids", []) or ()),
            exclude_subject_ids=tuple(str(value) for value in selection_raw.get("exclude_subject_ids", []) or ()),
            session_ids_file=_resolve_optional_path(config_dir, selection_raw.get("session_ids_file")),
            subject_ids_file=_resolve_optional_path(config_dir, selection_raw.get("subject_ids_file")),
            exclude_session_ids_file=_resolve_optional_path(config_dir, selection_raw.get("exclude_session_ids_file")),
            exclude_subject_ids_file=_resolve_optional_path(config_dir, selection_raw.get("exclude_subject_ids_file")),
            experience_levels=tuple(str(value) for value in selection_raw.get("experience_levels", []) or ()),
            session_types=tuple(str(value) for value in selection_raw.get("session_types", []) or ()),
            image_sets=tuple(str(value) for value in selection_raw.get("image_sets", []) or ()),
            session_numbers=tuple(int(value) for value in selection_raw.get("session_numbers", []) or ()),
            project_codes=tuple(str(value) for value in selection_raw.get("project_codes", []) or ()),
            brain_regions_any=tuple(str(value) for value in selection_raw.get("brain_regions_any", []) or ()),
            min_n_units=int(selection_raw["min_n_units"]) if selection_raw.get("min_n_units") is not None else None,
            max_n_units=int(selection_raw["max_n_units"]) if selection_raw.get("max_n_units") is not None else None,
            min_trial_count=(
                int(selection_raw["min_trial_count"]) if selection_raw.get("min_trial_count") is not None else None
            ),
            max_trial_count=(
                int(selection_raw["max_trial_count"]) if selection_raw.get("max_trial_count") is not None else None
            ),
            min_duration_s=(
                float(selection_raw["min_duration_s"]) if selection_raw.get("min_duration_s") is not None else None
            ),
            max_duration_s=(
                float(selection_raw["max_duration_s"]) if selection_raw.get("max_duration_s") is not None else None
            ),
            split_seed=int(selection_raw["split_seed"]) if selection_raw.get("split_seed") is not None else None,
            split_primary_axis=(
                str(selection_raw["split_primary_axis"]) if selection_raw.get("split_primary_axis") is not None else None
            ),
            train_fraction=(
                float(selection_raw["train_fraction"]) if selection_raw.get("train_fraction") is not None else None
            ),
            valid_fraction=(
                float(selection_raw["valid_fraction"]) if selection_raw.get("valid_fraction") is not None else None
            ),
            discovery_fraction=(
                float(selection_raw["discovery_fraction"]) if selection_raw.get("discovery_fraction") is not None else None
            ),
            test_fraction=(
                float(selection_raw["test_fraction"]) if selection_raw.get("test_fraction") is not None else None
            ),
        ),
        runtime_subset=(
            RuntimeSubsetConfig(
                split_manifest_path=_resolve_path(config_dir, str(runtime_subset_raw["split_manifest_path"])),
                session_catalog_path=_resolve_path(config_dir, str(runtime_subset_raw["session_catalog_path"])),
                config_dir=_resolve_path(config_dir, str(runtime_subset_raw["config_dir"])),
                config_name_prefix=str(runtime_subset_raw.get("config_name_prefix", "torch_brain_runtime")),
            )
            if runtime_subset_raw
            else None
        ),
        training=TrainingRuntimeConfig(
            num_epochs=int(training_raw.get("num_epochs", 1)),
            train_steps_per_epoch=int(training_raw.get("train_steps_per_epoch", 8)),
            validation_steps=int(training_raw.get("validation_steps", 2)),
            checkpoint_every_epochs=int(training_raw.get("checkpoint_every_epochs", 1)),
            evaluate_every_epochs=int(training_raw.get("evaluate_every_epochs", 1)),
            resume_checkpoint=_resolve_optional_path(config_dir, training_raw.get("resume_checkpoint")),
            dataloader_workers=int(training_raw.get("dataloader_workers", 0)),
            train_window_seed=int(training_raw.get("train_window_seed", 0)),
            log_every_steps=int(training_raw.get("log_every_steps", 1)),
        ),
        execution=ExecutionConfig(
            device=str(execution_raw.get("device", "cpu")),
            mixed_precision=bool(execution_raw.get("mixed_precision", False)),
        ),
        evaluation=EvaluationConfig(
            max_batches=int(evaluation_raw.get("max_batches", 4)),
            sequential_step_s=(
                float(evaluation_raw["sequential_step_s"])
                if evaluation_raw.get("sequential_step_s") is not None
                else None
            ),
        ),
        discovery=DiscoveryConfig(
            target_label=str(discovery_raw.get("target_label", "stimulus_change")),
            target_label_mode=str(discovery_raw.get("target_label_mode", "auto")),
            target_label_match_value=(
                str(discovery_raw["target_label_match_value"])
                if discovery_raw.get("target_label_match_value") is not None
                else None
            ),
            max_batches=int(discovery_raw.get("max_batches", 6)),
            sampling_strategy=str(discovery_raw.get("sampling_strategy", "sequential")),
            min_positive_windows=int(discovery_raw.get("min_positive_windows", 1)),
            negative_to_positive_ratio=float(discovery_raw.get("negative_to_positive_ratio", 1.0)),
            search_max_batches=(
                int(discovery_raw["search_max_batches"])
                if discovery_raw.get("search_max_batches") is not None
                else None
            ),
            probe_epochs=int(discovery_raw.get("probe_epochs", 25)),
            probe_learning_rate=float(discovery_raw.get("probe_learning_rate", 1.0e-2)),
            top_k_candidates=int(discovery_raw.get("top_k_candidates", 32)),
            candidate_session_balance_fraction=float(
                discovery_raw.get("candidate_session_balance_fraction", 0.2)
            ),
            min_candidate_score=float(discovery_raw.get("min_candidate_score", 0.0)),
            min_cluster_size=int(discovery_raw.get("min_cluster_size", 2)),
            stability_rounds=int(discovery_raw.get("stability_rounds", 4)),
            shuffle_seed=int(discovery_raw.get("shuffle_seed", 17)),
        ),
        config_path=config_path,
    )
    validate_experiment_config(cfg)
    return cfg


def validate_experiment_config(config: ExperimentConfig) -> None:
    if config.data_runtime.bin_width_ms <= 0:
        raise ValueError("data_runtime.bin_width_ms must be > 0")
    if config.data_runtime.context_bins <= 0:
        raise ValueError("data_runtime.context_bins must be > 0")
    if config.data_runtime.patch_bins <= 0:
        raise ValueError("data_runtime.patch_bins must be > 0")
    if config.data_runtime.context_bins % config.data_runtime.patch_bins != 0:
        raise ValueError("data_runtime.context_bins must be divisible by data_runtime.patch_bins")
    if config.data_runtime.padding_strategy != "mask":
        raise ValueError("Only mask padding_strategy is currently supported")
    if config.model.d_model <= 0:
        raise ValueError("model.d_model must be > 0")
    if config.model.num_heads <= 0:
        raise ValueError("model.num_heads must be > 0")
    if config.model.d_model % config.model.num_heads != 0:
        raise ValueError("model.d_model must be divisible by model.num_heads")
    if config.model.temporal_layers < 1:
        raise ValueError("model.temporal_layers must be >= 1")
    if config.model.spatial_layers < 1:
        raise ValueError("model.spatial_layers must be >= 1")
    if not 0.0 <= config.model.dropout < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")
    if config.objective.predictive_target_type not in {"delta", "next_patch_counts"}:
        raise ValueError("objective.predictive_target_type must be 'delta' or 'next_patch_counts'")
    if config.objective.continuation_baseline_type not in {"previous_patch", "zeros"}:
        raise ValueError("objective.continuation_baseline_type must be 'previous_patch' or 'zeros'")
    if config.objective.predictive_loss != "mse":
        raise ValueError("Only mse predictive_loss is currently supported")
    if config.objective.reconstruction_loss != "mse":
        raise ValueError("Only mse reconstruction_loss is currently supported")
    if config.objective.reconstruction_weight < 0:
        raise ValueError("objective.reconstruction_weight must be >= 0")
    if config.optimization.learning_rate <= 0:
        raise ValueError("optimization.learning_rate must be > 0")
    if config.optimization.weight_decay < 0:
        raise ValueError("optimization.weight_decay must be >= 0")
    if config.optimization.batch_size < 1:
        raise ValueError("optimization.batch_size must be >= 1")
    if config.optimization.scheduler_type not in {"none", "cosine"}:
        raise ValueError("optimization.scheduler_type must be 'none' or 'cosine'")
    if config.optimization.scheduler_warmup_steps < 0:
        raise ValueError("optimization.scheduler_warmup_steps must be >= 0")
    if config.training.num_epochs < 1:
        raise ValueError("training.num_epochs must be >= 1")
    if config.training.train_steps_per_epoch < 1:
        raise ValueError("training.train_steps_per_epoch must be >= 1")
    if config.training.validation_steps < 1:
        raise ValueError("training.validation_steps must be >= 1")
    if config.training.checkpoint_every_epochs < 1:
        raise ValueError("training.checkpoint_every_epochs must be >= 1")
    if config.training.evaluate_every_epochs < 1:
        raise ValueError("training.evaluate_every_epochs must be >= 1")
    if config.training.log_every_steps < 1:
        raise ValueError("training.log_every_steps must be >= 1")
    if config.training.dataloader_workers != 0:
        raise ValueError(
            "training.dataloader_workers is not implemented by the current in-process sampler path; set it to 0."
        )
    if config.dataset_selection.split_primary_axis not in {None, "subject", "session"}:
        raise ValueError("dataset_selection.split_primary_axis must be 'subject', 'session', or null")
    for name, value in (
        ("dataset_selection.train_fraction", config.dataset_selection.train_fraction),
        ("dataset_selection.valid_fraction", config.dataset_selection.valid_fraction),
        ("dataset_selection.discovery_fraction", config.dataset_selection.discovery_fraction),
        ("dataset_selection.test_fraction", config.dataset_selection.test_fraction),
    ):
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if any(
        value is not None
        for value in (
            config.dataset_selection.train_fraction,
            config.dataset_selection.valid_fraction,
            config.dataset_selection.discovery_fraction,
            config.dataset_selection.test_fraction,
        )
    ):
        total = sum(
            value or 0.0
            for value in (
                config.dataset_selection.train_fraction,
                config.dataset_selection.valid_fraction,
                config.dataset_selection.discovery_fraction,
                config.dataset_selection.test_fraction,
            )
        )
        if abs(total - 1.0) > 1.0e-6:
            raise ValueError("dataset_selection split fractions must sum to 1.0 when explicitly provided")
    if config.execution.device not in {"cpu", "cuda", "auto"}:
        raise ValueError("execution.device must be one of 'cpu', 'cuda', or 'auto'")
    if config.evaluation.max_batches < 1:
        raise ValueError("evaluation.max_batches must be >= 1")
    if config.runtime_subset is not None:
        if not config.runtime_subset.split_manifest_path.is_file():
            raise FileNotFoundError(
                f"runtime_subset.split_manifest_path not found: {config.runtime_subset.split_manifest_path}"
            )
        if not config.runtime_subset.session_catalog_path.is_file():
            raise FileNotFoundError(
                f"runtime_subset.session_catalog_path not found: {config.runtime_subset.session_catalog_path}"
            )
    if not config.discovery.target_label.strip():
        raise ValueError("discovery.target_label must not be empty")
    if (
        config.discovery.target_label_match_value is not None
        and not config.discovery.target_label_match_value.strip()
    ):
        raise ValueError("discovery.target_label_match_value must not be blank when provided")
    if config.discovery.target_label_mode not in {"auto", "overlap", "onset_within_window", "centered_onset"}:
        raise ValueError(
            "discovery.target_label_mode must be one of 'auto', 'overlap', 'onset_within_window', or "
            "'centered_onset'"
        )
    if config.discovery.max_batches < 1:
        raise ValueError("discovery.max_batches must be >= 1")
    if config.discovery.sampling_strategy not in {"sequential", "label_balanced"}:
        raise ValueError("discovery.sampling_strategy must be 'sequential' or 'label_balanced'")
    if config.discovery.min_positive_windows < 1:
        raise ValueError("discovery.min_positive_windows must be >= 1")
    if config.discovery.negative_to_positive_ratio < 0.0:
        raise ValueError("discovery.negative_to_positive_ratio must be >= 0")
    if config.discovery.search_max_batches is not None and config.discovery.search_max_batches < 1:
        raise ValueError("discovery.search_max_batches must be >= 1 when provided")
    if config.discovery.probe_epochs < 1:
        raise ValueError("discovery.probe_epochs must be >= 1")
    if config.discovery.top_k_candidates < 1:
        raise ValueError("discovery.top_k_candidates must be >= 1")
    if not 0.0 < config.discovery.candidate_session_balance_fraction <= 1.0:
        raise ValueError("discovery.candidate_session_balance_fraction must be in (0, 1]")
    if config.discovery.min_cluster_size < 2:
        raise ValueError("discovery.min_cluster_size must be >= 2")
    if config.discovery.stability_rounds < 1:
        raise ValueError("discovery.stability_rounds must be >= 1")
