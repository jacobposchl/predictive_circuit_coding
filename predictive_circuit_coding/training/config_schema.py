from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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
class ExperimentIdentityConfig:
    variant_name: str = "refined_core"


@dataclass(frozen=True)
class CountNormalizationConfig:
    mode: str = "none"
    stats_path: Path | None = None


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
    population_token_mode: str = "none"


@dataclass(frozen=True)
class ObjectiveConfig:
    predictive_target_type: str
    continuation_baseline_type: str
    predictive_loss: str
    reconstruction_loss: str
    reconstruction_weight: float
    exclude_final_prediction_patch: bool
    reconstruction_target_mode: str = "raw"


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
    def has_split_overrides(self) -> bool:
        return any(
            (
                self.split_seed is not None,
                self.split_primary_axis is not None,
                self.train_fraction is not None,
                self.valid_fraction is not None,
                self.discovery_fraction is not None,
                self.test_fraction is not None,
            )
        )

    @property
    def is_active(self) -> bool:
        return self.has_split_overrides or any(
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
    pooled_feature_mode: str = "mean_tokens"


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
    experiment: ExperimentIdentityConfig = field(default_factory=ExperimentIdentityConfig)
    count_normalization: CountNormalizationConfig = field(default_factory=CountNormalizationConfig)
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
        if self.count_normalization.stats_path is not None:
            payload["count_normalization"]["stats_path"] = str(self.count_normalization.stats_path)
        if self.runtime_subset is not None:
            payload["runtime_subset"] = {
                "split_manifest_path": str(self.runtime_subset.split_manifest_path),
                "session_catalog_path": str(self.runtime_subset.session_catalog_path),
                "config_dir": str(self.runtime_subset.config_dir),
                "config_name_prefix": self.runtime_subset.config_name_prefix,
            }
        return payload
