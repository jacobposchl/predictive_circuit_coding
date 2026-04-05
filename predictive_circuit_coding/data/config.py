from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DatasetPathsConfig:
    dataset_id: str
    source_name: str
    workspace_root: Path
    raw_subdir: str
    prepared_subdir: str
    manifests_subdir: str
    splits_subdir: str
    logs_subdir: str
    prepared_session_subdir: str
    session_manifest_name: str
    split_manifest_name: str


@dataclass(frozen=True)
class PreparationInputConfig:
    session_table_format: str
    session_id_field: str
    subject_id_field: str
    raw_path_field: str
    duration_field: str
    n_units_field: str
    brain_regions_field: str
    trial_count_field: str
    recording_id_template: str


@dataclass(frozen=True)
class SplitPlanningConfig:
    seed: int
    primary_axis: str
    train_fraction: float
    valid_fraction: float
    discovery_fraction: float
    test_fraction: float


@dataclass(frozen=True)
class RuntimeRulesConfig:
    local_cpu_only: bool
    training_surface: str


@dataclass(frozen=True)
class BrainsetsPipelineConfig:
    local_pipeline_path: Path
    runner_cores: int
    use_active_environment: bool
    processed_only_upload: bool
    keep_raw_cache: bool
    default_session_ids_file: Path | None
    default_max_sessions: int | None


@dataclass(frozen=True)
class AllenSdkConfig:
    cache_root: Path | None
    cleanup_raw_after_processing: bool


@dataclass(frozen=True)
class UnitFilteringConfig:
    filter_by_validity: bool
    filter_out_of_brain_units: bool
    amplitude_cutoff_maximum: float | None
    presence_ratio_minimum: float | None
    isi_violations_maximum: float | None


@dataclass(frozen=True)
class DataPreparationConfig:
    dataset: DatasetPathsConfig
    preparation: PreparationInputConfig
    splits: SplitPlanningConfig
    runtime: RuntimeRulesConfig
    brainsets_pipeline: BrainsetsPipelineConfig
    allen_sdk: AllenSdkConfig
    unit_filtering: UnitFilteringConfig
    config_path: Path


def _resolve_optional_path(base_dir: Path, value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_preparation_config(path: str | Path) -> DataPreparationConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config_dir = config_path.parent
    dataset = raw["dataset"]
    workspace_root = (config_dir.parent.parent / dataset["workspace_root"]).resolve()
    brainsets_pipeline = raw.get("brainsets_pipeline", {})
    allen_sdk = raw.get("allen_sdk", {})
    cfg = DataPreparationConfig(
        dataset=DatasetPathsConfig(
            dataset_id=str(dataset["dataset_id"]),
            source_name=str(dataset["source_name"]),
            workspace_root=workspace_root,
            raw_subdir=str(dataset["raw_subdir"]),
            prepared_subdir=str(dataset["prepared_subdir"]),
            manifests_subdir=str(dataset["manifests_subdir"]),
            splits_subdir=str(dataset["splits_subdir"]),
            logs_subdir=str(dataset["logs_subdir"]),
            prepared_session_subdir=str(dataset["prepared_session_subdir"]),
            session_manifest_name=str(dataset["session_manifest_name"]),
            split_manifest_name=str(dataset["split_manifest_name"]),
        ),
        preparation=PreparationInputConfig(
            session_table_format=str(raw["preparation"]["session_table_format"]),
            session_id_field=str(raw["preparation"]["session_id_field"]),
            subject_id_field=str(raw["preparation"]["subject_id_field"]),
            raw_path_field=str(raw["preparation"]["raw_path_field"]),
            duration_field=str(raw["preparation"]["duration_field"]),
            n_units_field=str(raw["preparation"]["n_units_field"]),
            brain_regions_field=str(raw["preparation"]["brain_regions_field"]),
            trial_count_field=str(raw["preparation"]["trial_count_field"]),
            recording_id_template=str(raw["preparation"]["recording_id_template"]),
        ),
        splits=SplitPlanningConfig(
            seed=int(raw["splits"]["seed"]),
            primary_axis=str(raw["splits"]["primary_axis"]),
            train_fraction=float(raw["splits"]["train_fraction"]),
            valid_fraction=float(raw["splits"]["valid_fraction"]),
            discovery_fraction=float(raw["splits"]["discovery_fraction"]),
            test_fraction=float(raw["splits"]["test_fraction"]),
        ),
        runtime=RuntimeRulesConfig(
            local_cpu_only=bool(raw["runtime"]["local_cpu_only"]),
            training_surface=str(raw["runtime"]["training_surface"]),
        ),
        brainsets_pipeline=BrainsetsPipelineConfig(
            local_pipeline_path=_resolve_optional_path(
                config_dir,
                str(
                    brainsets_pipeline.get(
                        "local_pipeline_path",
                        "brainsets_local_pipelines/allen_visual_behavior_neuropixels/pipeline.py",
                    )
                ),
            )
            or config_dir,
            runner_cores=int(brainsets_pipeline.get("runner_cores", 4)),
            use_active_environment=bool(brainsets_pipeline.get("use_active_environment", True)),
            processed_only_upload=bool(brainsets_pipeline.get("processed_only_upload", True)),
            keep_raw_cache=bool(brainsets_pipeline.get("keep_raw_cache", True)),
            default_session_ids_file=_resolve_optional_path(
                config_dir,
                brainsets_pipeline.get("default_session_ids_file"),
            ),
            default_max_sessions=(
                int(brainsets_pipeline["default_max_sessions"])
                if brainsets_pipeline.get("default_max_sessions") is not None
                else None
            ),
        ),
        allen_sdk=AllenSdkConfig(
            cache_root=_resolve_optional_path(
                config_dir,
                allen_sdk.get("cache_root"),
            ),
            cleanup_raw_after_processing=bool(allen_sdk.get("cleanup_raw_after_processing", False)),
        ),
        unit_filtering=UnitFilteringConfig(
            filter_by_validity=bool(raw.get("unit_filtering", {}).get("filter_by_validity", True)),
            filter_out_of_brain_units=bool(raw.get("unit_filtering", {}).get("filter_out_of_brain_units", True)),
            amplitude_cutoff_maximum=(
                float(raw["unit_filtering"]["amplitude_cutoff_maximum"])
                if raw.get("unit_filtering", {}).get("amplitude_cutoff_maximum") is not None
                else 0.1
            ),
            presence_ratio_minimum=(
                float(raw["unit_filtering"]["presence_ratio_minimum"])
                if raw.get("unit_filtering", {}).get("presence_ratio_minimum") is not None
                else 0.95
            ),
            isi_violations_maximum=(
                float(raw["unit_filtering"]["isi_violations_maximum"])
                if raw.get("unit_filtering", {}).get("isi_violations_maximum") is not None
                else 0.5
            ),
        ),
        config_path=config_path,
    )
    total = cfg.splits.train_fraction + cfg.splits.valid_fraction + cfg.splits.discovery_fraction + cfg.splits.test_fraction
    if abs(total - 1.0) > 1.0e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {total:.6f}")
    if cfg.splits.primary_axis not in {"subject", "session"}:
        raise ValueError("splits.primary_axis must be 'subject' or 'session'")
    if cfg.brainsets_pipeline.runner_cores < 1:
        raise ValueError("brainsets_pipeline.runner_cores must be >= 1")
    if (
        cfg.unit_filtering.amplitude_cutoff_maximum is not None
        and cfg.unit_filtering.amplitude_cutoff_maximum < 0
    ):
        raise ValueError("unit_filtering.amplitude_cutoff_maximum must be >= 0")
    if (
        cfg.unit_filtering.presence_ratio_minimum is not None
        and not 0.0 <= cfg.unit_filtering.presence_ratio_minimum <= 1.0
    ):
        raise ValueError("unit_filtering.presence_ratio_minimum must be in [0, 1]")
    if (
        cfg.unit_filtering.isi_violations_maximum is not None
        and cfg.unit_filtering.isi_violations_maximum < 0
    ):
        raise ValueError("unit_filtering.isi_violations_maximum must be >= 0")
    return cfg
