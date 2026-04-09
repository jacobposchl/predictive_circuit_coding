from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkTaskSpec:
    name: str
    target_label: str
    target_label_mode: str = "auto"
    target_label_match_value: str | None = None
    include_in_representation: bool = True
    include_in_motifs: bool = True
    optional: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkArmSpec:
    name: str
    feature_family: str
    geometry_mode: str
    use_pca: bool = False
    pca_components: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RepresentationBenchmarkResult:
    task_name: str
    arm_name: str
    status: str
    summary: dict[str, Any]
    summary_json_path: Path
    summary_csv_path: Path
    transform_summary_json_path: Path
    transform_summary_csv_path: Path


@dataclass(frozen=True)
class MotifBenchmarkResult:
    task_name: str
    arm_name: str
    status: str
    summary: dict[str, Any]
    summary_json_path: Path
    summary_csv_path: Path
    cluster_summary_json_path: Path
    cluster_summary_csv_path: Path
    discovery_artifact_path: Path
    transform_summary_json_path: Path
    transform_summary_csv_path: Path


@dataclass(frozen=True)
class PipelineStageState:
    stage_name: str
    status: str
    config_hash: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    created_at_utc: str = ""
    updated_at_utc: str = ""
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PipelineRunManifest:
    run_id: str
    dataset_id: str
    stage_order: tuple[str, ...]
    local_run_root: str
    drive_run_root: str | None
    config_snapshot_path: str
    created_at_utc: str
    updated_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
