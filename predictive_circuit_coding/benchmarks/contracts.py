from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkTaskSpec:
    name: str
    target_label: str
    target_label_mode: str = "auto"
    target_label_match_value: str | None = None
    include_in_motifs: bool = True
    optional: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkArmSpec:
    name: str
    geometry_mode: str
    candidate_geometry_mode: str = "embedding"
    claim_safe: bool = True
    supervision_level: str = "claim_safe"
    use_pca: bool = False
    pca_components: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
