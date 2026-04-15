from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PIPELINE_STAGE_ORDER = ("train", "evaluate", "refinement", "alignment_diagnostic", "final_reports")


@dataclass(frozen=True)
class PipelinePaths:
    run_id: str
    local_run_root: Path
    drive_run_root: Path | None
    train_root: Path
    evaluation_root: Path
    refinement_root: Path
    diagnostics_root: Path
    reports_root: Path
    pipeline_root: Path
    runtime_experiment_config_path: Path
    pipeline_config_snapshot_path: Path
    pipeline_manifest_path: Path
    pipeline_state_path: Path


@dataclass(frozen=True)
class PipelineRunResult:
    run_id: str
    local_run_root: Path
    drive_run_root: Path | None
    runtime_experiment_config_path: Path
    checkpoint_path: Path
    training_summary_path: Path
    training_history_json_path: Path | None
    training_history_csv_path: Path | None
    evaluation_summary_paths: tuple[Path, ...]
    refinement_summary_json_path: Path
    refinement_summary_csv_path: Path
    final_summary_json_path: Path
    final_summary_csv_path: Path
    alignment_summary_json_path: Path | None
    alignment_summary_csv_path: Path | None
    pipeline_manifest_path: Path
    pipeline_state_path: Path
