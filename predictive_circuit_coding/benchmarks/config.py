from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from predictive_circuit_coding.utils.notebook import NotebookProgressConfig


def _resolve_path(base_dir: Path, value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    text = str(value)
    candidate = Path(text)
    looks_posix_absolute = text.startswith("/")
    looks_windows_absolute = len(text) >= 3 and text[1] == ":" and text[2] in ("/", "\\")
    if not candidate.is_absolute() and not looks_posix_absolute and not looks_windows_absolute:
        candidate = (base_dir / candidate).resolve()
    return candidate


def _resolve_names(values: list[str] | tuple[str, ...] | None) -> tuple[str, ...] | None:
    if values in (None, []):
        return None
    return tuple(str(value) for value in values)


@dataclass(frozen=True)
class NotebookPipelineConfig:
    config_path: Path
    experiment_config_path: Path
    data_config_path: Path
    local_artifact_root: Path
    drive_export_root: Path | None
    source_dataset_root: Path | None
    run_stage_train: bool
    run_stage_evaluate: bool
    run_stage_representation_benchmark: bool
    run_stage_motif_benchmark: bool
    run_stage_alignment_diagnostic: bool
    run_stage_image_identity_appendix: bool
    stage_prepared_sessions_locally: bool
    step_log_every: int
    pca_components: int
    session_holdout_fraction: float
    session_holdout_seed: int | None
    neighbor_k: int
    debug_retain_intermediates: bool
    image_target_name: str | None
    image_target_names: tuple[str, ...] | None
    image_target_names_auto: bool
    representation_task_names: tuple[str, ...] | None
    motif_task_names: tuple[str, ...] | None
    representation_arm_names: tuple[str, ...] | None
    motif_arm_names: tuple[str, ...] | None
    notebook_ui: NotebookProgressConfig

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "config_path",
            "experiment_config_path",
            "data_config_path",
            "local_artifact_root",
            "drive_export_root",
            "source_dataset_root",
        ):
            value = payload.get(key)
            if value is not None:
                payload[key] = str(value)
        return payload


def load_notebook_pipeline_config(path: str | Path) -> NotebookPipelineConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config_dir = config_path.parent

    paths = dict(raw.get("paths") or {})
    stages = dict(raw.get("stages") or {})
    pipeline = dict(raw.get("pipeline") or {})
    tasks = dict(raw.get("tasks") or {})
    arms = dict(raw.get("arms") or {})
    notebook_ui = dict(raw.get("notebook_ui") or {})

    experiment_config_path = _resolve_path(config_dir, str(paths.get("experiment_config_path", "")))
    data_config_path = _resolve_path(config_dir, str(paths.get("data_config_path", "")))
    local_artifact_root = _resolve_path(config_dir, str(paths.get("local_artifact_root", "artifacts")))
    drive_export_root = _resolve_path(config_dir, paths.get("drive_export_root"))
    source_dataset_root = _resolve_path(config_dir, paths.get("source_dataset_root"))
    if experiment_config_path is None:
        raise ValueError("paths.experiment_config_path is required")
    if data_config_path is None:
        raise ValueError("paths.data_config_path is required")
    if local_artifact_root is None:
        raise ValueError("paths.local_artifact_root is required")

    return NotebookPipelineConfig(
        config_path=config_path,
        experiment_config_path=experiment_config_path,
        data_config_path=data_config_path,
        local_artifact_root=local_artifact_root,
        drive_export_root=drive_export_root,
        source_dataset_root=source_dataset_root,
        run_stage_train=bool(stages.get("train", True)),
        run_stage_evaluate=bool(stages.get("evaluate", True)),
        run_stage_representation_benchmark=bool(stages.get("representation_benchmark", True)),
        run_stage_motif_benchmark=bool(stages.get("motif_benchmark", True)),
        run_stage_alignment_diagnostic=bool(stages.get("alignment_diagnostic", False)),
        run_stage_image_identity_appendix=bool(stages.get("image_identity_appendix", False)),
        stage_prepared_sessions_locally=bool(pipeline.get("stage_prepared_sessions_locally", False)),
        step_log_every=int(pipeline.get("step_log_every", 10)),
        pca_components=int(pipeline.get("pca_components", 64)),
        session_holdout_fraction=float(pipeline.get("session_holdout_fraction", 0.5)),
        session_holdout_seed=(
            int(pipeline["session_holdout_seed"])
            if pipeline.get("session_holdout_seed") is not None
            else None
        ),
        neighbor_k=int(pipeline.get("neighbor_k", 5)),
        debug_retain_intermediates=bool(pipeline.get("debug_retain_intermediates", False)),
        image_target_name=(
            str(tasks["image_target_name"])
            if tasks.get("image_target_name") not in (None, "")
            else None
        ),
        image_target_names=(
            None
            if tasks.get("image_target_names") in (None, "", "auto")
            else _resolve_names(tasks.get("image_target_names"))
        ),
        image_target_names_auto=(str(tasks.get("image_target_names", "")).strip().lower() == "auto"),
        representation_task_names=_resolve_names(tasks.get("representation")),
        motif_task_names=_resolve_names(tasks.get("motifs")),
        representation_arm_names=_resolve_names(arms.get("representation")),
        motif_arm_names=_resolve_names(arms.get("motifs")),
        notebook_ui=NotebookProgressConfig(
            enabled=bool(notebook_ui.get("enabled", True)),
            progress_backend=str(notebook_ui.get("progress_backend", "tqdm")),
            mode=str(notebook_ui.get("mode", "clean_dashboard")),
            log_mode=str(notebook_ui.get("log_mode", "failures_only")),
            leave_pipeline_bar=bool(notebook_ui.get("leave_pipeline_bar", True)),
            leave_stage_bars=bool(notebook_ui.get("leave_stage_bars", False)),
            show_stage_summaries=bool(notebook_ui.get("show_stage_summaries", True)),
            show_artifact_paths=str(notebook_ui.get("show_artifact_paths", "compact")),
            metric_snapshot_every_n=(
                int(notebook_ui["metric_snapshot_every_n"])
                if notebook_ui.get("metric_snapshot_every_n") is not None
                else None
            ),
        ),
    )
