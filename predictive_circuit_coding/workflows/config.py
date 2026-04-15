from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from predictive_circuit_coding.data import build_workspace, load_preparation_config
from predictive_circuit_coding.training import load_experiment_config
from predictive_circuit_coding.utils.notebook import NotebookProgressConfig


_ALLOWED_NOTEBOOK_UI_KEYS = {
    "enabled",
    "leave_pipeline_bar",
    "leave_stage_bars",
    "show_stage_summaries",
    "show_artifact_paths",
    "metric_snapshot_every_n",
}
_COMPUTE_STAGE_NAMES = ("train", "evaluate", "refinement", "alignment_diagnostic")


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


def _has_prepared_sessions(root: Path | None) -> bool:
    return root is not None and any(root.glob("*.h5"))


@dataclass(frozen=True)
class PipelineConfig:
    config_path: Path
    experiment_config_path: Path
    data_config_path: Path
    local_artifact_root: Path
    drive_export_root: Path | None
    source_dataset_root: Path | None
    run_stage_train: bool
    run_stage_evaluate: bool
    run_stage_refinement: bool
    run_stage_alignment_diagnostic: bool
    stage_prepared_sessions_locally: bool
    step_log_every: int
    session_holdout_fraction: float
    session_holdout_seed: int | None
    neighbor_k: int
    debug_retain_intermediates: bool
    motif_task_names: tuple[str, ...] | None
    motif_arm_names: tuple[str, ...] | None
    notebook_ui: NotebookProgressConfig

    def enabled_stages(self) -> tuple[str, ...]:
        return tuple(
            stage_name
            for stage_name, enabled in (
                ("train", self.run_stage_train),
                ("evaluate", self.run_stage_evaluate),
                ("refinement", self.run_stage_refinement),
                ("alignment_diagnostic", self.run_stage_alignment_diagnostic),
            )
            if enabled
        )

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


@dataclass(frozen=True)
class PipelinePreflightIssue:
    severity: str
    message: str


@dataclass(frozen=True)
class PipelinePreflightReport:
    config: PipelineConfig
    path_status: dict[str, bool]
    enabled_stages: tuple[str, ...]
    local_dataset_root: Path | None
    local_prepared_root: Path | None
    source_dataset_root: Path | None
    source_prepared_root: Path | None
    configured_execution_device: str | None
    configured_mixed_precision: bool | None
    requires_cuda: bool
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str | None
    issues: tuple[PipelinePreflightIssue, ...]

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config_dir = config_path.parent

    paths = dict(raw.get("paths") or {})
    stages = dict(raw.get("stages") or {})
    pipeline = dict(raw.get("pipeline") or {})
    tasks = dict(raw.get("tasks") or {})
    arms = dict(raw.get("arms") or {})
    notebook_ui = dict(raw.get("notebook_ui") or {})

    unexpected_notebook_ui = sorted(set(notebook_ui) - _ALLOWED_NOTEBOOK_UI_KEYS)
    if unexpected_notebook_ui:
        raise ValueError(
            "Unsupported notebook_ui keys: "
            f"{unexpected_notebook_ui}. Keep only {sorted(_ALLOWED_NOTEBOOK_UI_KEYS)}."
        )

    experiment_config_path = _resolve_path(config_dir, str(paths.get("experiment_config_path", "")))
    data_config_path = _resolve_path(config_dir, str(paths.get("data_config_path", "")))
    local_artifact_root = _resolve_path(config_dir, str(paths.get("local_artifact_root", "artifacts")))
    drive_export_root = _resolve_path(config_dir, paths.get("drive_export_root"))
    source_dataset_root = _resolve_path(config_dir, paths.get("source_dataset_root"))

    if experiment_config_path is None:
        hint = ""
        if any(key in raw for key in ("dataset_id", "training", "objective", "model")):
            hint = (
                " This looks like an experiment config; pass a pipeline config such as "
                "configs/pcc/pipeline_refined_debug.yaml instead."
            )
        raise ValueError("paths.experiment_config_path is required." + hint)
    if data_config_path is None:
        raise ValueError("paths.data_config_path is required.")
    if local_artifact_root is None:
        raise ValueError("paths.local_artifact_root is required.")

    return PipelineConfig(
        config_path=config_path,
        experiment_config_path=experiment_config_path,
        data_config_path=data_config_path,
        local_artifact_root=local_artifact_root,
        drive_export_root=drive_export_root,
        source_dataset_root=source_dataset_root,
        run_stage_train=bool(stages.get("train", True)),
        run_stage_evaluate=bool(stages.get("evaluate", True)),
        run_stage_refinement=bool(stages.get("refinement", True)),
        run_stage_alignment_diagnostic=bool(stages.get("alignment_diagnostic", False)),
        stage_prepared_sessions_locally=bool(pipeline.get("stage_prepared_sessions_locally", False)),
        step_log_every=int(pipeline.get("step_log_every", 10)),
        session_holdout_fraction=float(pipeline.get("session_holdout_fraction", 0.5)),
        session_holdout_seed=(
            int(pipeline["session_holdout_seed"])
            if pipeline.get("session_holdout_seed") is not None
            else None
        ),
        neighbor_k=int(pipeline.get("neighbor_k", 5)),
        debug_retain_intermediates=bool(pipeline.get("debug_retain_intermediates", False)),
        motif_task_names=_resolve_names(tasks.get("motifs")),
        motif_arm_names=_resolve_names(arms.get("motifs")),
        notebook_ui=NotebookProgressConfig(
            enabled=bool(notebook_ui.get("enabled", True)),
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


def build_pipeline_preflight(config_or_path: PipelineConfig | str | Path) -> PipelinePreflightReport:
    config = (
        config_or_path
        if isinstance(config_or_path, PipelineConfig)
        else load_pipeline_config(config_or_path)
    )
    path_status = {
        "pipeline_config": config.config_path.exists(),
        "experiment_config": config.experiment_config_path.exists(),
        "data_config": config.data_config_path.exists(),
    }
    if config.source_dataset_root is not None:
        path_status["source_dataset_root"] = config.source_dataset_root.exists()

    enabled_stages = config.enabled_stages()
    local_dataset_root: Path | None = None
    local_prepared_root: Path | None = None
    source_prepared_root: Path | None = None
    configured_execution_device: str | None = None
    configured_mixed_precision: bool | None = None
    issues: list[PipelinePreflightIssue] = []

    if path_status["data_config"]:
        prep_config = load_preparation_config(config.data_config_path)
        workspace = build_workspace(prep_config)
        local_dataset_root = workspace.root.resolve()
        local_prepared_root = workspace.brainset_prepared_root.resolve()
        if config.source_dataset_root is not None:
            source_prepared_root = (
                config.source_dataset_root.resolve()
                / "prepared"
                / prep_config.dataset.dataset_id
            )
    if path_status["experiment_config"]:
        experiment_config = load_experiment_config(config.experiment_config_path)
        configured_execution_device = experiment_config.execution.device
        configured_mixed_precision = bool(experiment_config.execution.mixed_precision)

    requires_cuda = any(stage in _COMPUTE_STAGE_NAMES for stage in enabled_stages)
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count())
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_available and cuda_device_count > 0 else None

    if requires_cuda and not cuda_available:
        issues.append(
            PipelinePreflightIssue(
                severity="error",
                message=(
                    "CUDA is unavailable for compute stages. Attach the notebook to a Colab GPU runtime "
                    "before running train/evaluate/refinement/alignment_diagnostic."
                ),
            )
        )

    if requires_cuda and local_prepared_root is not None and not _has_prepared_sessions(local_prepared_root):
        if config.stage_prepared_sessions_locally:
            if config.source_dataset_root is None:
                issues.append(
                    PipelinePreflightIssue(
                        severity="error",
                        message=(
                            "stage_prepared_sessions_locally=true but paths.source_dataset_root is not set and "
                            f"no local prepared sessions were found under {local_prepared_root}."
                        ),
                    )
                )
            elif not config.source_dataset_root.exists():
                issues.append(
                    PipelinePreflightIssue(
                        severity="error",
                        message=f"paths.source_dataset_root does not exist: {config.source_dataset_root.resolve()}",
                    )
                )
            elif source_prepared_root is None or not _has_prepared_sessions(source_prepared_root):
                issues.append(
                    PipelinePreflightIssue(
                        severity="error",
                        message=(
                            "No prepared sessions were found under the configured source dataset root: "
                            f"{source_prepared_root}."
                        ),
                    )
                )
        else:
            issues.append(
                PipelinePreflightIssue(
                    severity="error",
                    message=(
                        "No local prepared sessions were found under "
                        f"{local_prepared_root} and pipeline.stage_prepared_sessions_locally is false."
                    ),
                )
            )

    return PipelinePreflightReport(
        config=config,
        path_status=path_status,
        enabled_stages=enabled_stages,
        local_dataset_root=local_dataset_root,
        local_prepared_root=local_prepared_root,
        source_dataset_root=config.source_dataset_root.resolve() if config.source_dataset_root is not None else None,
        source_prepared_root=source_prepared_root,
        configured_execution_device=configured_execution_device,
        configured_mixed_precision=configured_mixed_precision,
        requires_cuda=requires_cuda,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_device_name=cuda_device_name,
        issues=tuple(issues),
    )


def assert_pipeline_preflight(report: PipelinePreflightReport) -> None:
    if report.ok:
        return
    message = "\n".join(f"- {issue.message}" for issue in report.issues if issue.severity == "error")
    raise RuntimeError(f"Pipeline preflight failed:\n{message}")

