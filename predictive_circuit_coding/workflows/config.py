from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from predictive_circuit_coding.data import build_workspace, load_preparation_config
from predictive_circuit_coding.training import load_experiment_config
from predictive_circuit_coding.utils.notebook_progress import NotebookProgressConfig


_ALLOWED_TOP_LEVEL_KEYS = {
    "paths",
    "stages",
    "pipeline",
    "notebook_ui",
    "tasks",
    "arms",
}
_ALLOWED_PATH_KEYS = {
    "experiment_config_path",
    "data_config_path",
    "local_artifact_root",
    "drive_export_root",
    "source_dataset_root",
}
_ALLOWED_STAGE_KEYS = {
    "train",
    "evaluate",
    "refinement",
    "alignment_diagnostic",
}
_ALLOWED_PIPELINE_KEYS = {
    "stage_prepared_sessions_locally",
    "step_log_every",
    "session_holdout_fraction",
    "session_holdout_seed",
    "debug_retain_intermediates",
}
_ALLOWED_NOTEBOOK_UI_KEYS = {
    "enabled",
    "leave_pipeline_bar",
    "leave_stage_bars",
    "show_stage_summaries",
    "show_artifact_paths",
    "metric_snapshot_every_n",
}
_ALLOWED_TASK_KEYS = {"motifs"}
_ALLOWED_ARM_KEYS = {"motifs"}
_ALLOWED_ARTIFACT_PATH_MODES = {"compact", "hidden"}
_COMPUTE_STAGE_NAMES = ("train", "evaluate", "refinement", "alignment_diagnostic")
_EXPERIMENT_CONFIG_MARKER_KEYS = {"dataset_id", "training", "objective", "model"}


def _ensure_mapping(section_name: str, value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{section_name} must be a mapping.")
    return dict(value)


def _reject_unexpected_keys(section_name: str, payload: dict[str, Any], allowed_keys: set[str]) -> None:
    unexpected = sorted(set(payload) - allowed_keys)
    if unexpected:
        raise ValueError(
            f"Unsupported {section_name} keys: {unexpected}. Keep only {sorted(allowed_keys)}."
        )


def _resolve_path(base_dir: Path, value: Any, *, key_name: str) -> Path | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key_name} must be a string path or null.")
    text = value
    candidate = Path(text)
    looks_posix_absolute = text.startswith("/")
    looks_windows_absolute = len(text) >= 3 and text[1] == ":" and text[2] in ("/", "\\")
    if not candidate.is_absolute() and not looks_posix_absolute and not looks_windows_absolute:
        candidate = (base_dir / candidate).resolve()
    return candidate


def _resolve_names(values: Any, *, key_name: str) -> tuple[str, ...] | None:
    if values in (None, []):
        return None
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{key_name} must be a list of strings or null.")
    resolved: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key_name}[{index}] must be a non-empty string.")
        resolved.append(value.strip())
    return tuple(resolved) or None


def _has_prepared_sessions(root: Path | None) -> bool:
    return root is not None and any(root.glob("*.h5"))


def _read_bool(payload: dict[str, Any], key_name: str, *, default: bool) -> bool:
    if key_name not in payload:
        return bool(default)
    value = payload[key_name]
    if isinstance(value, bool):
        return value
    raise ValueError(f"{key_name} must be true or false.")


def _read_int(
    payload: dict[str, Any],
    key_name: str,
    *,
    default: int,
    minimum: int | None = None,
) -> int:
    if key_name not in payload:
        value = default
    else:
        value = payload[key_name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key_name} must be an integer.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{key_name} must be at least {minimum}.")
    return int(value)


def _read_optional_int(
    payload: dict[str, Any],
    key_name: str,
    *,
    minimum: int | None = None,
) -> int | None:
    if key_name not in payload or payload[key_name] is None:
        return None
    value = payload[key_name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key_name} must be an integer or null.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{key_name} must be at least {minimum}.")
    return int(value)


def _read_float(
    payload: dict[str, Any],
    key_name: str,
    *,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
    inclusive_minimum: bool = True,
    inclusive_maximum: bool = True,
) -> float:
    if key_name not in payload:
        value = default
    else:
        value = payload[key_name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key_name} must be a number.")
    resolved = float(value)
    if minimum is not None:
        if inclusive_minimum and resolved < minimum:
            raise ValueError(f"{key_name} must be at least {minimum}.")
        if not inclusive_minimum and resolved <= minimum:
            raise ValueError(f"{key_name} must be greater than {minimum}.")
    if maximum is not None:
        if inclusive_maximum and resolved > maximum:
            raise ValueError(f"{key_name} must be at most {maximum}.")
        if not inclusive_maximum and resolved >= maximum:
            raise ValueError(f"{key_name} must be less than {maximum}.")
    return resolved


def _read_artifact_path_mode(payload: dict[str, Any], key_name: str, *, default: str) -> str:
    if key_name not in payload:
        return default
    value = payload[key_name]
    if not isinstance(value, str):
        raise ValueError(
            f"{key_name} must be one of {sorted(_ALLOWED_ARTIFACT_PATH_MODES)}."
        )
    normalized = value.strip()
    if normalized not in _ALLOWED_ARTIFACT_PATH_MODES:
        raise ValueError(
            f"{key_name} must be one of {sorted(_ALLOWED_ARTIFACT_PATH_MODES)}."
        )
    return normalized


def _add_issue(
    issues: list["PipelinePreflightIssue"],
    *,
    severity: str,
    message: str,
) -> None:
    issues.append(PipelinePreflightIssue(severity=severity, message=message))


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
    if not isinstance(raw, dict):
        raise ValueError("Pipeline config must be a mapping.")
    if "paths" not in raw and any(key in raw for key in _EXPERIMENT_CONFIG_MARKER_KEYS):
        raise ValueError(
            "This looks like an experiment config; pass a pipeline config such as "
            "configs/pcc/pipeline_refined_debug.yaml instead."
        )
    _reject_unexpected_keys("top-level pipeline config", raw, _ALLOWED_TOP_LEVEL_KEYS)
    config_dir = config_path.parent

    paths = _ensure_mapping("paths", raw.get("paths"))
    stages = _ensure_mapping("stages", raw.get("stages"))
    pipeline = _ensure_mapping("pipeline", raw.get("pipeline"))
    tasks = _ensure_mapping("tasks", raw.get("tasks"))
    arms = _ensure_mapping("arms", raw.get("arms"))
    notebook_ui = _ensure_mapping("notebook_ui", raw.get("notebook_ui"))

    _reject_unexpected_keys("paths", paths, _ALLOWED_PATH_KEYS)
    _reject_unexpected_keys("stages", stages, _ALLOWED_STAGE_KEYS)
    _reject_unexpected_keys("pipeline", pipeline, _ALLOWED_PIPELINE_KEYS)
    _reject_unexpected_keys("tasks", tasks, _ALLOWED_TASK_KEYS)
    _reject_unexpected_keys("arms", arms, _ALLOWED_ARM_KEYS)
    _reject_unexpected_keys("notebook_ui", notebook_ui, _ALLOWED_NOTEBOOK_UI_KEYS)

    experiment_config_path = _resolve_path(
        config_dir,
        paths.get("experiment_config_path", ""),
        key_name="paths.experiment_config_path",
    )
    data_config_path = _resolve_path(
        config_dir,
        paths.get("data_config_path", ""),
        key_name="paths.data_config_path",
    )
    local_artifact_root = _resolve_path(
        config_dir,
        paths.get("local_artifact_root", "artifacts"),
        key_name="paths.local_artifact_root",
    )
    drive_export_root = _resolve_path(
        config_dir,
        paths.get("drive_export_root"),
        key_name="paths.drive_export_root",
    )
    source_dataset_root = _resolve_path(
        config_dir,
        paths.get("source_dataset_root"),
        key_name="paths.source_dataset_root",
    )

    if experiment_config_path is None:
        raise ValueError("paths.experiment_config_path is required.")
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
        run_stage_train=_read_bool(stages, "train", default=True),
        run_stage_evaluate=_read_bool(stages, "evaluate", default=True),
        run_stage_refinement=_read_bool(stages, "refinement", default=True),
        run_stage_alignment_diagnostic=_read_bool(stages, "alignment_diagnostic", default=False),
        stage_prepared_sessions_locally=_read_bool(pipeline, "stage_prepared_sessions_locally", default=False),
        step_log_every=_read_int(pipeline, "step_log_every", default=10, minimum=1),
        session_holdout_fraction=_read_float(
            pipeline,
            "session_holdout_fraction",
            default=0.5,
            minimum=0.0,
            maximum=1.0,
            inclusive_minimum=False,
            inclusive_maximum=False,
        ),
        session_holdout_seed=_read_optional_int(pipeline, "session_holdout_seed", minimum=0),
        debug_retain_intermediates=_read_bool(pipeline, "debug_retain_intermediates", default=False),
        motif_task_names=_resolve_names(tasks.get("motifs"), key_name="tasks.motifs"),
        motif_arm_names=_resolve_names(arms.get("motifs"), key_name="arms.motifs"),
        notebook_ui=NotebookProgressConfig(
            enabled=_read_bool(notebook_ui, "enabled", default=True),
            leave_pipeline_bar=_read_bool(notebook_ui, "leave_pipeline_bar", default=True),
            leave_stage_bars=_read_bool(notebook_ui, "leave_stage_bars", default=False),
            show_stage_summaries=_read_bool(notebook_ui, "show_stage_summaries", default=True),
            show_artifact_paths=_read_artifact_path_mode(
                notebook_ui,
                "show_artifact_paths",
                default="compact",
            ),
            metric_snapshot_every_n=_read_optional_int(
                notebook_ui,
                "metric_snapshot_every_n",
                minimum=1,
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

    if not path_status["pipeline_config"]:
        _add_issue(
            issues,
            severity="error",
            message=f"Pipeline config not found: {config.config_path}",
        )
    if not path_status["experiment_config"]:
        _add_issue(
            issues,
            severity="error",
            message=f"Experiment config not found: {config.experiment_config_path}",
        )
    if not path_status["data_config"]:
        _add_issue(
            issues,
            severity="error",
            message=f"Data config not found: {config.data_config_path}",
        )

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

    if requires_cuda and configured_execution_device is not None:
        normalized_device = configured_execution_device.strip().lower()
        if normalized_device == "cpu":
            _add_issue(
                issues,
                severity="error",
                message=(
                    "execution.device is set to cpu, but compute stages are enabled. "
                    "Use auto or a CUDA device for train/evaluate/refinement/alignment_diagnostic."
                ),
            )
        elif normalized_device not in {"auto", "cuda"} and not normalized_device.startswith("cuda:"):
            _add_issue(
                issues,
                severity="warning",
                message=(
                    f"execution.device is set to {configured_execution_device!r}. "
                    "This workflow is validated primarily for auto or CUDA devices."
                ),
            )

    if requires_cuda and not cuda_available:
        _add_issue(
            issues,
            severity="error",
            message=(
                "CUDA is unavailable for compute stages. Attach the notebook to a Colab GPU runtime "
                "before running train/evaluate/refinement/alignment_diagnostic."
            ),
        )

    if requires_cuda and local_prepared_root is not None and not _has_prepared_sessions(local_prepared_root):
        if config.stage_prepared_sessions_locally:
            if config.source_dataset_root is None:
                _add_issue(
                    issues,
                    severity="error",
                    message=(
                        "stage_prepared_sessions_locally=true but paths.source_dataset_root is not set and "
                        f"no local prepared sessions were found under {local_prepared_root}."
                    ),
                )
            elif not config.source_dataset_root.exists():
                _add_issue(
                    issues,
                    severity="error",
                    message=f"paths.source_dataset_root does not exist: {config.source_dataset_root.resolve()}",
                )
            elif source_prepared_root is None or not _has_prepared_sessions(source_prepared_root):
                _add_issue(
                    issues,
                    severity="error",
                    message=(
                        "No prepared sessions were found under the configured source dataset root: "
                        f"{source_prepared_root}."
                    ),
                )
        else:
            _add_issue(
                issues,
                severity="error",
                message=(
                    "No local prepared sessions were found under "
                    f"{local_prepared_root} and pipeline.stage_prepared_sessions_locally is false."
                ),
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
