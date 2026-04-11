from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Any

from predictive_circuit_coding.benchmarks.config import NotebookPipelineConfig, load_notebook_pipeline_config
from predictive_circuit_coding.benchmarks.run import (
    _task_config,
    default_benchmark_task_specs,
    default_motif_arm_specs,
    default_representation_arm_specs,
)
from predictive_circuit_coding.data import resolve_runtime_dataset_view
from predictive_circuit_coding.discovery.run import prepare_discovery_collection
from predictive_circuit_coding.training import ExperimentConfig, load_experiment_config
from predictive_circuit_coding.windowing.dataset import split_session_ids
from predictive_circuit_coding.utils.notebook import collect_notebook_target_value_counts


_FULL_REPRESENTATION_ARMS = {
    "count_patch_mean_raw",
    "untrained_encoder_raw",
    "encoder_raw",
    "encoder_whitened",
}
_FULL_MOTIF_ARMS = {
    "untrained_encoder_raw",
    "encoder_raw",
    "encoder_whitened",
}
_ALLOWED_FULL_TASKS = {"stimulus_change", "trials_go", "stimulus_omitted"}
_REQUIRED_FULL_TASKS = {"stimulus_change", "trials_go"}
_MIN_POSITIVE_SESSION_COUNT = 2


@dataclass(frozen=True)
class VerificationIssue:
    gate: str
    severity: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskCoverageRow:
    task_name: str
    target_label: str
    split_name: str
    status: str
    total_scanned_windows: int
    positive_window_count: int
    negative_window_count: int
    selected_positive_count: int
    selected_negative_count: int
    selected_window_count: int
    positive_session_count: int
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FullRunVerificationResult:
    status: str
    pipeline_config_path: str
    experiment_config_path: str
    data_config_path: str
    training_num_epochs: int
    training_variant_name: str
    split_counts: dict[str, int]
    issues: tuple[VerificationIssue, ...]
    coverage_rows: tuple[TaskCoverageRow, ...]
    summary_json_path: Path
    coverage_csv_path: Path

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["summary_json_path"] = str(self.summary_json_path)
        payload["coverage_csv_path"] = str(self.coverage_csv_path)
        return payload


def _add_issue(issues: list[VerificationIssue], gate: str, message: str, *, severity: str = "error") -> None:
    issues.append(VerificationIssue(gate=gate, severity=severity, message=message))


def _check_full_config(
    *,
    pipeline_config: NotebookPipelineConfig,
    experiment_config: ExperimentConfig,
    issues: list[VerificationIssue],
) -> None:
    if pipeline_config.experiment_config_path.name != "predictive_circuit_coding_cross_session_aug_full.yaml":
        _add_issue(
            issues,
            "full_config",
            "Pipeline does not reference predictive_circuit_coding_cross_session_aug_full.yaml.",
        )
    if int(experiment_config.training.num_epochs) < 50:
        _add_issue(
            issues,
            "full_config",
            f"training.num_epochs must be >= 50 for the full augmented run; got {experiment_config.training.num_epochs}.",
        )
    aug = experiment_config.objective.cross_session_aug
    if not aug.enabled:
        _add_issue(issues, "full_config", "objective.cross_session_aug.enabled must be true.")
    if aug.training_variant_name != "cross_session_aug_full":
        _add_issue(
            issues,
            "full_config",
            f"training_variant_name must be cross_session_aug_full; got {aug.training_variant_name!r}.",
        )
    if experiment_config.training.resume_checkpoint is not None:
        _add_issue(issues, "full_config", "training.resume_checkpoint must be empty for a fresh full run.")
    required_stages = {
        "train": pipeline_config.run_stage_train,
        "evaluate": pipeline_config.run_stage_evaluate,
        "representation_benchmark": pipeline_config.run_stage_representation_benchmark,
        "motif_benchmark": pipeline_config.run_stage_motif_benchmark,
    }
    disabled = [name for name, enabled in required_stages.items() if not enabled]
    if disabled:
        _add_issue(issues, "full_config", f"Required full-run stages are disabled: {disabled}.")
    if pipeline_config.debug_retain_intermediates:
        _add_issue(issues, "full_config", "debug_retain_intermediates must be false for the full run.")
    if pipeline_config.run_stage_image_identity_appendix:
        has_image_targets = bool(pipeline_config.image_target_name) or bool(pipeline_config.image_target_names)
        if not has_image_targets and not pipeline_config.image_target_names_auto:
            _add_issue(
                issues,
                "full_config",
                "image_identity_appendix is enabled but neither tasks.image_target_names nor tasks.image_target_name is configured.",
            )

    representation_arms = set(pipeline_config.representation_arm_names or ())
    motif_arms = set(pipeline_config.motif_arm_names or ())
    if representation_arms != _FULL_REPRESENTATION_ARMS:
        _add_issue(
            issues,
            "full_config",
            f"Representation arms are not the full encoder-baseline set. Expected {sorted(_FULL_REPRESENTATION_ARMS)}, got {sorted(representation_arms)}.",
        )
    if motif_arms != _FULL_MOTIF_ARMS:
        _add_issue(
            issues,
            "full_config",
            f"Motif arms are not the full motif set. Expected {sorted(_FULL_MOTIF_ARMS)}, got {sorted(motif_arms)}.",
        )
    representation_tasks = set(pipeline_config.representation_task_names or ())
    motif_tasks = set(pipeline_config.motif_task_names or ())
    if not _REQUIRED_FULL_TASKS.issubset(representation_tasks):
        _add_issue(
            issues,
            "full_config",
            f"Representation tasks must include {sorted(_REQUIRED_FULL_TASKS)}; got {sorted(representation_tasks)}.",
        )
    if not _REQUIRED_FULL_TASKS.issubset(motif_tasks):
        _add_issue(
            issues,
            "full_config",
            f"Motif tasks must include {sorted(_REQUIRED_FULL_TASKS)}; got {sorted(motif_tasks)}.",
        )
    unknown_representation_tasks = representation_tasks - _ALLOWED_FULL_TASKS
    unknown_motif_tasks = motif_tasks - _ALLOWED_FULL_TASKS
    if unknown_representation_tasks:
        _add_issue(
            issues,
            "full_config",
            f"Representation tasks include unsupported full-run tasks: {sorted(unknown_representation_tasks)}.",
        )
    if unknown_motif_tasks:
        _add_issue(
            issues,
            "full_config",
            f"Motif tasks include unsupported full-run tasks: {sorted(unknown_motif_tasks)}.",
        )


def _split_counts(dataset_view) -> dict[str, int]:
    return {
        split_name: len(split_session_ids(dataset_view.split_manifest, split_name))
        for split_name in ("train", "valid", "discovery", "test")
    }


def _coverage_row_from_plan(*, task_name: str, target_label: str, split_name: str, plan) -> TaskCoverageRow:
    summary = plan.coverage_summary
    positive_session_count = len(summary.sessions_with_positive_windows)
    ok = (
        int(summary.selected_positive_count) > 0
        and int(summary.selected_negative_count) > 0
        and positive_session_count >= _MIN_POSITIVE_SESSION_COUNT
    )
    reason = None
    status = "ok"
    if int(summary.selected_positive_count) <= 0 or int(summary.selected_negative_count) <= 0:
        status = "blocked_missing_class"
        reason = (
            "selected split is missing both classes for this task "
            f"(selected_positive={summary.selected_positive_count}, selected_negative={summary.selected_negative_count})"
        )
    elif positive_session_count < _MIN_POSITIVE_SESSION_COUNT:
        status = "blocked_insufficient_positive_sessions"
        reason = (
            "selected scan has too few sessions with positive windows for a cross-session benchmark "
            f"(positive_session_count={positive_session_count}, required_min={_MIN_POSITIVE_SESSION_COUNT})"
        )
    return TaskCoverageRow(
        task_name=task_name,
        target_label=target_label,
        split_name=split_name,
        status=status,
        total_scanned_windows=int(summary.total_scanned_windows),
        positive_window_count=int(summary.positive_window_count),
        negative_window_count=int(summary.negative_window_count),
        selected_positive_count=int(summary.selected_positive_count),
        selected_negative_count=int(summary.selected_negative_count),
        selected_window_count=int(summary.selected_window_count),
        positive_session_count=positive_session_count,
        failure_reason=reason,
    )


def _run_task_coverage_gate(
    *,
    experiment_config: ExperimentConfig,
    data_config_path: str | Path,
    dataset_view,
    task_specs: tuple,
    issues: list[VerificationIssue],
) -> tuple[TaskCoverageRow, ...]:
    rows: list[TaskCoverageRow] = []
    for task in task_specs:
        task_config = _task_config(experiment_config, task)
        for split_name in (experiment_config.splits.discovery, experiment_config.splits.test):
            try:
                plan = prepare_discovery_collection(
                    experiment_config=task_config,
                    data_config_path=data_config_path,
                    split_name=split_name,
                    dataset_view=dataset_view,
                )
                row = _coverage_row_from_plan(
                    task_name=task.name,
                    target_label=task.target_label,
                    split_name=split_name,
                    plan=plan,
                )
            except Exception as exc:
                row = TaskCoverageRow(
                    task_name=task.name,
                    target_label=task.target_label,
                    split_name=split_name,
                    status="blocked_exception",
                    total_scanned_windows=0,
                    positive_window_count=0,
                    negative_window_count=0,
                    selected_positive_count=0,
                    selected_negative_count=0,
                    selected_window_count=0,
                    positive_session_count=0,
                    failure_reason=str(exc),
                )
            rows.append(row)
            if row.status != "ok":
                _add_issue(
                    issues,
                    "task_coverage",
                    f"{task.name} / {split_name} is blocked: {row.failure_reason}",
                )
    return tuple(rows)


def _write_rows_csv(rows: tuple[TaskCoverageRow, ...], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(TaskCoverageRow.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())
    return path


def verify_full_run_readiness(
    *,
    pipeline_config_path: str | Path,
    output_root: str | Path,
) -> FullRunVerificationResult:
    pipeline_config = load_notebook_pipeline_config(pipeline_config_path)
    experiment_config = load_experiment_config(pipeline_config.experiment_config_path)
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    issues: list[VerificationIssue] = []
    _check_full_config(
        pipeline_config=pipeline_config,
        experiment_config=experiment_config,
        issues=issues,
    )

    dataset_view = resolve_runtime_dataset_view(
        experiment_config=experiment_config,
        data_config_path=pipeline_config.data_config_path,
    )
    counts = _split_counts(dataset_view)
    for split_name in ("train", "valid", "discovery", "test"):
        if counts.get(split_name, 0) <= 0:
            _add_issue(issues, "task_coverage", f"Runtime split '{split_name}' has no sessions.")

    image_target_names = tuple(pipeline_config.image_target_names or ())
    if pipeline_config.run_stage_image_identity_appendix and pipeline_config.image_target_names_auto:
        image_rows = collect_notebook_target_value_counts(
            experiment_config_path=pipeline_config.experiment_config_path,
            data_config_path=pipeline_config.data_config_path,
            split_name=experiment_config.splits.discovery,
            target_label="stimulus_presentations.image_name",
        )
        image_target_names = tuple(str(row["value"]) for row in image_rows)
        if not image_target_names:
            _add_issue(
                issues,
                "image_identity",
                "image_identity_appendix is enabled with image_target_names=auto, but no stimulus_presentations.image_name values were found in the discovery split.",
            )
    if pipeline_config.image_target_name:
        image_target_names = tuple(dict.fromkeys((pipeline_config.image_target_name, *image_target_names)))

    default_specs = default_benchmark_task_specs(
        include_image_identity=pipeline_config.run_stage_image_identity_appendix,
        image_target_name=pipeline_config.image_target_name,
        image_target_names=image_target_names,
    )
    requested_names = set(pipeline_config.representation_task_names or ()) | set(pipeline_config.motif_task_names or ())
    configured_task_specs = tuple(
        spec
        for spec in default_specs
        if spec.name in requested_names
        or (
            pipeline_config.run_stage_image_identity_appendix
            and spec.target_label == "stimulus_presentations.image_name"
            and spec.target_label_match_value is not None
        )
    )
    coverage_rows = _run_task_coverage_gate(
        experiment_config=experiment_config,
        data_config_path=pipeline_config.data_config_path,
        dataset_view=dataset_view,
        task_specs=configured_task_specs,
        issues=issues,
    )

    status = "ok" if not any(issue.severity == "error" for issue in issues) else "blocked"
    summary_json_path = output_root_path / "full_run_verification_summary.json"
    coverage_csv_path = output_root_path / "full_run_task_coverage.csv"
    result = FullRunVerificationResult(
        status=status,
        pipeline_config_path=str(Path(pipeline_config_path).resolve()),
        experiment_config_path=str(pipeline_config.experiment_config_path),
        data_config_path=str(pipeline_config.data_config_path),
        training_num_epochs=int(experiment_config.training.num_epochs),
        training_variant_name=experiment_config.objective.cross_session_aug.training_variant_name,
        split_counts=counts,
        issues=tuple(issues),
        coverage_rows=coverage_rows,
        summary_json_path=summary_json_path,
        coverage_csv_path=coverage_csv_path,
    )
    _write_rows_csv(coverage_rows, coverage_csv_path)
    summary_payload = result.to_dict()
    summary_payload["issues"] = [issue.to_dict() for issue in issues]
    summary_payload["coverage_rows"] = [row.to_dict() for row in coverage_rows]
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return result
