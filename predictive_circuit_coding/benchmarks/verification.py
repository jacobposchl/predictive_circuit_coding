from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Any

from predictive_circuit_coding.benchmarks.config import load_notebook_pipeline_config
from predictive_circuit_coding.benchmarks.run import _task_config, default_benchmark_task_specs, default_motif_arm_specs
from predictive_circuit_coding.data import resolve_runtime_dataset_view
from predictive_circuit_coding.discovery.run import prepare_discovery_collection
from predictive_circuit_coding.training import ExperimentConfig, load_experiment_config
from predictive_circuit_coding.windowing.dataset import split_session_ids


_REQUIRED_REFINEMENT_ARMS = {
    "encoder_raw",
    "encoder_token_normalized",
    "encoder_probe_weighted",
    "encoder_aligned_oracle",
}
_ALLOWED_REFINEMENT_TASKS = {"stimulus_change", "trials_go"}
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
class RefinementVerificationResult:
    status: str
    pipeline_config_path: str
    experiment_config_path: str
    data_config_path: str
    variant_name: str
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


def _check_refinement_config(
    *,
    experiment_config: ExperimentConfig,
    pipeline_config,
    issues: list[VerificationIssue],
) -> None:
    if not experiment_config.experiment.variant_name.strip():
        _add_issue(issues, "config", "experiment.variant_name must not be blank.")
    if not pipeline_config.run_stage_refinement:
        _add_issue(issues, "config", "stages.refinement must be enabled.")
    arms = set(pipeline_config.motif_arm_names or [arm.name for arm in default_motif_arm_specs()])
    if arms != _REQUIRED_REFINEMENT_ARMS:
        _add_issue(
            issues,
            "config",
            f"Refinement arms must be {sorted(_REQUIRED_REFINEMENT_ARMS)}, got {sorted(arms)}.",
        )
    tasks = set(pipeline_config.motif_task_names or [task.name for task in default_benchmark_task_specs()])
    unknown = tasks - _ALLOWED_REFINEMENT_TASKS
    if unknown:
        _add_issue(issues, "config", f"Unsupported refinement task names: {sorted(unknown)}.")


def _coverage_row_from_summary(*, task_name: str, target_label: str, split_name: str, summary, status: str, failure_reason: str | None = None) -> TaskCoverageRow:
    sessions = tuple(getattr(summary, "sessions_with_positive_windows", ()) or ())
    return TaskCoverageRow(
        task_name=task_name,
        target_label=target_label,
        split_name=split_name,
        status=status,
        total_scanned_windows=int(getattr(summary, "total_scanned_windows", 0)),
        positive_window_count=int(getattr(summary, "positive_window_count", 0)),
        negative_window_count=int(getattr(summary, "negative_window_count", 0)),
        selected_positive_count=int(getattr(summary, "selected_positive_count", 0)),
        selected_negative_count=int(getattr(summary, "selected_negative_count", 0)),
        selected_window_count=int(getattr(summary, "selected_window_count", 0)),
        positive_session_count=len(sessions),
        failure_reason=failure_reason,
    )


def _verify_task_coverage(
    *,
    experiment_config: ExperimentConfig,
    data_config_path: str | Path,
    issues: list[VerificationIssue],
) -> tuple[TaskCoverageRow, ...]:
    dataset_view = resolve_runtime_dataset_view(experiment_config=experiment_config, data_config_path=data_config_path)
    rows: list[TaskCoverageRow] = []
    for task in default_benchmark_task_specs():
        task_config = _task_config(experiment_config, task)
        for split_name in (experiment_config.splits.discovery, experiment_config.splits.test):
            try:
                plan = prepare_discovery_collection(
                    experiment_config=task_config,
                    data_config_path=data_config_path,
                    split_name=split_name,
                    dataset_view=dataset_view,
                )
                summary = plan.coverage_summary
                row = _coverage_row_from_summary(
                    task_name=task.name,
                    target_label=task.target_label,
                    split_name=split_name,
                    summary=summary,
                    status="ok",
                )
                if row.positive_session_count < _MIN_POSITIVE_SESSION_COUNT:
                    _add_issue(
                        issues,
                        "coverage",
                        f"{task.name}/{split_name} has only {row.positive_session_count} positive sessions.",
                    )
                rows.append(row)
            except Exception as exc:
                rows.append(
                    TaskCoverageRow(
                        task_name=task.name,
                        target_label=task.target_label,
                        split_name=split_name,
                        status="error",
                        total_scanned_windows=0,
                        positive_window_count=0,
                        negative_window_count=0,
                        selected_positive_count=0,
                        selected_negative_count=0,
                        selected_window_count=0,
                        positive_session_count=0,
                        failure_reason=str(exc),
                    )
                )
                _add_issue(issues, "coverage", f"{task.name}/{split_name} failed coverage scan: {exc}")
    return tuple(rows)


def _write_coverage_csv(rows: tuple[TaskCoverageRow, ...], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(TaskCoverageRow.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())
    return path


def verify_refinement_readiness(
    *,
    pipeline_config_path: str | Path,
    output_root: str | Path,
) -> RefinementVerificationResult:
    pipeline_config = load_notebook_pipeline_config(pipeline_config_path)
    experiment_config = load_experiment_config(pipeline_config.experiment_config_path)
    issues: list[VerificationIssue] = []
    _check_refinement_config(experiment_config=experiment_config, pipeline_config=pipeline_config, issues=issues)
    dataset_view = resolve_runtime_dataset_view(
        experiment_config=experiment_config,
        data_config_path=pipeline_config.data_config_path,
    )
    split_counts = {
        split_name: len(split_session_ids(dataset_view.split_manifest, split_name))
        for split_name in (
            experiment_config.splits.train,
            experiment_config.splits.valid,
            experiment_config.splits.discovery,
            experiment_config.splits.test,
        )
    }
    coverage_rows = _verify_task_coverage(
        experiment_config=experiment_config,
        data_config_path=pipeline_config.data_config_path,
        issues=issues,
    )
    status = "ok" if not any(issue.severity == "error" for issue in issues) else "blocked"
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path = output_dir / "refinement_verification_summary.json"
    coverage_csv_path = output_dir / "refinement_task_coverage.csv"
    result = RefinementVerificationResult(
        status=status,
        pipeline_config_path=str(Path(pipeline_config_path).resolve()),
        experiment_config_path=str(pipeline_config.experiment_config_path),
        data_config_path=str(pipeline_config.data_config_path),
        variant_name=experiment_config.experiment.variant_name,
        split_counts=split_counts,
        issues=tuple(issues),
        coverage_rows=coverage_rows,
        summary_json_path=summary_json_path,
        coverage_csv_path=coverage_csv_path,
    )
    summary_json_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    _write_coverage_csv(coverage_rows, coverage_csv_path)
    return result
