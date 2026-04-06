from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import yaml

from predictive_circuit_coding.utils.console import get_console


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def verify_paths_exist(paths: dict[str, str | Path]) -> dict[str, bool]:
    return {label: Path(path).exists() for label, path in paths.items()}


def _selection_disabled_payload() -> dict[str, object | None]:
    return {
        "output_name": "full_dataset_colab",
        "session_ids": [],
        "subject_ids": [],
        "exclude_session_ids": [],
        "exclude_subject_ids": [],
        "session_ids_file": None,
        "subject_ids_file": None,
        "exclude_session_ids_file": None,
        "exclude_subject_ids_file": None,
        "experience_levels": [],
        "session_types": [],
        "image_sets": [],
        "session_numbers": [],
        "project_codes": [],
        "brain_regions_any": [],
        "min_n_units": None,
        "max_n_units": None,
        "min_trial_count": None,
        "max_trial_count": None,
        "min_duration_s": None,
        "max_duration_s": None,
        "split_seed": None,
        "split_primary_axis": None,
        "train_fraction": None,
        "valid_fraction": None,
        "discovery_fraction": None,
        "test_fraction": None,
    }


@dataclass(frozen=True)
class NotebookDatasetConfig:
    use_full_dataset: bool = False
    experience_level: str = "Familiar"
    max_sessions: int = 10
    split_seed: int = 7
    split_primary_axis: str = "session"
    train_fraction: float = 0.6
    valid_fraction: float = 0.2
    discovery_fraction: float = 0.1
    test_fraction: float = 0.1
    output_name: str | None = None

    def resolved_output_name(self) -> str:
        if self.use_full_dataset:
            return "full_dataset_colab"
        if self.output_name:
            return self.output_name
        return f"{self.experience_level.lower()}_{self.max_sessions}_session_subset"


@dataclass(frozen=True)
class NotebookRuntimeContext:
    experiment_config_path: Path
    checkpoint_dir: Path
    summary_path: Path
    checkpoint_path: Path
    dataset_selection_active: bool
    selected_session_count: int
    profile_path: Path
    selection_output_name: str
    exported_runtime_config_path: Path


def _load_selected_session_ids(
    session_catalog_csv: Path,
    *,
    experience_level: str,
    max_sessions: int,
) -> tuple[list[str], list[dict[str, str]]]:
    selected_rows: list[dict[str, str]] = []
    with session_catalog_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("experience_level") != experience_level:
                continue
            selected_rows.append(row)
    selected_rows.sort(key=lambda row: row.get("session_id", ""))
    selected_rows = selected_rows[:max_sessions]
    if not selected_rows:
        raise ValueError(f"No sessions found for experience_level={experience_level}")
    session_ids = [str(row["session_id"]) for row in selected_rows]
    return session_ids, selected_rows


def _resolve_checkpoint_reference(path_value: str | Path, *, summary_path: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    repo_root = summary_path.parent.parent
    repo_relative = (repo_root / candidate).resolve()
    if repo_relative.exists():
        return repo_relative
    artifact_relative = (summary_path.parent / candidate).resolve()
    if artifact_relative.exists():
        return artifact_relative
    return repo_relative


def prepare_notebook_runtime_context(
    *,
    base_experiment_config: str | Path,
    runtime_experiment_config: str | Path,
    session_catalog_csv: str | Path,
    artifact_root: str | Path,
    dataset_config: NotebookDatasetConfig,
    step_log_every: int,
) -> NotebookRuntimeContext:
    base_path = Path(base_experiment_config)
    runtime_path = Path(runtime_experiment_config)
    catalog_path = Path(session_catalog_csv)
    artifact_path = Path(artifact_root)
    checkpoint_dir = artifact_path / "checkpoints"
    summary_path = artifact_path / "training_summary.json"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    artifact_path.mkdir(parents=True, exist_ok=True)

    payload = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    payload.setdefault("training", {})["log_every_steps"] = int(step_log_every)
    discovery = payload.setdefault("discovery", {})
    discovery["sampling_strategy"] = "label_balanced"
    discovery.pop("search_max_batches", None)
    artifacts = payload.setdefault("artifacts", {})
    artifacts["checkpoint_dir"] = str(checkpoint_dir.resolve())
    artifacts["summary_path"] = str(summary_path.resolve())

    selected_rows: list[dict[str, str]] = []
    selected_session_count = 0
    selection_output_name = dataset_config.resolved_output_name()
    if dataset_config.use_full_dataset:
        payload["dataset_selection"] = _selection_disabled_payload()
    else:
        session_ids, selected_rows = _load_selected_session_ids(
            catalog_path,
            experience_level=dataset_config.experience_level,
            max_sessions=dataset_config.max_sessions,
        )
        session_ids_file = artifact_path / f"{dataset_config.experience_level.lower()}_{dataset_config.max_sessions}_session_ids.txt"
        session_ids_file.write_text("\n".join(session_ids) + "\n", encoding="utf-8")
        payload["dataset_selection"] = {
            "output_name": selection_output_name,
            "session_ids": [],
            "subject_ids": [],
            "exclude_session_ids": [],
            "exclude_subject_ids": [],
            "session_ids_file": str(session_ids_file),
            "subject_ids_file": None,
            "exclude_session_ids_file": None,
            "exclude_subject_ids_file": None,
            "experience_levels": [],
            "session_types": [],
            "image_sets": [],
            "session_numbers": [],
            "project_codes": [],
            "brain_regions_any": [],
            "min_n_units": None,
            "max_n_units": None,
            "min_trial_count": None,
            "max_trial_count": None,
            "min_duration_s": None,
            "max_duration_s": None,
            "split_seed": dataset_config.split_seed,
            "split_primary_axis": dataset_config.split_primary_axis,
            "train_fraction": dataset_config.train_fraction,
            "valid_fraction": dataset_config.valid_fraction,
            "discovery_fraction": dataset_config.discovery_fraction,
            "test_fraction": dataset_config.test_fraction,
        }
        selected_session_count = len(session_ids)

    runtime_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    exported_runtime_config_path = artifact_path / "colab_runtime_experiment.yaml"
    exported_runtime_config_path.write_text(runtime_path.read_text(encoding="utf-8"), encoding="utf-8")
    dataset_selection_active = any(
        value not in (None, [], "", {})
        for key, value in payload.get("dataset_selection", {}).items()
        if key != "output_name"
    )
    checkpoint_prefix = str(artifacts.get("checkpoint_prefix", "pcc"))
    checkpoint_path = checkpoint_dir / f"{checkpoint_prefix}_best.pt"
    profile_path = artifact_path / "colab_notebook_profile.json"
    profile_payload = {
        "runtime_experiment_config": str(runtime_path.resolve()),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "summary_path": str(summary_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "dataset_selection_active": dataset_selection_active,
        "selection_output_name": selection_output_name,
        "step_log_every": int(step_log_every),
        "dataset_config": {
            "use_full_dataset": dataset_config.use_full_dataset,
            "experience_level": dataset_config.experience_level,
            "max_sessions": dataset_config.max_sessions,
            "split_seed": dataset_config.split_seed,
            "split_primary_axis": dataset_config.split_primary_axis,
            "train_fraction": dataset_config.train_fraction,
            "valid_fraction": dataset_config.valid_fraction,
            "discovery_fraction": dataset_config.discovery_fraction,
            "test_fraction": dataset_config.test_fraction,
        },
        "selected_session_count": selected_session_count,
        "selected_sessions_preview": [
            {
                "session_id": row.get("session_id"),
                "subject_id": row.get("subject_id"),
                "experience_level": row.get("experience_level"),
                "image_set": row.get("image_set"),
                "session_type": row.get("session_type"),
            }
            for row in selected_rows[: min(len(selected_rows), 10)]
        ],
    }
    profile_path.write_text(json.dumps(profile_payload, indent=2), encoding="utf-8")
    return NotebookRuntimeContext(
        experiment_config_path=runtime_path,
        checkpoint_dir=checkpoint_dir,
        summary_path=summary_path,
        checkpoint_path=checkpoint_path,
        dataset_selection_active=dataset_selection_active,
        selected_session_count=selected_session_count,
        profile_path=profile_path,
        selection_output_name=selection_output_name,
        exported_runtime_config_path=exported_runtime_config_path,
    )


def build_notebook_discovery_runtime_config(
    *,
    source_experiment_config: str | Path,
    runtime_experiment_config: str | Path,
    artifact_root: str | Path,
    decode_type: str,
    step_log_every: int,
) -> Path:
    source_path = Path(source_experiment_config)
    runtime_path = Path(runtime_experiment_config)
    artifact_path = Path(artifact_root)
    checkpoint_dir = artifact_path / "checkpoints"
    summary_path = artifact_path / "training_summary.json"
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    payload.setdefault("training", {})["log_every_steps"] = int(step_log_every)
    discovery = payload.setdefault("discovery", {})
    discovery["target_label"] = str(decode_type)
    discovery["sampling_strategy"] = "label_balanced"
    discovery.pop("search_max_batches", None)
    artifacts = payload.setdefault("artifacts", {})
    artifacts["checkpoint_dir"] = str(checkpoint_dir.resolve())
    artifacts["summary_path"] = str(summary_path.resolve())
    runtime_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return runtime_path


def load_notebook_split_counts(
    *,
    data_config_path: str | Path,
    dataset_selection_active: bool,
    selection_output_name: str | None = None,
) -> dict[str, int]:
    from predictive_circuit_coding.data import build_workspace, load_preparation_config, load_split_manifest

    prep_config = load_preparation_config(data_config_path)
    workspace = build_workspace(prep_config)
    if dataset_selection_active:
        if not selection_output_name:
            raise ValueError("selection_output_name is required when dataset_selection_active is True.")
        split_manifest_path = workspace.splits / "selections" / selection_output_name / "selected_split_manifest.json"
    else:
        split_manifest_path = workspace.split_manifest_path
    if not split_manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {split_manifest_path}")
    split_manifest = load_split_manifest(split_manifest_path)
    counts: dict[str, int] = {}
    for assignment in split_manifest.assignments:
        counts[assignment.split] = counts.get(assignment.split, 0) + 1
    return counts


_STEP_LOG_PATTERN = re.compile(r"epoch=(?P<epoch>\d+) step=(?P<step>\d+):(?P<metrics>.*)")
_METRIC_CONTINUATION_PATTERN = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_]*=")
_NO_POSITIVE_LABELS_PATTERN = re.compile(r"no positive '.*?' labels")


def _extract_metric(metrics_blob: str, metric_name: str) -> str:
    match = re.search(rf"{metric_name}=(-?\d+(?:\.\d+)?)", metrics_blob)
    return match.group(1) if match else "n/a"


@dataclass
class _PendingStepMetrics:
    epoch: str
    step: int
    metrics_parts: list[str]
    should_emit: bool


class NotebookCommandStreamFormatter:
    def __init__(self, *, step_log_every: int = 16) -> None:
        self.step_log_every = max(1, int(step_log_every))
        self._pending: _PendingStepMetrics | None = None

    def _flush_pending(self) -> list[str]:
        if self._pending is None:
            return []
        pending = self._pending
        self._pending = None
        if not pending.should_emit:
            return []
        metrics_blob = " ".join(part.strip() for part in pending.metrics_parts if part.strip())
        return [
            (
                f"epoch={pending.epoch} step={pending.step}: "
                f"predictive_improvement={_extract_metric(metrics_blob, 'predictive_improvement')}, "
                f"predictive_loss={_extract_metric(metrics_blob, 'predictive_loss')}, "
                f"total_loss={_extract_metric(metrics_blob, 'total_loss')}\n"
            )
        ]

    def feed(self, line: str) -> list[str]:
        outputs: list[str] = []
        step_match = _STEP_LOG_PATTERN.search(line)
        if step_match is not None:
            outputs.extend(self._flush_pending())
            step = int(step_match.group("step"))
            self._pending = _PendingStepMetrics(
                epoch=step_match.group("epoch"),
                step=step,
                metrics_parts=[step_match.group("metrics")],
                should_emit=(step == 1 or step % self.step_log_every == 0),
            )
            return outputs
        if self._pending is not None and _METRIC_CONTINUATION_PATTERN.match(line):
            self._pending.metrics_parts.append(line)
            return outputs
        outputs.extend(self._flush_pending())
        outputs.append(line)
        return outputs

    def finalize(self) -> list[str]:
        return self._flush_pending()


def run_streaming_command(
    command: list[str],
    *,
    cwd: str | Path | None = None,
    step_log_every: int = 16,
    stream: TextIO | None = None,
) -> int:
    output_stream = stream or sys.stdout
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    formatter = NotebookCommandStreamFormatter(step_log_every=step_log_every)
    captured_output: list[str] = []
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert process.stdout is not None
    for line in process.stdout:
        for output_line in formatter.feed(line):
            captured_output.append(output_line)
            output_stream.write(output_line)
    for output_line in formatter.finalize():
        captured_output.append(output_line)
        output_stream.write(output_line)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command, output="".join(captured_output))
    return return_code


def output_indicates_missing_positive_labels(output: str) -> bool:
    return bool(_NO_POSITIVE_LABELS_PATTERN.search(output.lower()))


def resolve_notebook_checkpoint(
    *,
    summary_path: str | Path,
    checkpoint_dir: str | Path,
    checkpoint_prefix: str = "pcc",
) -> Path:
    resolved_summary_path = Path(summary_path)
    resolved_checkpoint_dir = Path(checkpoint_dir)
    if resolved_summary_path.exists():
        payload = json.loads(resolved_summary_path.read_text(encoding="utf-8"))
        checkpoint_path = payload.get("checkpoint_path")
        if checkpoint_path:
            resolved = _resolve_checkpoint_reference(checkpoint_path, summary_path=resolved_summary_path)
            if resolved.exists():
                return resolved
    best_checkpoint = resolved_checkpoint_dir / f"{checkpoint_prefix}_best.pt"
    if best_checkpoint.exists():
        return best_checkpoint
    epoch_candidates = sorted(resolved_checkpoint_dir.glob(f"{checkpoint_prefix}_epoch_*.pt"))
    if epoch_candidates:
        return epoch_candidates[-1]
    raise FileNotFoundError(f"No checkpoint found under {resolved_checkpoint_dir}")


def restore_latest_exported_artifacts(
    *,
    drive_export_root: str | Path,
    local_artifact_root: str | Path,
    runtime_experiment_config: str | Path | None = None,
    run_prefix: str = "train_run_",
) -> Path | None:
    export_root = Path(drive_export_root)
    if not export_root.exists():
        return None
    run_candidates = sorted(
        [path for path in export_root.iterdir() if path.is_dir() and path.name.startswith(run_prefix)],
        key=lambda path: path.name,
    )
    if not run_candidates:
        return None
    latest_run = run_candidates[-1]
    artifact_root = Path(local_artifact_root)
    if artifact_root.exists():
        shutil.rmtree(artifact_root)
    shutil.copytree(latest_run, artifact_root)
    if runtime_experiment_config is not None:
        runtime_target = Path(runtime_experiment_config)
        exported_runtime_config = artifact_root / "colab_runtime_experiment.yaml"
        if exported_runtime_config.exists():
            runtime_target.write_text(exported_runtime_config.read_text(encoding="utf-8"), encoding="utf-8")
    return latest_run


@dataclass
class NotebookStageReporter:
    name: str
    expected_duration: str | None = None
    console: object = field(default_factory=get_console)
    started_at: float = field(default_factory=time.perf_counter)
    current_stage_started_at: float | None = None

    def banner(self, title: str, *, subtitle: str | None = None) -> None:
        self.console.print(f"[bold]{title}[/bold]")
        if subtitle:
            self.console.print(subtitle)

    def begin(self, stage_name: str, *, next_artifact: str | None = None) -> None:
        self.current_stage_started_at = time.perf_counter()
        message = f"[{self.name}] starting {stage_name}"
        if self.expected_duration:
            message += f" | expected: {self.expected_duration}"
        if next_artifact:
            message += f" | next artifact: {next_artifact}"
        self.console.print(message)

    def finish(self, stage_name: str) -> None:
        total_elapsed = format_duration(time.perf_counter() - self.started_at)
        if self.current_stage_started_at is None:
            stage_elapsed = total_elapsed
        else:
            stage_elapsed = format_duration(time.perf_counter() - self.current_stage_started_at)
        self.console.print(f"[{self.name}] finished {stage_name} in {stage_elapsed} (total {total_elapsed})")
        self.current_stage_started_at = None

    def note_checkpoint(self, checkpoint_path: str | Path) -> None:
        self.console.print(f"[green]checkpoint saved[/green] {Path(checkpoint_path)}")

    def run_command(self, command: list[str], *, cwd: str | Path | None = None) -> subprocess.CompletedProcess[str]:
        self.console.print("$ " + " ".join(command))
        return subprocess.run(command, cwd=str(cwd) if cwd else None, check=True, text=True, capture_output=True)
