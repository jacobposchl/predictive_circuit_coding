from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
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
class NotebookTrainingConfig:
    num_epochs: int = 8
    train_steps_per_epoch: int = 128
    validation_steps: int = 16


@dataclass(frozen=True)
class NotebookRuntimeContext:
    experiment_config_path: Path
    checkpoint_dir: Path
    summary_path: Path
    checkpoint_path: Path
    run_id: str
    selected_session_count: int
    profile_path: Path
    runtime_split_manifest_path: Path
    runtime_session_catalog_path: Path
    runtime_config_dir: Path
    config_name_prefix: str
    exported_runtime_config_path: Path


@dataclass(frozen=True)
class NotebookLocalDatasetStageResult:
    target_dataset_root: Path
    target_prepared_root: Path
    staged_session_ids: tuple[str, ...]
    copied_support_files: tuple[Path, ...]


def _load_selected_catalog_records(
    session_catalog_json: Path,
    *,
    experience_level: str,
    max_sessions: int,
) -> list[dict[str, object]]:
    with session_catalog_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    selected_records = [
        dict(record)
        for record in payload.get("records", [])
        if record.get("experience_level") == experience_level
    ]
    selected_records.sort(key=lambda row: str(row.get("session_id", "")))
    selected_records = selected_records[:max_sessions]
    if not selected_records:
        raise ValueError(f"No sessions found for experience_level={experience_level}")
    return selected_records


def _write_runtime_subset_artifacts(
    *,
    artifact_root: Path,
    dataset_id: str,
    selected_records: list[dict[str, object]],
    dataset_config: NotebookDatasetConfig,
) -> tuple[Path, Path, Path]:
    from predictive_circuit_coding.data.catalog import write_session_catalog, write_session_catalog_csv
    from predictive_circuit_coding.data.manifest import SessionManifest, SessionRecord
    from predictive_circuit_coding.data.splits import build_split_manifest, write_split_manifest
    from predictive_circuit_coding.data.config import SplitPlanningConfig
    from predictive_circuit_coding.windowing import build_torch_brain_config

    runtime_dir = artifact_root / "runtime_subset"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    split_dir = runtime_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    catalog_payload = {
        "dataset_id": dataset_id,
        "source_name": dataset_id,
        "records": selected_records,
    }
    catalog_json = runtime_dir / "selected_session_catalog.json"
    catalog_csv = runtime_dir / "selected_session_catalog.csv"
    with catalog_json.open("w", encoding="utf-8") as handle:
        json.dump(catalog_payload, handle, indent=2)
    fieldnames = sorted({key for record in selected_records for key in record.keys()})
    with catalog_csv.open("w", encoding="utf-8", newline="") as handle:
        import csv
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in selected_records:
            writer.writerow(record)

    manifest = SessionManifest(
        dataset_id=dataset_id,
        source_name=dataset_id,
        records=tuple(
            SessionRecord(
                recording_id=str(record["recording_id"]),
                session_id=str(record["session_id"]),
                subject_id=str(record["subject_id"]),
                raw_data_path=str(record.get("raw_data_path", "")),
                duration_s=float(record.get("duration_s", 0.0)),
                n_units=int(record.get("n_units", 0)),
                brain_regions=tuple(str(region) for region in record.get("brain_regions", []) or ()),
                trial_count=int(record.get("trial_count", 0)),
            )
            for record in selected_records
        ),
    )
    split_manifest = build_split_manifest(
        manifest,
        config=SplitPlanningConfig(
            seed=dataset_config.split_seed,
            primary_axis=dataset_config.split_primary_axis,
            train_fraction=dataset_config.train_fraction,
            valid_fraction=dataset_config.valid_fraction,
            discovery_fraction=dataset_config.discovery_fraction,
            test_fraction=dataset_config.test_fraction,
        ),
    )
    split_manifest_path = runtime_dir / "selected_split_manifest.json"
    write_split_manifest(split_manifest, split_manifest_path)

    for split_name in ("train", "valid", "discovery", "test"):
        build_torch_brain_config(
            workspace=type("NotebookWorkspace", (), {"splits": split_dir})(),
            dataset_id=dataset_id,
            session_ids=[
                assignment.recording_id.split("/", 1)[1]
                for assignment in split_manifest.assignments
                if assignment.split == split_name
            ],
            split=split_name,
            output_dir=split_dir,
            filename_prefix="torch_brain_runtime",
        )

    return catalog_json, split_manifest_path, split_dir


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
    training_config: NotebookTrainingConfig | None = None,
    step_log_every: int,
    run_id: str | None = None,
) -> NotebookRuntimeContext:
    base_path = Path(base_experiment_config)
    runtime_path = Path(runtime_experiment_config)
    catalog_path = Path(session_catalog_csv)
    catalog_json_path = catalog_path.with_name("session_catalog.json")
    artifact_path = Path(artifact_root)
    checkpoint_dir = artifact_path / "checkpoints"
    summary_path = artifact_path / "training_summary.json"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    artifact_path.mkdir(parents=True, exist_ok=True)

    resolved_run_id = str(run_id).strip() if run_id is not None else datetime.now().strftime("run_%Y%m%d_%H%M%S")
    if not resolved_run_id:
        raise ValueError("run_id must not be empty")

    payload = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    training_payload = payload.setdefault("training", {})
    training_payload["log_every_steps"] = int(step_log_every)
    if training_config is not None:
        training_payload["num_epochs"] = int(training_config.num_epochs)
        training_payload["train_steps_per_epoch"] = int(training_config.train_steps_per_epoch)
        training_payload["validation_steps"] = int(training_config.validation_steps)
    discovery = payload.setdefault("discovery", {})
    discovery["sampling_strategy"] = "label_balanced"
    discovery.pop("search_max_batches", None)
    artifacts = payload.setdefault("artifacts", {})
    artifacts["checkpoint_dir"] = str(checkpoint_dir.resolve())
    artifacts["summary_path"] = str(summary_path.resolve())

    selected_rows: list[dict[str, object]] = []
    selected_session_count = 0
    runtime_session_catalog_path = catalog_json_path
    runtime_split_manifest_path = artifact_path / "runtime_subset" / "selected_split_manifest.json"
    runtime_config_dir = artifact_path / "runtime_subset" / "splits"
    config_name_prefix = "torch_brain_runtime"
    if dataset_config.use_full_dataset:
        payload["dataset_selection"] = {}
        payload["runtime_subset"] = None
    else:
        if not catalog_json_path.exists():
            raise FileNotFoundError(
                f"Canonical session catalog JSON not found: {catalog_json_path}. "
                "Run local data preparation so the canonical catalog exists before using the notebook subset flow."
            )
        selected_rows = _load_selected_catalog_records(
            catalog_json_path,
            experience_level=dataset_config.experience_level,
            max_sessions=dataset_config.max_sessions,
        )
        runtime_session_catalog_path, runtime_split_manifest_path, runtime_config_dir = _write_runtime_subset_artifacts(
            artifact_root=artifact_path,
            dataset_id=str(payload["dataset_id"]),
            selected_records=selected_rows,
            dataset_config=dataset_config,
        )
        payload["dataset_selection"] = {}
        payload["runtime_subset"] = {
            "split_manifest_path": str(runtime_split_manifest_path.resolve()),
            "session_catalog_path": str(runtime_session_catalog_path.resolve()),
            "config_dir": str(runtime_config_dir.resolve()),
            "config_name_prefix": config_name_prefix,
        }
        selected_session_count = len(selected_rows)

    runtime_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    exported_runtime_config_path = artifact_path / "colab_runtime_experiment.yaml"
    exported_runtime_config_path.write_text(runtime_path.read_text(encoding="utf-8"), encoding="utf-8")
    checkpoint_prefix = str(artifacts.get("checkpoint_prefix", "pcc"))
    checkpoint_path = checkpoint_dir / f"{checkpoint_prefix}_best.pt"
    profile_path = artifact_path / "colab_notebook_profile.json"
    profile_payload = {
        "run_id": resolved_run_id,
        "runtime_experiment_config": str(runtime_path.resolve()),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "summary_path": str(summary_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "runtime_split_manifest_path": str(runtime_split_manifest_path.resolve()) if runtime_split_manifest_path.exists() else None,
        "runtime_session_catalog_path": str(runtime_session_catalog_path.resolve()) if runtime_session_catalog_path.exists() else None,
        "runtime_config_dir": str(runtime_config_dir.resolve()) if runtime_config_dir.exists() else None,
        "config_name_prefix": config_name_prefix,
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
        "training_config": {
            "num_epochs": int(training_payload.get("num_epochs", 0)),
            "train_steps_per_epoch": int(training_payload.get("train_steps_per_epoch", 0)),
            "validation_steps": int(training_payload.get("validation_steps", 0)),
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
        run_id=resolved_run_id,
        selected_session_count=selected_session_count,
        profile_path=profile_path,
        runtime_split_manifest_path=runtime_split_manifest_path,
        runtime_session_catalog_path=runtime_session_catalog_path,
        runtime_config_dir=runtime_config_dir,
        config_name_prefix=config_name_prefix,
        exported_runtime_config_path=exported_runtime_config_path,
    )


def build_notebook_discovery_runtime_config(
    *,
    source_experiment_config: str | Path,
    runtime_experiment_config: str | Path,
    artifact_root: str | Path,
    decode_type: str,
    target_label_mode: str | None = None,
    target_label_match_value: str | None = None,
    device_mode: str = "auto",
    step_log_every: int,
    discovery_max_batches: int | None = None,
    discovery_top_k_candidates: int | None = None,
    discovery_candidate_session_balance_fraction: float | None = None,
    discovery_min_candidate_score: float | None = None,
    discovery_min_cluster_size: int | None = None,
    discovery_probe_epochs: int | None = None,
    discovery_probe_learning_rate: float | None = None,
    validation_max_batches: int | None = None,
    validation_shuffle_seed: int | None = None,
) -> Path:
    source_path = Path(source_experiment_config)
    runtime_path = Path(runtime_experiment_config)
    artifact_path = Path(artifact_root)
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = artifact_path / "checkpoints"
    summary_path = artifact_path / "training_summary.json"
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    payload.setdefault("training", {})["log_every_steps"] = int(step_log_every)
    execution = payload.setdefault("execution", {})
    execution["device"] = "cpu" if str(device_mode).lower() == "cpu" else "auto"
    discovery = payload.setdefault("discovery", {})
    discovery["target_label"] = str(decode_type)
    if target_label_mode is not None:
        discovery["target_label_mode"] = str(target_label_mode)
    if target_label_match_value is not None:
        discovery["target_label_match_value"] = str(target_label_match_value)
    else:
        discovery.pop("target_label_match_value", None)
    discovery["sampling_strategy"] = "label_balanced"
    if discovery_max_batches is not None:
        discovery["max_batches"] = int(discovery_max_batches)
    if discovery_top_k_candidates is not None:
        discovery["top_k_candidates"] = int(discovery_top_k_candidates)
    if discovery_candidate_session_balance_fraction is not None:
        discovery["candidate_session_balance_fraction"] = float(discovery_candidate_session_balance_fraction)
    if discovery_min_candidate_score is not None:
        discovery["min_candidate_score"] = float(discovery_min_candidate_score)
    if discovery_min_cluster_size is not None:
        discovery["min_cluster_size"] = int(discovery_min_cluster_size)
    if discovery_probe_epochs is not None:
        discovery["probe_epochs"] = int(discovery_probe_epochs)
    if discovery_probe_learning_rate is not None:
        discovery["probe_learning_rate"] = float(discovery_probe_learning_rate)
    if validation_shuffle_seed is not None:
        discovery["shuffle_seed"] = int(validation_shuffle_seed)
    discovery.pop("search_max_batches", None)
    evaluation = payload.setdefault("evaluation", {})
    if validation_max_batches is not None:
        evaluation["max_batches"] = int(validation_max_batches)
    payload["dataset_selection"] = {}
    artifacts = payload.setdefault("artifacts", {})
    artifacts["checkpoint_dir"] = str(checkpoint_dir.resolve())
    artifacts["summary_path"] = str(summary_path.resolve())
    runtime_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return runtime_path


def load_notebook_split_counts(
    *,
    split_manifest_path: str | Path,
) -> dict[str, int]:
    from predictive_circuit_coding.data import load_split_manifest

    split_manifest_path = Path(split_manifest_path)
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


def _sanitize_notebook_export_segment(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    sanitized = sanitized.strip(".-_")
    return sanitized or "default"


def materialize_notebook_prepared_sessions(
    *,
    source_dataset_root: str | Path,
    target_dataset_root: str | Path,
    session_ids: list[str] | tuple[str, ...],
    dataset_id: str,
    reset_target: bool = True,
) -> NotebookLocalDatasetStageResult:
    source_root = Path(source_dataset_root).resolve()
    target_root = Path(target_dataset_root).expanduser()
    staged_session_ids = tuple(sorted(dict.fromkeys(str(session_id) for session_id in session_ids if str(session_id).strip())))
    if not staged_session_ids:
        raise ValueError("session_ids must contain at least one session to stage locally")
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source dataset root not found: {source_root}")

    if target_root.exists() or target_root.is_symlink():
        if not reset_target:
            raise FileExistsError(f"Target dataset root already exists: {target_root}")
        if target_root.is_symlink() or target_root.is_file():
            target_root.unlink()
        else:
            shutil.rmtree(target_root)

    target_root.mkdir(parents=True, exist_ok=True)
    source_prepared_root = source_root / "prepared" / str(dataset_id)
    if not source_prepared_root.is_dir():
        raise FileNotFoundError(f"Prepared dataset root not found: {source_prepared_root}")
    target_prepared_root = target_root / "prepared" / str(dataset_id)
    target_prepared_root.mkdir(parents=True, exist_ok=True)

    for session_id in staged_session_ids:
        source_path = source_prepared_root / f"{session_id}.h5"
        if not source_path.is_file():
            raise FileNotFoundError(f"Prepared session not found: {source_path}")
        shutil.copy2(source_path, target_prepared_root / source_path.name)

    copied_support_files: list[Path] = []
    for relative_path in (
        Path("manifests") / "session_catalog.json",
        Path("manifests") / "session_catalog.csv",
        Path("splits") / "split_manifest.json",
    ):
        source_path = source_root / relative_path
        if not source_path.is_file():
            continue
        destination = target_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        copied_support_files.append(destination.resolve())

    return NotebookLocalDatasetStageResult(
        target_dataset_root=target_root.resolve(),
        target_prepared_root=target_prepared_root.resolve(),
        staged_session_ids=staged_session_ids,
        copied_support_files=tuple(copied_support_files),
    )


def _resolve_training_export_path(
    *,
    drive_export_root: str | Path,
    training_run_id: str | None = None,
    run_name: str = "run_1",
) -> Path | None:
    export_root = Path(drive_export_root)
    if training_run_id is not None:
        selected_run_id = str(training_run_id).strip()
        if not selected_run_id:
            raise ValueError("training_run_id must not be empty when provided")
        if not export_root.exists():
            raise FileNotFoundError(
                f"Requested training run_id '{selected_run_id}' was not found under {export_root}."
            )
        train_dir = export_root / selected_run_id / run_name / "train"
        if not train_dir.is_dir():
            raise FileNotFoundError(
                f"Requested training run_id '{selected_run_id}' was not found under {export_root}."
            )
        return train_dir
    if not export_root.exists():
        return None
    run_candidates = sorted(
        [
            path / run_name / "train"
            for path in export_root.iterdir()
            if path.is_dir() and path.name.startswith("run_") and (path / run_name / "train").is_dir()
        ],
        key=lambda path: path.parent.parent.name,
    )
    if not run_candidates:
        return None
    return run_candidates[-1]


def build_notebook_training_export_path(
    *,
    drive_export_root: str | Path,
    run_id: str,
    run_name: str = "run_1",
) -> Path:
    return Path(drive_export_root) / str(run_id) / run_name / "train"


def export_notebook_training_artifacts(
    *,
    drive_export_root: str | Path,
    local_artifact_root: str | Path,
    run_id: str,
    run_name: str = "run_1",
) -> Path:
    artifact_root = Path(local_artifact_root)
    target = build_notebook_training_export_path(
        drive_export_root=drive_export_root,
        run_id=run_id,
        run_name=run_name,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(artifact_root, target)
    return target


def restore_latest_exported_artifacts(
    *,
    drive_export_root: str | Path,
    local_artifact_root: str | Path,
    runtime_experiment_config: str | Path | None = None,
    training_run_id: str | None = None,
    run_name: str = "run_1",
) -> Path | None:
    train_export_path = _resolve_training_export_path(
        drive_export_root=drive_export_root,
        training_run_id=training_run_id,
        run_name=run_name,
    )
    if train_export_path is None:
        return None
    artifact_root = Path(local_artifact_root)
    if artifact_root.exists():
        shutil.rmtree(artifact_root)
    shutil.copytree(train_export_path, artifact_root)
    if runtime_experiment_config is not None:
        runtime_target = Path(runtime_experiment_config)
        exported_runtime_config = artifact_root / "colab_runtime_experiment.yaml"
        if exported_runtime_config.exists():
            runtime_target.write_text(exported_runtime_config.read_text(encoding="utf-8"), encoding="utf-8")
    return train_export_path


def build_notebook_discovery_export_path(
    *,
    drive_export_root: str | Path,
    run_id: str,
    decode_type: str,
    attempt_timestamp: str,
    run_name: str = "run_1",
) -> Path:
    attempt_name = f"{_sanitize_notebook_export_segment(decode_type)}__{attempt_timestamp}"
    return Path(drive_export_root) / str(run_id) / run_name / "discovery" / attempt_name


def export_notebook_discovery_artifacts(
    *,
    drive_export_root: str | Path,
    local_artifact_root: str | Path,
    run_id: str,
    decode_type: str,
    attempt_timestamp: str,
    runtime_experiment_config: str | Path,
    checkpoint_path: str | Path,
    discovery_run: "NotebookDiscoveryRunResult",
    validation_run: "NotebookValidationRunResult | None" = None,
    run_name: str = "run_1",
) -> Path:
    artifact_root = Path(local_artifact_root).resolve()
    target = build_notebook_discovery_export_path(
        drive_export_root=drive_export_root,
        run_id=run_id,
        decode_type=decode_type,
        attempt_timestamp=attempt_timestamp,
        run_name=run_name,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    export_candidates = [
        discovery_run.discovery_artifact_path,
        discovery_run.decode_coverage_summary_path,
        discovery_run.cluster_summary_json_path,
        discovery_run.cluster_summary_csv_path,
    ]
    if validation_run is not None:
        export_candidates.extend(
            [
                validation_run.validation_summary_json_path,
                validation_run.validation_summary_csv_path,
            ]
        )

    for src_candidate in export_candidates:
        src = Path(src_candidate)
        if not src.exists():
            continue
        try:
            relative_path = src.resolve().relative_to(artifact_root)
        except ValueError:
            relative_path = Path(src.name)
        destination = target / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, destination)

    runtime_copy = target / Path(runtime_experiment_config).name
    runtime_copy.write_text(Path(runtime_experiment_config).read_text(encoding="utf-8"), encoding="utf-8")
    metadata_path = target / "discovery_export_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "training_run_id": str(run_id),
                "run_name": str(run_name),
                "decode_type": str(decode_type),
                "checkpoint_name": Path(checkpoint_path).name,
                "local_discovery_runtime_config_path": str(Path(runtime_experiment_config).resolve()),
                "exported_discovery_runtime_config_path": str(runtime_copy),
                "attempt_timestamp": str(attempt_timestamp),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return target


def restore_latest_discovery_artifacts(
    *,
    drive_export_root: str | Path,
    local_artifact_root: str | Path,
    training_run_id: str | None = None,
    decode_type: str | None = None,
    run_name: str = "run_1",
) -> Path | None:
    train_export_path = _resolve_training_export_path(
        drive_export_root=drive_export_root,
        training_run_id=training_run_id,
        run_name=run_name,
    )
    if train_export_path is None:
        return None
    discovery_root = train_export_path.parent / "discovery"
    if not discovery_root.is_dir():
        return None

    decode_prefix = None
    if decode_type is not None:
        decode_prefix = f"{_sanitize_notebook_export_segment(decode_type)}__"

    run_candidates = sorted(
        [
            path
            for path in discovery_root.iterdir()
            if path.is_dir() and (decode_prefix is None or path.name.startswith(decode_prefix))
        ],
        key=lambda path: path.name,
    )
    if not run_candidates:
        return None

    latest_run = run_candidates[-1]
    artifact_root = Path(local_artifact_root)
    for src_file in latest_run.rglob("*"):
        if not src_file.is_file():
            continue
        relative_path = src_file.relative_to(latest_run)
        if not relative_path.parts or relative_path.parts[0] != "checkpoints":
            continue
        dst = artifact_root / relative_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst)
    return latest_run


@dataclass(frozen=True)
class NotebookDiscoveryRunResult:
    discovery_artifact_path: Path
    decode_coverage_summary_path: Path
    cluster_summary_json_path: Path
    cluster_summary_csv_path: Path


@dataclass(frozen=True)
class NotebookValidationRunResult:
    validation_summary_json_path: Path
    validation_summary_csv_path: Path


@dataclass(frozen=True)
class NotebookGeometryDiagnosticsRunResult:
    window_geometry_summary_json_path: Path
    window_geometry_summary_csv_path: Path
    candidate_geometry_summary_json_path: Path
    candidate_geometry_summary_csv_path: Path


@dataclass(frozen=True)
class NotebookAlignmentDiagnosticsRunResult:
    alignment_summary_json_path: Path
    alignment_summary_csv_path: Path


@dataclass(frozen=True)
class NotebookDiagnosticsExperimentPaths:
    experiment_name: str
    experiment_root: Path
    runtime_experiment_config_path: Path
    discovery_artifact_path: Path
    decode_coverage_summary_path: Path
    cluster_summary_json_path: Path
    cluster_summary_csv_path: Path
    validation_summary_json_path: Path
    validation_summary_csv_path: Path
    window_geometry_summary_json_path: Path
    window_geometry_summary_csv_path: Path
    candidate_geometry_summary_json_path: Path
    candidate_geometry_summary_csv_path: Path
    alignment_summary_json_path: Path
    alignment_summary_csv_path: Path


def build_notebook_diagnostics_local_root(*, local_artifact_root: str | Path) -> Path:
    return Path(local_artifact_root) / "diagnostics"


def build_notebook_diagnostics_experiment_paths(
    *,
    local_artifact_root: str | Path,
    checkpoint_path: str | Path,
    experiment_name: str,
    split_name: str = "discovery",
) -> NotebookDiagnosticsExperimentPaths:
    experiment_root = build_notebook_diagnostics_local_root(local_artifact_root=local_artifact_root) / str(
        experiment_name
    )
    checkpoints_root = experiment_root / "checkpoints"
    checkpoint_stem = Path(checkpoint_path).stem
    discovery_artifact_path = checkpoints_root / f"{checkpoint_stem}_{split_name}_discovery.json"
    decode_coverage_summary_path = _coverage_summary_path(discovery_artifact_path)
    cluster_summary_json_path, cluster_summary_csv_path = _cluster_report_paths(discovery_artifact_path)
    validation_summary_json_path, validation_summary_csv_path = _default_validation_output_paths(discovery_artifact_path)
    return NotebookDiagnosticsExperimentPaths(
        experiment_name=str(experiment_name),
        experiment_root=experiment_root,
        runtime_experiment_config_path=experiment_root / "colab_runtime_experiment.yaml",
        discovery_artifact_path=discovery_artifact_path,
        decode_coverage_summary_path=decode_coverage_summary_path,
        cluster_summary_json_path=cluster_summary_json_path,
        cluster_summary_csv_path=cluster_summary_csv_path,
        validation_summary_json_path=validation_summary_json_path,
        validation_summary_csv_path=validation_summary_csv_path,
        window_geometry_summary_json_path=experiment_root / "window_geometry_summary.json",
        window_geometry_summary_csv_path=experiment_root / "window_geometry_summary.csv",
        candidate_geometry_summary_json_path=experiment_root / "candidate_geometry_summary.json",
        candidate_geometry_summary_csv_path=experiment_root / "candidate_geometry_summary.csv",
        alignment_summary_json_path=experiment_root / "session_alignment_summary.json",
        alignment_summary_csv_path=experiment_root / "session_alignment_summary.csv",
    )


def build_notebook_diagnostics_export_path(
    *,
    drive_export_root: str | Path,
    run_id: str,
    diagnostics_timestamp: str,
    run_name: str = "run_1",
) -> Path:
    return Path(drive_export_root) / str(run_id) / run_name / "diagnostics" / str(diagnostics_timestamp)


def export_notebook_diagnostics_artifacts(
    *,
    drive_export_root: str | Path,
    local_artifact_root: str | Path,
    run_id: str,
    diagnostics_timestamp: str,
    run_name: str = "run_1",
) -> Path:
    source_root = build_notebook_diagnostics_local_root(local_artifact_root=local_artifact_root).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Local diagnostics root not found: {source_root}")
    target = build_notebook_diagnostics_export_path(
        drive_export_root=drive_export_root,
        run_id=run_id,
        diagnostics_timestamp=diagnostics_timestamp,
        run_name=run_name,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source_root, target)
    return target


def describe_notebook_compute_targets(*, experiment_config_path: str | Path) -> dict[str, str]:
    from predictive_circuit_coding.training import load_experiment_config
    from predictive_circuit_coding.training.runtime import resolve_device

    config = load_experiment_config(experiment_config_path)
    encoder_device = str(resolve_device(config.execution.device))
    return {
        "encoder_device": encoder_device,
        "probe_device": "cpu",
        "clustering_device": "cpu",
        "metrics_device": "cpu",
    }


def _default_discovery_output_path(checkpoint_path: str | Path, split_name: str) -> Path:
    checkpoint = Path(checkpoint_path)
    return checkpoint.with_name(f"{checkpoint.stem}_{split_name}_discovery.json")


def _cluster_report_paths(discovery_output_path: str | Path) -> tuple[Path, Path]:
    output = Path(discovery_output_path)
    return (
        output.with_name(f"{output.stem}_cluster_summary.json"),
        output.with_name(f"{output.stem}_cluster_summary.csv"),
    )


def _coverage_summary_path(discovery_output_path: str | Path) -> Path:
    output = Path(discovery_output_path)
    return output.with_name(f"{output.stem}_decode_coverage.json")


def _load_discovery_target_label(discovery_artifact_path: str | Path) -> str | None:
    artifact_path = Path(discovery_artifact_path)
    if not artifact_path.exists():
        return None
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    decoder_summary = payload.get("decoder_summary") or {}
    target_label = decoder_summary.get("target_label")
    if isinstance(target_label, str) and target_label.strip():
        return target_label
    config_snapshot = payload.get("config_snapshot") or {}
    discovery_config = config_snapshot.get("discovery") or {}
    target_label = discovery_config.get("target_label")
    if isinstance(target_label, str) and target_label.strip():
        return target_label
    return None


def _load_discovery_config_snapshot(discovery_artifact_path: str | Path) -> dict | None:
    artifact_path = Path(discovery_artifact_path)
    if not artifact_path.exists():
        return None
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    config_snapshot = payload.get("config_snapshot")
    return config_snapshot if isinstance(config_snapshot, dict) else None


def _normalize_discovery_reuse_config(payload: dict) -> dict:
    return {
        "dataset_id": payload.get("dataset_id"),
        "split_name": payload.get("split_name"),
        "data_runtime": payload.get("data_runtime") or {},
        "execution": payload.get("execution") or {},
        "evaluation": payload.get("evaluation") or {},
        "discovery": payload.get("discovery") or {},
        "runtime_subset": payload.get("runtime_subset") or {},
        "splits": payload.get("splits") or {},
    }


def _discovery_artifact_matches_runtime_config(
    *,
    discovery_artifact_path: str | Path,
    experiment_config_path: str | Path,
) -> bool:
    artifact_config_snapshot = _load_discovery_config_snapshot(discovery_artifact_path)
    if artifact_config_snapshot is None:
        return False
    runtime_payload = yaml.safe_load(Path(experiment_config_path).read_text(encoding="utf-8")) or {}
    return _normalize_discovery_reuse_config(artifact_config_snapshot) == _normalize_discovery_reuse_config(
        runtime_payload
    )


def find_existing_discovery_run(
    *,
    checkpoint_path: str | Path,
    split_name: str = "discovery",
    target_label: str | None = None,
    experiment_config_path: str | Path | None = None,
    discovery_output_path: str | Path | None = None,
) -> "NotebookDiscoveryRunResult | None":
    """Return a NotebookDiscoveryRunResult if all expected discovery artifact files exist locally,
    otherwise return None so the caller can decide whether to re-run discovery."""
    discovery_path = (
        Path(discovery_output_path)
        if discovery_output_path is not None
        else _default_discovery_output_path(checkpoint_path, split_name)
    )
    if not discovery_path.exists():
        return None
    coverage_path = _coverage_summary_path(discovery_path)
    cluster_json, cluster_csv = _cluster_report_paths(discovery_path)
    if not all(p.exists() for p in (coverage_path, cluster_json, cluster_csv)):
        return None
    if target_label is not None:
        existing_target_label = _load_discovery_target_label(discovery_path)
        if existing_target_label != str(target_label):
            return None
    if experiment_config_path is not None and not _discovery_artifact_matches_runtime_config(
        discovery_artifact_path=discovery_path,
        experiment_config_path=experiment_config_path,
    ):
        return None
    return NotebookDiscoveryRunResult(
        discovery_artifact_path=discovery_path,
        decode_coverage_summary_path=coverage_path,
        cluster_summary_json_path=cluster_json,
        cluster_summary_csv_path=cluster_csv,
    )


def _default_validation_output_paths(discovery_artifact_path: str | Path) -> tuple[Path, Path]:
    artifact = Path(discovery_artifact_path)
    return (
        artifact.with_name(f"{artifact.stem}_validation.json"),
        artifact.with_name(f"{artifact.stem}_validation.csv"),
    )


def run_notebook_discovery(
    *,
    experiment_config_path: str | Path,
    data_config_path: str | Path,
    checkpoint_path: str | Path,
    split_name: str = "discovery",
    output_path: str | Path | None = None,
    progress_ui: bool = True,
) -> NotebookDiscoveryRunResult:
    from tqdm.auto import tqdm

    from predictive_circuit_coding.cli.common import (
        require_checkpoint_matches_dataset,
        require_non_empty_split,
        require_runtime_view,
    )
    from predictive_circuit_coding.discovery import (
        build_discovery_cluster_report,
        discover_motifs_from_plan,
        prepare_discovery_collection,
        write_discovery_artifact,
        write_discovery_cluster_report_csv,
        write_discovery_cluster_report_json,
        write_discovery_coverage_summary,
    )
    from predictive_circuit_coding.training import load_experiment_config

    config = load_experiment_config(experiment_config_path)
    dataset_view = require_runtime_view(experiment_config=config, data_config_path=data_config_path)
    require_non_empty_split(dataset_view=dataset_view, split_name=split_name)
    checkpoint = require_checkpoint_matches_dataset(checkpoint_path=checkpoint_path, dataset_id=config.dataset_id)
    discovery_output_path = Path(output_path) if output_path is not None else _default_discovery_output_path(checkpoint, split_name)
    coverage_path = _coverage_summary_path(discovery_output_path)
    cluster_json_path, cluster_csv_path = _cluster_report_paths(discovery_output_path)

    plan_bar = tqdm(total=0, desc="Discovery coverage scan", unit="window", leave=False, disable=not progress_ui)

    def _plan_progress(current: int, total: int | None) -> None:
        if total is not None and plan_bar.total != total:
            plan_bar.total = total
        plan_bar.n = current
        plan_bar.refresh()

    window_plan = prepare_discovery_collection(
        experiment_config=config,
        data_config_path=data_config_path,
        split_name=split_name,
        dataset_view=dataset_view,
        progress_callback=_plan_progress if progress_ui else None,
    )
    plan_bar.close()
    write_discovery_coverage_summary(window_plan.coverage_summary, coverage_path)

    encode_bar = tqdm(
        total=int(window_plan.coverage_summary.selected_positive_count + window_plan.coverage_summary.selected_negative_count),
        desc="Selected-window discovery",
        unit="window",
        leave=False,
        disable=not progress_ui,
    )

    def _encode_progress(current: int, total: int | None) -> None:
        if total is not None and encode_bar.total != total:
            encode_bar.total = total
        encode_bar.n = current
        encode_bar.refresh()

    result = discover_motifs_from_plan(
        experiment_config=config,
        data_config_path=data_config_path,
        checkpoint_path=checkpoint,
        split_name=split_name,
        window_plan=window_plan,
        dataset_view=dataset_view,
        progress_callback=_encode_progress if progress_ui else None,
    )
    encode_bar.close()
    artifact = result.artifact
    write_discovery_artifact(artifact, discovery_output_path)
    cluster_report = build_discovery_cluster_report(artifact)
    write_discovery_cluster_report_json(cluster_report, cluster_json_path)
    write_discovery_cluster_report_csv(cluster_report, cluster_csv_path)

    return NotebookDiscoveryRunResult(
        discovery_artifact_path=discovery_output_path,
        decode_coverage_summary_path=coverage_path,
        cluster_summary_json_path=cluster_json_path,
        cluster_summary_csv_path=cluster_csv_path,
    )


def run_notebook_validation(
    *,
    experiment_config_path: str | Path,
    data_config_path: str | Path,
    checkpoint_path: str | Path,
    discovery_artifact_path: str | Path,
    output_json_path: str | Path | None = None,
    output_csv_path: str | Path | None = None,
    progress_ui: bool = True,
) -> NotebookValidationRunResult:
    from tqdm.auto import tqdm

    from predictive_circuit_coding.cli.common import (
        require_checkpoint_matches_dataset,
        require_discovery_artifact_matches_dataset,
        require_non_empty_split,
        require_runtime_view,
    )
    from predictive_circuit_coding.training import (
        load_experiment_config,
        write_validation_summary,
        write_validation_summary_csv,
    )
    from predictive_circuit_coding.validation import validate_discovery_artifact

    config = load_experiment_config(experiment_config_path)
    dataset_view = require_runtime_view(experiment_config=config, data_config_path=data_config_path)
    require_non_empty_split(dataset_view=dataset_view, split_name=config.splits.discovery)
    require_non_empty_split(dataset_view=dataset_view, split_name=config.splits.test)
    checkpoint = require_checkpoint_matches_dataset(checkpoint_path=checkpoint_path, dataset_id=config.dataset_id)
    artifact_path = require_discovery_artifact_matches_dataset(artifact_path=discovery_artifact_path, dataset_id=config.dataset_id)
    default_json, default_csv = _default_validation_output_paths(artifact_path)
    target_json = Path(output_json_path) if output_json_path is not None else default_json
    target_csv = Path(output_csv_path) if output_csv_path is not None else default_csv
    validation_bar = tqdm(
        total=int(config.evaluation.max_batches),
        desc="Held-out extraction / validation",
        unit="batch",
        leave=False,
        disable=not progress_ui,
    )

    def _validation_progress(current: int, total: int | None) -> None:
        if total is not None and validation_bar.total != total:
            validation_bar.total = total
        validation_bar.n = current
        validation_bar.refresh()

    summary = validate_discovery_artifact(
        experiment_config=config,
        data_config_path=data_config_path,
        checkpoint_path=checkpoint,
        discovery_artifact_path=artifact_path,
        dataset_view=dataset_view,
        progress_callback=_validation_progress if progress_ui else None,
    )
    validation_bar.close()
    write_validation_summary(summary, target_json)
    write_validation_summary_csv(summary, target_csv)
    return NotebookValidationRunResult(
        validation_summary_json_path=target_json,
        validation_summary_csv_path=target_csv,
    )


def collect_notebook_target_value_counts(
    *,
    experiment_config_path: str | Path,
    data_config_path: str | Path,
    split_name: str,
    target_label: str,
    target_label_mode: str = "auto",
) -> tuple[dict[str, object], ...]:
    import numpy as np

    from predictive_circuit_coding.cli.common import require_non_empty_split, require_runtime_view
    from predictive_circuit_coding.data import load_temporaldata_session
    from predictive_circuit_coding.decoding.labels import extract_matching_values_from_annotations
    from predictive_circuit_coding.tokenization import extract_sample_event_annotations
    from predictive_circuit_coding.training import load_experiment_config
    from predictive_circuit_coding.windowing import (
        FixedWindowConfig,
        build_dataset_bundle,
        build_sequential_fixed_window_sampler,
    )
    from predictive_circuit_coding.windowing.dataset import split_session_ids

    def _normalize_value(value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, bytes):
            normalized = value.decode("utf-8", errors="replace").strip()
        else:
            normalized = str(value).strip()
        if not normalized or normalized.lower() == "nan":
            return None
        return normalized

    config = load_experiment_config(experiment_config_path)
    dataset_view = require_runtime_view(experiment_config=config, data_config_path=data_config_path)
    require_non_empty_split(dataset_view=dataset_view, split_name=split_name)

    direct_counts: dict[str, int] = {}
    namespace, dot, field = str(target_label).partition(".")
    if dot and namespace and field and "." not in field:
        for session_id in split_session_ids(dataset_view.split_manifest, split_name):
            session_path = dataset_view.workspace.brainset_prepared_root / f"{session_id}.h5"
            if not session_path.is_file():
                continue
            session = load_temporaldata_session(session_path, lazy=False)
            interval = getattr(session, namespace, None)
            raw_values = getattr(interval, field, None) if interval is not None else None
            if raw_values is None:
                continue
            for raw_value in np.asarray(raw_values, dtype=object).reshape(-1):
                normalized = _normalize_value(raw_value)
                if normalized is None:
                    continue
                direct_counts[normalized] = direct_counts.get(normalized, 0) + 1
    if direct_counts:
        return tuple(
            {"value": value, "count": count}
            for value, count in sorted(direct_counts.items(), key=lambda item: (-item[1], item[0]))
        )

    bundle = build_dataset_bundle(
        workspace=dataset_view.workspace,
        split_manifest=dataset_view.split_manifest,
        split=split_name,
        config_dir=dataset_view.config_dir,
        config_name_prefix=dataset_view.config_name_prefix,
        dataset_split=dataset_view.dataset_split,
    )
    sampler = build_sequential_fixed_window_sampler(
        bundle.dataset,
        window=FixedWindowConfig(
            window_length_s=config.data_runtime.context_duration_s,
            step_s=config.evaluation.sequential_step_s,
        ),
    )
    counts: dict[str, int] = {}
    try:
        for item in sampler:
            sample = bundle.dataset.get(item.recording_id, item.start, item.end)
            annotations = extract_sample_event_annotations(
                sample,
                config.data_runtime,
                window_start_s=float(item.start),
                window_end_s=float(item.end),
            )
            values = extract_matching_values_from_annotations(
                annotations,
                target_label=target_label,
                target_label_mode=target_label_mode,
                window_duration_s=float(item.end) - float(item.start),
            )
            for value in values:
                counts[value] = counts.get(value, 0) + 1
    finally:
        if hasattr(bundle.dataset, "_close_open_files"):
            bundle.dataset._close_open_files()
    return tuple(
        {"value": value, "count": count}
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )


def inspect_notebook_target_field_availability(
    *,
    experiment_config_path: str | Path,
    data_config_path: str | Path,
    split_name: str,
    target_label: str,
    preview_limit: int = 5,
) -> dict[str, object]:
    import numpy as np

    from predictive_circuit_coding.cli.common import require_non_empty_split, require_runtime_view
    from predictive_circuit_coding.data import load_temporaldata_session
    from predictive_circuit_coding.training import load_experiment_config
    from predictive_circuit_coding.windowing.dataset import split_session_ids

    def _normalize_value(value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, bytes):
            normalized = value.decode("utf-8", errors="replace").strip()
        else:
            normalized = str(value).strip()
        if not normalized or normalized.lower() == "nan":
            return None
        return normalized

    namespace, dot, field = str(target_label).partition(".")
    if not dot or not namespace or not field or "." in field:
        raise ValueError(
            "inspect_notebook_target_field_availability requires a direct interval field target like "
            "'stimulus_presentations.image_name'."
        )

    config = load_experiment_config(experiment_config_path)
    dataset_view = require_runtime_view(experiment_config=config, data_config_path=data_config_path)
    require_non_empty_split(dataset_view=dataset_view, split_name=split_name)

    value_counts: dict[str, int] = {}
    session_rows: list[dict[str, object]] = []
    session_ids = split_session_ids(dataset_view.split_manifest, split_name)
    for session_id in session_ids:
        session_path = dataset_view.workspace.brainset_prepared_root / f"{session_id}.h5"
        has_session_file = session_path.is_file()
        has_namespace = False
        has_field = False
        non_null_value_count = 0
        preview_values: list[str] = []
        if has_session_file:
            session = load_temporaldata_session(session_path, lazy=False)
            interval = getattr(session, namespace, None)
            has_namespace = interval is not None
            raw_values = getattr(interval, field, None) if interval is not None else None
            has_field = raw_values is not None
            if raw_values is not None:
                for raw_value in np.asarray(raw_values, dtype=object).reshape(-1):
                    normalized = _normalize_value(raw_value)
                    if normalized is None:
                        continue
                    non_null_value_count += 1
                    if len(preview_values) < preview_limit:
                        preview_values.append(normalized)
                    value_counts[normalized] = value_counts.get(normalized, 0) + 1
        session_rows.append(
            {
                "session_id": str(session_id),
                "session_path": str(session_path),
                "has_session_file": has_session_file,
                "has_namespace": has_namespace,
                "has_field": has_field,
                "non_null_value_count": non_null_value_count,
                "preview_values": tuple(preview_values),
            }
        )

    return {
        "split_name": str(split_name),
        "target_label": str(target_label),
        "sessions_scanned": len(session_rows),
        "sessions_with_namespace": sum(1 for row in session_rows if row["has_namespace"]),
        "sessions_with_field": sum(1 for row in session_rows if row["has_field"]),
        "session_rows": tuple(session_rows),
        "value_counts": tuple(
            {"value": value, "count": count}
            for value, count in sorted(value_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
    }


def run_notebook_session_alignment_diagnostics(
    *,
    experiment_config_path: str | Path,
    data_config_path: str | Path,
    checkpoint_path: str | Path,
    neighbor_k: int,
    output_json_path: str | Path,
    output_csv_path: str | Path,
    split_name: str = "discovery",
    progress_ui: bool = True,
) -> NotebookAlignmentDiagnosticsRunResult:
    from tqdm.auto import tqdm

    from predictive_circuit_coding.cli.common import (
        require_checkpoint_matches_dataset,
        require_non_empty_split,
        require_runtime_view,
    )
    from predictive_circuit_coding.decoding.extract import extract_selected_discovery_windows
    from predictive_circuit_coding.decoding.geometry import (
        summarize_session_alignment_geometry,
        write_session_alignment_csv,
        write_session_alignment_json,
    )
    from predictive_circuit_coding.discovery import prepare_discovery_collection
    from predictive_circuit_coding.training import load_experiment_config

    config = load_experiment_config(experiment_config_path)
    dataset_view = require_runtime_view(experiment_config=config, data_config_path=data_config_path)
    require_non_empty_split(dataset_view=dataset_view, split_name=split_name)
    checkpoint = require_checkpoint_matches_dataset(checkpoint_path=checkpoint_path, dataset_id=config.dataset_id)
    shard_dir = Path(output_json_path).parent / "_tmp_alignment_token_shards"

    plan_bar = tqdm(total=0, desc="Alignment window scan", unit="window", leave=False, disable=not progress_ui)

    def _plan_progress(current: int, total: int | None) -> None:
        if total is not None and plan_bar.total != total:
            plan_bar.total = total
        plan_bar.n = current
        plan_bar.refresh()

    window_plan = prepare_discovery_collection(
        experiment_config=config,
        data_config_path=data_config_path,
        split_name=split_name,
        dataset_view=dataset_view,
        progress_callback=_plan_progress if progress_ui else None,
    )
    plan_bar.close()

    encode_bar = tqdm(
        total=int(window_plan.selected_indices.numel()),
        desc="Alignment encoding",
        unit="window",
        leave=False,
        disable=not progress_ui,
    )

    def _encode_progress(current: int, total: int | None) -> None:
        if total is not None and encode_bar.total != total:
            encode_bar.total = total
        encode_bar.n = current
        encode_bar.refresh()

    encoded = extract_selected_discovery_windows(
        experiment_config=config,
        data_config_path=data_config_path,
        checkpoint_path=checkpoint,
        window_plan=window_plan,
        dataset_view=dataset_view,
        shard_dir=shard_dir,
        progress_callback=_encode_progress if progress_ui else None,
    )
    encode_bar.close()

    summary = summarize_session_alignment_geometry(
        features=encoded.pooled_features,
        labels=encoded.labels,
        session_ids=encoded.window_session_ids,
        subject_ids=encoded.window_subject_ids,
        neighbor_k=neighbor_k,
    )
    write_session_alignment_json(summary, output_json_path)
    write_session_alignment_csv(summary, output_csv_path)
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    return NotebookAlignmentDiagnosticsRunResult(
        alignment_summary_json_path=Path(output_json_path),
        alignment_summary_csv_path=Path(output_csv_path),
    )


def run_notebook_geometry_diagnostics(
    *,
    experiment_config_path: str | Path,
    data_config_path: str | Path,
    checkpoint_path: str | Path,
    discovery_artifact_path: str | Path,
    neighbor_k: int,
    output_json_path: str | Path,
    output_csv_path: str | Path,
    candidate_output_json_path: str | Path,
    candidate_output_csv_path: str | Path,
    split_name: str = "discovery",
    progress_ui: bool = True,
) -> NotebookGeometryDiagnosticsRunResult:
    from tqdm.auto import tqdm

    from predictive_circuit_coding.cli.common import (
        require_checkpoint_matches_dataset,
        require_discovery_artifact_matches_dataset,
        require_non_empty_split,
        require_runtime_view,
    )
    from predictive_circuit_coding.decoding.extract import extract_selected_discovery_windows
    from predictive_circuit_coding.decoding.geometry import (
        summarize_candidate_neighbor_geometry,
        summarize_neighbor_geometry,
        write_neighbor_geometry_csv,
        write_neighbor_geometry_json,
    )
    from predictive_circuit_coding.discovery import prepare_discovery_collection
    from predictive_circuit_coding.training import load_experiment_config
    from predictive_circuit_coding.training.contracts import CandidateTokenRecord

    config = load_experiment_config(experiment_config_path)
    dataset_view = require_runtime_view(experiment_config=config, data_config_path=data_config_path)
    require_non_empty_split(dataset_view=dataset_view, split_name=split_name)
    checkpoint = require_checkpoint_matches_dataset(checkpoint_path=checkpoint_path, dataset_id=config.dataset_id)
    artifact_path = require_discovery_artifact_matches_dataset(
        artifact_path=discovery_artifact_path,
        dataset_id=config.dataset_id,
    )

    plan_bar = tqdm(total=0, desc="Geometry window scan", unit="window", leave=False, disable=not progress_ui)

    def _plan_progress(current: int, total: int | None) -> None:
        if total is not None and plan_bar.total != total:
            plan_bar.total = total
        plan_bar.n = current
        plan_bar.refresh()

    window_plan = prepare_discovery_collection(
        experiment_config=config,
        data_config_path=data_config_path,
        split_name=split_name,
        dataset_view=dataset_view,
        progress_callback=_plan_progress if progress_ui else None,
    )
    plan_bar.close()

    encode_bar = tqdm(
        total=int(window_plan.coverage_summary.selected_positive_count + window_plan.coverage_summary.selected_negative_count),
        desc="Geometry selected-window extraction",
        unit="window",
        leave=False,
        disable=not progress_ui,
    )

    def _encode_progress(current: int, total: int | None) -> None:
        if total is not None and encode_bar.total != total:
            encode_bar.total = total
        encode_bar.n = current
        encode_bar.refresh()

    shard_dir = Path(output_json_path).parent / "geometry_tmp"
    try:
        encoded = extract_selected_discovery_windows(
            experiment_config=config,
            data_config_path=data_config_path,
            checkpoint_path=checkpoint,
            window_plan=window_plan,
            dataset_view=dataset_view,
            shard_dir=shard_dir,
            progress_callback=_encode_progress if progress_ui else None,
        )
    finally:
        encode_bar.close()
        if shard_dir.exists():
            shutil.rmtree(shard_dir)

    window_summary = summarize_neighbor_geometry(
        features=encoded.pooled_features,
        attributes={
            "label": tuple("positive" if float(value) > 0.0 else "negative" for value in encoded.labels.tolist()),
            "session_id": encoded.window_session_ids,
            "subject_id": encoded.window_subject_ids,
        },
        neighbor_k=neighbor_k,
    )
    artifact_payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    candidates = tuple(
        CandidateTokenRecord(**candidate_payload)
        for candidate_payload in artifact_payload.get("candidates", [])
    )
    candidate_summary = summarize_candidate_neighbor_geometry(candidates=candidates, neighbor_k=neighbor_k)
    write_neighbor_geometry_json(window_summary, output_json_path)
    write_neighbor_geometry_csv(window_summary, output_csv_path)
    write_neighbor_geometry_json(candidate_summary, candidate_output_json_path)
    write_neighbor_geometry_csv(candidate_summary, candidate_output_csv_path)
    return NotebookGeometryDiagnosticsRunResult(
        window_geometry_summary_json_path=Path(output_json_path),
        window_geometry_summary_csv_path=Path(output_csv_path),
        candidate_geometry_summary_json_path=Path(candidate_output_json_path),
        candidate_geometry_summary_csv_path=Path(candidate_output_csv_path),
    )


def build_notebook_diagnostics_summary_row(
    *,
    experiment_name: str,
    discovery_artifact_path: str | Path,
    validation_summary_path: str | Path,
    cluster_summary_path: str | Path,
    window_geometry_summary_path: str | Path | None = None,
    candidate_geometry_summary_path: str | Path | None = None,
) -> dict[str, object]:
    discovery_payload = json.loads(Path(discovery_artifact_path).read_text(encoding="utf-8"))
    validation_payload = json.loads(Path(validation_summary_path).read_text(encoding="utf-8"))
    cluster_payload = json.loads(Path(cluster_summary_path).read_text(encoding="utf-8"))
    window_geometry_payload = (
        json.loads(Path(window_geometry_summary_path).read_text(encoding="utf-8"))
        if window_geometry_summary_path is not None and Path(window_geometry_summary_path).is_file()
        else {}
    )
    candidate_geometry_payload = (
        json.loads(Path(candidate_geometry_summary_path).read_text(encoding="utf-8"))
        if candidate_geometry_summary_path is not None and Path(candidate_geometry_summary_path).is_file()
        else {}
    )
    discovery_config = (discovery_payload.get("config_snapshot") or {}).get("discovery") or {}
    return {
        "experiment_name": str(experiment_name),
        "target_label": discovery_payload.get("decoder_summary", {}).get("target_label"),
        "target_label_match_value": discovery_config.get("target_label_match_value"),
        "candidate_count": validation_payload.get("candidate_count"),
        "cluster_count": validation_payload.get("cluster_count"),
        "real_probe_accuracy": validation_payload.get("real_label_metrics", {}).get("probe_accuracy"),
        "shuffled_probe_accuracy": validation_payload.get("shuffled_label_metrics", {}).get("probe_accuracy"),
        "held_out_test_probe_accuracy": validation_payload.get("held_out_test_metrics", {}).get("probe_accuracy"),
        "held_out_similarity_roc_auc": validation_payload.get("held_out_similarity_summary", {}).get("window_roc_auc"),
        "held_out_similarity_pr_auc": validation_payload.get("held_out_similarity_summary", {}).get("window_pr_auc"),
        "cluster_persistence_mean": validation_payload.get("cluster_quality_summary", {}).get("cluster_persistence_mean"),
        "silhouette_score": validation_payload.get("cluster_quality_summary", {}).get("silhouette_score"),
        "window_label_neighbor_enrichment": (((window_geometry_payload.get("metrics") or {}).get("label") or {}).get("enrichment_over_base")),
        "window_session_neighbor_enrichment": (((window_geometry_payload.get("metrics") or {}).get("session_id") or {}).get("enrichment_over_base")),
        "window_subject_neighbor_enrichment": (((window_geometry_payload.get("metrics") or {}).get("subject_id") or {}).get("enrichment_over_base")),
        "candidate_session_neighbor_enrichment": (((candidate_geometry_payload.get("metrics") or {}).get("session_id") or {}).get("enrichment_over_base")),
        "candidate_subject_neighbor_enrichment": (((candidate_geometry_payload.get("metrics") or {}).get("subject_id") or {}).get("enrichment_over_base")),
        "candidate_unit_region_neighbor_enrichment": (((candidate_geometry_payload.get("metrics") or {}).get("unit_region") or {}).get("enrichment_over_base")),
        "cluster_summary_path": str(Path(cluster_summary_path)),
    }


def build_notebook_alignment_summary_row(
    *,
    experiment_name: str,
    alignment_summary_path: str | Path,
) -> dict[str, object]:
    payload = json.loads(Path(alignment_summary_path).read_text(encoding="utf-8"))
    original_geometry = payload.get("geometry_original") or {}
    whitened_geometry = payload.get("geometry_whitened") or {}
    aligned_geometry = payload.get("geometry_aligned") or {}
    aggregate = payload.get("aggregate_metrics") or {}

    def _metric(summary: dict[str, object], attribute: str) -> float | None:
        return (((summary.get("metrics") or {}).get(attribute) or {}).get("enrichment_over_base"))

    return {
        "experiment_name": str(experiment_name),
        "experiment_type": "session_alignment",
        "reference_session_id": payload.get("reference_session_id"),
        "session_count": payload.get("session_count"),
        "sample_count": payload.get("sample_count"),
        "label_axis_cosine_before": aggregate.get("mean_label_axis_cosine_before"),
        "label_axis_cosine_after": aggregate.get("mean_label_axis_cosine_after"),
        "positive_centroid_cosine_before": aggregate.get("mean_positive_centroid_cosine_before"),
        "positive_centroid_cosine_after": aggregate.get("mean_positive_centroid_cosine_after"),
        "negative_centroid_cosine_before": aggregate.get("mean_negative_centroid_cosine_before"),
        "negative_centroid_cosine_after": aggregate.get("mean_negative_centroid_cosine_after"),
        "anchor_rmse_after_alignment": aggregate.get("mean_anchor_rmse_after_alignment"),
        "raw_label_neighbor_enrichment": _metric(original_geometry, "label"),
        "raw_session_neighbor_enrichment": _metric(original_geometry, "session_id"),
        "raw_subject_neighbor_enrichment": _metric(original_geometry, "subject_id"),
        "whitened_label_neighbor_enrichment": _metric(whitened_geometry, "label"),
        "whitened_session_neighbor_enrichment": _metric(whitened_geometry, "session_id"),
        "whitened_subject_neighbor_enrichment": _metric(whitened_geometry, "subject_id"),
        "aligned_label_neighbor_enrichment": _metric(aligned_geometry, "label"),
        "aligned_session_neighbor_enrichment": _metric(aligned_geometry, "session_id"),
        "aligned_subject_neighbor_enrichment": _metric(aligned_geometry, "subject_id"),
        "alignment_summary_path": str(Path(alignment_summary_path)),
    }


def write_notebook_diagnostics_summary(
    rows: list[dict[str, object]],
    *,
    output_json_path: str | Path,
    output_csv_path: str | Path,
) -> tuple[Path, Path]:
    json_path = Path(output_json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({"experiments": rows}, indent=2), encoding="utf-8")
    csv_path = Path(output_csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return json_path, csv_path


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
