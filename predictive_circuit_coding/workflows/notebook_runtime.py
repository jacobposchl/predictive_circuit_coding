from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import yaml

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


def prepare_notebook_runtime_context_from_experiment_config(
    *,
    base_experiment_config: str | Path,
    runtime_experiment_config: str | Path,
    artifact_root: str | Path,
    step_log_every: int,
    run_id: str | None = None,
) -> NotebookRuntimeContext:
    from predictive_circuit_coding.training import load_experiment_config

    base_config = load_experiment_config(base_experiment_config)
    runtime_path = Path(runtime_experiment_config)
    artifact_path = Path(artifact_root)
    checkpoint_dir = artifact_path / "checkpoints"
    summary_path = artifact_path / "training_summary.json"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    artifact_path.mkdir(parents=True, exist_ok=True)

    resolved_run_id = str(run_id).strip() if run_id is not None else datetime.now().strftime("run_%Y%m%d_%H%M%S")
    if not resolved_run_id:
        raise ValueError("run_id must not be empty")

    payload = base_config.to_dict()
    payload.pop("config_path", None)
    payload.setdefault("training", {})["log_every_steps"] = int(step_log_every)
    artifacts = payload.setdefault("artifacts", {})
    artifacts["checkpoint_dir"] = str(checkpoint_dir.resolve())
    artifacts["summary_path"] = str(summary_path.resolve())

    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    exported_runtime_config_path = artifact_path / "colab_runtime_experiment.yaml"
    exported_runtime_config_path.write_text(runtime_path.read_text(encoding="utf-8"), encoding="utf-8")

    checkpoint_prefix = str(artifacts.get("checkpoint_prefix", "pcc"))
    checkpoint_path = checkpoint_dir / f"{checkpoint_prefix}_best.pt"
    runtime_subset = payload.get("runtime_subset") or {}
    runtime_split_manifest_path = (
        Path(str(runtime_subset["split_manifest_path"]))
        if runtime_subset.get("split_manifest_path")
        else artifact_path / "runtime_subset" / "selected_split_manifest.json"
    )
    runtime_session_catalog_path = (
        Path(str(runtime_subset["session_catalog_path"]))
        if runtime_subset.get("session_catalog_path")
        else artifact_path / "runtime_subset" / "selected_session_catalog.json"
    )
    runtime_config_dir = (
        Path(str(runtime_subset["config_dir"]))
        if runtime_subset.get("config_dir")
        else artifact_path / "runtime_subset" / "splits"
    )
    config_name_prefix = str(runtime_subset.get("config_name_prefix", "torch_brain_runtime"))

    profile_path = artifact_path / "colab_notebook_profile.json"
    profile_payload = {
        "run_id": resolved_run_id,
        "runtime_experiment_config": str(runtime_path.resolve()),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "summary_path": str(summary_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "runtime_split_manifest_path": str(runtime_split_manifest_path.resolve()),
        "runtime_session_catalog_path": str(runtime_session_catalog_path.resolve()),
        "runtime_config_dir": str(runtime_config_dir.resolve()),
        "config_name_prefix": config_name_prefix,
        "step_log_every": int(step_log_every),
        "dataset_selection_active": base_config.dataset_selection.is_active,
        "runtime_subset_active": base_config.runtime_subset is not None,
    }
    profile_path.write_text(json.dumps(profile_payload, indent=2), encoding="utf-8")
    return NotebookRuntimeContext(
        experiment_config_path=runtime_path,
        checkpoint_dir=checkpoint_dir,
        summary_path=summary_path,
        checkpoint_path=checkpoint_path,
        run_id=resolved_run_id,
        selected_session_count=0,
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


def materialize_notebook_prepared_sessions(
    *,
    source_dataset_root: str | Path,
    target_dataset_root: str | Path,
    session_ids: list[str] | tuple[str, ...],
    dataset_id: str,
    reset_target: bool = True,
    note_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> NotebookLocalDatasetStageResult:
    def _note(message: str) -> None:
        if note_callback is not None:
            note_callback(message)

    source_root = Path(source_dataset_root).resolve()
    target_root = Path(target_dataset_root).expanduser()
    staged_session_ids = tuple(sorted(dict.fromkeys(str(session_id) for session_id in session_ids if str(session_id).strip())))
    if not staged_session_ids:
        raise ValueError("session_ids must contain at least one session to stage locally")
    if not source_root.is_dir():
        _note(f"Source dataset root was not found: {source_root}")
        raise FileNotFoundError(f"Source dataset root not found: {source_root}")

    if target_root.exists() or target_root.is_symlink():
        if not reset_target:
            raise FileExistsError(f"Target dataset root already exists: {target_root}")
        _note(f"Refreshing local dataset workspace: {target_root}")
        if target_root.is_symlink() or target_root.is_file():
            target_root.unlink()
        else:
            shutil.rmtree(target_root)

    target_root.mkdir(parents=True, exist_ok=True)
    source_prepared_root = source_root / "prepared" / str(dataset_id)
    if not source_prepared_root.is_dir():
        _note(f"Prepared session directory was not found: {source_prepared_root}")
        raise FileNotFoundError(f"Prepared dataset root not found: {source_prepared_root}")
    target_prepared_root = target_root / "prepared" / str(dataset_id)
    target_prepared_root.mkdir(parents=True, exist_ok=True)

    _note(f"Copying {len(staged_session_ids)} prepared session files into the local workspace.")
    for index, session_id in enumerate(staged_session_ids, start=1):
        source_path = source_prepared_root / f"{session_id}.h5"
        if not source_path.is_file():
            _note(f"Prepared session file was not found: {source_path}")
            raise FileNotFoundError(f"Prepared session not found: {source_path}")
        shutil.copy2(source_path, target_prepared_root / source_path.name)
        if progress_callback is not None:
            progress_callback(index, len(staged_session_ids))

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

    _note(f"Local data staging complete: {len(staged_session_ids)} sessions ready.")
    return NotebookLocalDatasetStageResult(
        target_dataset_root=target_root.resolve(),
        target_prepared_root=target_prepared_root.resolve(),
        staged_session_ids=staged_session_ids,
        copied_support_files=tuple(copied_support_files),
    )


def resolve_notebook_training_export_path(
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
    train_export_path = resolve_notebook_training_export_path(
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


def describe_notebook_compute_targets(*, experiment_config_path: str | Path) -> dict[str, str]:
    from predictive_circuit_coding.training.runtime import resolve_device

    config_path = Path(experiment_config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Notebook experiment config must contain a mapping: {config_path}")
    execution = raw.get("execution") or {}
    if not isinstance(execution, dict):
        raise ValueError(f"'execution' section must be a mapping in {config_path}")
    encoder_device = str(resolve_device(str(execution.get("device", "cpu"))))
    return {
        "encoder_device": encoder_device,
        "probe_device": "cpu",
        "clustering_device": "cpu",
        "metrics_device": "cpu",
    }
