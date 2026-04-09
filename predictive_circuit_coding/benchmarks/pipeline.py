from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import shutil
from pathlib import Path
import sys
from typing import Any, Iterable, TextIO

import yaml

from predictive_circuit_coding.benchmarks.config import load_notebook_pipeline_config
from predictive_circuit_coding.benchmarks.contracts import (
    BenchmarkArmSpec,
    BenchmarkTaskSpec,
    PipelineRunManifest,
    PipelineStageState,
)
from predictive_circuit_coding.benchmarks.reports import build_final_project_summary, write_single_row_summary, write_summary_rows
from predictive_circuit_coding.benchmarks.run import (
    default_benchmark_task_specs,
    default_motif_arm_specs,
    default_representation_arm_specs,
    run_motif_benchmark_matrix,
    run_representation_benchmark_matrix,
)
from predictive_circuit_coding.data import create_workspace, load_preparation_config
from predictive_circuit_coding.evaluation import evaluate_checkpoint_on_split
from predictive_circuit_coding.training import load_experiment_config, write_evaluation_summary
from predictive_circuit_coding.utils.notebook import (
    NotebookDatasetConfig,
    NotebookTrainingConfig,
    prepare_notebook_runtime_context,
    prepare_notebook_runtime_context_from_experiment_config,
    resolve_notebook_checkpoint,
    run_notebook_session_alignment_diagnostics,
    run_streaming_command,
)


_PIPELINE_STAGE_ORDER = (
    "train",
    "evaluate",
    "representation_benchmark",
    "motif_benchmark",
    "alignment_diagnostic",
    "final_reports",
)


@dataclass(frozen=True)
class NotebookPipelinePaths:
    run_id: str
    local_run_root: Path
    drive_run_root: Path | None
    train_root: Path
    evaluation_root: Path
    representation_root: Path
    motif_root: Path
    diagnostics_root: Path
    reports_root: Path
    pipeline_root: Path
    runtime_experiment_config_path: Path
    pipeline_config_snapshot_path: Path
    pipeline_manifest_path: Path
    pipeline_state_path: Path


@dataclass(frozen=True)
class NotebookPipelineRunResult:
    run_id: str
    local_run_root: Path
    drive_run_root: Path | None
    runtime_experiment_config_path: Path
    checkpoint_path: Path
    training_summary_path: Path
    evaluation_summary_paths: tuple[Path, ...]
    representation_summary_json_path: Path
    representation_summary_csv_path: Path
    motif_summary_json_path: Path
    motif_summary_csv_path: Path
    final_summary_json_path: Path
    final_summary_csv_path: Path
    alignment_summary_json_path: Path | None
    alignment_summary_csv_path: Path | None
    pipeline_manifest_path: Path
    pipeline_state_path: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _resolve_latest_run_id(drive_export_root: str | Path) -> str | None:
    export_root = Path(drive_export_root)
    if not export_root.is_dir():
        return None
    candidates = sorted(
        [
            path.name
            for path in export_root.iterdir()
            if path.is_dir() and path.name.startswith("run_") and (path / "run_1" / "train").is_dir()
        ]
    )
    return candidates[-1] if candidates else None


def _build_pipeline_paths(
    *,
    local_artifact_root: str | Path,
    drive_export_root: str | Path | None,
    run_id: str,
    run_name: str = "run_1",
) -> NotebookPipelinePaths:
    local_run_root = Path(local_artifact_root) / "pipeline_runs" / str(run_id) / run_name
    drive_run_root = Path(drive_export_root) / str(run_id) / run_name if drive_export_root is not None else None
    pipeline_root = local_run_root / "pipeline"
    return NotebookPipelinePaths(
        run_id=str(run_id),
        local_run_root=local_run_root,
        drive_run_root=drive_run_root,
        train_root=local_run_root / "train",
        evaluation_root=local_run_root / "evaluation",
        representation_root=local_run_root / "benchmarks" / "representation",
        motif_root=local_run_root / "benchmarks" / "motifs",
        diagnostics_root=local_run_root / "benchmarks" / "diagnostics",
        reports_root=local_run_root / "reports",
        pipeline_root=pipeline_root,
        runtime_experiment_config_path=local_run_root / "train" / "colab_runtime_experiment.yaml",
        pipeline_config_snapshot_path=pipeline_root / "pipeline_config_snapshot.yaml",
        pipeline_manifest_path=pipeline_root / "pipeline_manifest.json",
        pipeline_state_path=pipeline_root / "pipeline_state.json",
    )


def _sync_path(source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        return
    if target.exists():
        target.unlink()
    shutil.copy2(source, target)


def _restore_drive_run_if_present(paths: NotebookPipelinePaths) -> None:
    if paths.drive_run_root is None or not paths.drive_run_root.exists() or paths.local_run_root.exists():
        return
    paths.local_run_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(paths.drive_run_root, paths.local_run_root)


def _sync_run_relatives(paths: NotebookPipelinePaths, relatives: Iterable[str]) -> None:
    if paths.drive_run_root is None:
        return
    for relative in relatives:
        source = paths.local_run_root / relative
        target = paths.drive_run_root / relative
        _sync_path(source, target)


def _load_pipeline_state(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload.get("stages", {}))


def _write_pipeline_state(
    *,
    path: Path,
    states: dict[str, dict[str, Any]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"stages": states}, indent=2), encoding="utf-8")
    return path


def _write_pipeline_manifest(
    *,
    paths: NotebookPipelinePaths,
    dataset_id: str,
    created_at_utc: str,
) -> Path:
    manifest = PipelineRunManifest(
        run_id=paths.run_id,
        dataset_id=str(dataset_id),
        stage_order=_PIPELINE_STAGE_ORDER,
        local_run_root=str(paths.local_run_root),
        drive_run_root=(str(paths.drive_run_root) if paths.drive_run_root is not None else None),
        config_snapshot_path=str(paths.pipeline_config_snapshot_path),
        created_at_utc=created_at_utc,
        updated_at_utc=_utc_now(),
    )
    paths.pipeline_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    paths.pipeline_manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return paths.pipeline_manifest_path


def _path_identity(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.exists():
        return {"path": str(resolved), "exists": False}
    if resolved.is_dir():
        return {"path": str(resolved), "exists": True, "type": "dir"}
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "exists": True,
        "type": "file",
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _outputs_exist(outputs: dict[str, Any]) -> bool:
    for value in outputs.values():
        if value in (None, "", [], {}):
            continue
        if isinstance(value, (list, tuple)):
            if not all(Path(item).exists() for item in value):
                return False
            continue
        if not Path(value).exists():
            return False
    return True


def _stage_is_reusable(
    *,
    states: dict[str, dict[str, Any]],
    stage_name: str,
    config_hash: str,
    inputs: dict[str, Any],
) -> bool:
    state = states.get(stage_name)
    if state is None:
        return False
    if state.get("status") != "complete":
        return False
    if state.get("config_hash") != config_hash:
        return False
    if state.get("inputs") != inputs:
        return False
    outputs = dict(state.get("outputs") or {})
    return _outputs_exist(outputs)


def _set_stage_state(
    *,
    states: dict[str, dict[str, Any]],
    paths: NotebookPipelinePaths,
    stage_name: str,
    status: str,
    config_hash: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> None:
    created_at = states.get(stage_name, {}).get("created_at_utc", _utc_now())
    payload = PipelineStageState(
        stage_name=stage_name,
        status=status,
        config_hash=config_hash,
        inputs=inputs,
        outputs=outputs or {},
        created_at_utc=created_at,
        updated_at_utc=_utc_now(),
        error_message=error_message,
    )
    states[stage_name] = payload.to_dict()
    _write_pipeline_state(path=paths.pipeline_state_path, states=states)


def _summary_rows_from_json(path: Path, root_key: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get(root_key, []))


def _select_task_specs(
    *,
    include_image_identity: bool,
    image_target_name: str | None,
    requested_names: tuple[str, ...] | None,
) -> tuple[BenchmarkTaskSpec, ...]:
    default_specs = default_benchmark_task_specs(
        include_image_identity=include_image_identity,
        image_target_name=image_target_name,
    )
    if not requested_names:
        return default_specs
    requested = {str(name) for name in requested_names}
    return tuple(spec for spec in default_specs if spec.name in requested)


def _select_arm_specs(
    specs: tuple[BenchmarkArmSpec, ...],
    requested_names: tuple[str, ...] | None,
) -> tuple[BenchmarkArmSpec, ...]:
    if not requested_names:
        return specs
    requested = {str(name) for name in requested_names}
    return tuple(spec for spec in specs if spec.name in requested)


def prepare_or_restore_training_stage(
    *,
    paths: NotebookPipelinePaths,
    states: dict[str, dict[str, Any]],
    base_experiment_config: str | Path,
    data_config_path: str | Path,
    dataset_config: NotebookDatasetConfig,
    training_config: NotebookTrainingConfig,
    step_log_every: int,
    run_stage_train: bool,
    use_experiment_config_as_is: bool,
    output_stream: TextIO,
) -> dict[str, Any]:
    prep_config = load_preparation_config(data_config_path)
    workspace = create_workspace(prep_config)
    stage_config = {
        "base_experiment_config": str(Path(base_experiment_config).resolve()),
        "data_config_path": str(Path(data_config_path).resolve()),
        "dataset_config": dataset_config.__dict__,
        "training_config": training_config.__dict__,
        "step_log_every": int(step_log_every),
        "run_stage_train": bool(run_stage_train),
    }
    config_hash = _json_hash(stage_config)
    inputs = {
        "base_experiment_config": _path_identity(base_experiment_config),
        "data_config_path": _path_identity(data_config_path),
        "workspace_root": _path_identity(workspace.root),
    }
    if _stage_is_reusable(states=states, stage_name="train", config_hash=config_hash, inputs=inputs):
        return dict(states["train"]["outputs"])

    _set_stage_state(
        states=states,
        paths=paths,
        stage_name="train",
        status="running",
        config_hash=config_hash,
        inputs=inputs,
    )
    try:
        if use_experiment_config_as_is:
            context = prepare_notebook_runtime_context_from_experiment_config(
                base_experiment_config=base_experiment_config,
                runtime_experiment_config=paths.runtime_experiment_config_path,
                artifact_root=paths.train_root,
                step_log_every=step_log_every,
                run_id=paths.run_id,
            )
        else:
            context = prepare_notebook_runtime_context(
                base_experiment_config=base_experiment_config,
                runtime_experiment_config=paths.runtime_experiment_config_path,
                session_catalog_csv=workspace.session_catalog_csv_path,
                artifact_root=paths.train_root,
                dataset_config=dataset_config,
                training_config=training_config,
                step_log_every=step_log_every,
                run_id=paths.run_id,
            )
        runtime_config = load_experiment_config(context.experiment_config_path)
        checkpoint_prefix = runtime_config.artifacts.checkpoint_prefix
        checkpoint_path = context.checkpoint_path
        if context.summary_path.exists() or any(context.checkpoint_dir.glob(f"{checkpoint_prefix}_*.pt")):
            checkpoint_path = resolve_notebook_checkpoint(
                summary_path=context.summary_path,
                checkpoint_dir=context.checkpoint_dir,
                checkpoint_prefix=checkpoint_prefix,
            )

        if run_stage_train:
            run_streaming_command(
                [
                    sys.executable,
                    "-m",
                    "predictive_circuit_coding.cli.train",
                    "--config",
                    str(context.experiment_config_path),
                    "--data-config",
                    str(Path(data_config_path).resolve()),
                ],
                stream=output_stream,
                cwd=Path.cwd(),
                step_log_every=step_log_every,
            )
            checkpoint_path = resolve_notebook_checkpoint(
                summary_path=context.summary_path,
                checkpoint_dir=context.checkpoint_dir,
                checkpoint_prefix=checkpoint_prefix,
            )
        elif not checkpoint_path.exists():
            raise FileNotFoundError(
                "Training stage is disabled, but no checkpoint was found for the selected run. "
                "Enable RUN_STAGE_TRAIN or point PIPELINE_RUN_ID at an exported run with train artifacts."
            )

        outputs = {
            "runtime_experiment_config_path": str(context.experiment_config_path),
            "exported_runtime_config_path": str(context.exported_runtime_config_path),
            "checkpoint_path": str(checkpoint_path),
            "training_summary_path": str(context.summary_path),
            "train_root": str(paths.train_root),
        }
        _set_stage_state(
            states=states,
            paths=paths,
            stage_name="train",
            status="complete",
            config_hash=config_hash,
            inputs=inputs,
            outputs=outputs,
        )
        _sync_run_relatives(paths, ("train", "pipeline"))
        return outputs
    except Exception as exc:
        _set_stage_state(
            states=states,
            paths=paths,
            stage_name="train",
            status="failed",
            config_hash=config_hash,
            inputs=inputs,
            error_message=str(exc),
        )
        raise


def run_standard_evaluation_stage(
    *,
    paths: NotebookPipelinePaths,
    states: dict[str, dict[str, Any]],
    runtime_experiment_config_path: str | Path,
    data_config_path: str | Path,
    checkpoint_path: str | Path,
    run_stage_evaluate: bool,
    evaluation_splits: tuple[str, ...] = ("valid", "test"),
) -> dict[str, Any]:
    stage_config = {
        "runtime_experiment_config_path": str(Path(runtime_experiment_config_path).resolve()),
        "data_config_path": str(Path(data_config_path).resolve()),
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "run_stage_evaluate": bool(run_stage_evaluate),
        "evaluation_splits": list(evaluation_splits),
    }
    config_hash = _json_hash(stage_config)
    inputs = {
        "runtime_experiment_config_path": _path_identity(runtime_experiment_config_path),
        "data_config_path": _path_identity(data_config_path),
        "checkpoint_path": _path_identity(checkpoint_path),
    }
    if _stage_is_reusable(states=states, stage_name="evaluate", config_hash=config_hash, inputs=inputs):
        return dict(states["evaluate"]["outputs"])
    if not run_stage_evaluate:
        outputs = {"evaluation_summary_paths": []}
        _set_stage_state(
            states=states,
            paths=paths,
            stage_name="evaluate",
            status="complete",
            config_hash=config_hash,
            inputs=inputs,
            outputs=outputs,
        )
        _sync_run_relatives(paths, ("pipeline",))
        return outputs

    _set_stage_state(
        states=states,
        paths=paths,
        stage_name="evaluate",
        status="running",
        config_hash=config_hash,
        inputs=inputs,
    )
    try:
        config = load_experiment_config(runtime_experiment_config_path)
        outputs_list: list[str] = []
        paths.evaluation_root.mkdir(parents=True, exist_ok=True)
        for split_name in evaluation_splits:
            summary = evaluate_checkpoint_on_split(
                experiment_config=config,
                data_config_path=data_config_path,
                checkpoint_path=str(checkpoint_path),
                split_name=split_name,
            )
            output_path = paths.evaluation_root / f"{split_name}_evaluation.json"
            write_evaluation_summary(summary, output_path)
            outputs_list.append(str(output_path))
        outputs = {"evaluation_summary_paths": outputs_list}
        _set_stage_state(
            states=states,
            paths=paths,
            stage_name="evaluate",
            status="complete",
            config_hash=config_hash,
            inputs=inputs,
            outputs=outputs,
        )
        _sync_run_relatives(paths, ("evaluation", "pipeline"))
        return outputs
    except Exception as exc:
        _set_stage_state(
            states=states,
            paths=paths,
            stage_name="evaluate",
            status="failed",
            config_hash=config_hash,
            inputs=inputs,
            error_message=str(exc),
        )
        raise


def run_representation_benchmark_stage(
    *,
    paths: NotebookPipelinePaths,
    states: dict[str, dict[str, Any]],
    runtime_experiment_config_path: str | Path,
    data_config_path: str | Path,
    checkpoint_path: str | Path,
    run_stage_representation_benchmark: bool,
    task_specs: tuple[BenchmarkTaskSpec, ...],
    arm_specs: tuple[BenchmarkArmSpec, ...],
    session_holdout_fraction: float,
    session_holdout_seed: int | None,
    neighbor_k: int,
) -> dict[str, Any]:
    stage_config = {
        "runtime_experiment_config_path": str(Path(runtime_experiment_config_path).resolve()),
        "data_config_path": str(Path(data_config_path).resolve()),
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "run_stage_representation_benchmark": bool(run_stage_representation_benchmark),
        "task_specs": [task.to_dict() for task in task_specs],
        "arm_specs": [arm.to_dict() for arm in arm_specs],
        "session_holdout_fraction": float(session_holdout_fraction),
        "session_holdout_seed": session_holdout_seed,
        "neighbor_k": int(neighbor_k),
    }
    config_hash = _json_hash(stage_config)
    inputs = {
        "runtime_experiment_config_path": _path_identity(runtime_experiment_config_path),
        "checkpoint_path": _path_identity(checkpoint_path),
        "data_config_path": _path_identity(data_config_path),
    }
    if _stage_is_reusable(states=states, stage_name="representation_benchmark", config_hash=config_hash, inputs=inputs):
        return dict(states["representation_benchmark"]["outputs"])
    if not run_stage_representation_benchmark:
        outputs = {
            "representation_summary_json_path": str(paths.reports_root / "representation_benchmark_summary.json"),
            "representation_summary_csv_path": str(paths.reports_root / "representation_benchmark_summary.csv"),
        }
        _set_stage_state(
            states=states,
            paths=paths,
            stage_name="representation_benchmark",
            status="complete",
            config_hash=config_hash,
            inputs=inputs,
            outputs=outputs,
        )
        _sync_run_relatives(paths, ("pipeline",))
        return outputs

    _set_stage_state(
        states=states,
        paths=paths,
        stage_name="representation_benchmark",
        status="running",
        config_hash=config_hash,
        inputs=inputs,
    )
    try:
        config = load_experiment_config(runtime_experiment_config_path)
        results = run_representation_benchmark_matrix(
            experiment_config=config,
            data_config_path=data_config_path,
            checkpoint_path=checkpoint_path,
            output_root=paths.representation_root,
            task_specs=task_specs,
            arm_specs=arm_specs,
            session_holdout_fraction=session_holdout_fraction,
            session_holdout_seed=session_holdout_seed,
            neighbor_k=neighbor_k,
        )
        rows = [result.summary for result in results]
        summary_json_path, summary_csv_path = write_summary_rows(
            rows,
            output_json_path=paths.reports_root / "representation_benchmark_summary.json",
            output_csv_path=paths.reports_root / "representation_benchmark_summary.csv",
            root_key="representation_benchmark",
        )
        outputs = {
            "representation_summary_json_path": str(summary_json_path),
            "representation_summary_csv_path": str(summary_csv_path),
        }
        _set_stage_state(
            states=states,
            paths=paths,
            stage_name="representation_benchmark",
            status="complete",
            config_hash=config_hash,
            inputs=inputs,
            outputs=outputs,
        )
        _sync_run_relatives(paths, ("benchmarks", "reports", "pipeline"))
        return outputs
    except Exception as exc:
        _set_stage_state(
            states=states,
            paths=paths,
            stage_name="representation_benchmark",
            status="failed",
            config_hash=config_hash,
            inputs=inputs,
            error_message=str(exc),
        )
        raise


def run_motif_benchmark_stage(
    *,
    paths: NotebookPipelinePaths,
    states: dict[str, dict[str, Any]],
    runtime_experiment_config_path: str | Path,
    data_config_path: str | Path,
    checkpoint_path: str | Path,
    run_stage_motif_benchmark: bool,
    task_specs: tuple[BenchmarkTaskSpec, ...],
    arm_specs: tuple[BenchmarkArmSpec, ...],
    session_holdout_fraction: float,
    session_holdout_seed: int | None,
    debug_retain_intermediates: bool,
) -> dict[str, Any]:
    stage_config = {
        "runtime_experiment_config_path": str(Path(runtime_experiment_config_path).resolve()),
        "data_config_path": str(Path(data_config_path).resolve()),
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "run_stage_motif_benchmark": bool(run_stage_motif_benchmark),
        "task_specs": [task.to_dict() for task in task_specs],
        "arm_specs": [arm.to_dict() for arm in arm_specs],
        "session_holdout_fraction": float(session_holdout_fraction),
        "session_holdout_seed": session_holdout_seed,
        "debug_retain_intermediates": bool(debug_retain_intermediates),
    }
    config_hash = _json_hash(stage_config)
    inputs = {
        "runtime_experiment_config_path": _path_identity(runtime_experiment_config_path),
        "checkpoint_path": _path_identity(checkpoint_path),
        "data_config_path": _path_identity(data_config_path),
    }
    if _stage_is_reusable(states=states, stage_name="motif_benchmark", config_hash=config_hash, inputs=inputs):
        return dict(states["motif_benchmark"]["outputs"])
    if not run_stage_motif_benchmark:
        outputs = {
            "motif_summary_json_path": str(paths.reports_root / "motif_benchmark_summary.json"),
            "motif_summary_csv_path": str(paths.reports_root / "motif_benchmark_summary.csv"),
        }
        _set_stage_state(
            states=states,
            paths=paths,
            stage_name="motif_benchmark",
            status="complete",
            config_hash=config_hash,
            inputs=inputs,
            outputs=outputs,
        )
        _sync_run_relatives(paths, ("pipeline",))
        return outputs

    _set_stage_state(
        states=states,
        paths=paths,
        stage_name="motif_benchmark",
        status="running",
        config_hash=config_hash,
        inputs=inputs,
    )
    try:
        config = load_experiment_config(runtime_experiment_config_path)
        results = run_motif_benchmark_matrix(
            experiment_config=config,
            data_config_path=data_config_path,
            checkpoint_path=checkpoint_path,
            output_root=paths.motif_root,
            task_specs=task_specs,
            arm_specs=arm_specs,
            session_holdout_fraction=session_holdout_fraction,
            session_holdout_seed=session_holdout_seed,
            debug_retain_intermediates=debug_retain_intermediates,
        )
        rows = [result.summary for result in results]
        summary_json_path, summary_csv_path = write_summary_rows(
            rows,
            output_json_path=paths.reports_root / "motif_benchmark_summary.json",
            output_csv_path=paths.reports_root / "motif_benchmark_summary.csv",
            root_key="motif_benchmark",
        )
        outputs = {
            "motif_summary_json_path": str(summary_json_path),
            "motif_summary_csv_path": str(summary_csv_path),
        }
        _set_stage_state(
            states=states,
            paths=paths,
            stage_name="motif_benchmark",
            status="complete",
            config_hash=config_hash,
            inputs=inputs,
            outputs=outputs,
        )
        _sync_run_relatives(paths, ("benchmarks", "reports", "pipeline"))
        return outputs
    except Exception as exc:
        _set_stage_state(
            states=states,
            paths=paths,
            stage_name="motif_benchmark",
            status="failed",
            config_hash=config_hash,
            inputs=inputs,
            error_message=str(exc),
        )
        raise


def run_optional_alignment_diagnostic_stage(
    *,
    paths: NotebookPipelinePaths,
    states: dict[str, dict[str, Any]],
    runtime_experiment_config_path: str | Path,
    data_config_path: str | Path,
    checkpoint_path: str | Path,
    run_stage_alignment_diagnostic: bool,
    neighbor_k: int,
) -> dict[str, Any]:
    stage_config = {
        "runtime_experiment_config_path": str(Path(runtime_experiment_config_path).resolve()),
        "data_config_path": str(Path(data_config_path).resolve()),
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "run_stage_alignment_diagnostic": bool(run_stage_alignment_diagnostic),
        "neighbor_k": int(neighbor_k),
    }
    config_hash = _json_hash(stage_config)
    inputs = {
        "runtime_experiment_config_path": _path_identity(runtime_experiment_config_path),
        "checkpoint_path": _path_identity(checkpoint_path),
        "data_config_path": _path_identity(data_config_path),
    }
    if _stage_is_reusable(states=states, stage_name="alignment_diagnostic", config_hash=config_hash, inputs=inputs):
        return dict(states["alignment_diagnostic"]["outputs"])
    if not run_stage_alignment_diagnostic:
        outputs = {
            "alignment_summary_json_path": None,
            "alignment_summary_csv_path": None,
        }
        _set_stage_state(
            states=states,
            paths=paths,
            stage_name="alignment_diagnostic",
            status="complete",
            config_hash=config_hash,
            inputs=inputs,
            outputs=outputs,
        )
        _sync_run_relatives(paths, ("pipeline",))
        return outputs

    _set_stage_state(
        states=states,
        paths=paths,
        stage_name="alignment_diagnostic",
        status="running",
        config_hash=config_hash,
        inputs=inputs,
    )
    try:
        output_json_path = paths.diagnostics_root / "session_alignment_summary.json"
        output_csv_path = paths.diagnostics_root / "session_alignment_summary.csv"
        paths.diagnostics_root.mkdir(parents=True, exist_ok=True)
        run_notebook_session_alignment_diagnostics(
            experiment_config_path=runtime_experiment_config_path,
            data_config_path=data_config_path,
            checkpoint_path=checkpoint_path,
            neighbor_k=neighbor_k,
            output_json_path=output_json_path,
            output_csv_path=output_csv_path,
            progress_ui=False,
        )
        outputs = {
            "alignment_summary_json_path": str(output_json_path),
            "alignment_summary_csv_path": str(output_csv_path),
        }
        _set_stage_state(
            states=states,
            paths=paths,
            stage_name="alignment_diagnostic",
            status="complete",
            config_hash=config_hash,
            inputs=inputs,
            outputs=outputs,
        )
        _sync_run_relatives(paths, ("benchmarks", "pipeline"))
        return outputs
    except Exception as exc:
        _set_stage_state(
            states=states,
            paths=paths,
            stage_name="alignment_diagnostic",
            status="failed",
            config_hash=config_hash,
            inputs=inputs,
            error_message=str(exc),
        )
        raise


def write_final_project_reports(
    *,
    paths: NotebookPipelinePaths,
    states: dict[str, dict[str, Any]],
    run_stage_final_reports: bool = True,
) -> dict[str, Any]:
    representation_rows = _summary_rows_from_json(
        paths.reports_root / "representation_benchmark_summary.json",
        "representation_benchmark",
    )
    motif_rows = _summary_rows_from_json(
        paths.reports_root / "motif_benchmark_summary.json",
        "motif_benchmark",
    )
    stage_config = {
        "run_stage_final_reports": bool(run_stage_final_reports),
        "representation_row_count": len(representation_rows),
        "motif_row_count": len(motif_rows),
    }
    config_hash = _json_hash(stage_config)
    inputs = {
        "representation_summary_json_path": _path_identity(paths.reports_root / "representation_benchmark_summary.json"),
        "motif_summary_json_path": _path_identity(paths.reports_root / "motif_benchmark_summary.json"),
    }
    if _stage_is_reusable(states=states, stage_name="final_reports", config_hash=config_hash, inputs=inputs):
        return dict(states["final_reports"]["outputs"])

    _set_stage_state(
        states=states,
        paths=paths,
        stage_name="final_reports",
        status="running",
        config_hash=config_hash,
        inputs=inputs,
    )
    try:
        summary_payload = build_final_project_summary(
            representation_rows=representation_rows,
            motif_rows=motif_rows,
        )
        summary_json_path, summary_csv_path = write_single_row_summary(
            summary_payload,
            output_json_path=paths.reports_root / "final_project_summary.json",
            output_csv_path=paths.reports_root / "final_project_summary.csv",
            root_key="final_project_summary",
        )
        outputs = {
            "final_summary_json_path": str(summary_json_path),
            "final_summary_csv_path": str(summary_csv_path),
        }
        _set_stage_state(
            states=states,
            paths=paths,
            stage_name="final_reports",
            status="complete",
            config_hash=config_hash,
            inputs=inputs,
            outputs=outputs,
        )
        _sync_run_relatives(paths, ("reports", "pipeline"))
        return outputs
    except Exception as exc:
        _set_stage_state(
            states=states,
            paths=paths,
            stage_name="final_reports",
            status="failed",
            config_hash=config_hash,
            inputs=inputs,
            error_message=str(exc),
        )
        raise


def run_notebook_pipeline(
    *,
    base_experiment_config: str | Path,
    data_config_path: str | Path,
    drive_export_root: str | Path | None,
    local_artifact_root: str | Path,
    pipeline_run_id: str | None,
    dataset_config: NotebookDatasetConfig,
    training_config: NotebookTrainingConfig,
    step_log_every: int,
    run_stage_train: bool,
    run_stage_evaluate: bool,
    run_stage_representation_benchmark: bool,
    run_stage_motif_benchmark: bool,
    run_stage_alignment_diagnostic: bool,
    run_stage_image_identity_appendix: bool,
    image_target_name: str | None,
    representation_task_names: tuple[str, ...] | None = None,
    motif_task_names: tuple[str, ...] | None = None,
    representation_arm_names: tuple[str, ...] | None = None,
    motif_arm_names: tuple[str, ...] | None = None,
    pca_components: int = 64,
    session_holdout_fraction: float = 0.5,
    session_holdout_seed: int | None = None,
    neighbor_k: int = 5,
    debug_retain_intermediates: bool = False,
    use_experiment_config_as_is: bool = False,
    output_stream: TextIO | None = None,
) -> NotebookPipelineRunResult:
    resolved_run_id = str(pipeline_run_id).strip() if pipeline_run_id is not None else ""
    if not resolved_run_id:
        if run_stage_train:
            resolved_run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        else:
            if drive_export_root is None:
                raise FileNotFoundError("PIPELINE_RUN_ID is required when training is disabled and no Drive export root is available.")
            latest_run_id = _resolve_latest_run_id(drive_export_root)
            if latest_run_id is None:
                raise FileNotFoundError("No exported training runs were found under the Drive export root.")
            resolved_run_id = latest_run_id

    paths = _build_pipeline_paths(
        local_artifact_root=local_artifact_root,
        drive_export_root=drive_export_root,
        run_id=resolved_run_id,
    )
    _restore_drive_run_if_present(paths)
    paths.local_run_root.mkdir(parents=True, exist_ok=True)
    paths.pipeline_root.mkdir(parents=True, exist_ok=True)
    pipeline_config_payload = {
        "base_experiment_config": str(Path(base_experiment_config).resolve()),
        "data_config_path": str(Path(data_config_path).resolve()),
        "drive_export_root": (str(Path(drive_export_root).resolve()) if drive_export_root is not None else None),
        "dataset_config": dataset_config.__dict__,
        "training_config": training_config.__dict__,
        "step_log_every": int(step_log_every),
        "run_stage_train": bool(run_stage_train),
        "run_stage_evaluate": bool(run_stage_evaluate),
        "run_stage_representation_benchmark": bool(run_stage_representation_benchmark),
        "run_stage_motif_benchmark": bool(run_stage_motif_benchmark),
        "run_stage_alignment_diagnostic": bool(run_stage_alignment_diagnostic),
        "run_stage_image_identity_appendix": bool(run_stage_image_identity_appendix),
        "image_target_name": image_target_name,
        "representation_task_names": list(representation_task_names or ()),
        "motif_task_names": list(motif_task_names or ()),
        "representation_arm_names": list(representation_arm_names or ()),
        "motif_arm_names": list(motif_arm_names or ()),
        "pca_components": int(pca_components),
        "session_holdout_fraction": float(session_holdout_fraction),
        "session_holdout_seed": session_holdout_seed,
        "neighbor_k": int(neighbor_k),
        "debug_retain_intermediates": bool(debug_retain_intermediates),
        "use_experiment_config_as_is": bool(use_experiment_config_as_is),
    }
    paths.pipeline_config_snapshot_path.write_text(
        yaml.safe_dump(pipeline_config_payload, sort_keys=False),
        encoding="utf-8",
    )
    train_config = load_experiment_config(base_experiment_config)
    created_at = (
        json.loads(paths.pipeline_manifest_path.read_text(encoding="utf-8")).get("created_at_utc")
        if paths.pipeline_manifest_path.is_file()
        else _utc_now()
    )
    _write_pipeline_manifest(paths=paths, dataset_id=train_config.dataset_id, created_at_utc=str(created_at))
    states = _load_pipeline_state(paths.pipeline_state_path)

    output_stream = output_stream or __import__("sys").stdout
    train_outputs = prepare_or_restore_training_stage(
        paths=paths,
        states=states,
        base_experiment_config=base_experiment_config,
        data_config_path=data_config_path,
        dataset_config=dataset_config,
        training_config=training_config,
        step_log_every=step_log_every,
        run_stage_train=run_stage_train,
        use_experiment_config_as_is=use_experiment_config_as_is,
        output_stream=output_stream,
    )
    runtime_experiment_config_path = Path(train_outputs["runtime_experiment_config_path"])
    checkpoint_path = Path(train_outputs["checkpoint_path"])
    training_summary_path = Path(train_outputs["training_summary_path"])

    evaluation_outputs = run_standard_evaluation_stage(
        paths=paths,
        states=states,
        runtime_experiment_config_path=runtime_experiment_config_path,
        data_config_path=data_config_path,
        checkpoint_path=checkpoint_path,
        run_stage_evaluate=run_stage_evaluate,
    )

    representation_task_specs = _select_task_specs(
        include_image_identity=run_stage_image_identity_appendix,
        image_target_name=image_target_name,
        requested_names=representation_task_names,
    )
    motif_task_specs = _select_task_specs(
        include_image_identity=run_stage_image_identity_appendix,
        image_target_name=image_target_name,
        requested_names=motif_task_names,
    )
    representation_task_specs = tuple(spec for spec in representation_task_specs if spec.include_in_representation)
    motif_task_specs = tuple(spec for spec in motif_task_specs if spec.include_in_motifs)
    representation_arm_specs = _select_arm_specs(
        default_representation_arm_specs(pca_components=pca_components),
        representation_arm_names,
    )
    motif_arm_specs = _select_arm_specs(
        default_motif_arm_specs(pca_components=pca_components),
        motif_arm_names,
    )

    representation_outputs = run_representation_benchmark_stage(
        paths=paths,
        states=states,
        runtime_experiment_config_path=runtime_experiment_config_path,
        data_config_path=data_config_path,
        checkpoint_path=checkpoint_path,
        run_stage_representation_benchmark=run_stage_representation_benchmark,
        task_specs=representation_task_specs,
        arm_specs=representation_arm_specs,
        session_holdout_fraction=session_holdout_fraction,
        session_holdout_seed=session_holdout_seed,
        neighbor_k=neighbor_k,
    )
    motif_outputs = run_motif_benchmark_stage(
        paths=paths,
        states=states,
        runtime_experiment_config_path=runtime_experiment_config_path,
        data_config_path=data_config_path,
        checkpoint_path=checkpoint_path,
        run_stage_motif_benchmark=run_stage_motif_benchmark,
        task_specs=motif_task_specs,
        arm_specs=motif_arm_specs,
        session_holdout_fraction=session_holdout_fraction,
        session_holdout_seed=session_holdout_seed,
        debug_retain_intermediates=debug_retain_intermediates,
    )
    alignment_outputs = run_optional_alignment_diagnostic_stage(
        paths=paths,
        states=states,
        runtime_experiment_config_path=runtime_experiment_config_path,
        data_config_path=data_config_path,
        checkpoint_path=checkpoint_path,
        run_stage_alignment_diagnostic=run_stage_alignment_diagnostic,
        neighbor_k=neighbor_k,
    )
    final_outputs = write_final_project_reports(paths=paths, states=states)

    return NotebookPipelineRunResult(
        run_id=paths.run_id,
        local_run_root=paths.local_run_root,
        drive_run_root=paths.drive_run_root,
        runtime_experiment_config_path=runtime_experiment_config_path,
        checkpoint_path=checkpoint_path,
        training_summary_path=training_summary_path,
        evaluation_summary_paths=tuple(Path(path) for path in evaluation_outputs.get("evaluation_summary_paths", [])),
        representation_summary_json_path=Path(representation_outputs["representation_summary_json_path"]),
        representation_summary_csv_path=Path(representation_outputs["representation_summary_csv_path"]),
        motif_summary_json_path=Path(motif_outputs["motif_summary_json_path"]),
        motif_summary_csv_path=Path(motif_outputs["motif_summary_csv_path"]),
        final_summary_json_path=Path(final_outputs["final_summary_json_path"]),
        final_summary_csv_path=Path(final_outputs["final_summary_csv_path"]),
        alignment_summary_json_path=(
            Path(alignment_outputs["alignment_summary_json_path"])
            if alignment_outputs.get("alignment_summary_json_path")
            else None
        ),
        alignment_summary_csv_path=(
            Path(alignment_outputs["alignment_summary_csv_path"])
            if alignment_outputs.get("alignment_summary_csv_path")
            else None
        ),
        pipeline_manifest_path=paths.pipeline_manifest_path,
        pipeline_state_path=paths.pipeline_state_path,
    )


def run_notebook_pipeline_from_config(
    *,
    pipeline_config_path: str | Path,
    pipeline_run_id: str | None = None,
    output_stream: TextIO | None = None,
) -> NotebookPipelineRunResult:
    config = load_notebook_pipeline_config(pipeline_config_path)
    return run_notebook_pipeline(
        base_experiment_config=config.experiment_config_path,
        data_config_path=config.data_config_path,
        drive_export_root=config.drive_export_root,
        local_artifact_root=config.local_artifact_root,
        pipeline_run_id=pipeline_run_id,
        dataset_config=NotebookDatasetConfig(),
        training_config=NotebookTrainingConfig(),
        step_log_every=config.step_log_every,
        run_stage_train=config.run_stage_train,
        run_stage_evaluate=config.run_stage_evaluate,
        run_stage_representation_benchmark=config.run_stage_representation_benchmark,
        run_stage_motif_benchmark=config.run_stage_motif_benchmark,
        run_stage_alignment_diagnostic=config.run_stage_alignment_diagnostic,
        run_stage_image_identity_appendix=config.run_stage_image_identity_appendix,
        image_target_name=config.image_target_name,
        representation_task_names=config.representation_task_names,
        motif_task_names=config.motif_task_names,
        representation_arm_names=config.representation_arm_names,
        motif_arm_names=config.motif_arm_names,
        pca_components=config.pca_components,
        session_holdout_fraction=config.session_holdout_fraction,
        session_holdout_seed=config.session_holdout_seed,
        neighbor_k=config.neighbor_k,
        debug_retain_intermediates=config.debug_retain_intermediates,
        use_experiment_config_as_is=True,
        output_stream=output_stream,
    )


def resume_notebook_pipeline(**kwargs) -> NotebookPipelineRunResult:
    return run_notebook_pipeline(**kwargs)
