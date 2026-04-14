from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, TextIO

import yaml

from predictive_circuit_coding.benchmarks.config import load_notebook_pipeline_config
from predictive_circuit_coding.benchmarks.contracts import BenchmarkArmSpec, BenchmarkTaskSpec, PipelineRunManifest, PipelineStageState
from predictive_circuit_coding.benchmarks.reports import build_final_project_summary, write_single_row_summary, write_summary_rows
from predictive_circuit_coding.benchmarks.run import default_benchmark_task_specs, default_motif_arm_specs, run_motif_benchmark_matrix
from predictive_circuit_coding.data import build_workspace, load_preparation_config, load_session_catalog
from predictive_circuit_coding.evaluation import evaluate_checkpoint_on_split
from predictive_circuit_coding.training import load_experiment_config, train_model, write_evaluation_summary
from predictive_circuit_coding.utils.notebook import (
    NotebookDatasetConfig,
    NotebookProgressConfig,
    NotebookProgressUI,
    NotebookStageSummary,
    NotebookTrainingConfig,
    materialize_notebook_prepared_sessions,
    prepare_notebook_runtime_context_from_experiment_config,
    resolve_notebook_checkpoint,
)


_PIPELINE_STAGE_ORDER = ("train", "evaluate", "refinement", "alignment_diagnostic", "final_reports")


@dataclass(frozen=True)
class NotebookPipelinePaths:
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
class NotebookPipelineRunResult:
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _path_identity(path: str | Path | None) -> str | None:
    return str(Path(path).resolve()) if path is not None else None


def _resolve_latest_run_id(drive_export_root: str | Path) -> str | None:
    root = Path(drive_export_root)
    if not root.is_dir():
        return None
    candidates = sorted(path.name for path in root.iterdir() if path.is_dir() and path.name.startswith("run_"))
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
        refinement_root=local_run_root / "refinement",
        diagnostics_root=local_run_root / "diagnostics",
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
    else:
        shutil.copy2(source, target)


def _sync_run_relatives(paths: NotebookPipelinePaths, relatives: Iterable[str]) -> None:
    if paths.drive_run_root is None:
        return
    for relative in relatives:
        _sync_path(paths.local_run_root / relative, paths.drive_run_root / relative)


def _load_pipeline_state(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    stages = payload.get("stages", payload)
    return {str(key): dict(value) for key, value in stages.items() if isinstance(value, dict)}


def _write_pipeline_state(path: Path, states: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"stages": states}, indent=2), encoding="utf-8")


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
    previous = states.get(stage_name, {})
    states[stage_name] = PipelineStageState(
        stage_name=stage_name,
        status=status,
        config_hash=config_hash,
        inputs=inputs,
        outputs=outputs or {},
        created_at_utc=str(previous.get("created_at_utc") or _utc_now()),
        updated_at_utc=_utc_now(),
        error_message=error_message,
    ).to_dict()
    _write_pipeline_state(paths.pipeline_state_path, states)


def _stage_is_reusable(
    *,
    states: dict[str, dict[str, Any]],
    stage_name: str,
    config_hash: str,
    inputs: dict[str, Any],
) -> bool:
    state = states.get(stage_name) or {}
    return (
        state.get("status") == "complete"
        and state.get("config_hash") == config_hash
        and dict(state.get("inputs") or {}) == inputs
    )


def _write_pipeline_manifest(paths: NotebookPipelinePaths, *, dataset_id: str, created_at_utc: str) -> None:
    manifest = PipelineRunManifest(
        run_id=paths.run_id,
        dataset_id=dataset_id,
        stage_order=_PIPELINE_STAGE_ORDER,
        local_run_root=str(paths.local_run_root),
        drive_run_root=str(paths.drive_run_root) if paths.drive_run_root is not None else None,
        config_snapshot_path=str(paths.pipeline_config_snapshot_path),
        created_at_utc=created_at_utc,
        updated_at_utc=_utc_now(),
    )
    paths.pipeline_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    paths.pipeline_manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")


def _select_task_specs(requested_names: tuple[str, ...] | None) -> tuple[BenchmarkTaskSpec, ...]:
    specs = default_benchmark_task_specs()
    if requested_names is None:
        return specs
    by_name = {spec.name: spec for spec in specs}
    missing = [name for name in requested_names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown refinement task names: {missing}")
    return tuple(by_name[name] for name in requested_names)


def _select_arm_specs(specs: tuple[BenchmarkArmSpec, ...], requested_names: tuple[str, ...] | None) -> tuple[BenchmarkArmSpec, ...]:
    if requested_names is None:
        return specs
    by_name = {spec.name: spec for spec in specs}
    missing = [name for name in requested_names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown refinement arm names: {missing}")
    return tuple(by_name[name] for name in requested_names)


def _summary_rows_from_json(path: Path, root_key: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get(root_key, [])
    return [dict(row) for row in rows if isinstance(row, dict)]


def _stage_summary(stage_name: str, outputs: dict[str, Any], status: str) -> NotebookStageSummary:
    return NotebookStageSummary(
        stage_name=stage_name,
        status=status,
        headline=f"{stage_name}: {status}",
        metrics={},
        artifact_paths={key: str(value) for key, value in outputs.items() if value},
    )


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
    use_experiment_config_as_is: bool = False,
    output_stream: TextIO | None = None,
    progress_ui: NotebookProgressUI | None = None,
    debug_retain_intermediates: bool = False,
) -> dict[str, Any]:
    del debug_retain_intermediates
    stage_config = {
        "base_experiment_config": _path_identity(base_experiment_config),
        "data_config_path": _path_identity(data_config_path),
        "run_stage_train": bool(run_stage_train),
        "use_experiment_config_as_is": bool(use_experiment_config_as_is),
        "dataset_config": dataset_config.__dict__,
        "training_config": training_config.__dict__,
        "step_log_every": int(step_log_every),
    }
    config_hash = _json_hash(stage_config)
    inputs = {"base_experiment_config": _path_identity(base_experiment_config), "data_config_path": _path_identity(data_config_path)}
    if _stage_is_reusable(states=states, stage_name="train", config_hash=config_hash, inputs=inputs):
        return dict(states["train"]["outputs"])

    paths.train_root.mkdir(parents=True, exist_ok=True)
    if progress_ui is not None:
        progress_ui.start_stage(stage_name="train", total=1, description="Train")

    if run_stage_train:
        if use_experiment_config_as_is:
            source_config = load_experiment_config(base_experiment_config)
            paths.runtime_experiment_config_path.write_text(
                yaml.safe_dump(source_config.to_dict(), sort_keys=False),
                encoding="utf-8",
            )
            runtime_config = load_experiment_config(paths.runtime_experiment_config_path)
        else:
            context = prepare_notebook_runtime_context_from_experiment_config(
                base_experiment_config=base_experiment_config,
                runtime_experiment_config=paths.runtime_experiment_config_path,
                artifact_root=paths.train_root,
                step_log_every=step_log_every,
                run_id=paths.run_id,
            )
            runtime_config = load_experiment_config(context.experiment_config_path)
        result = train_model(
            experiment_config=runtime_config,
            data_config_path=data_config_path,
            train_split=runtime_config.splits.train,
            valid_split=runtime_config.splits.valid,
        )
        outputs = {
            "runtime_experiment_config_path": str(paths.runtime_experiment_config_path),
            "checkpoint_path": str(result.checkpoint_path),
            "training_summary_path": str(result.summary_path),
            "training_history_json_path": str(result.history_json_path),
            "training_history_csv_path": str(result.history_csv_path),
        }
    else:
        checkpoint = resolve_notebook_checkpoint(
            summary_path=paths.train_root / "training_summary.json",
            checkpoint_dir=paths.train_root / "checkpoints",
            checkpoint_prefix=load_experiment_config(paths.runtime_experiment_config_path).artifacts.checkpoint_prefix,
        )
        outputs = {
            "runtime_experiment_config_path": str(paths.runtime_experiment_config_path),
            "checkpoint_path": str(checkpoint),
            "training_summary_path": str(paths.train_root / "training_summary.json"),
            "training_history_json_path": None,
            "training_history_csv_path": None,
        }

    _set_stage_state(states=states, paths=paths, stage_name="train", status="complete", config_hash=config_hash, inputs=inputs, outputs=outputs)
    _sync_run_relatives(paths, ("train", "pipeline"))
    if progress_ui is not None:
        progress_ui.finish_stage(_stage_summary("train", outputs, "complete"))
        progress_ui.advance_pipeline()
    return outputs


def run_standard_evaluation_stage(
    *,
    paths: NotebookPipelinePaths,
    states: dict[str, dict[str, Any]],
    runtime_experiment_config_path: str | Path,
    data_config_path: str | Path,
    checkpoint_path: str | Path,
    run_stage_evaluate: bool,
    progress_ui: NotebookProgressUI | None = None,
    debug_retain_intermediates: bool = False,
) -> dict[str, Any]:
    del debug_retain_intermediates
    stage_config = {
        "runtime_experiment_config_path": _path_identity(runtime_experiment_config_path),
        "data_config_path": _path_identity(data_config_path),
        "checkpoint_path": _path_identity(checkpoint_path),
        "run_stage_evaluate": bool(run_stage_evaluate),
    }
    config_hash = _json_hash(stage_config)
    inputs = {
        "runtime_experiment_config_path": _path_identity(runtime_experiment_config_path),
        "checkpoint_path": _path_identity(checkpoint_path),
        "data_config_path": _path_identity(data_config_path),
    }
    if _stage_is_reusable(states=states, stage_name="evaluate", config_hash=config_hash, inputs=inputs):
        return dict(states["evaluate"]["outputs"])
    if not run_stage_evaluate:
        outputs = {"evaluation_summary_paths": []}
        _set_stage_state(states=states, paths=paths, stage_name="evaluate", status="complete", config_hash=config_hash, inputs=inputs, outputs=outputs)
        return outputs

    config = load_experiment_config(runtime_experiment_config_path)
    paths.evaluation_root.mkdir(parents=True, exist_ok=True)
    summaries: list[str] = []
    for split_name in (config.splits.valid, config.splits.test):
        summary = evaluate_checkpoint_on_split(
            experiment_config=config,
            data_config_path=data_config_path,
            checkpoint_path=str(checkpoint_path),
            split_name=split_name,
        )
        summary_path = paths.evaluation_root / f"{split_name}_summary.json"
        write_evaluation_summary(summary, summary_path)
        summaries.append(str(summary_path))
    outputs = {"evaluation_summary_paths": summaries}
    _set_stage_state(states=states, paths=paths, stage_name="evaluate", status="complete", config_hash=config_hash, inputs=inputs, outputs=outputs)
    _sync_run_relatives(paths, ("evaluation", "pipeline"))
    if progress_ui is not None:
        progress_ui.finish_stage(_stage_summary("evaluate", outputs, "complete"))
        progress_ui.advance_pipeline()
    return outputs


def run_refinement_stage(
    *,
    paths: NotebookPipelinePaths,
    states: dict[str, dict[str, Any]],
    runtime_experiment_config_path: str | Path,
    data_config_path: str | Path,
    checkpoint_path: str | Path,
    run_stage_refinement: bool,
    task_specs: tuple[BenchmarkTaskSpec, ...],
    arm_specs: tuple[BenchmarkArmSpec, ...],
    session_holdout_fraction: float,
    session_holdout_seed: int | None,
    debug_retain_intermediates: bool,
    progress_ui: NotebookProgressUI | None = None,
) -> dict[str, Any]:
    stage_config = {
        "runtime_experiment_config_path": _path_identity(runtime_experiment_config_path),
        "data_config_path": _path_identity(data_config_path),
        "checkpoint_path": _path_identity(checkpoint_path),
        "run_stage_refinement": bool(run_stage_refinement),
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
    if _stage_is_reusable(states=states, stage_name="refinement", config_hash=config_hash, inputs=inputs):
        return dict(states["refinement"]["outputs"])
    if not run_stage_refinement:
        outputs = {
            "refinement_summary_json_path": str(paths.reports_root / "refinement_summary.json"),
            "refinement_summary_csv_path": str(paths.reports_root / "refinement_summary.csv"),
        }
        _set_stage_state(states=states, paths=paths, stage_name="refinement", status="complete", config_hash=config_hash, inputs=inputs, outputs=outputs)
        return outputs

    config = load_experiment_config(runtime_experiment_config_path)
    results = run_motif_benchmark_matrix(
        experiment_config=config,
        data_config_path=data_config_path,
        checkpoint_path=checkpoint_path,
        output_root=paths.refinement_root,
        task_specs=task_specs,
        arm_specs=arm_specs,
        session_holdout_fraction=session_holdout_fraction,
        session_holdout_seed=session_holdout_seed,
        debug_retain_intermediates=debug_retain_intermediates,
    )
    rows = [result.summary for result in results]
    summary_json_path, summary_csv_path = write_summary_rows(
        rows,
        output_json_path=paths.reports_root / "refinement_summary.json",
        output_csv_path=paths.reports_root / "refinement_summary.csv",
        root_key="refinement",
    )
    outputs = {
        "refinement_summary_json_path": str(summary_json_path),
        "refinement_summary_csv_path": str(summary_csv_path),
    }
    _set_stage_state(states=states, paths=paths, stage_name="refinement", status="complete", config_hash=config_hash, inputs=inputs, outputs=outputs)
    _sync_run_relatives(paths, ("refinement", "reports", "pipeline"))
    if progress_ui is not None:
        progress_ui.finish_stage(_stage_summary("refinement", outputs, "complete"))
        progress_ui.advance_pipeline()
    return outputs


def run_optional_alignment_diagnostic_stage(
    *,
    paths: NotebookPipelinePaths,
    states: dict[str, dict[str, Any]],
    runtime_experiment_config_path: str | Path,
    data_config_path: str | Path,
    checkpoint_path: str | Path,
    run_stage_alignment_diagnostic: bool,
    neighbor_k: int,
    progress_ui: NotebookProgressUI | None = None,
    debug_retain_intermediates: bool = False,
) -> dict[str, Any]:
    del runtime_experiment_config_path, data_config_path, checkpoint_path, neighbor_k, debug_retain_intermediates
    config_hash = _json_hash({"run_stage_alignment_diagnostic": bool(run_stage_alignment_diagnostic)})
    inputs: dict[str, Any] = {}
    outputs = {"alignment_summary_json_path": None, "alignment_summary_csv_path": None}
    _set_stage_state(states=states, paths=paths, stage_name="alignment_diagnostic", status="complete", config_hash=config_hash, inputs=inputs, outputs=outputs)
    if progress_ui is not None and run_stage_alignment_diagnostic:
        progress_ui.finish_stage(_stage_summary("alignment_diagnostic", outputs, "complete"))
        progress_ui.advance_pipeline()
    return outputs


def write_final_project_reports(
    *,
    paths: NotebookPipelinePaths,
    states: dict[str, dict[str, Any]],
    run_stage_final_reports: bool = True,
    progress_ui: NotebookProgressUI | None = None,
) -> dict[str, Any]:
    rows = _summary_rows_from_json(paths.reports_root / "refinement_summary.json", "refinement")
    config_hash = _json_hash({"run_stage_final_reports": bool(run_stage_final_reports), "row_count": len(rows)})
    inputs = {"refinement_summary_json_path": _path_identity(paths.reports_root / "refinement_summary.json")}
    if _stage_is_reusable(states=states, stage_name="final_reports", config_hash=config_hash, inputs=inputs):
        return dict(states["final_reports"]["outputs"])
    payload = build_final_project_summary(motif_rows=rows)
    summary_json_path, summary_csv_path = write_single_row_summary(
        payload,
        output_json_path=paths.reports_root / "final_project_summary.json",
        output_csv_path=paths.reports_root / "final_project_summary.csv",
        root_key="final_project_summary",
    )
    outputs = {"final_summary_json_path": str(summary_json_path), "final_summary_csv_path": str(summary_csv_path)}
    _set_stage_state(states=states, paths=paths, stage_name="final_reports", status="complete", config_hash=config_hash, inputs=inputs, outputs=outputs)
    _sync_run_relatives(paths, ("reports", "pipeline"))
    if progress_ui is not None:
        progress_ui.finish_stage(_stage_summary("final_reports", outputs, "complete"))
        progress_ui.advance_pipeline()
    return outputs


def _resolve_source_session_ids(*, source_dataset_root: str | Path, dataset_id: str) -> list[str]:
    source_root = Path(source_dataset_root).resolve()
    catalog_path = source_root / "manifests" / "session_catalog.json"
    if catalog_path.is_file():
        catalog = load_session_catalog(catalog_path)
        session_ids = sorted({record.session_id for record in catalog.records})
        if session_ids:
            return session_ids
    prepared_root = source_root / "prepared" / str(dataset_id)
    session_ids = sorted(path.stem for path in prepared_root.glob("*.h5"))
    if session_ids:
        return session_ids
    raise FileNotFoundError(
        f"No prepared sessions found under {prepared_root}. Populate the source dataset root or disable local staging."
    )


def _ensure_local_prepared_sessions(
    *,
    data_config_path: str | Path,
    source_dataset_root: str | Path | None,
    stage_prepared_sessions_locally: bool,
) -> None:
    prep_config = load_preparation_config(data_config_path)
    workspace = build_workspace(prep_config)
    if any(workspace.brainset_prepared_root.glob("*.h5")):
        return
    if not stage_prepared_sessions_locally:
        return
    if source_dataset_root is None:
        raise FileNotFoundError(
            "No prepared sessions were found under the local workspace and paths.source_dataset_root is not set. "
            "Either stage prepared sessions locally or run local data preparation before training."
        )
    if Path(source_dataset_root).resolve() == workspace.root.resolve():
        return
    session_ids = _resolve_source_session_ids(
        source_dataset_root=source_dataset_root,
        dataset_id=prep_config.dataset.dataset_id,
    )
    materialize_notebook_prepared_sessions(
        source_dataset_root=source_dataset_root,
        target_dataset_root=workspace.root,
        session_ids=session_ids,
        dataset_id=prep_config.dataset.dataset_id,
        reset_target=True,
    )


def run_notebook_pipeline(
    *,
    base_experiment_config: str | Path,
    data_config_path: str | Path,
    drive_export_root: str | Path | None,
    local_artifact_root: str | Path,
    source_dataset_root: str | Path | None = None,
    pipeline_run_id: str | None,
    dataset_config: NotebookDatasetConfig,
    training_config: NotebookTrainingConfig,
    step_log_every: int,
    run_stage_train: bool,
    run_stage_evaluate: bool,
    run_stage_refinement: bool,
    run_stage_alignment_diagnostic: bool,
    motif_task_names: tuple[str, ...] | None = None,
    motif_arm_names: tuple[str, ...] | None = None,
    session_holdout_fraction: float = 0.5,
    session_holdout_seed: int | None = None,
    neighbor_k: int = 5,
    debug_retain_intermediates: bool = False,
    stage_prepared_sessions_locally: bool = False,
    use_experiment_config_as_is: bool = False,
    output_stream: TextIO | None = None,
    progress_ui: bool = True,
    notebook_progress_config: NotebookProgressConfig | None = None,
) -> NotebookPipelineRunResult:
    _ensure_local_prepared_sessions(
        data_config_path=data_config_path,
        source_dataset_root=source_dataset_root,
        stage_prepared_sessions_locally=stage_prepared_sessions_locally,
    )
    resolved_run_id = str(pipeline_run_id).strip() if pipeline_run_id is not None else ""
    if not resolved_run_id:
        if run_stage_train:
            resolved_run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        else:
            latest = _resolve_latest_run_id(drive_export_root) if drive_export_root is not None else None
            if latest is None:
                raise FileNotFoundError("PIPELINE_RUN_ID is required when training is disabled.")
            resolved_run_id = latest

    paths = _build_pipeline_paths(local_artifact_root=local_artifact_root, drive_export_root=drive_export_root, run_id=resolved_run_id)
    paths.local_run_root.mkdir(parents=True, exist_ok=True)
    paths.pipeline_root.mkdir(parents=True, exist_ok=True)
    paths.pipeline_config_snapshot_path.write_text(
        yaml.safe_dump(
            {
                "base_experiment_config": _path_identity(base_experiment_config),
                "data_config_path": _path_identity(data_config_path),
                "run_stage_train": bool(run_stage_train),
                "run_stage_evaluate": bool(run_stage_evaluate),
                "run_stage_refinement": bool(run_stage_refinement),
                "run_stage_alignment_diagnostic": bool(run_stage_alignment_diagnostic),
                "motif_task_names": list(motif_task_names or ()),
                "motif_arm_names": list(motif_arm_names or ()),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config = load_experiment_config(base_experiment_config)
    created_at = (
        json.loads(paths.pipeline_manifest_path.read_text(encoding="utf-8")).get("created_at_utc")
        if paths.pipeline_manifest_path.is_file()
        else _utc_now()
    )
    _write_pipeline_manifest(paths, dataset_id=config.dataset_id, created_at_utc=str(created_at))
    states = _load_pipeline_state(paths.pipeline_state_path)

    output_stream = output_stream or __import__("sys").stdout
    progress_config = notebook_progress_config or NotebookProgressConfig(enabled=bool(progress_ui))
    ui = NotebookProgressUI(config=progress_config, stream=output_stream) if progress_ui and progress_config.enabled else None
    if ui is not None:
        enabled = [run_stage_train, run_stage_evaluate, run_stage_refinement, run_stage_alignment_diagnostic, True]
        ui.start_pipeline(total_stages=sum(1 for value in enabled if value), completed_stages=0)

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
        progress_ui=ui,
        debug_retain_intermediates=debug_retain_intermediates,
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
        progress_ui=ui,
        debug_retain_intermediates=debug_retain_intermediates,
    )
    refinement_outputs = run_refinement_stage(
        paths=paths,
        states=states,
        runtime_experiment_config_path=runtime_experiment_config_path,
        data_config_path=data_config_path,
        checkpoint_path=checkpoint_path,
        run_stage_refinement=run_stage_refinement,
        task_specs=_select_task_specs(motif_task_names),
        arm_specs=_select_arm_specs(default_motif_arm_specs(), motif_arm_names),
        session_holdout_fraction=session_holdout_fraction,
        session_holdout_seed=session_holdout_seed,
        debug_retain_intermediates=debug_retain_intermediates,
        progress_ui=ui,
    )
    alignment_outputs = run_optional_alignment_diagnostic_stage(
        paths=paths,
        states=states,
        runtime_experiment_config_path=runtime_experiment_config_path,
        data_config_path=data_config_path,
        checkpoint_path=checkpoint_path,
        run_stage_alignment_diagnostic=run_stage_alignment_diagnostic,
        neighbor_k=neighbor_k,
        progress_ui=ui,
        debug_retain_intermediates=debug_retain_intermediates,
    )
    final_outputs = write_final_project_reports(paths=paths, states=states, progress_ui=ui)
    if ui is not None:
        ui.finish_pipeline()

    return NotebookPipelineRunResult(
        run_id=paths.run_id,
        local_run_root=paths.local_run_root,
        drive_run_root=paths.drive_run_root,
        runtime_experiment_config_path=runtime_experiment_config_path,
        checkpoint_path=checkpoint_path,
        training_summary_path=training_summary_path,
        training_history_json_path=Path(train_outputs["training_history_json_path"]) if train_outputs.get("training_history_json_path") else None,
        training_history_csv_path=Path(train_outputs["training_history_csv_path"]) if train_outputs.get("training_history_csv_path") else None,
        evaluation_summary_paths=tuple(Path(path) for path in evaluation_outputs.get("evaluation_summary_paths", [])),
        refinement_summary_json_path=Path(refinement_outputs["refinement_summary_json_path"]),
        refinement_summary_csv_path=Path(refinement_outputs["refinement_summary_csv_path"]),
        final_summary_json_path=Path(final_outputs["final_summary_json_path"]),
        final_summary_csv_path=Path(final_outputs["final_summary_csv_path"]),
        alignment_summary_json_path=Path(alignment_outputs["alignment_summary_json_path"]) if alignment_outputs.get("alignment_summary_json_path") else None,
        alignment_summary_csv_path=Path(alignment_outputs["alignment_summary_csv_path"]) if alignment_outputs.get("alignment_summary_csv_path") else None,
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
        source_dataset_root=config.source_dataset_root,
        pipeline_run_id=pipeline_run_id,
        dataset_config=NotebookDatasetConfig(),
        training_config=NotebookTrainingConfig(),
        step_log_every=config.step_log_every,
        run_stage_train=config.run_stage_train,
        run_stage_evaluate=config.run_stage_evaluate,
        run_stage_refinement=config.run_stage_refinement,
        run_stage_alignment_diagnostic=config.run_stage_alignment_diagnostic,
        motif_task_names=config.motif_task_names,
        motif_arm_names=config.motif_arm_names,
        session_holdout_fraction=config.session_holdout_fraction,
        session_holdout_seed=config.session_holdout_seed,
        neighbor_k=config.neighbor_k,
        debug_retain_intermediates=config.debug_retain_intermediates,
        stage_prepared_sessions_locally=config.stage_prepared_sessions_locally,
        use_experiment_config_as_is=True,
        output_stream=output_stream,
        progress_ui=config.notebook_ui.enabled,
        notebook_progress_config=config.notebook_ui,
    )


def resume_notebook_pipeline(**kwargs) -> NotebookPipelineRunResult:
    return run_notebook_pipeline(**kwargs)


