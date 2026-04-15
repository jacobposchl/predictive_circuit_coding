from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, TextIO

import yaml

import predictive_circuit_coding.workflows.runtime as workflow_runtime
import predictive_circuit_coding.workflows.stages as workflow_stages
import predictive_circuit_coding.workflows.state as workflow_state
from predictive_circuit_coding.benchmarks.contracts import BenchmarkArmSpec, BenchmarkTaskSpec
from predictive_circuit_coding.benchmarks.reports import (
    build_final_project_summary,
    write_single_row_summary,
    write_summary_rows,
)
from predictive_circuit_coding.benchmarks.run import (
    default_benchmark_task_specs,
    default_motif_arm_specs,
    run_motif_benchmark_matrix,
)
from predictive_circuit_coding.evaluation import evaluate_checkpoint_on_split
from predictive_circuit_coding.training import (
    load_experiment_config,
    train_model,
    write_evaluation_summary,
)
from predictive_circuit_coding.utils.notebook_progress import (
    NotebookProgressConfig,
    NotebookProgressUI,
)
from predictive_circuit_coding.workflows.notebook_runtime import (
    resolve_notebook_checkpoint,
)
from predictive_circuit_coding.workflows.config import (
    PipelineConfig,
    assert_pipeline_preflight,
    build_pipeline_preflight,
    load_pipeline_config,
)
from predictive_circuit_coding.workflows.contracts import (
    PIPELINE_STAGE_ORDER,
    PipelinePaths,
    PipelineRunResult,
)


_PIPELINE_STAGE_ORDER = PIPELINE_STAGE_ORDER


def _utc_now() -> str:
    return workflow_state.utc_now()


def _json_hash(payload: dict[str, Any]) -> str:
    return workflow_state.json_hash(payload)


def _path_identity(path: str | Path | None) -> str | None:
    return workflow_state.path_identity(path)


def _resolve_latest_run_id(drive_export_root: str | Path) -> str | None:
    return workflow_state.resolve_latest_run_id(drive_export_root)


def _build_pipeline_paths(
    *,
    local_artifact_root: str | Path,
    drive_export_root: str | Path | None,
    run_id: str,
    run_name: str = "run_1",
) -> PipelinePaths:
    return workflow_state.build_pipeline_paths(
        local_artifact_root=local_artifact_root,
        drive_export_root=drive_export_root,
        run_id=run_id,
        run_name=run_name,
    )


def _load_pipeline_state(path: Path) -> dict[str, dict[str, Any]]:
    return workflow_state.load_pipeline_state(path)


def _write_pipeline_manifest(paths: PipelinePaths, *, dataset_id: str, created_at_utc: str) -> None:
    workflow_state.write_pipeline_manifest(paths, dataset_id=dataset_id, created_at_utc=created_at_utc)


def _summary_rows_from_json(path: Path, root_key: str) -> list[dict[str, Any]]:
    return workflow_state.summary_rows_from_json(path, root_key)


def _write_runtime_experiment_config(
    *,
    base_experiment_config: str | Path,
    runtime_experiment_config_path: Path,
    step_log_every: int,
    artifact_root: Path | None = None,
) -> Path:
    return workflow_runtime.write_runtime_experiment_config(
        base_experiment_config=base_experiment_config,
        runtime_experiment_config_path=runtime_experiment_config_path,
        step_log_every=step_log_every,
        artifact_root=artifact_root,
    )


def _resolve_source_session_ids(*, source_dataset_root: str | Path, dataset_id: str) -> list[str]:
    return workflow_runtime.resolve_source_session_ids(
        source_dataset_root=source_dataset_root,
        dataset_id=dataset_id,
    )


def _ensure_local_prepared_sessions(
    *,
    data_config_path: str | Path,
    source_dataset_root: str | Path | None,
    stage_prepared_sessions_locally: bool,
) -> None:
    workflow_runtime.ensure_local_prepared_sessions(
        data_config_path=data_config_path,
        source_dataset_root=source_dataset_root,
        stage_prepared_sessions_locally=stage_prepared_sessions_locally,
    )


def _select_task_specs(requested_names: tuple[str, ...] | None) -> tuple[BenchmarkTaskSpec, ...]:
    specs = default_benchmark_task_specs()
    if requested_names is None:
        return specs
    by_name = {spec.name: spec for spec in specs}
    missing = [name for name in requested_names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown refinement task names: {missing}")
    return tuple(by_name[name] for name in requested_names)


def _select_arm_specs(
    specs: tuple[BenchmarkArmSpec, ...],
    requested_names: tuple[str, ...] | None,
) -> tuple[BenchmarkArmSpec, ...]:
    if requested_names is None:
        return specs
    by_name = {spec.name: spec for spec in specs}
    missing = [name for name in requested_names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown refinement arm names: {missing}")
    return tuple(by_name[name] for name in requested_names)


def run_training_stage(
    *,
    paths: PipelinePaths,
    states: dict[str, dict[str, Any]],
    base_experiment_config: str | Path,
    data_config_path: str | Path,
    step_log_every: int,
    run_stage_train: bool,
    progress_ui: NotebookProgressUI | None = None,
) -> dict[str, Any]:
    return workflow_stages.run_training_stage(
        paths=paths,
        states=states,
        base_experiment_config=base_experiment_config,
        data_config_path=data_config_path,
        step_log_every=step_log_every,
        run_stage_train=run_stage_train,
        progress_ui=progress_ui,
        json_hash_func=_json_hash,
        path_identity_func=_path_identity,
        write_runtime_experiment_config_func=_write_runtime_experiment_config,
        load_experiment_config_func=load_experiment_config,
        resolve_notebook_checkpoint_func=resolve_notebook_checkpoint,
        train_model_func=train_model,
    )


def run_evaluation_stage(
    *,
    paths: PipelinePaths,
    states: dict[str, dict[str, Any]],
    runtime_experiment_config_path: str | Path,
    data_config_path: str | Path,
    checkpoint_path: str | Path,
    run_stage_evaluate: bool,
    progress_ui: NotebookProgressUI | None = None,
) -> dict[str, Any]:
    return workflow_stages.run_evaluation_stage(
        paths=paths,
        states=states,
        runtime_experiment_config_path=runtime_experiment_config_path,
        data_config_path=data_config_path,
        checkpoint_path=checkpoint_path,
        run_stage_evaluate=run_stage_evaluate,
        progress_ui=progress_ui,
        json_hash_func=_json_hash,
        path_identity_func=_path_identity,
        load_experiment_config_func=load_experiment_config,
        evaluate_checkpoint_on_split_func=evaluate_checkpoint_on_split,
        write_evaluation_summary_func=write_evaluation_summary,
    )


def run_refinement_stage(
    *,
    paths: PipelinePaths,
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
    return workflow_stages.run_refinement_stage(
        paths=paths,
        states=states,
        runtime_experiment_config_path=runtime_experiment_config_path,
        data_config_path=data_config_path,
        checkpoint_path=checkpoint_path,
        run_stage_refinement=run_stage_refinement,
        task_specs=task_specs,
        arm_specs=arm_specs,
        session_holdout_fraction=session_holdout_fraction,
        session_holdout_seed=session_holdout_seed,
        debug_retain_intermediates=debug_retain_intermediates,
        progress_ui=progress_ui,
        json_hash_func=_json_hash,
        path_identity_func=_path_identity,
        load_experiment_config_func=load_experiment_config,
        run_motif_benchmark_matrix_func=run_motif_benchmark_matrix,
        write_summary_rows_func=write_summary_rows,
    )


def run_alignment_diagnostic_stage(
    *,
    paths: PipelinePaths,
    states: dict[str, dict[str, Any]],
    runtime_experiment_config_path: str | Path,
    data_config_path: str | Path,
    checkpoint_path: str | Path,
    run_stage_alignment_diagnostic: bool,
    progress_ui: NotebookProgressUI | None = None,
) -> dict[str, Any]:
    return workflow_stages.run_alignment_diagnostic_stage(
        paths=paths,
        states=states,
        runtime_experiment_config_path=runtime_experiment_config_path,
        data_config_path=data_config_path,
        checkpoint_path=checkpoint_path,
        run_stage_alignment_diagnostic=run_stage_alignment_diagnostic,
        progress_ui=progress_ui,
        json_hash_func=_json_hash,
        path_identity_func=_path_identity,
    )


def run_final_reports_stage(
    *,
    paths: PipelinePaths,
    states: dict[str, dict[str, Any]],
    progress_ui: NotebookProgressUI | None = None,
) -> dict[str, Any]:
    return workflow_stages.run_final_reports_stage(
        paths=paths,
        states=states,
        progress_ui=progress_ui,
        json_hash_func=_json_hash,
        path_identity_func=_path_identity,
        build_final_project_summary_func=build_final_project_summary,
        write_single_row_summary_func=write_single_row_summary,
    )


def run_pipeline(
    *,
    config: PipelineConfig,
    pipeline_run_id: str | None = None,
    output_stream: TextIO | None = None,
    validate_preflight: bool = True,
) -> PipelineRunResult:
    if validate_preflight:
        assert_pipeline_preflight(build_pipeline_preflight(config))
    _ensure_local_prepared_sessions(
        data_config_path=config.data_config_path,
        source_dataset_root=config.source_dataset_root,
        stage_prepared_sessions_locally=config.stage_prepared_sessions_locally,
    )
    resolved_run_id = str(pipeline_run_id).strip() if pipeline_run_id is not None else ""
    if not resolved_run_id:
        if config.run_stage_train:
            resolved_run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        else:
            latest = _resolve_latest_run_id(config.drive_export_root) if config.drive_export_root is not None else None
            if latest is None:
                raise FileNotFoundError("pipeline_run_id is required when training is disabled.")
            resolved_run_id = latest

    paths = _build_pipeline_paths(
        local_artifact_root=config.local_artifact_root,
        drive_export_root=config.drive_export_root,
        run_id=resolved_run_id,
    )
    paths.local_run_root.mkdir(parents=True, exist_ok=True)
    paths.pipeline_root.mkdir(parents=True, exist_ok=True)
    paths.pipeline_config_snapshot_path.write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    experiment_config = load_experiment_config(config.experiment_config_path)
    created_at = (
        json.loads(paths.pipeline_manifest_path.read_text(encoding="utf-8")).get("created_at_utc")
        if paths.pipeline_manifest_path.is_file()
        else _utc_now()
    )
    _write_pipeline_manifest(paths, dataset_id=experiment_config.dataset_id, created_at_utc=str(created_at))
    states = _load_pipeline_state(paths.pipeline_state_path)

    output_stream = output_stream or sys.stdout
    progress_config = config.notebook_ui or NotebookProgressConfig()
    ui = NotebookProgressUI(config=progress_config, stream=output_stream) if progress_config.enabled else None
    if ui is not None:
        ui.start_pipeline(total_stages=len(config.enabled_stages()) + 1, completed_stages=0)

    train_outputs = run_training_stage(
        paths=paths,
        states=states,
        base_experiment_config=config.experiment_config_path,
        data_config_path=config.data_config_path,
        step_log_every=config.step_log_every,
        run_stage_train=config.run_stage_train,
        progress_ui=ui,
    )
    runtime_experiment_config_path = Path(train_outputs["runtime_experiment_config_path"])
    checkpoint_path = Path(train_outputs["checkpoint_path"])
    training_summary_path = Path(train_outputs["training_summary_path"])

    evaluation_outputs = run_evaluation_stage(
        paths=paths,
        states=states,
        runtime_experiment_config_path=runtime_experiment_config_path,
        data_config_path=config.data_config_path,
        checkpoint_path=checkpoint_path,
        run_stage_evaluate=config.run_stage_evaluate,
        progress_ui=ui,
    )
    refinement_outputs = run_refinement_stage(
        paths=paths,
        states=states,
        runtime_experiment_config_path=runtime_experiment_config_path,
        data_config_path=config.data_config_path,
        checkpoint_path=checkpoint_path,
        run_stage_refinement=config.run_stage_refinement,
        task_specs=_select_task_specs(config.motif_task_names),
        arm_specs=_select_arm_specs(default_motif_arm_specs(), config.motif_arm_names),
        session_holdout_fraction=config.session_holdout_fraction,
        session_holdout_seed=config.session_holdout_seed,
        debug_retain_intermediates=config.debug_retain_intermediates,
        progress_ui=ui,
    )
    alignment_outputs = run_alignment_diagnostic_stage(
        paths=paths,
        states=states,
        runtime_experiment_config_path=runtime_experiment_config_path,
        data_config_path=config.data_config_path,
        checkpoint_path=checkpoint_path,
        run_stage_alignment_diagnostic=config.run_stage_alignment_diagnostic,
        progress_ui=ui,
    )
    final_outputs = run_final_reports_stage(paths=paths, states=states, progress_ui=ui)
    if ui is not None:
        ui.finish_pipeline()

    return PipelineRunResult(
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


def run_pipeline_from_config(
    *,
    pipeline_config_path: str | Path,
    pipeline_run_id: str | None = None,
    output_stream: TextIO | None = None,
) -> PipelineRunResult:
    config = load_pipeline_config(pipeline_config_path)
    return run_pipeline(config=config, pipeline_run_id=pipeline_run_id, output_stream=output_stream, validate_preflight=True)


def resume_pipeline(**kwargs) -> PipelineRunResult:
    return run_pipeline(**kwargs)
