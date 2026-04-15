from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
from typing import TextIO

import yaml

from predictive_circuit_coding.benchmarks.contracts import BenchmarkArmSpec, BenchmarkTaskSpec
from predictive_circuit_coding.benchmarks.reports import build_final_project_summary, write_single_row_summary, write_summary_rows
from predictive_circuit_coding.benchmarks.run import default_benchmark_task_specs, default_motif_arm_specs, run_motif_benchmark_matrix
from predictive_circuit_coding.evaluation import evaluate_checkpoint_on_split
from predictive_circuit_coding.training import load_experiment_config, train_model, write_evaluation_summary
from predictive_circuit_coding.utils.notebook_progress import NotebookProgressConfig, NotebookProgressUI
import predictive_circuit_coding.workflows.runtime as workflow_runtime
import predictive_circuit_coding.workflows.stages as workflow_stages
import predictive_circuit_coding.workflows.state as workflow_state
from predictive_circuit_coding.workflows.config import (
    PipelineConfig,
    assert_pipeline_preflight,
    build_pipeline_preflight,
    load_pipeline_config,
)
from predictive_circuit_coding.workflows.contracts import PipelineRunResult
from predictive_circuit_coding.workflows.notebook_runtime import resolve_notebook_checkpoint


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


def run_pipeline(
    *,
    config: PipelineConfig,
    pipeline_run_id: str | None = None,
    output_stream: TextIO | None = None,
    validate_preflight: bool = True,
) -> PipelineRunResult:
    progress_config = config.notebook_ui or NotebookProgressConfig()
    ui = NotebookProgressUI(config=progress_config, stream=output_stream or sys.stdout) if progress_config.enabled else None
    if ui is not None:
        ui.start_pipeline(total_stages=len(config.enabled_stages()) + 1, completed_stages=0)
        ui.milestone("Starting predictive circuit coding pipeline.")
    if validate_preflight:
        assert_pipeline_preflight(build_pipeline_preflight(config))
        if ui is not None:
            ui.milestone("Preflight complete.")
    workflow_runtime.ensure_local_prepared_sessions(
        data_config_path=config.data_config_path,
        source_dataset_root=config.source_dataset_root,
        stage_prepared_sessions_locally=config.stage_prepared_sessions_locally,
        note_callback=ui.note if ui is not None else None,
        progress_callback=ui.make_copy_callback(label="Staging prepared sessions") if ui is not None else None,
    )
    if ui is not None:
        ui.clear_detail()
    resolved_run_id = str(pipeline_run_id).strip() if pipeline_run_id is not None else ""
    if not resolved_run_id:
        if config.run_stage_train:
            resolved_run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        else:
            latest = (
                workflow_state.resolve_latest_run_id(config.drive_export_root)
                if config.drive_export_root is not None
                else None
            )
            if latest is None:
                raise FileNotFoundError("pipeline_run_id is required when training is disabled.")
            resolved_run_id = latest

    paths = workflow_state.build_pipeline_paths(
        local_artifact_root=config.local_artifact_root,
        drive_export_root=config.drive_export_root,
        run_id=resolved_run_id,
    )
    paths.local_run_root.mkdir(parents=True, exist_ok=True)
    paths.pipeline_root.mkdir(parents=True, exist_ok=True)
    paths.pipeline_config_snapshot_path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
    experiment_config = load_experiment_config(config.experiment_config_path)
    created_at = (
        json.loads(paths.pipeline_manifest_path.read_text(encoding="utf-8")).get("created_at_utc")
        if paths.pipeline_manifest_path.is_file()
        else workflow_state.utc_now()
    )
    workflow_state.write_pipeline_manifest(paths, dataset_id=experiment_config.dataset_id, created_at_utc=str(created_at))
    states = workflow_state.load_pipeline_state(paths.pipeline_state_path)

    train_outputs = workflow_stages.run_training_stage(
        paths=paths,
        states=states,
        base_experiment_config=config.experiment_config_path,
        data_config_path=config.data_config_path,
        step_log_every=config.step_log_every,
        run_stage_train=config.run_stage_train,
        progress_ui=ui,
        json_hash_func=workflow_state.json_hash,
        path_identity_func=workflow_state.path_identity,
        write_runtime_experiment_config_func=workflow_runtime.write_runtime_experiment_config,
        load_experiment_config_func=load_experiment_config,
        resolve_notebook_checkpoint_func=resolve_notebook_checkpoint,
        train_model_func=train_model,
    )
    runtime_experiment_config_path = Path(train_outputs["runtime_experiment_config_path"])
    checkpoint_path = Path(train_outputs["checkpoint_path"])
    training_summary_path = Path(train_outputs["training_summary_path"])

    evaluation_outputs = workflow_stages.run_evaluation_stage(
        paths=paths,
        states=states,
        runtime_experiment_config_path=runtime_experiment_config_path,
        data_config_path=config.data_config_path,
        checkpoint_path=checkpoint_path,
        run_stage_evaluate=config.run_stage_evaluate,
        progress_ui=ui,
        json_hash_func=workflow_state.json_hash,
        path_identity_func=workflow_state.path_identity,
        load_experiment_config_func=load_experiment_config,
        evaluate_checkpoint_on_split_func=evaluate_checkpoint_on_split,
        write_evaluation_summary_func=write_evaluation_summary,
    )
    refinement_outputs = workflow_stages.run_refinement_stage(
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
        json_hash_func=workflow_state.json_hash,
        path_identity_func=workflow_state.path_identity,
        load_experiment_config_func=load_experiment_config,
        run_motif_benchmark_matrix_func=run_motif_benchmark_matrix,
        write_summary_rows_func=write_summary_rows,
    )
    alignment_outputs = workflow_stages.run_alignment_diagnostic_stage(
        paths=paths,
        states=states,
        runtime_experiment_config_path=runtime_experiment_config_path,
        data_config_path=config.data_config_path,
        checkpoint_path=checkpoint_path,
        run_stage_alignment_diagnostic=config.run_stage_alignment_diagnostic,
        progress_ui=ui,
        json_hash_func=workflow_state.json_hash,
        path_identity_func=workflow_state.path_identity,
    )
    final_outputs = workflow_stages.run_final_reports_stage(
        paths=paths,
        states=states,
        progress_ui=ui,
        json_hash_func=workflow_state.json_hash,
        path_identity_func=workflow_state.path_identity,
        build_final_project_summary_func=build_final_project_summary,
        write_single_row_summary_func=write_single_row_summary,
    )
    if ui is not None:
        ui.render_artifacts(
            "Run artifacts",
            {
                "run_id": paths.run_id,
                "local_run_root": paths.local_run_root,
                "drive_run_root": paths.drive_run_root,
                "checkpoint": checkpoint_path,
                "training_summary": training_summary_path,
                "training_history": train_outputs.get("training_history_json_path"),
                "refinement_summary": refinement_outputs.get("refinement_summary_json_path"),
                "final_summary": final_outputs.get("final_summary_json_path"),
                "pipeline_manifest": paths.pipeline_manifest_path,
                "pipeline_state": paths.pipeline_state_path,
            },
        )
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
