from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from predictive_circuit_coding.benchmarks.contracts import BenchmarkArmSpec, BenchmarkTaskSpec
from predictive_circuit_coding.utils.notebook_progress import NotebookProgressUI
from predictive_circuit_coding.workflows.contracts import PipelinePaths
from predictive_circuit_coding.workflows.state import (
    mark_stage_complete,
    mark_stage_failed,
    mark_stage_reused,
    mark_stage_running,
    mark_stage_skipped,
    stage_is_reusable,
    summary_rows_from_json,
)


def run_training_stage(
    *,
    paths: PipelinePaths,
    states: dict[str, dict[str, Any]],
    base_experiment_config: str | Path,
    data_config_path: str | Path,
    step_log_every: int,
    run_stage_train: bool,
    progress_ui: NotebookProgressUI | None,
    json_hash_func: Callable[[dict[str, Any]], str],
    path_identity_func: Callable[[str | Path | None], str | None],
    write_runtime_experiment_config_func: Callable[..., Path],
    load_experiment_config_func: Callable[[str | Path], Any],
    resolve_notebook_checkpoint_func: Callable[..., Path],
    train_model_func: Callable[..., Any],
) -> dict[str, Any]:
    runtime_config_path = write_runtime_experiment_config_func(
        base_experiment_config=base_experiment_config,
        runtime_experiment_config_path=paths.runtime_experiment_config_path,
        step_log_every=step_log_every,
        artifact_root=paths.train_root,
    )
    stage_config = {
        "base_experiment_config": path_identity_func(base_experiment_config),
        "data_config_path": path_identity_func(data_config_path),
        "run_stage_train": bool(run_stage_train),
        "step_log_every": int(step_log_every),
        "stage_root": path_identity_func(paths.train_root),
    }
    config_hash = json_hash_func(stage_config)
    inputs = {
        "base_experiment_config": path_identity_func(base_experiment_config),
        "data_config_path": path_identity_func(data_config_path),
        "stage_root": path_identity_func(paths.train_root),
    }
    if run_stage_train and stage_is_reusable(states=states, stage_name="train", config_hash=config_hash, inputs=inputs):
        return mark_stage_reused(
            paths=paths,
            states=states,
            stage_name="train",
            config_hash=config_hash,
            inputs=inputs,
            outputs=dict(states["train"]["outputs"]),
            progress_ui=progress_ui,
        )

    runtime_config = load_experiment_config_func(runtime_config_path)
    if not run_stage_train:
        history_json_path = runtime_config.artifacts.summary_path.with_name("training_history.json")
        history_csv_path = runtime_config.artifacts.summary_path.with_name("training_history.csv")
        checkpoint = resolve_notebook_checkpoint_func(
            summary_path=runtime_config.artifacts.summary_path,
            checkpoint_dir=runtime_config.artifacts.checkpoint_dir,
            checkpoint_prefix=runtime_config.artifacts.checkpoint_prefix,
        )
        outputs = {
            "runtime_experiment_config_path": str(runtime_config_path),
            "checkpoint_path": str(checkpoint),
            "training_summary_path": str(runtime_config.artifacts.summary_path),
            "training_history_json_path": str(history_json_path) if history_json_path.is_file() else None,
            "training_history_csv_path": str(history_csv_path) if history_csv_path.is_file() else None,
        }
        return mark_stage_skipped(
            paths=paths,
            states=states,
            stage_name="train",
            config_hash=config_hash,
            inputs=inputs,
            outputs=outputs,
        )

    paths.train_root.mkdir(parents=True, exist_ok=True)
    if progress_ui is not None:
        progress_ui.start_stage(stage_name="train", total=1, description="Training")
    training_progress_callback = (
        progress_ui.make_training_callback(stage_name="train")
        if progress_ui is not None
        else None
    )
    mark_stage_running(
        paths=paths,
        states=states,
        stage_name="train",
        config_hash=config_hash,
        inputs=inputs,
    )
    try:
        result = train_model_func(
            experiment_config=runtime_config,
            data_config_path=data_config_path,
            train_split=runtime_config.splits.train,
            valid_split=runtime_config.splits.valid,
            progress_callback=training_progress_callback,
            emit_logs=progress_ui is None,
        )
    except Exception as exc:
        mark_stage_failed(
            paths=paths,
            states=states,
            stage_name="train",
            config_hash=config_hash,
            inputs=inputs,
            error=exc,
            progress_ui=progress_ui,
        )
        raise

    outputs = {
        "runtime_experiment_config_path": str(runtime_config_path),
        "checkpoint_path": str(result.checkpoint_path),
        "training_summary_path": str(result.summary_path),
        "training_history_json_path": str(result.history_json_path),
        "training_history_csv_path": str(result.history_csv_path),
    }
    return mark_stage_complete(
        paths=paths,
        states=states,
        stage_name="train",
        config_hash=config_hash,
        inputs=inputs,
        outputs=outputs,
        synced_relatives=("train",),
        progress_ui=progress_ui,
    )


def run_evaluation_stage(
    *,
    paths: PipelinePaths,
    states: dict[str, dict[str, Any]],
    runtime_experiment_config_path: str | Path,
    data_config_path: str | Path,
    checkpoint_path: str | Path,
    run_stage_evaluate: bool,
    progress_ui: NotebookProgressUI | None,
    json_hash_func: Callable[[dict[str, Any]], str],
    path_identity_func: Callable[[str | Path | None], str | None],
    load_experiment_config_func: Callable[[str | Path], Any],
    evaluate_checkpoint_on_split_func: Callable[..., Any],
    write_evaluation_summary_func: Callable[[Any, str | Path], Any],
) -> dict[str, Any]:
    stage_config = {
        "runtime_experiment_config_path": path_identity_func(runtime_experiment_config_path),
        "data_config_path": path_identity_func(data_config_path),
        "checkpoint_path": path_identity_func(checkpoint_path),
        "run_stage_evaluate": bool(run_stage_evaluate),
        "stage_root": path_identity_func(paths.evaluation_root),
    }
    config_hash = json_hash_func(stage_config)
    inputs = {
        "runtime_experiment_config_path": path_identity_func(runtime_experiment_config_path),
        "checkpoint_path": path_identity_func(checkpoint_path),
        "data_config_path": path_identity_func(data_config_path),
        "stage_root": path_identity_func(paths.evaluation_root),
    }
    if run_stage_evaluate and stage_is_reusable(states=states, stage_name="evaluate", config_hash=config_hash, inputs=inputs):
        return mark_stage_reused(
            paths=paths,
            states=states,
            stage_name="evaluate",
            config_hash=config_hash,
            inputs=inputs,
            outputs=dict(states["evaluate"]["outputs"]),
            progress_ui=progress_ui,
        )
    if not run_stage_evaluate:
        return mark_stage_skipped(
            paths=paths,
            states=states,
            stage_name="evaluate",
            config_hash=config_hash,
            inputs=inputs,
            outputs={"evaluation_summary_paths": []},
        )

    config = load_experiment_config_func(runtime_experiment_config_path)
    paths.evaluation_root.mkdir(parents=True, exist_ok=True)
    if progress_ui is not None:
        progress_ui.start_stage(stage_name="evaluate", total=2, description="Evaluation")
    evaluation_progress_callback = (
        progress_ui.make_evaluation_callback(split_total=2)
        if progress_ui is not None
        else None
    )
    mark_stage_running(
        paths=paths,
        states=states,
        stage_name="evaluate",
        config_hash=config_hash,
        inputs=inputs,
    )
    summaries: list[str] = []
    try:
        for split_name in (config.splits.valid, config.splits.test):
            summary = evaluate_checkpoint_on_split_func(
                experiment_config=config,
                data_config_path=data_config_path,
                checkpoint_path=str(checkpoint_path),
                split_name=split_name,
                progress_callback=evaluation_progress_callback,
            )
            summary_path = paths.evaluation_root / f"{split_name}_summary.json"
            write_evaluation_summary_func(summary, summary_path)
            summaries.append(str(summary_path))
    except Exception as exc:
        mark_stage_failed(
            paths=paths,
            states=states,
            stage_name="evaluate",
            config_hash=config_hash,
            inputs=inputs,
            error=exc,
            progress_ui=progress_ui,
        )
        raise

    return mark_stage_complete(
        paths=paths,
        states=states,
        stage_name="evaluate",
        config_hash=config_hash,
        inputs=inputs,
        outputs={"evaluation_summary_paths": summaries},
        synced_relatives=("evaluation",),
        progress_ui=progress_ui,
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
    progress_ui: NotebookProgressUI | None,
    json_hash_func: Callable[[dict[str, Any]], str],
    path_identity_func: Callable[[str | Path | None], str | None],
    load_experiment_config_func: Callable[[str | Path], Any],
    run_motif_benchmark_matrix_func: Callable[..., Any],
    write_summary_rows_func: Callable[..., tuple[Path, Path]],
) -> dict[str, Any]:
    stage_config = {
        "runtime_experiment_config_path": path_identity_func(runtime_experiment_config_path),
        "data_config_path": path_identity_func(data_config_path),
        "checkpoint_path": path_identity_func(checkpoint_path),
        "run_stage_refinement": bool(run_stage_refinement),
        "task_specs": [task.to_dict() for task in task_specs],
        "arm_specs": [arm.to_dict() for arm in arm_specs],
        "session_holdout_fraction": float(session_holdout_fraction),
        "session_holdout_seed": session_holdout_seed,
        "debug_retain_intermediates": bool(debug_retain_intermediates),
        "stage_root": path_identity_func(paths.refinement_root),
    }
    config_hash = json_hash_func(stage_config)
    inputs = {
        "runtime_experiment_config_path": path_identity_func(runtime_experiment_config_path),
        "checkpoint_path": path_identity_func(checkpoint_path),
        "data_config_path": path_identity_func(data_config_path),
        "stage_root": path_identity_func(paths.refinement_root),
    }
    if run_stage_refinement and stage_is_reusable(states=states, stage_name="refinement", config_hash=config_hash, inputs=inputs):
        return mark_stage_reused(
            paths=paths,
            states=states,
            stage_name="refinement",
            config_hash=config_hash,
            inputs=inputs,
            outputs=dict(states["refinement"]["outputs"]),
            progress_ui=progress_ui,
        )
    if not run_stage_refinement:
        return mark_stage_skipped(
            paths=paths,
            states=states,
            stage_name="refinement",
            config_hash=config_hash,
            inputs=inputs,
            outputs={
                "refinement_summary_json_path": str(paths.reports_root / "refinement_summary.json"),
                "refinement_summary_csv_path": str(paths.reports_root / "refinement_summary.csv"),
            },
        )

    config = load_experiment_config_func(runtime_experiment_config_path)
    total_refinement_arms = len(task_specs) * len(arm_specs)
    if progress_ui is not None:
        progress_ui.start_stage(stage_name="refinement", total=total_refinement_arms, description="Refinement")
    benchmark_progress_callback = (
        progress_ui.make_benchmark_callback(benchmark_name="refinement", total_arms=total_refinement_arms)
        if progress_ui is not None
        else None
    )
    mark_stage_running(
        paths=paths,
        states=states,
        stage_name="refinement",
        config_hash=config_hash,
        inputs=inputs,
    )
    try:
        results = run_motif_benchmark_matrix_func(
            experiment_config=config,
            data_config_path=data_config_path,
            checkpoint_path=checkpoint_path,
            output_root=paths.refinement_root,
            task_specs=task_specs,
            arm_specs=arm_specs,
            session_holdout_fraction=session_holdout_fraction,
            session_holdout_seed=session_holdout_seed,
            debug_retain_intermediates=debug_retain_intermediates,
            progress_callback=benchmark_progress_callback,
        )
        rows = [result.summary for result in results]
        summary_json_path, summary_csv_path = write_summary_rows_func(
            rows,
            output_json_path=paths.reports_root / "refinement_summary.json",
            output_csv_path=paths.reports_root / "refinement_summary.csv",
            root_key="refinement",
        )
    except Exception as exc:
        mark_stage_failed(
            paths=paths,
            states=states,
            stage_name="refinement",
            config_hash=config_hash,
            inputs=inputs,
            error=exc,
            progress_ui=progress_ui,
        )
        raise

    return mark_stage_complete(
        paths=paths,
        states=states,
        stage_name="refinement",
        config_hash=config_hash,
        inputs=inputs,
        outputs={
            "refinement_summary_json_path": str(summary_json_path),
            "refinement_summary_csv_path": str(summary_csv_path),
        },
        synced_relatives=("refinement", "reports"),
        progress_ui=progress_ui,
    )


def run_alignment_diagnostic_stage(
    *,
    paths: PipelinePaths,
    states: dict[str, dict[str, Any]],
    runtime_experiment_config_path: str | Path,
    data_config_path: str | Path,
    checkpoint_path: str | Path,
    run_stage_alignment_diagnostic: bool,
    progress_ui: NotebookProgressUI | None,
    json_hash_func: Callable[[dict[str, Any]], str],
    path_identity_func: Callable[[str | Path | None], str | None],
) -> dict[str, Any]:
    del runtime_experiment_config_path, data_config_path, checkpoint_path
    config_hash = json_hash_func(
        {
            "run_stage_alignment_diagnostic": bool(run_stage_alignment_diagnostic),
            "stage_root": path_identity_func(paths.diagnostics_root),
        }
    )
    inputs = {"stage_root": path_identity_func(paths.diagnostics_root)}
    outputs = {"alignment_summary_json_path": None, "alignment_summary_csv_path": None}
    if not run_stage_alignment_diagnostic:
        return mark_stage_skipped(
            paths=paths,
            states=states,
            stage_name="alignment_diagnostic",
            config_hash=config_hash,
            inputs=inputs,
            outputs=outputs,
        )
    if stage_is_reusable(states=states, stage_name="alignment_diagnostic", config_hash=config_hash, inputs=inputs):
        return mark_stage_reused(
            paths=paths,
            states=states,
            stage_name="alignment_diagnostic",
            config_hash=config_hash,
            inputs=inputs,
            outputs=dict(states["alignment_diagnostic"]["outputs"]),
            progress_ui=progress_ui,
        )

    if progress_ui is not None:
        progress_ui.start_stage(stage_name="alignment_diagnostic", total=1, description="Alignment")
    mark_stage_running(
        paths=paths,
        states=states,
        stage_name="alignment_diagnostic",
        config_hash=config_hash,
        inputs=inputs,
    )
    return mark_stage_complete(
        paths=paths,
        states=states,
        stage_name="alignment_diagnostic",
        config_hash=config_hash,
        inputs=inputs,
        outputs=outputs,
        synced_relatives=("diagnostics",),
        progress_ui=progress_ui,
    )


def run_final_reports_stage(
    *,
    paths: PipelinePaths,
    states: dict[str, dict[str, Any]],
    progress_ui: NotebookProgressUI | None,
    json_hash_func: Callable[[dict[str, Any]], str],
    path_identity_func: Callable[[str | Path | None], str | None],
    build_final_project_summary_func: Callable[..., dict[str, Any]],
    write_single_row_summary_func: Callable[..., tuple[Path, Path]],
) -> dict[str, Any]:
    rows = summary_rows_from_json(paths.reports_root / "refinement_summary.json", "refinement")
    config_hash = json_hash_func(
        {
            "run_stage_final_reports": True,
            "row_count": len(rows),
            "stage_root": path_identity_func(paths.reports_root),
        }
    )
    inputs = {
        "refinement_summary_json_path": path_identity_func(paths.reports_root / "refinement_summary.json"),
        "stage_root": path_identity_func(paths.reports_root),
    }
    if stage_is_reusable(states=states, stage_name="final_reports", config_hash=config_hash, inputs=inputs):
        return mark_stage_reused(
            paths=paths,
            states=states,
            stage_name="final_reports",
            config_hash=config_hash,
            inputs=inputs,
            outputs=dict(states["final_reports"]["outputs"]),
            progress_ui=progress_ui,
        )
    if progress_ui is not None:
        progress_ui.start_stage(stage_name="final_reports", total=1, description="Final Reports")
    mark_stage_running(
        paths=paths,
        states=states,
        stage_name="final_reports",
        config_hash=config_hash,
        inputs=inputs,
    )
    try:
        payload = build_final_project_summary_func(motif_rows=rows)
        summary_json_path, summary_csv_path = write_single_row_summary_func(
            payload,
            output_json_path=paths.reports_root / "final_project_summary.json",
            output_csv_path=paths.reports_root / "final_project_summary.csv",
            root_key="final_project_summary",
        )
    except Exception as exc:
        mark_stage_failed(
            paths=paths,
            states=states,
            stage_name="final_reports",
            config_hash=config_hash,
            inputs=inputs,
            error=exc,
            progress_ui=progress_ui,
        )
        raise
    return mark_stage_complete(
        paths=paths,
        states=states,
        stage_name="final_reports",
        config_hash=config_hash,
        inputs=inputs,
        outputs={
            "final_summary_json_path": str(summary_json_path),
            "final_summary_csv_path": str(summary_csv_path),
        },
        synced_relatives=("reports",),
        progress_ui=progress_ui,
    )
