from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from predictive_circuit_coding.benchmarks.contracts import BenchmarkArmSpec, BenchmarkTaskSpec
from predictive_circuit_coding.workflows import state as workflow_state
from predictive_circuit_coding.workflows.contracts import PipelinePaths
from predictive_circuit_coding.workflows.stages import (
    run_evaluation_stage,
    run_refinement_stage,
    run_training_stage,
)


class RecordingProgressUI:
    def __init__(self) -> None:
        self.started: list[tuple[str, int | None, str | None]] = []
        self.finished: list[object] = []
        self.advanced = 0
        self.training_callback = object()
        self.evaluation_callback = object()
        self.benchmark_callback = object()

    def start_stage(self, *, stage_name: str, total: int | None = None, description: str | None = None) -> None:
        self.started.append((stage_name, total, description))

    def finish_stage(self, summary) -> None:
        self.finished.append(summary)

    def advance_pipeline(self, steps: int = 1) -> None:
        self.advanced += int(steps)

    def make_training_callback(self, *, stage_name: str):
        return self.training_callback

    def make_evaluation_callback(self, *, split_total: int):
        return self.evaluation_callback

    def make_benchmark_callback(self, *, benchmark_name: str, total_arms: int | None = None):
        return self.benchmark_callback


def _make_paths(tmp_path: Path) -> PipelinePaths:
    local_run_root = tmp_path / "run_1"
    return PipelinePaths(
        run_id="run_1",
        local_run_root=local_run_root,
        drive_run_root=None,
        train_root=local_run_root / "train",
        evaluation_root=local_run_root / "evaluation",
        refinement_root=local_run_root / "refinement",
        diagnostics_root=local_run_root / "diagnostics",
        reports_root=local_run_root / "reports",
        pipeline_root=local_run_root / "pipeline",
        runtime_experiment_config_path=local_run_root / "train" / "colab_runtime_experiment.yaml",
        pipeline_config_snapshot_path=local_run_root / "pipeline" / "pipeline_config_snapshot.yaml",
        pipeline_manifest_path=local_run_root / "pipeline" / "pipeline_manifest.json",
        pipeline_state_path=local_run_root / "pipeline" / "pipeline_state.json",
    )


def _runtime_config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        splits=SimpleNamespace(train="train", valid="valid"),
        artifacts=SimpleNamespace(
            summary_path=tmp_path / "training_summary.json",
            checkpoint_dir=tmp_path / "checkpoints",
            checkpoint_prefix="pcc_test",
        ),
    )


def test_prepare_training_stage_passes_training_progress_callback(tmp_path: Path, monkeypatch):
    paths = _make_paths(tmp_path)
    progress_ui = RecordingProgressUI()
    captured: dict[str, object] = {}

    def fake_train_model(**kwargs):
        captured["progress_callback"] = kwargs["progress_callback"]
        return SimpleNamespace(
            checkpoint_path=tmp_path / "best.pt",
            summary_path=tmp_path / "training_summary.json",
            history_json_path=tmp_path / "training_history.json",
            history_csv_path=tmp_path / "training_history.csv",
        )

    run_training_stage(
        paths=paths,
        states={},
        base_experiment_config=tmp_path / "base.yaml",
        data_config_path=tmp_path / "data.yaml",
        step_log_every=1,
        run_stage_train=True,
        progress_ui=progress_ui,
        json_hash_func=workflow_state.json_hash,
        path_identity_func=workflow_state.path_identity,
        write_runtime_experiment_config_func=lambda **kwargs: paths.runtime_experiment_config_path,
        load_experiment_config_func=lambda path: _runtime_config(tmp_path),
        resolve_notebook_checkpoint_func=lambda **kwargs: tmp_path / "best.pt",
        train_model_func=fake_train_model,
    )

    assert captured["progress_callback"] is progress_ui.training_callback
    assert progress_ui.started == [("train", 1, "Train")]
    assert progress_ui.finished[0].status == "complete"
    assert progress_ui.advanced == 1


def test_evaluation_stage_passes_evaluation_progress_callback(tmp_path: Path, monkeypatch):
    paths = _make_paths(tmp_path)
    progress_ui = RecordingProgressUI()
    captured_callbacks: list[object] = []

    def fake_evaluate_checkpoint_on_split(**kwargs):
        captured_callbacks.append(kwargs["progress_callback"])
        return SimpleNamespace(metrics={"predictive_improvement": 0.5}, losses={"predictive_loss": 1.0})

    run_evaluation_stage(
        paths=paths,
        states={},
        runtime_experiment_config_path=tmp_path / "runtime.yaml",
        data_config_path=tmp_path / "data.yaml",
        checkpoint_path=tmp_path / "best.pt",
        run_stage_evaluate=True,
        progress_ui=progress_ui,
        json_hash_func=workflow_state.json_hash,
        path_identity_func=workflow_state.path_identity,
        load_experiment_config_func=lambda path: SimpleNamespace(splits=SimpleNamespace(valid="valid", test="test")),
        evaluate_checkpoint_on_split_func=fake_evaluate_checkpoint_on_split,
        write_evaluation_summary_func=lambda summary, path: Path(path).write_text("{}", encoding="utf-8"),
    )

    assert captured_callbacks == [progress_ui.evaluation_callback, progress_ui.evaluation_callback]
    assert progress_ui.started == [("evaluate", 2, "Evaluate")]
    assert progress_ui.finished[0].status == "complete"
    assert progress_ui.advanced == 1


def test_refinement_stage_passes_benchmark_progress_callback(tmp_path: Path, monkeypatch):
    paths = _make_paths(tmp_path)
    progress_ui = RecordingProgressUI()
    captured: dict[str, object] = {}

    def fake_run_motif_benchmark_matrix(**kwargs):
        captured["progress_callback"] = kwargs["progress_callback"]
        return (
            SimpleNamespace(summary={"task_name": "stimulus_change", "arm_name": "encoder_raw", "status": "ok"}),
        )

    run_refinement_stage(
        paths=paths,
        states={},
        runtime_experiment_config_path=tmp_path / "runtime.yaml",
        data_config_path=tmp_path / "data.yaml",
        checkpoint_path=tmp_path / "best.pt",
        run_stage_refinement=True,
        task_specs=(BenchmarkTaskSpec(name="stimulus_change", target_label="stimulus_change"),),
        arm_specs=(BenchmarkArmSpec(name="encoder_raw", geometry_mode="raw"),),
        session_holdout_fraction=0.5,
        session_holdout_seed=7,
        debug_retain_intermediates=False,
        progress_ui=progress_ui,
        json_hash_func=workflow_state.json_hash,
        path_identity_func=workflow_state.path_identity,
        load_experiment_config_func=lambda path: SimpleNamespace(),
        run_motif_benchmark_matrix_func=fake_run_motif_benchmark_matrix,
        write_summary_rows_func=lambda rows, output_json_path, output_csv_path, root_key: (
            Path(output_json_path),
            Path(output_csv_path),
        ),
    )

    assert captured["progress_callback"] is progress_ui.benchmark_callback
    assert progress_ui.started == [("refinement", 1, "Refinement")]
    assert progress_ui.finished[0].status == "complete"
    assert progress_ui.advanced == 1


def test_reused_training_stage_advances_pipeline(tmp_path: Path, monkeypatch):
    paths = _make_paths(tmp_path)
    progress_ui = RecordingProgressUI()
    states = {
        "train": {
            "status": "complete",
            "config_hash": "same",
            "inputs": {
                "base_experiment_config": str((tmp_path / "base.yaml").resolve()),
                "data_config_path": str((tmp_path / "data.yaml").resolve()),
                "stage_root": str(paths.train_root.resolve()),
            },
            "outputs": {
                "runtime_experiment_config_path": str(paths.runtime_experiment_config_path),
                "checkpoint_path": str(tmp_path / "best.pt"),
                "training_summary_path": str(tmp_path / "training_summary.json"),
                "training_history_json_path": None,
                "training_history_csv_path": None,
            },
            "created_at_utc": "2026-01-01T00:00:00Z",
            "updated_at_utc": "2026-01-01T00:00:00Z",
        }
    }

    run_training_stage(
        paths=paths,
        states=states,
        base_experiment_config=tmp_path / "base.yaml",
        data_config_path=tmp_path / "data.yaml",
        step_log_every=1,
        run_stage_train=True,
        progress_ui=progress_ui,
        json_hash_func=lambda payload: "same",
        path_identity_func=workflow_state.path_identity,
        write_runtime_experiment_config_func=lambda **kwargs: paths.runtime_experiment_config_path,
        load_experiment_config_func=lambda path: _runtime_config(tmp_path),
        resolve_notebook_checkpoint_func=lambda **kwargs: tmp_path / "best.pt",
        train_model_func=lambda **kwargs: None,
    )

    assert progress_ui.started == []
    assert progress_ui.finished[0].status == "reused"
    assert progress_ui.advanced == 1
