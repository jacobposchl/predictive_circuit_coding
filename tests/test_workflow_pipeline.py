from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml
import pytest

from predictive_circuit_coding.workflows import (
    assert_pipeline_preflight,
    build_pipeline_preflight,
    load_pipeline_config,
    run_pipeline_from_config,
)
from predictive_circuit_coding.workflows import state as workflow_state
from predictive_circuit_coding.workflows.contracts import PipelinePaths
from predictive_circuit_coding.workflows.stages import run_training_stage


def _write_prep_config(tmp_path: Path) -> Path:
    config_dir = tmp_path / "configs" / "pcc"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "prep.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset:",
                "  dataset_id: allen_visual_behavior_neuropixels",
                "  source_name: allen_visual_behavior_neuropixels",
                "  workspace_root: data/allen_visual_behavior_neuropixels",
                "  raw_subdir: raw",
                "  prepared_subdir: prepared",
                "  manifests_subdir: manifests",
                "  splits_subdir: splits",
                "  logs_subdir: logs",
                "  prepared_session_subdir: sessions",
                "  session_manifest_name: session_manifest.json",
                "  split_manifest_name: split_manifest.json",
                "preparation:",
                "  session_table_format: csv",
                "  session_id_field: session_id",
                "  subject_id_field: subject_id",
                "  raw_path_field: raw_data_path",
                "  duration_field: duration_s",
                "  n_units_field: n_units",
                "  brain_regions_field: brain_regions",
                "  trial_count_field: trial_count",
                '  recording_id_template: "{dataset_id}/{session_id}"',
                "splits:",
                "  seed: 7",
                "  primary_axis: session",
                "  train_fraction: 0.5",
                "  valid_fraction: 0.2",
                "  discovery_fraction: 0.15",
                "  test_fraction: 0.15",
                "runtime:",
                "  local_cpu_only: true",
                "  training_surface: colab_a100",
                "brainsets_pipeline:",
                "  local_pipeline_path: ../../brainsets_local_pipelines/allen_visual_behavior_neuropixels/pipeline.py",
                "  runner_cores: 1",
                "  use_active_environment: true",
                "  processed_only_upload: true",
                "  keep_raw_cache: true",
                "  default_session_ids_file:",
                "  default_max_sessions:",
                "allen_sdk:",
                "  cache_root:",
                "  cleanup_raw_after_processing: false",
                "unit_filtering:",
                "  filter_by_validity: true",
                "  filter_out_of_brain_units: true",
                "  amplitude_cutoff_maximum: 0.1",
                "  presence_ratio_minimum: 0.95",
                "  isi_violations_maximum: 0.5",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _write_experiment_config(tmp_path: Path, *, device: str = "auto") -> Path:
    config_dir = tmp_path / "configs" / "pcc"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "experiment.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset_id: allen_visual_behavior_neuropixels",
                "split_name: train",
                "seed: 13",
                "experiment:",
                "  variant_name: refined_core",
                "splits:",
                "  train: train",
                "  valid: valid",
                "  discovery: discovery",
                "  test: test",
                "dataset_selection:",
                "  output_name: runtime_selection",
                "data_runtime:",
                "  bin_width_ms: 100.0",
                "  context_bins: 20",
                "  patch_bins: 5",
                "  min_unit_spikes: 0",
                "  max_units:",
                "  padding_strategy: mask",
                "  include_trials: true",
                "  include_stimulus_presentations: true",
                "  include_optotagging: false",
                "count_normalization:",
                "  mode: none",
                "  stats_path:",
                "model:",
                "  d_model: 16",
                "  num_heads: 4",
                "  temporal_layers: 1",
                "  spatial_layers: 1",
                "  dropout: 0.0",
                "  mlp_ratio: 2.0",
                "  l2_normalize_tokens: true",
                "  norm_eps: 1.0e-5",
                "  population_token_mode: none",
                "objective:",
                "  predictive_target_type: delta",
                "  continuation_baseline_type: previous_patch",
                "  predictive_loss: mse",
                "  reconstruction_loss: mse",
                "  reconstruction_weight: 0.05",
                "  reconstruction_target_mode: window_zscore",
                "  exclude_final_prediction_patch: true",
                "optimization:",
                "  learning_rate: 1.0e-3",
                "  weight_decay: 0.0",
                "  grad_clip_norm: 1.0",
                "  batch_size: 2",
                "  scheduler_type: none",
                "  scheduler_warmup_steps: 0",
                "training:",
                "  num_epochs: 1",
                "  train_steps_per_epoch: 1",
                "  validation_steps: 1",
                "  checkpoint_every_epochs: 1",
                "  evaluate_every_epochs: 1",
                "  resume_checkpoint:",
                "  dataloader_workers: 0",
                "  train_window_seed: 5",
                "  log_every_steps: 9",
                "execution:",
                f"  device: {device}",
                "  mixed_precision: true",
                "evaluation:",
                "  max_batches: 1",
                "  sequential_step_s: 2.0",
                "discovery:",
                "  target_label: stimulus_change",
                "  target_label_mode: auto",
                "  max_batches: 1",
                "  sampling_strategy: sequential",
                "  min_positive_windows: 1",
                "  negative_to_positive_ratio: 1.0",
                "  search_max_batches:",
                "  probe_epochs: 5",
                "  probe_learning_rate: 0.05",
                "  top_k_candidates: 8",
                "  min_candidate_score: -100.0",
                "  min_cluster_size: 2",
                "  stability_rounds: 2",
                "  shuffle_seed: 19",
                "  pooled_feature_mode: mean_tokens",
                "artifacts:",
                "  checkpoint_dir: ../../artifacts/checkpoints",
                "  summary_path: ../../artifacts/training_summary.json",
                "  checkpoint_prefix: pcc_test",
                "  save_config_snapshot: true",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _write_pipeline_config(
    tmp_path: Path,
    *,
    source_dataset_root: Path | None = None,
    extra_pipeline_lines: list[str] | None = None,
    extra_notebook_ui_lines: list[str] | None = None,
    extra_top_level_lines: list[str] | None = None,
) -> Path:
    config_dir = tmp_path / "configs" / "pcc"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "pipeline.yaml"
    lines = [
        "paths:",
        "  experiment_config_path: experiment.yaml",
        "  data_config_path: prep.yaml",
        "  local_artifact_root: ../../artifacts",
        f"  source_dataset_root: {source_dataset_root.as_posix()}" if source_dataset_root is not None else "  source_dataset_root:",
        "stages:",
        "  train: true",
        "  evaluate: true",
        "  refinement: true",
        "  alignment_diagnostic: false",
        "pipeline:",
        "  stage_prepared_sessions_locally: true",
        "  step_log_every: 3",
        "  session_holdout_fraction: 0.5",
        "  session_holdout_seed: 7",
        "  debug_retain_intermediates: false",
        "notebook_ui:",
        "  enabled: true",
        "  leave_pipeline_bar: true",
        "  leave_stage_bars: false",
        "  show_stage_summaries: true",
        "  show_artifact_paths: compact",
        "  metric_snapshot_every_n: 1",
        "tasks:",
        "  motifs: [stimulus_change]",
        "arms:",
        "  motifs: [encoder_raw]",
    ]
    if extra_pipeline_lines:
        insert_at = lines.index("notebook_ui:")
        lines = lines[:insert_at] + extra_pipeline_lines + lines[insert_at:]
    if extra_notebook_ui_lines:
        insert_at = lines.index("tasks:")
        lines = lines[:insert_at] + extra_notebook_ui_lines + lines[insert_at:]
    if extra_top_level_lines:
        lines.extend(extra_top_level_lines)
    config_path.write_text("\n".join(lines), encoding="utf-8")
    return config_path


def _write_source_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "drive_dataset"
    prepared_root = root / "prepared" / "allen_visual_behavior_neuropixels"
    prepared_root.mkdir(parents=True, exist_ok=True)
    (prepared_root / "101.h5").write_text("101", encoding="utf-8")
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "session_catalog.json").write_text(
        json.dumps(
            {
                "dataset_id": "allen_visual_behavior_neuropixels",
                "source_name": "allen_visual_behavior_neuropixels",
                "records": [
                    {
                        "recording_id": "allen_visual_behavior_neuropixels/101",
                        "session_id": "101",
                        "subject_id": "mouse_a",
                        "raw_data_path": "raw/101.nwb",
                        "duration_s": 10.0,
                        "n_units": 4,
                        "brain_regions": ["VISp"],
                        "trial_count": 5,
                        "prepared_session_path": str(prepared_root / "101.h5"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (manifests / "session_catalog.csv").write_text("session_id\n101\n", encoding="utf-8")
    splits = root / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    (splits / "split_manifest.json").write_text("{}", encoding="utf-8")
    return root


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


def test_load_pipeline_config_rejects_unused_notebook_ui_keys(tmp_path: Path) -> None:
    _write_prep_config(tmp_path)
    _write_experiment_config(tmp_path)
    config_path = _write_pipeline_config(
        tmp_path,
        extra_notebook_ui_lines=["  progress_backend: tqdm"],
    )

    with pytest.raises(ValueError, match="Unsupported notebook_ui keys"):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_unused_pipeline_keys(tmp_path: Path) -> None:
    _write_prep_config(tmp_path)
    _write_experiment_config(tmp_path)
    config_path = _write_pipeline_config(
        tmp_path,
        extra_pipeline_lines=["  neighbor_k: 3"],
    )

    with pytest.raises(ValueError, match="Unsupported pipeline keys"):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_unused_top_level_keys(tmp_path: Path) -> None:
    _write_prep_config(tmp_path)
    _write_experiment_config(tmp_path)
    config_path = _write_pipeline_config(
        tmp_path,
        extra_top_level_lines=["mystery_section:", "  enabled: true"],
    )

    with pytest.raises(ValueError, match="Unsupported top-level pipeline config keys"):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_invalid_numeric_values(tmp_path: Path) -> None:
    _write_prep_config(tmp_path)
    _write_experiment_config(tmp_path)
    config_path = _write_pipeline_config(
        tmp_path,
        extra_pipeline_lines=["  step_log_every: 0"],
    )

    with pytest.raises(ValueError, match="step_log_every must be at least 1"):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_invalid_holdout_fraction(tmp_path: Path) -> None:
    _write_prep_config(tmp_path)
    _write_experiment_config(tmp_path)
    config_path = _write_pipeline_config(
        tmp_path,
        extra_pipeline_lines=["  session_holdout_fraction: 1.0"],
    )

    with pytest.raises(ValueError, match="session_holdout_fraction must be less than 1.0"):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_invalid_metric_snapshot_interval(tmp_path: Path) -> None:
    _write_prep_config(tmp_path)
    _write_experiment_config(tmp_path)
    config_path = _write_pipeline_config(
        tmp_path,
        extra_notebook_ui_lines=["  metric_snapshot_every_n: 0"],
    )

    with pytest.raises(ValueError, match="metric_snapshot_every_n must be at least 1"):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_invalid_notebook_artifact_mode(tmp_path: Path) -> None:
    _write_prep_config(tmp_path)
    _write_experiment_config(tmp_path)
    config_path = _write_pipeline_config(
        tmp_path,
        extra_notebook_ui_lines=["  show_artifact_paths: verbose"],
    )

    with pytest.raises(ValueError, match="show_artifact_paths must be one of"):
        load_pipeline_config(config_path)


def test_pipeline_preflight_requires_cuda_for_compute_stages(tmp_path: Path, monkeypatch) -> None:
    _write_prep_config(tmp_path)
    _write_experiment_config(tmp_path, device="auto")
    source_dataset_root = _write_source_dataset(tmp_path)
    config_path = _write_pipeline_config(tmp_path, source_dataset_root=source_dataset_root)

    monkeypatch.setattr("predictive_circuit_coding.workflows.config.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("predictive_circuit_coding.workflows.config.torch.cuda.device_count", lambda: 0)

    report = build_pipeline_preflight(config_path)
    assert not report.ok
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        assert_pipeline_preflight(report)


def test_pipeline_preflight_fails_when_referenced_configs_are_missing(tmp_path: Path, monkeypatch) -> None:
    _write_prep_config(tmp_path)
    config_path = _write_pipeline_config(tmp_path)

    monkeypatch.setattr("predictive_circuit_coding.workflows.config.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("predictive_circuit_coding.workflows.config.torch.cuda.device_count", lambda: 1)
    monkeypatch.setattr("predictive_circuit_coding.workflows.config.torch.cuda.get_device_name", lambda index: "Fake GPU")

    report = build_pipeline_preflight(config_path)
    assert not report.ok
    with pytest.raises(RuntimeError, match="Experiment config not found"):
        assert_pipeline_preflight(report)


def test_pipeline_preflight_rejects_cpu_execution_for_compute_stages(tmp_path: Path, monkeypatch) -> None:
    _write_prep_config(tmp_path)
    _write_experiment_config(tmp_path, device="cpu")
    source_dataset_root = _write_source_dataset(tmp_path)
    config_path = _write_pipeline_config(tmp_path, source_dataset_root=source_dataset_root)

    monkeypatch.setattr("predictive_circuit_coding.workflows.config.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("predictive_circuit_coding.workflows.config.torch.cuda.device_count", lambda: 1)
    monkeypatch.setattr("predictive_circuit_coding.workflows.config.torch.cuda.get_device_name", lambda index: "Fake GPU")

    report = build_pipeline_preflight(config_path)
    assert not report.ok
    with pytest.raises(RuntimeError, match="execution.device is set to cpu"):
        assert_pipeline_preflight(report)


def test_training_stage_records_running_then_failed(tmp_path: Path, monkeypatch) -> None:
    paths = _make_paths(tmp_path)
    observed_statuses: list[str] = []

    def fake_train_model(**kwargs):
        payload = json.loads(paths.pipeline_state_path.read_text(encoding="utf-8"))
        observed_statuses.append(payload["stages"]["train"]["status"])
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_training_stage(
            paths=paths,
            states={},
            base_experiment_config=tmp_path / "experiment.yaml",
            data_config_path=tmp_path / "prep.yaml",
            step_log_every=3,
            run_stage_train=True,
            progress_ui=None,
            json_hash_func=workflow_state.json_hash,
            path_identity_func=workflow_state.path_identity,
            write_runtime_experiment_config_func=lambda **kwargs: paths.runtime_experiment_config_path,
            load_experiment_config_func=lambda path: SimpleNamespace(
                splits=SimpleNamespace(train="train", valid="valid"),
                artifacts=SimpleNamespace(
                    summary_path=tmp_path / "training_summary.json",
                    checkpoint_dir=tmp_path / "checkpoints",
                    checkpoint_prefix="pcc_test",
                ),
            ),
            resolve_notebook_checkpoint_func=lambda **kwargs: tmp_path / "best.pt",
            train_model_func=fake_train_model,
        )

    payload = json.loads(paths.pipeline_state_path.read_text(encoding="utf-8"))
    assert observed_statuses == ["running"]
    assert payload["stages"]["train"]["status"] == "failed"
    assert payload["stages"]["train"]["error_message"] == "boom"


def test_run_pipeline_from_config_smoke_stages_dataset_and_writes_state(tmp_path: Path, monkeypatch) -> None:
    _write_prep_config(tmp_path)
    _write_experiment_config(tmp_path, device="auto")
    source_dataset_root = _write_source_dataset(tmp_path)
    config_path = _write_pipeline_config(tmp_path, source_dataset_root=source_dataset_root)

    monkeypatch.setattr("predictive_circuit_coding.workflows.config.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("predictive_circuit_coding.workflows.config.torch.cuda.device_count", lambda: 1)
    monkeypatch.setattr("predictive_circuit_coding.workflows.config.torch.cuda.get_device_name", lambda index: "Fake GPU")

    ui_events: list[object] = []

    class RecordingPipelineUI:
        def __init__(self, *, config=None, stream=None) -> None:
            del config, stream
            ui_events.append("ui_created")

        def start_pipeline(self, *, total_stages: int, completed_stages: int = 0) -> None:
            ui_events.append(("start_pipeline", total_stages, completed_stages))

        def finish_pipeline(self) -> None:
            ui_events.append("finish_pipeline")

        def note(self, message: str) -> None:
            ui_events.append(("note", message))

        def milestone(self, message: str) -> None:
            ui_events.append(("milestone", message))

        def clear_detail(self) -> None:
            ui_events.append("clear_detail")

        def start_stage(self, *, stage_name: str, total: int | None = None, description: str | None = None) -> None:
            ui_events.append(("start_stage", stage_name, total, description))

        def finish_stage(self, summary) -> None:
            ui_events.append(("finish_stage", summary.stage_name, summary.status))

        def advance_pipeline(self, steps: int = 1) -> None:
            ui_events.append(("advance_pipeline", steps))

        def fail_stage(self, *, stage_name: str, error_message: str, debug_log_path: str | None, tail_lines=()) -> None:
            del debug_log_path, tail_lines
            ui_events.append(("fail_stage", stage_name, error_message))

        def make_copy_callback(self, *, label: str):
            ui_events.append(("make_copy_callback", label))

            def _callback(current: int, total: int) -> None:
                ui_events.append(("copy_progress", current, total))

            return _callback

        def make_training_callback(self, *, stage_name: str):
            ui_events.append(("make_training_callback", stage_name))
            return None

        def make_evaluation_callback(self, *, split_total: int):
            ui_events.append(("make_evaluation_callback", split_total))
            return None

        def make_benchmark_callback(self, *, benchmark_name: str, total_arms: int | None = None):
            ui_events.append(("make_benchmark_callback", benchmark_name, total_arms))
            return None

        def render_artifacts(self, title: str, artifacts: dict) -> None:
            ui_events.append(("render_artifacts", title, tuple(artifacts)))

    monkeypatch.setattr("predictive_circuit_coding.workflows.pipeline.NotebookProgressUI", RecordingPipelineUI)

    captured_train_kwargs: dict[str, object] = {}

    def fake_train_model(**kwargs):
        captured_train_kwargs.update(kwargs)
        experiment_config = kwargs["experiment_config"]
        checkpoint_dir = experiment_config.artifacts.checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "pcc_test_best.pt"
        checkpoint_path.write_text("checkpoint", encoding="utf-8")
        summary_path = experiment_config.artifacts.summary_path
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text("{}", encoding="utf-8")
        history_json_path = summary_path.with_name("training_history.json")
        history_json_path.write_text("{}", encoding="utf-8")
        history_csv_path = summary_path.with_name("training_history.csv")
        history_csv_path.write_text("epoch,total_loss\n1,0.1\n", encoding="utf-8")
        return SimpleNamespace(
            checkpoint_path=checkpoint_path,
            summary_path=summary_path,
            history_json_path=history_json_path,
            history_csv_path=history_csv_path,
        )

    monkeypatch.setattr("predictive_circuit_coding.workflows.pipeline.train_model", fake_train_model)
    monkeypatch.setattr(
        "predictive_circuit_coding.workflows.pipeline.evaluate_checkpoint_on_split",
        lambda **kwargs: SimpleNamespace(metrics={"predictive_improvement": 0.5}, losses={"total_loss": 0.1}),
    )
    monkeypatch.setattr(
        "predictive_circuit_coding.workflows.pipeline.write_evaluation_summary",
        lambda summary, path: Path(path).write_text("{}", encoding="utf-8"),
    )
    monkeypatch.setattr(
        "predictive_circuit_coding.workflows.pipeline.run_motif_benchmark_matrix",
        lambda **kwargs: (
            SimpleNamespace(
                summary={
                    "task_name": "stimulus_change",
                    "arm_name": "encoder_raw",
                    "status": "ok",
                    "claim_safe": True,
                }
            ),
        ),
    )

    result = run_pipeline_from_config(pipeline_config_path=config_path, pipeline_run_id="run_test")

    local_prepared = tmp_path / "data" / "allen_visual_behavior_neuropixels" / "prepared" / "allen_visual_behavior_neuropixels" / "101.h5"
    assert local_prepared.is_file()
    assert result.final_summary_json_path.is_file()
    assert result.checkpoint_path == result.local_run_root / "train" / "checkpoints" / "pcc_test_best.pt"
    assert result.training_summary_path == result.local_run_root / "train" / "training_summary.json"
    assert result.training_history_json_path == result.local_run_root / "train" / "training_history.json"
    assert result.training_history_csv_path == result.local_run_root / "train" / "training_history.csv"
    assert captured_train_kwargs["emit_logs"] is False
    assert ui_events[0] == "ui_created"
    assert ui_events[1] == ("start_pipeline", 4, 0)
    assert ("copy_progress", 1, 1) in ui_events
    assert ui_events.index(("start_pipeline", 4, 0)) < ui_events.index(("copy_progress", 1, 1))
    state_payload = json.loads(result.pipeline_state_path.read_text(encoding="utf-8"))
    assert state_payload["stages"]["train"]["status"] == "complete"
    assert state_payload["stages"]["evaluate"]["status"] == "complete"
    assert state_payload["stages"]["refinement"]["status"] == "complete"
    assert state_payload["stages"]["alignment_diagnostic"]["status"] == "skipped"
    assert state_payload["stages"]["final_reports"]["status"] == "complete"

    runtime_payload = yaml.safe_load(result.runtime_experiment_config_path.read_text(encoding="utf-8"))
    assert runtime_payload["training"]["log_every_steps"] == 3
    assert runtime_payload["artifacts"]["checkpoint_dir"] == str((result.local_run_root / "train" / "checkpoints").resolve())
    assert runtime_payload["artifacts"]["summary_path"] == str((result.local_run_root / "train" / "training_summary.json").resolve())


def test_committed_colab_notebook_does_not_patch_repo_source() -> None:
    notebook_path = Path("notebooks/run_predictive_circuit_coding_pipeline_colab.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code")

    assert "patch_once" not in code
    assert "Runtime patch for notebook pipeline" not in code
    assert "pipeline_path.write_text" not in code
    assert "predictive_circuit_coding/benchmarks/pipeline.py" not in code


def test_committed_colab_notebook_defines_pipeline_config_cell_and_fallbacks() -> None:
    notebook_path = Path("notebooks/run_predictive_circuit_coding_pipeline_colab.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    assert code_cells, "Expected code cells in the Colab notebook."

    first_code = "".join(code_cells[0].get("source", []))
    assert "# Pipeline config" in first_code
    assert "PIPELINE_CONFIG_PATH" in first_code
    assert "PIPELINE_RUN_ID" in first_code

    all_code = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
    assert "globals().get('PIPELINE_CONFIG_PATH'" in all_code
    assert "globals().get('PIPELINE_RUN_ID'" in all_code
