from __future__ import annotations

import io
import json
from pathlib import Path
import textwrap

import torch
import yaml

from predictive_circuit_coding.benchmarks import (
    load_notebook_pipeline_config,
    run_notebook_pipeline,
    run_notebook_pipeline_from_config,
    verify_full_run_readiness,
)
from predictive_circuit_coding.benchmarks.contracts import BenchmarkArmSpec, BenchmarkTaskSpec
from predictive_circuit_coding.benchmarks.features import extract_benchmark_selected_windows
from predictive_circuit_coding.benchmarks.run import (
    default_motif_arm_specs,
    default_representation_arm_specs,
    run_motif_benchmark_matrix,
    run_representation_benchmark_matrix,
)
from predictive_circuit_coding.data import (
    build_session_catalog_from_manifest,
    create_workspace,
    load_preparation_config,
    load_session_manifest,
    write_session_catalog,
    write_session_catalog_csv,
)
from predictive_circuit_coding.discovery.run import prepare_discovery_collection
from predictive_circuit_coding.training import load_experiment_config, load_training_checkpoint
from predictive_circuit_coding.training.loop import train_model
from predictive_circuit_coding.utils.notebook import (
    build_pipeline_summary_figure,
    build_synthetic_pipeline_summary_tables,
    load_pipeline_display_tables,
    NotebookDatasetConfig,
    NotebookProgressConfig,
    NotebookProgressUI,
    NotebookTrainingConfig,
    write_synthetic_pipeline_summary_preview,
)
from tests.test_stage5_stage6_workflow import _create_prepared_workspace


def _write_augmented_experiment_config(base_config_path: Path, *, output_path: Path) -> Path:
    payload = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    objective = dict(payload.get("objective") or {})
    objective["cross_session_aug"] = {
        "enabled": True,
        "training_variant_name": "test_cross_session_aug",
        "target_label": "stimulus_change",
        "target_label_mode": "auto",
        "target_label_match_value": None,
        "aug_prob_start": 0.5,
        "aug_prob_end": 0.5,
        "region_loss_weight_start": 0.2,
        "region_loss_weight_end": 0.2,
        "warmup_epochs": 0,
        "canonical_regions": [],
        "donor_cache_size_per_label_session": 4,
        "min_shared_regions": 1,
        "geometry_monitor_every_epochs": 1,
        "geometry_monitor_split": "discovery",
        "geometry_monitor_max_batches": 2,
        "geometry_monitor_neighbor_k": 3,
    }
    artifacts = dict(payload.get("artifacts") or {})
    artifacts["checkpoint_prefix"] = "pcc_aug_test"
    payload["objective"] = objective
    payload["artifacts"] = artifacts
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_path


def _write_full_verification_pipeline_config(
    *,
    output_path: Path,
    experiment_config_path: Path,
    data_config_path: Path,
    local_artifact_root: Path,
    include_stimulus_omitted: bool = True,
) -> Path:
    task_names = ["stimulus_change", "trials_go"]
    if include_stimulus_omitted:
        task_names.append("stimulus_omitted")
    output_path.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "experiment_config_path": experiment_config_path.name,
                    "data_config_path": data_config_path.name,
                    "local_artifact_root": str(local_artifact_root),
                    "drive_export_root": "",
                    "source_dataset_root": "",
                },
                "stages": {
                    "train": True,
                    "evaluate": True,
                    "representation_benchmark": True,
                    "motif_benchmark": True,
                    "alignment_diagnostic": False,
                    "image_identity_appendix": False,
                },
                "pipeline": {
                    "stage_prepared_sessions_locally": False,
                    "step_log_every": 10,
                    "session_holdout_fraction": 0.5,
                    "session_holdout_seed": 7,
                    "neighbor_k": 5,
                    "debug_retain_intermediates": False,
                },
                "tasks": {
                    "image_target_name": None,
                    "representation": task_names,
                    "motifs": task_names,
                },
                "arms": {
                    "representation": [
                        "count_patch_mean_raw",
                        "untrained_encoder_raw",
                        "encoder_raw",
                        "encoder_whitened",
                    ],
                    "motifs": [
                        "untrained_encoder_raw",
                        "encoder_raw",
                        "encoder_whitened",
                    ],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return output_path


def test_unified_pipeline_notebook_parses_and_has_stage_runner_cell() -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "run_predictive_circuit_coding_pipeline_colab.ipynb"
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert payload["nbformat"] == 4
    cells = payload["cells"]
    assert len(cells) >= 5
    sources = ["".join(cell.get("source", [])) for cell in cells]
    assert any("run_notebook_pipeline_from_config" in source for source in sources)
    assert any("PIPELINE_CONFIG_PATH" in source for source in sources)


def test_default_benchmark_arms_use_untrained_encoder_baseline_without_pca() -> None:
    representation_arms = default_representation_arm_specs()
    motif_arms = default_motif_arm_specs()
    assert tuple(arm.name for arm in representation_arms) == (
        "count_patch_mean_raw",
        "untrained_encoder_raw",
        "encoder_raw",
        "encoder_whitened",
    )
    assert tuple(arm.name for arm in motif_arms) == (
        "untrained_encoder_raw",
        "encoder_raw",
        "encoder_whitened",
    )
    assert not any(arm.use_pca for arm in representation_arms + motif_arms)


def test_load_notebook_pipeline_config_resolves_relative_paths(tmp_path: Path) -> None:
    pipeline_config_path = tmp_path / "configs" / "pcc" / "pipeline_test.yaml"
    pipeline_config_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline_config_path.write_text(
        textwrap.dedent(
            """
            paths:
              experiment_config_path: predictive_circuit_coding_debug.yaml
              data_config_path: allen_visual_behavior_neuropixels_local.yaml
              local_artifact_root: ../../artifacts
              drive_export_root: /content/drive/MyDrive/pcc_colab_outputs
              source_dataset_root: /content/drive/MyDrive/pcc_runs/data/allen_visual_behavior_neuropixels

            stages:
              train: true
              evaluate: true
              representation_benchmark: true
              motif_benchmark: false
              alignment_diagnostic: false
              image_identity_appendix: false

            pipeline:
              stage_prepared_sessions_locally: true
              step_log_every: 3
              session_holdout_fraction: 0.5
              session_holdout_seed: 7
              neighbor_k: 4
              debug_retain_intermediates: false

            notebook_ui:
              enabled: true
              progress_backend: tqdm
              mode: clean_dashboard
              log_mode: failures_only
              leave_pipeline_bar: true
              leave_stage_bars: false
              show_stage_summaries: true
              show_artifact_paths: compact
              metric_snapshot_every_n:

            tasks:
              representation: [stimulus_change]
              motifs: [stimulus_change]

            arms:
              representation: [encoder_raw]
              motifs: [encoder_raw]
            """
        ).strip(),
        encoding="utf-8",
    )

    config = load_notebook_pipeline_config(pipeline_config_path)
    assert config.experiment_config_path == (pipeline_config_path.parent / "predictive_circuit_coding_debug.yaml").resolve()
    assert config.data_config_path == (pipeline_config_path.parent / "allen_visual_behavior_neuropixels_local.yaml").resolve()
    assert config.local_artifact_root == (pipeline_config_path.parent / "../../artifacts").resolve()
    assert str(config.source_dataset_root).replace("\\", "/").endswith(
        "/content/drive/MyDrive/pcc_runs/data/allen_visual_behavior_neuropixels"
    )
    assert config.stage_prepared_sessions_locally is True
    assert config.representation_task_names == ("stimulus_change",)
    assert config.representation_arm_names == ("encoder_raw",)
    assert config.notebook_ui.mode == "clean_dashboard"
    assert config.notebook_ui.log_mode == "failures_only"


def test_representation_benchmark_skips_optional_missing_field(tmp_path: Path) -> None:
    prep_config_path, experiment_config_path, _ = _create_prepared_workspace(tmp_path)
    experiment_config = load_experiment_config(experiment_config_path)
    results = run_representation_benchmark_matrix(
        experiment_config=experiment_config,
        data_config_path=prep_config_path,
        checkpoint_path=tmp_path / "unused.pt",
        output_root=tmp_path / "representation_benchmarks",
        task_specs=(
            BenchmarkTaskSpec(
                name="missing_optional_field",
                target_label="stimulus_presentations.not_a_field",
                target_label_match_value="im0",
                optional=True,
            ),
        ),
        arm_specs=(BenchmarkArmSpec("count_patch_mean_raw", "count_patch_mean", "raw"),),
    )

    assert len(results) == 1
    assert results[0].status == "skipped_missing_field"
    payload = json.loads(results[0].summary_json_path.read_text(encoding="utf-8"))
    assert payload["representation_benchmark"][0]["status"] == "skipped_missing_field"


def test_benchmark_matrices_write_summary_artifacts(tmp_path: Path) -> None:
    prep_config_path, experiment_config_path, _ = _create_prepared_workspace(tmp_path)
    experiment_config = load_experiment_config(experiment_config_path)
    training_result = train_model(
        experiment_config=experiment_config,
        data_config_path=prep_config_path,
        train_split="train",
        valid_split="valid",
    )

    representation_results = run_representation_benchmark_matrix(
        experiment_config=experiment_config,
        data_config_path=prep_config_path,
        checkpoint_path=training_result.checkpoint_path,
        output_root=tmp_path / "representation_benchmarks",
        task_specs=(BenchmarkTaskSpec(name="stimulus_change", target_label="stimulus_change"),),
        arm_specs=(BenchmarkArmSpec("encoder_raw", "encoder", "raw"),),
        session_holdout_fraction=0.5,
        session_holdout_seed=11,
        neighbor_k=3,
    )
    motif_results = run_motif_benchmark_matrix(
        experiment_config=experiment_config,
        data_config_path=prep_config_path,
        checkpoint_path=training_result.checkpoint_path,
        output_root=tmp_path / "motif_benchmarks",
        task_specs=(BenchmarkTaskSpec(name="stimulus_change", target_label="stimulus_change"),),
        arm_specs=(BenchmarkArmSpec("encoder_raw", "encoder", "raw"),),
        session_holdout_fraction=0.5,
        session_holdout_seed=11,
    )

    assert len(representation_results) == 1
    assert representation_results[0].summary_json_path.is_file()
    assert representation_results[0].transform_summary_json_path.is_file()
    assert "status" in representation_results[0].summary

    assert len(motif_results) == 1
    assert motif_results[0].summary_json_path.is_file()
    assert motif_results[0].cluster_summary_json_path.is_file()
    assert motif_results[0].discovery_artifact_path.is_file()
    assert "status" in motif_results[0].summary


def test_untrained_encoder_features_are_deterministic_and_checkpoint_free(tmp_path: Path) -> None:
    prep_config_path, experiment_config_path, _ = _create_prepared_workspace(tmp_path)
    experiment_config = load_experiment_config(experiment_config_path)
    training_result = train_model(
        experiment_config=experiment_config,
        data_config_path=prep_config_path,
        train_split="train",
        valid_split="valid",
    )
    window_plan = prepare_discovery_collection(
        experiment_config=experiment_config,
        data_config_path=prep_config_path,
        split_name="discovery",
    )

    untrained_a = extract_benchmark_selected_windows(
        experiment_config=experiment_config,
        data_config_path=prep_config_path,
        feature_family="untrained_encoder",
        window_plan=window_plan,
    )
    untrained_b = extract_benchmark_selected_windows(
        experiment_config=experiment_config,
        data_config_path=prep_config_path,
        feature_family="untrained_encoder",
        window_plan=window_plan,
    )
    trained = extract_benchmark_selected_windows(
        experiment_config=experiment_config,
        data_config_path=prep_config_path,
        feature_family="encoder",
        window_plan=window_plan,
        checkpoint_path=training_result.checkpoint_path,
    )

    assert untrained_a.feature_family == "untrained_encoder"
    assert untrained_a.pooled_features.shape == trained.pooled_features.shape
    assert untrained_a.token_tensors.shape == trained.token_tensors.shape
    assert torch.allclose(untrained_a.pooled_features, untrained_b.pooled_features)


def test_augmented_training_writes_geometry_monitor_and_auxiliary_state(tmp_path: Path) -> None:
    prep_config_path, experiment_config_path, _ = _create_prepared_workspace(tmp_path)
    augmented_config_path = _write_augmented_experiment_config(
        experiment_config_path,
        output_path=experiment_config_path.with_name("experiment_cross_session_aug.yaml"),
    )
    experiment_config = load_experiment_config(augmented_config_path)

    first = train_model(
        experiment_config=experiment_config,
        data_config_path=prep_config_path,
        train_split="train",
        valid_split="valid",
    )
    assert first.geometry_monitor_json_path is not None and first.geometry_monitor_json_path.is_file()
    assert first.geometry_monitor_csv_path is not None and first.geometry_monitor_csv_path.is_file()
    checkpoint_payload = load_training_checkpoint(first.checkpoint_path, map_location="cpu")
    assert checkpoint_payload.get("auxiliary_state") is not None
    assert checkpoint_payload["auxiliary_state"].get("region_head_state") is not None
    assert checkpoint_payload["auxiliary_state"].get("donor_cache_state") is not None

    resumed_payload = yaml.safe_load(augmented_config_path.read_text(encoding="utf-8"))
    resumed_payload["training"]["num_epochs"] = 3
    resumed_payload["training"]["resume_checkpoint"] = str(first.checkpoint_path)
    resumed_path = augmented_config_path.with_name("experiment_cross_session_aug_resume.yaml")
    resumed_path.write_text(yaml.safe_dump(resumed_payload, sort_keys=False), encoding="utf-8")
    resumed_config = load_experiment_config(resumed_path)
    second = train_model(
        experiment_config=resumed_config,
        data_config_path=prep_config_path,
        train_split="train",
        valid_split="valid",
    )
    assert second.checkpoint_path.is_file()
    assert second.geometry_monitor_json_path is not None and second.geometry_monitor_json_path.is_file()


def test_augmented_pipeline_run_surfaces_training_variant_and_geometry(tmp_path: Path) -> None:
    prep_config_path, experiment_config_path, _ = _create_prepared_workspace(tmp_path)
    augmented_config_path = _write_augmented_experiment_config(
        experiment_config_path,
        output_path=experiment_config_path.with_name("experiment_cross_session_aug.yaml"),
    )
    drive_export_root = tmp_path / "drive" / "pcc_colab_outputs"
    local_artifact_root = tmp_path / "local_artifacts"

    result = run_notebook_pipeline(
        base_experiment_config=augmented_config_path,
        data_config_path=prep_config_path,
        drive_export_root=drive_export_root,
        local_artifact_root=local_artifact_root,
        pipeline_run_id="run_augmented_pipeline",
        dataset_config=NotebookDatasetConfig(use_full_dataset=True),
        training_config=NotebookTrainingConfig(num_epochs=2, train_steps_per_epoch=2, validation_steps=1),
        step_log_every=1,
        run_stage_train=True,
        run_stage_evaluate=True,
        run_stage_representation_benchmark=True,
        run_stage_motif_benchmark=True,
        run_stage_alignment_diagnostic=False,
        run_stage_image_identity_appendix=False,
        image_target_name=None,
        representation_task_names=("stimulus_change",),
        motif_task_names=("stimulus_change",),
        representation_arm_names=("encoder_raw",),
        motif_arm_names=("encoder_raw",),
        session_holdout_fraction=0.5,
        session_holdout_seed=5,
        neighbor_k=3,
        debug_retain_intermediates=False,
        output_stream=io.StringIO(),
    )

    assert result.training_geometry_monitor_json_path is not None
    assert result.training_geometry_monitor_csv_path is not None
    tables = load_pipeline_display_tables(
        representation_summary_csv_path=result.representation_summary_csv_path,
        motif_summary_csv_path=result.motif_summary_csv_path,
        final_summary_csv_path=result.final_summary_csv_path,
        training_geometry_monitor_csv_path=result.training_geometry_monitor_csv_path,
    )
    assert "training_variant_name" in tables["representation"].columns
    assert not tables["training_geometry"].empty
    assert "training_variant_names" in json.loads(result.final_summary_json_path.read_text(encoding="utf-8"))["final_project_summary"][0]


def test_full_run_verification_blocks_known_degraded_task_panel(tmp_path: Path) -> None:
    prep_config_path, experiment_config_path, _ = _create_prepared_workspace(tmp_path)
    full_config_path = _write_augmented_experiment_config(
        experiment_config_path,
        output_path=experiment_config_path.with_name("predictive_circuit_coding_cross_session_aug_full.yaml"),
    )
    payload = yaml.safe_load(full_config_path.read_text(encoding="utf-8"))
    payload["training"]["num_epochs"] = 50
    payload["objective"]["cross_session_aug"]["training_variant_name"] = "cross_session_aug_full"
    payload["discovery"]["max_batches"] = 2
    payload["evaluation"]["max_batches"] = 2
    full_config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    pipeline_config_path = _write_full_verification_pipeline_config(
        output_path=full_config_path.with_name("pipeline_cross_session_aug_full.yaml"),
        experiment_config_path=full_config_path,
        data_config_path=prep_config_path,
        local_artifact_root=tmp_path / "artifacts",
    )

    result = verify_full_run_readiness(
        pipeline_config_path=pipeline_config_path,
        output_root=tmp_path / "verification",
    )

    assert result.status == "blocked"
    assert result.training_num_epochs == 50
    assert result.coverage_csv_path.is_file()
    assert result.summary_json_path.is_file()
    assert any(row.task_name == "stimulus_omitted" and row.status != "ok" for row in result.coverage_rows)
    assert any(issue.gate == "task_coverage" for issue in result.issues)


def test_full_run_verification_blocks_not_full_epoch_config(tmp_path: Path) -> None:
    prep_config_path, experiment_config_path, _ = _create_prepared_workspace(tmp_path)
    full_config_path = _write_augmented_experiment_config(
        experiment_config_path,
        output_path=experiment_config_path.with_name("predictive_circuit_coding_cross_session_aug_full.yaml"),
    )
    payload = yaml.safe_load(full_config_path.read_text(encoding="utf-8"))
    payload["training"]["num_epochs"] = 49
    payload["objective"]["cross_session_aug"]["training_variant_name"] = "cross_session_aug_full"
    full_config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    pipeline_config_path = _write_full_verification_pipeline_config(
        output_path=full_config_path.with_name("pipeline_cross_session_aug_full.yaml"),
        experiment_config_path=full_config_path,
        data_config_path=prep_config_path,
        local_artifact_root=tmp_path / "artifacts",
    )

    result = verify_full_run_readiness(
        pipeline_config_path=pipeline_config_path,
        output_root=tmp_path / "verification",
    )

    assert result.status == "blocked"
    assert any("num_epochs" in issue.message for issue in result.issues)


def test_full_run_verification_blocks_single_positive_session_panel(tmp_path: Path) -> None:
    prep_config_path, experiment_config_path, _ = _create_prepared_workspace(tmp_path)
    full_config_path = _write_augmented_experiment_config(
        experiment_config_path,
        output_path=experiment_config_path.with_name("predictive_circuit_coding_cross_session_aug_full.yaml"),
    )
    payload = yaml.safe_load(full_config_path.read_text(encoding="utf-8"))
    payload["training"]["num_epochs"] = 50
    payload["objective"]["cross_session_aug"]["training_variant_name"] = "cross_session_aug_full"
    payload["discovery"]["max_batches"] = 2
    payload["evaluation"]["max_batches"] = 2
    full_config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    pipeline_config_path = _write_full_verification_pipeline_config(
        output_path=full_config_path.with_name("pipeline_cross_session_aug_full.yaml"),
        experiment_config_path=full_config_path,
        data_config_path=prep_config_path,
        local_artifact_root=tmp_path / "artifacts",
        include_stimulus_omitted=False,
    )

    result = verify_full_run_readiness(
        pipeline_config_path=pipeline_config_path,
        output_root=tmp_path / "verification",
    )

    assert result.status == "blocked"
    assert {row.task_name for row in result.coverage_rows} == {"stimulus_change", "trials_go"}
    assert any(row.status == "blocked_insufficient_positive_sessions" for row in result.coverage_rows)
    assert any("too few sessions" in (issue.message or "") for issue in result.issues)


def test_run_notebook_pipeline_writes_resume_state(tmp_path: Path) -> None:
    prep_config_path, experiment_config_path, _ = _create_prepared_workspace(tmp_path)
    drive_export_root = tmp_path / "drive" / "pcc_colab_outputs"
    local_artifact_root = tmp_path / "local_artifacts"
    stream = io.StringIO()

    first = run_notebook_pipeline(
        base_experiment_config=experiment_config_path,
        data_config_path=prep_config_path,
        drive_export_root=drive_export_root,
        local_artifact_root=local_artifact_root,
        pipeline_run_id="run_test_pipeline",
        dataset_config=NotebookDatasetConfig(use_full_dataset=True),
        training_config=NotebookTrainingConfig(num_epochs=1, train_steps_per_epoch=1, validation_steps=1),
        step_log_every=1,
        run_stage_train=True,
        run_stage_evaluate=True,
        run_stage_representation_benchmark=True,
        run_stage_motif_benchmark=True,
        run_stage_alignment_diagnostic=False,
        run_stage_image_identity_appendix=False,
        image_target_name=None,
        representation_task_names=("stimulus_change",),
        motif_task_names=("stimulus_change",),
        representation_arm_names=("encoder_raw",),
        motif_arm_names=("encoder_raw",),
        session_holdout_fraction=0.5,
        session_holdout_seed=5,
        neighbor_k=3,
        debug_retain_intermediates=False,
        output_stream=stream,
    )

    state_before = json.loads(first.pipeline_state_path.read_text(encoding="utf-8"))
    assert first.pipeline_manifest_path.is_file()
    assert first.training_summary_path.is_file()
    assert first.representation_summary_json_path.is_file()
    assert first.motif_summary_json_path.is_file()
    assert first.final_summary_json_path.is_file()
    assert "epoch=1 step=" not in stream.getvalue()

    second = run_notebook_pipeline(
        base_experiment_config=experiment_config_path,
        data_config_path=prep_config_path,
        drive_export_root=drive_export_root,
        local_artifact_root=local_artifact_root,
        pipeline_run_id="run_test_pipeline",
        dataset_config=NotebookDatasetConfig(use_full_dataset=True),
        training_config=NotebookTrainingConfig(num_epochs=1, train_steps_per_epoch=1, validation_steps=1),
        step_log_every=1,
        run_stage_train=True,
        run_stage_evaluate=True,
        run_stage_representation_benchmark=True,
        run_stage_motif_benchmark=True,
        run_stage_alignment_diagnostic=False,
        run_stage_image_identity_appendix=False,
        image_target_name=None,
        representation_task_names=("stimulus_change",),
        motif_task_names=("stimulus_change",),
        representation_arm_names=("encoder_raw",),
        motif_arm_names=("encoder_raw",),
        session_holdout_fraction=0.5,
        session_holdout_seed=5,
        neighbor_k=3,
        debug_retain_intermediates=False,
        output_stream=stream,
    )

    state_after = json.loads(second.pipeline_state_path.read_text(encoding="utf-8"))
    assert first.run_id == second.run_id
    assert state_before["stages"]["train"]["updated_at_utc"] == state_after["stages"]["train"]["updated_at_utc"]
    assert state_after["stages"]["representation_benchmark"]["status"] == "complete"
    assert state_after["stages"]["motif_benchmark"]["status"] == "complete"
    assert state_after["stages"]["final_reports"]["status"] == "complete"


def test_notebook_progress_ui_falls_back_without_widgets() -> None:
    ui = NotebookProgressUI(config=NotebookProgressConfig(enabled=True))
    ui.start_pipeline(total_stages=2, completed_stages=0)
    ui.start_stage(stage_name="train", total=1, description="Training")
    ui.update_stage(current=1, total=1, description="Training")
    ui.finish_stage()
    ui.finish_pipeline()


def test_train_model_emits_structured_progress_events(tmp_path: Path) -> None:
    prep_config_path, experiment_config_path, _ = _create_prepared_workspace(tmp_path)
    experiment_config = load_experiment_config(experiment_config_path)
    events = []

    train_model(
        experiment_config=experiment_config,
        data_config_path=prep_config_path,
        train_split="train",
        valid_split="valid",
        progress_callback=events.append,
        emit_logs=False,
    )

    event_types = [event.event_type for event in events]
    assert "epoch_start" in event_types
    assert "step" in event_types
    assert "validation_start" in event_types
    assert "validation_end" in event_types
    assert "checkpoint_saved" in event_types
    assert "epoch_end" in event_types
    assert event_types[-1] == "training_complete"


def test_evaluate_checkpoint_on_split_emits_progress_events(tmp_path: Path) -> None:
    prep_config_path, experiment_config_path, _ = _create_prepared_workspace(tmp_path)
    experiment_config = load_experiment_config(experiment_config_path)
    training_result = train_model(
        experiment_config=experiment_config,
        data_config_path=prep_config_path,
        train_split="train",
        valid_split="valid",
        emit_logs=False,
    )
    from predictive_circuit_coding.evaluation import evaluate_checkpoint_on_split

    events = []
    evaluate_checkpoint_on_split(
        experiment_config=experiment_config,
        data_config_path=prep_config_path,
        checkpoint_path=str(training_result.checkpoint_path),
        split_name="test",
        max_batches=1,
        progress_callback=events.append,
    )
    assert [event.event_type for event in events] == ["split_start", "batch", "split_end"]


def test_benchmark_matrices_emit_progress_events(tmp_path: Path) -> None:
    prep_config_path, experiment_config_path, _ = _create_prepared_workspace(tmp_path)
    experiment_config = load_experiment_config(experiment_config_path)
    training_result = train_model(
        experiment_config=experiment_config,
        data_config_path=prep_config_path,
        train_split="train",
        valid_split="valid",
        emit_logs=False,
    )

    representation_events = []
    motif_events = []
    run_representation_benchmark_matrix(
        experiment_config=experiment_config,
        data_config_path=prep_config_path,
        checkpoint_path=training_result.checkpoint_path,
        output_root=tmp_path / "representation_benchmarks",
        task_specs=(BenchmarkTaskSpec(name="stimulus_change", target_label="stimulus_change"),),
        arm_specs=(BenchmarkArmSpec("encoder_raw", "encoder", "raw"),),
        session_holdout_fraction=0.5,
        session_holdout_seed=11,
        neighbor_k=3,
        progress_callback=representation_events.append,
    )
    run_motif_benchmark_matrix(
        experiment_config=experiment_config,
        data_config_path=prep_config_path,
        checkpoint_path=training_result.checkpoint_path,
        output_root=tmp_path / "motif_benchmarks",
        task_specs=(BenchmarkTaskSpec(name="stimulus_change", target_label="stimulus_change"),),
        arm_specs=(BenchmarkArmSpec("encoder_raw", "encoder", "raw"),),
        session_holdout_fraction=0.5,
        session_holdout_seed=11,
        progress_callback=motif_events.append,
    )

    assert "task_start" in [event.event_type for event in representation_events]
    assert "arm_start" in [event.event_type for event in representation_events]
    assert "arm_end" in [event.event_type for event in representation_events]
    assert any(event.step_name == "discovery_extraction" for event in representation_events if event.event_type == "arm_step")

    assert "task_start" in [event.event_type for event in motif_events]
    assert "arm_start" in [event.event_type for event in motif_events]
    assert "arm_end" in [event.event_type for event in motif_events]
    assert any(
        event.step_name in {"discovery_extraction", "test_extraction", "token_shards", "probe_fit", "candidate_selection"}
        for event in motif_events
        if event.event_type == "arm_step"
    )


def test_build_pipeline_summary_figure_renders_synthetic_dashboard() -> None:
    import matplotlib.pyplot as plt

    tables = build_synthetic_pipeline_summary_tables()

    figure = build_pipeline_summary_figure(
        representation_df=tables["representation"],
        motif_df=tables["motif"],
        final_df=tables["final"],
    )

    assert len(figure.axes) >= 4
    titles = [axis.get_title() for axis in figure.axes[:4]]
    assert "Representation Benchmark" in titles
    assert "Motif Benchmark" in titles
    assert "Generalization Gap" in titles
    assert "Run Summary" in titles
    plt.close(figure)


def test_write_synthetic_pipeline_summary_preview_writes_png(tmp_path: Path) -> None:
    output_path = write_synthetic_pipeline_summary_preview(tmp_path / "synthetic_pipeline_summary.png")

    assert output_path.is_file()
    assert output_path.suffix == ".png"
    assert output_path.stat().st_size > 0


def test_run_notebook_pipeline_from_config_uses_repo_config_surface(tmp_path: Path) -> None:
    prep_config_path, experiment_config_path, _ = _create_prepared_workspace(tmp_path)
    pipeline_config_path = tmp_path / "configs" / "pcc" / "pipeline_test.yaml"
    pipeline_config_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline_config_path.write_text(
        textwrap.dedent(
            f"""
            paths:
              experiment_config_path: {experiment_config_path.as_posix()}
              data_config_path: {prep_config_path.as_posix()}
              local_artifact_root: { (tmp_path / 'local_artifacts').as_posix() }
              drive_export_root: { (tmp_path / 'drive' / 'pcc_colab_outputs').as_posix() }

            stages:
              train: true
              evaluate: true
              representation_benchmark: true
              motif_benchmark: false
              alignment_diagnostic: false
              image_identity_appendix: false

            pipeline:
              step_log_every: 1
              session_holdout_fraction: 0.5
              session_holdout_seed: 5
              neighbor_k: 3
              debug_retain_intermediates: false

            tasks:
              representation: [stimulus_change]
              motifs: [stimulus_change]

            arms:
              representation: [encoder_raw]
              motifs: [encoder_raw]
            """
        ).strip(),
        encoding="utf-8",
    )

    result = run_notebook_pipeline_from_config(
        pipeline_config_path=pipeline_config_path,
        pipeline_run_id="run_test_pipeline_from_config",
        output_stream=io.StringIO(),
    )

    state_payload = json.loads(result.pipeline_state_path.read_text(encoding="utf-8"))
    assert result.training_summary_path.is_file()
    assert result.representation_summary_json_path.is_file()
    assert state_payload["stages"]["train"]["status"] == "complete"
    assert state_payload["stages"]["representation_benchmark"]["status"] == "complete"


def test_run_notebook_pipeline_from_config_stages_source_dataset_root(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    prep_config_path, experiment_config_path, source_workspace_root = _create_prepared_workspace(source_root)
    source_prep_config = load_preparation_config(prep_config_path)
    source_workspace = create_workspace(source_prep_config)
    source_manifest = load_session_manifest(source_workspace.session_manifest_path)
    source_catalog = build_session_catalog_from_manifest(source_manifest)
    write_session_catalog(source_catalog, source_workspace.session_catalog_path)
    write_session_catalog_csv(source_catalog, source_workspace.session_catalog_csv_path)
    target_config_dir = tmp_path / "target" / "configs" / "pcc"
    target_config_dir.mkdir(parents=True, exist_ok=True)
    target_prep_config_path = target_config_dir / "prep.yaml"
    target_prep_config_path.write_text(
        Path(prep_config_path).read_text(encoding="utf-8").replace(
            "workspace_root: data/allen_visual_behavior_neuropixels",
            "workspace_root: data/local_stage_target",
        ),
        encoding="utf-8",
    )
    pipeline_config_path = target_config_dir / "pipeline_stage_from_source.yaml"
    pipeline_config_path.write_text(
        textwrap.dedent(
            f"""
            paths:
              experiment_config_path: {experiment_config_path.as_posix()}
              data_config_path: {target_prep_config_path.as_posix()}
              local_artifact_root: { (tmp_path / 'local_artifacts').as_posix() }
              drive_export_root: { (tmp_path / 'drive' / 'pcc_colab_outputs').as_posix() }
              source_dataset_root: { source_workspace_root.as_posix() }

            stages:
              train: true
              evaluate: true
              representation_benchmark: true
              motif_benchmark: false
              alignment_diagnostic: false
              image_identity_appendix: false

            pipeline:
              stage_prepared_sessions_locally: true
              step_log_every: 1
              session_holdout_fraction: 0.5
              session_holdout_seed: 5
              neighbor_k: 3
              debug_retain_intermediates: false

            tasks:
              representation: [stimulus_change]
              motifs: [stimulus_change]

            arms:
              representation: [encoder_raw]
              motifs: [encoder_raw]
            """
        ).strip(),
        encoding="utf-8",
    )

    result = run_notebook_pipeline_from_config(
        pipeline_config_path=pipeline_config_path,
        pipeline_run_id="run_stage_from_source",
        output_stream=io.StringIO(),
    )

    staged_workspace_root = tmp_path / "target" / "data" / "local_stage_target"
    assert (staged_workspace_root / "manifests" / "session_catalog.json").is_file()
    assert any((staged_workspace_root / "prepared" / "allen_visual_behavior_neuropixels").glob("*.h5"))
    assert result.training_summary_path.is_file()
    assert result.representation_summary_json_path.is_file()
