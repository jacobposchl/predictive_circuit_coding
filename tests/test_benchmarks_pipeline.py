from __future__ import annotations

import io
import json
from pathlib import Path
import textwrap

from predictive_circuit_coding.benchmarks import (
    load_notebook_pipeline_config,
    run_notebook_pipeline,
    run_notebook_pipeline_from_config,
)
from predictive_circuit_coding.benchmarks.contracts import BenchmarkArmSpec, BenchmarkTaskSpec
from predictive_circuit_coding.benchmarks.run import (
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
from predictive_circuit_coding.training import load_experiment_config
from predictive_circuit_coding.training.loop import train_model
from predictive_circuit_coding.utils.notebook import NotebookDatasetConfig, NotebookTrainingConfig
from tests.test_stage5_stage6_workflow import _create_prepared_workspace


def test_unified_pipeline_notebook_parses_and_has_stage_runner_cell() -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "run_predictive_circuit_coding_pipeline_colab.ipynb"
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert payload["nbformat"] == 4
    cells = payload["cells"]
    assert len(cells) >= 5
    sources = ["".join(cell.get("source", [])) for cell in cells]
    assert any("run_notebook_pipeline_from_config" in source for source in sources)
    assert any("PIPELINE_CONFIG_PATH" in source for source in sources)


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
              pca_components: 16
              session_holdout_fraction: 0.5
              session_holdout_seed: 7
              neighbor_k: 4
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
        pca_components=8,
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
        pca_components=8,
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
              pca_components: 8
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
              pca_components: 8
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
