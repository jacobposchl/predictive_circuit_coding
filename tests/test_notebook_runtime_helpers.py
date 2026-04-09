from __future__ import annotations

import json
from pathlib import Path

import yaml

from predictive_circuit_coding.utils import (
    NotebookCommandStreamFormatter,
    NotebookDatasetConfig,
    NotebookDiscoveryRunResult,
    NotebookDiscoveryComparisonPaths,
    NotebookAlignmentDiagnosticsRunResult,
    NotebookDiagnosticsExperimentPaths,
    NotebookLocalDatasetStageResult,
    NotebookTrainingConfig,
    NotebookValidationRunResult,
    build_notebook_alignment_summary_row,
    build_notebook_discovery_comparison_local_root,
    build_notebook_discovery_comparison_paths,
    build_notebook_discovery_comparison_summary_row,
    build_notebook_discovery_runtime_config,
    build_notebook_discovery_export_path,
    build_notebook_diagnostics_experiment_paths,
    build_notebook_diagnostics_export_path,
    describe_notebook_compute_targets,
    export_notebook_discovery_comparison_artifacts,
    export_notebook_discovery_artifacts,
    export_notebook_diagnostics_artifacts,
    export_notebook_training_artifacts,
    find_existing_discovery_run,
    load_notebook_split_counts,
    materialize_notebook_prepared_sessions,
    output_indicates_missing_positive_labels,
    prepare_notebook_runtime_context,
    resolve_notebook_checkpoint,
    restore_latest_discovery_artifacts,
    restore_latest_exported_artifacts,
    run_streaming_command,
)
import pytest
import subprocess
import re


def test_notebook_command_stream_formatter_compacts_wrapped_step_logs() -> None:
    formatter = NotebookCommandStreamFormatter(step_log_every=16)
    outputs: list[str] = []
    outputs.extend(formatter.feed(" +10.0s Stage: epoch 1/8 | next artifact: checkpoint or training summary\n"))
    outputs.extend(formatter.feed(" +10.1s epoch=1 step=16: predictive_baseline_mse=0.3396,\n"))
    outputs.extend(formatter.feed("predictive_improvement=0.0002, predictive_loss=0.3394,\n"))
    outputs.extend(formatter.feed("predictive_raw_mse=0.3394, reconstruction_loss=0.2520, total_loss=0.3898\n"))
    outputs.extend(formatter.feed(" +10.2s best checkpoint: /tmp/pcc_best.pt\n"))
    outputs.extend(formatter.finalize())

    assert outputs[0].startswith(" +10.0s Stage: epoch 1/8")
    assert outputs[1] == (
        "epoch=1 step=16: predictive_improvement=0.0002, "
        "predictive_loss=0.3394, total_loss=0.3898\n"
    )
    assert outputs[2].startswith(" +10.2s best checkpoint:")


def test_prepare_notebook_runtime_context_writes_notebook_only_subset_config(tmp_path: Path) -> None:
    base_config_path = tmp_path / "predictive_circuit_coding_base.yaml"
    base_config_path.write_text(
        "\n".join(
            [
                "dataset_id: allen_visual_behavior_neuropixels",
                "split_name: train",
                "seed: 7",
                "training:",
                "  log_every_steps: 10",
                "artifacts:",
                "  checkpoint_dir: ../../artifacts/checkpoints",
                "  summary_path: ../../artifacts/training_summary.json",
                "  checkpoint_prefix: pcc",
            ]
        ),
        encoding="utf-8",
    )
    catalog_path = tmp_path / "session_catalog.csv"
    catalog_path.write_text(
        "\n".join(
            [
                "session_id,subject_id,experience_level,image_set,session_type",
                "101,mouse_a,Familiar,G,type_a",
                "102,mouse_b,Familiar,H,type_b",
                "103,mouse_c,Novel,G,type_c",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "session_catalog.json").write_text(
        json.dumps(
            {
                "dataset_id": "allen_visual_behavior_neuropixels",
                "source_name": "allen_visual_behavior_neuropixels",
                "records": [
                    {
                        "recording_id": "allen_visual_behavior_neuropixels/101",
                        "session_id": "101",
                        "subject_id": "mouse_a",
                        "experience_level": "Familiar",
                        "image_set": "G",
                        "session_type": "type_a",
                        "raw_data_path": "raw/101.nwb",
                        "duration_s": 10.0,
                        "n_units": 32,
                        "brain_regions": ["VISp"],
                        "trial_count": 20,
                    },
                    {
                        "recording_id": "allen_visual_behavior_neuropixels/102",
                        "session_id": "102",
                        "subject_id": "mouse_b",
                        "experience_level": "Familiar",
                        "image_set": "H",
                        "session_type": "type_b",
                        "raw_data_path": "raw/102.nwb",
                        "duration_s": 11.0,
                        "n_units": 28,
                        "brain_regions": ["VISam"],
                        "trial_count": 18,
                    },
                    {
                        "recording_id": "allen_visual_behavior_neuropixels/103",
                        "session_id": "103",
                        "subject_id": "mouse_c",
                        "experience_level": "Novel",
                        "image_set": "G",
                        "session_type": "type_c",
                        "raw_data_path": "raw/103.nwb",
                        "duration_s": 12.0,
                        "n_units": 16,
                        "brain_regions": ["CA1"],
                        "trial_count": 15,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    context = prepare_notebook_runtime_context(
        base_experiment_config=base_config_path,
        runtime_experiment_config=tmp_path / "colab_runtime_experiment.yaml",
        session_catalog_csv=catalog_path,
        artifact_root=tmp_path / "artifacts",
        dataset_config=NotebookDatasetConfig(
            use_full_dataset=False,
            experience_level="Familiar",
            max_sessions=2,
        ),
        training_config=NotebookTrainingConfig(
            num_epochs=20,
            train_steps_per_epoch=256,
            validation_steps=64,
        ),
        step_log_every=16,
    )

    runtime_payload = yaml.safe_load(context.experiment_config_path.read_text(encoding="utf-8"))
    assert runtime_payload["training"]["log_every_steps"] == 16
    assert runtime_payload["training"]["num_epochs"] == 20
    assert runtime_payload["training"]["train_steps_per_epoch"] == 256
    assert runtime_payload["training"]["validation_steps"] == 64
    assert runtime_payload["artifacts"]["checkpoint_dir"] == str((tmp_path / "artifacts" / "checkpoints").resolve())
    assert runtime_payload["artifacts"]["summary_path"] == str((tmp_path / "artifacts" / "training_summary.json").resolve())
    assert runtime_payload["dataset_selection"] == {}
    assert runtime_payload["runtime_subset"]["split_manifest_path"] == str(
        (tmp_path / "artifacts" / "runtime_subset" / "selected_split_manifest.json").resolve()
    )
    assert runtime_payload["runtime_subset"]["session_catalog_path"] == str(
        (tmp_path / "artifacts" / "runtime_subset" / "selected_session_catalog.json").resolve()
    )
    assert runtime_payload["discovery"]["sampling_strategy"] == "label_balanced"
    assert "search_max_batches" not in runtime_payload["discovery"]
    assert (tmp_path / "artifacts" / "runtime_subset" / "selected_session_catalog.json").is_file()
    assert (tmp_path / "artifacts" / "runtime_subset" / "selected_session_catalog.csv").is_file()
    assert (tmp_path / "artifacts" / "runtime_subset" / "selected_split_manifest.json").is_file()
    assert (tmp_path / "artifacts" / "runtime_subset" / "splits" / "torch_brain_runtime_train.yaml").is_file()
    profile_payload = json.loads(context.profile_path.read_text(encoding="utf-8"))
    assert re.fullmatch(r"run_\d{8}_\d{6}", context.run_id)
    assert profile_payload["run_id"] == context.run_id
    assert profile_payload["selected_session_count"] == 2
    assert profile_payload["selected_sessions_preview"][0]["session_id"] == "101"
    assert profile_payload["training_config"]["num_epochs"] == 20
    assert profile_payload["training_config"]["train_steps_per_epoch"] == 256
    assert profile_payload["training_config"]["validation_steps"] == 64


def test_resolve_notebook_checkpoint_prefers_training_summary_checkpoint_path(tmp_path: Path) -> None:
    summary_path = tmp_path / "artifacts" / "training_summary.json"
    checkpoint_dir = tmp_path / "artifacts" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "pcc_best.pt"
    checkpoint_path.write_text("checkpoint", encoding="utf-8")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "dataset_id": "allen_visual_behavior_neuropixels",
                "split_name": "valid",
                "epoch": 3,
                "best_epoch": 2,
                "metrics": {"predictive_improvement": 0.12},
                "losses": {"total_loss": 0.2},
                "checkpoint_path": "artifacts/checkpoints/pcc_best.pt",
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_notebook_checkpoint(summary_path=summary_path, checkpoint_dir=checkpoint_dir)
    assert resolved == checkpoint_path


def test_export_notebook_training_artifacts_writes_nested_train_bundle(tmp_path: Path) -> None:
    local_artifact_root = tmp_path / "artifacts"
    checkpoint_dir = local_artifact_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "pcc_best.pt").write_text("checkpoint", encoding="utf-8")
    (local_artifact_root / "training_summary.json").write_text("{}", encoding="utf-8")
    (local_artifact_root / "colab_runtime_experiment.yaml").write_text(
        "dataset_id: allen_visual_behavior_neuropixels\n",
        encoding="utf-8",
    )

    export_path = export_notebook_training_artifacts(
        drive_export_root=tmp_path / "exports",
        local_artifact_root=local_artifact_root,
        run_id="run_20240102_010101",
    )

    assert export_path == tmp_path / "exports" / "run_20240102_010101" / "run_1" / "train"
    assert (export_path / "checkpoints" / "pcc_best.pt").read_text(encoding="utf-8") == "checkpoint"


def test_restore_latest_exported_artifacts_restores_latest_nested_train_run(tmp_path: Path) -> None:
    export_root = tmp_path / "exports"
    old_run = export_root / "run_20240101_010101" / "run_1" / "train"
    new_run = export_root / "run_20240102_010101" / "run_1" / "train"
    for run_dir in (old_run, new_run):
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "pcc_best.pt").write_text(run_dir.parent.parent.name, encoding="utf-8")
        (run_dir / "training_summary.json").write_text("{}", encoding="utf-8")
        (run_dir / "colab_runtime_experiment.yaml").write_text("dataset_id: allen_visual_behavior_neuropixels\n", encoding="utf-8")

    local_artifact_root = tmp_path / "artifacts"
    runtime_experiment_config = tmp_path / "colab_runtime_experiment.yaml"
    restored = restore_latest_exported_artifacts(
        drive_export_root=export_root,
        local_artifact_root=local_artifact_root,
        runtime_experiment_config=runtime_experiment_config,
    )

    assert restored == new_run
    assert (local_artifact_root / "checkpoints" / "pcc_best.pt").read_text(encoding="utf-8") == "run_20240102_010101"
    assert runtime_experiment_config.read_text(encoding="utf-8").startswith("dataset_id:")


def test_restore_latest_exported_artifacts_honors_explicit_run_id(tmp_path: Path) -> None:
    export_root = tmp_path / "exports"
    for run_id in ("run_20240101_010101", "run_20240102_010101"):
        run_dir = export_root / run_id / "run_1" / "train"
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "pcc_best.pt").write_text(run_id, encoding="utf-8")
        (run_dir / "training_summary.json").write_text("{}", encoding="utf-8")
        (run_dir / "colab_runtime_experiment.yaml").write_text("dataset_id: allen_visual_behavior_neuropixels\n", encoding="utf-8")

    local_artifact_root = tmp_path / "artifacts"
    restored = restore_latest_exported_artifacts(
        drive_export_root=export_root,
        local_artifact_root=local_artifact_root,
        training_run_id="run_20240101_010101",
    )

    assert restored == export_root / "run_20240101_010101" / "run_1" / "train"
    assert (local_artifact_root / "checkpoints" / "pcc_best.pt").read_text(encoding="utf-8") == "run_20240101_010101"


def test_restore_latest_exported_artifacts_rejects_unknown_run_id(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run_id 'run_20990101_010101'"):
        restore_latest_exported_artifacts(
            drive_export_root=tmp_path / "exports",
            local_artifact_root=tmp_path / "artifacts",
            training_run_id="run_20990101_010101",
        )


def test_materialize_notebook_prepared_sessions_copies_selected_h5s_and_support_files(tmp_path: Path) -> None:
    source_dataset_root = tmp_path / "drive_dataset"
    source_prepared_root = source_dataset_root / "prepared" / "allen_visual_behavior_neuropixels"
    source_prepared_root.mkdir(parents=True, exist_ok=True)
    for session_id in ("101", "202"):
        (source_prepared_root / f"{session_id}.h5").write_text(session_id, encoding="utf-8")
    manifests_dir = source_dataset_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    (manifests_dir / "session_catalog.json").write_text("{}", encoding="utf-8")
    (manifests_dir / "session_catalog.csv").write_text("session_id\n101\n", encoding="utf-8")
    splits_dir = source_dataset_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "split_manifest.json").write_text("{}", encoding="utf-8")

    target_dataset_root = tmp_path / "repo_dataset"
    result = materialize_notebook_prepared_sessions(
        source_dataset_root=source_dataset_root,
        target_dataset_root=target_dataset_root,
        session_ids=["202", "101"],
        dataset_id="allen_visual_behavior_neuropixels",
    )

    assert result.target_dataset_root == target_dataset_root.resolve()
    assert result.staged_session_ids == ("101", "202")
    assert (result.target_prepared_root / "101.h5").read_text(encoding="utf-8") == "101"
    assert (result.target_prepared_root / "202.h5").read_text(encoding="utf-8") == "202"
    assert (target_dataset_root / "manifests" / "session_catalog.json").is_file()
    assert (target_dataset_root / "splits" / "split_manifest.json").is_file()


def test_materialize_notebook_prepared_sessions_replaces_existing_directory(tmp_path: Path) -> None:
    source_dataset_root = tmp_path / "drive_dataset"
    source_prepared_root = source_dataset_root / "prepared" / "allen_visual_behavior_neuropixels"
    source_prepared_root.mkdir(parents=True, exist_ok=True)
    (source_prepared_root / "101.h5").write_text("101", encoding="utf-8")

    target_dataset_root = tmp_path / "repo_dataset"
    target_dataset_root.mkdir(parents=True, exist_ok=True)
    (target_dataset_root / "stale.txt").write_text("stale", encoding="utf-8")

    result = materialize_notebook_prepared_sessions(
        source_dataset_root=source_dataset_root,
        target_dataset_root=target_dataset_root,
        session_ids=["101"],
        dataset_id="allen_visual_behavior_neuropixels",
    )

    assert target_dataset_root.exists()
    assert not target_dataset_root.is_symlink()
    assert (result.target_prepared_root / "101.h5").is_file()
    assert not (target_dataset_root / "stale.txt").exists()


def test_materialize_notebook_prepared_sessions_replaces_existing_symlink_without_touching_source(
    tmp_path: Path,
) -> None:
    source_dataset_root = tmp_path / "drive_dataset"
    source_prepared_root = source_dataset_root / "prepared" / "allen_visual_behavior_neuropixels"
    source_prepared_root.mkdir(parents=True, exist_ok=True)
    (source_prepared_root / "101.h5").write_text("101", encoding="utf-8")

    target_dataset_root = tmp_path / "repo_dataset"
    try:
        target_dataset_root.symlink_to(source_dataset_root, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable in this environment")

    result = materialize_notebook_prepared_sessions(
        source_dataset_root=source_dataset_root,
        target_dataset_root=target_dataset_root,
        session_ids=["101"],
        dataset_id="allen_visual_behavior_neuropixels",
    )

    assert target_dataset_root.exists()
    assert not target_dataset_root.is_symlink()
    assert (result.target_prepared_root / "101.h5").read_text(encoding="utf-8") == "101"
    assert (source_prepared_root / "101.h5").read_text(encoding="utf-8") == "101"


def test_build_notebook_discovery_runtime_config_only_overrides_decode_settings(tmp_path: Path) -> None:
    source_config = tmp_path / "colab_runtime_experiment.yaml"
    source_config.write_text(
        "\n".join(
            [
                "dataset_id: allen_visual_behavior_neuropixels",
                "training:",
                "  log_every_steps: 8",
                "dataset_selection: {}",
                "runtime_subset:",
                f"  split_manifest_path: {(tmp_path / 'artifacts' / 'runtime_subset' / 'selected_split_manifest.json').resolve()}",
                f"  session_catalog_path: {(tmp_path / 'artifacts' / 'runtime_subset' / 'selected_session_catalog.json').resolve()}",
                f"  config_dir: {(tmp_path / 'artifacts' / 'runtime_subset' / 'splits').resolve()}",
                "  config_name_prefix: torch_brain_runtime",
                "discovery:",
                "  target_label: stimulus_change",
                "  max_batches: 12",
                "  sampling_strategy: sequential",
                "  search_max_batches: 32",
                "  top_k_candidates: 48",
                "  candidate_session_balance_fraction: 0.2",
                "  min_candidate_score: 0.25",
                "  min_cluster_size: 3",
                "  probe_epochs: 33",
                "  probe_learning_rate: 0.02",
                "  shuffle_seed: 29",
                "evaluation:",
                "  max_batches: 9",
                "artifacts:",
                "  checkpoint_dir: artifacts/checkpoints",
                "  summary_path: artifacts/training_summary.json",
            ]
        ),
        encoding="utf-8",
    )

    runtime_config = build_notebook_discovery_runtime_config(
        source_experiment_config=source_config,
        runtime_experiment_config=tmp_path / "colab_discovery_runtime_experiment.yaml",
        artifact_root=tmp_path / "artifacts",
        decode_type="behavior.outcome.hit",
        target_label_match_value="hit",
        step_log_every=16,
    )

    payload = yaml.safe_load(runtime_config.read_text(encoding="utf-8"))
    assert payload["discovery"]["target_label"] == "behavior.outcome.hit"
    assert payload["discovery"]["target_label_match_value"] == "hit"
    assert payload["discovery"]["sampling_strategy"] == "label_balanced"
    assert "search_max_batches" not in payload["discovery"]
    assert payload["discovery"]["max_batches"] == 12
    assert payload["discovery"]["top_k_candidates"] == 48
    assert payload["discovery"]["candidate_session_balance_fraction"] == 0.2
    assert payload["discovery"]["min_candidate_score"] == 0.25
    assert payload["discovery"]["min_cluster_size"] == 3
    assert payload["discovery"]["probe_epochs"] == 33
    assert payload["discovery"]["probe_learning_rate"] == 0.02
    assert payload["discovery"]["shuffle_seed"] == 29
    assert payload["training"]["log_every_steps"] == 16
    assert payload["evaluation"]["max_batches"] == 9
    assert payload["dataset_selection"] == {}
    assert payload["runtime_subset"]["config_name_prefix"] == "torch_brain_runtime"
    assert payload["artifacts"]["checkpoint_dir"] == str((tmp_path / "artifacts" / "checkpoints").resolve())
    assert payload["artifacts"]["summary_path"] == str((tmp_path / "artifacts" / "training_summary.json").resolve())


def test_build_notebook_discovery_runtime_config_applies_requested_overrides(tmp_path: Path) -> None:
    source_config = tmp_path / "colab_runtime_experiment.yaml"
    source_config.write_text(
        "\n".join(
            [
                "dataset_id: allen_visual_behavior_neuropixels",
                "training:",
                "  log_every_steps: 8",
                "dataset_selection: {}",
                "runtime_subset: {}",
                "discovery:",
                "  target_label: stimulus_change",
                "  max_batches: 12",
                "  sampling_strategy: sequential",
                "  top_k_candidates: 48",
                "  candidate_session_balance_fraction: 0.2",
                "  min_candidate_score: 0.25",
                "  min_cluster_size: 3",
                "  probe_epochs: 33",
                "  probe_learning_rate: 0.02",
                "  shuffle_seed: 29",
                "evaluation:",
                "  max_batches: 9",
                "artifacts:",
                "  checkpoint_dir: artifacts/checkpoints",
                "  summary_path: artifacts/training_summary.json",
            ]
        ),
        encoding="utf-8",
    )

    runtime_config = build_notebook_discovery_runtime_config(
        source_experiment_config=source_config,
        runtime_experiment_config=tmp_path / "colab_discovery_runtime_experiment.yaml",
        artifact_root=tmp_path / "artifacts",
        decode_type="trials.is_change",
        target_label_mode="centered_onset",
        target_label_match_value="trial_a",
        step_log_every=32,
        discovery_max_batches=20,
        discovery_top_k_candidates=80,
        discovery_candidate_session_balance_fraction=0.35,
        discovery_min_candidate_score=-0.1,
        discovery_min_cluster_size=5,
        discovery_probe_epochs=60,
        discovery_probe_learning_rate=0.005,
        validation_max_batches=14,
        validation_shuffle_seed=41,
    )

    payload = yaml.safe_load(runtime_config.read_text(encoding="utf-8"))
    assert payload["discovery"]["target_label"] == "trials.is_change"
    assert payload["discovery"]["target_label_mode"] == "centered_onset"
    assert payload["discovery"]["target_label_match_value"] == "trial_a"
    assert payload["discovery"]["max_batches"] == 20
    assert payload["discovery"]["top_k_candidates"] == 80
    assert payload["discovery"]["candidate_session_balance_fraction"] == 0.35
    assert payload["discovery"]["min_candidate_score"] == -0.1
    assert payload["discovery"]["min_cluster_size"] == 5
    assert payload["discovery"]["probe_epochs"] == 60
    assert payload["discovery"]["probe_learning_rate"] == 0.005
    assert payload["discovery"]["shuffle_seed"] == 41
    assert payload["evaluation"]["max_batches"] == 14
    assert payload["training"]["log_every_steps"] == 32


def test_build_notebook_discovery_runtime_config_respects_cpu_device_override(tmp_path: Path) -> None:
    source_config = tmp_path / "colab_runtime_experiment.yaml"
    source_config.write_text(
        "\n".join(
            [
                "dataset_id: allen_visual_behavior_neuropixels",
                "execution:",
                "  device: auto",
                "training:",
                "  log_every_steps: 8",
                "dataset_selection: {}",
                "runtime_subset: {}",
                "discovery:",
                "  target_label: stimulus_change",
                "artifacts:",
                "  checkpoint_dir: artifacts/checkpoints",
                "  summary_path: artifacts/training_summary.json",
            ]
        ),
        encoding="utf-8",
    )

    runtime_config = build_notebook_discovery_runtime_config(
        source_experiment_config=source_config,
        runtime_experiment_config=tmp_path / "colab_discovery_runtime_experiment.yaml",
        artifact_root=tmp_path / "artifacts",
        decode_type="stimulus_change",
        device_mode="cpu",
        step_log_every=16,
    )

    payload = yaml.safe_load(runtime_config.read_text(encoding="utf-8"))
    assert payload["execution"]["device"] == "cpu"


def test_build_notebook_discovery_runtime_config_creates_nested_parent_directories(tmp_path: Path) -> None:
    source_config = tmp_path / "colab_runtime_experiment.yaml"
    source_config.write_text(
        "\n".join(
            [
                "dataset_id: allen_visual_behavior_neuropixels",
                "training:",
                "  log_every_steps: 8",
                "dataset_selection: {}",
                "runtime_subset: {}",
                "discovery:",
                "  target_label: stimulus_change",
                "artifacts:",
                "  checkpoint_dir: artifacts/checkpoints",
                "  summary_path: artifacts/training_summary.json",
            ]
        ),
        encoding="utf-8",
    )

    runtime_config = build_notebook_discovery_runtime_config(
        source_experiment_config=source_config,
        runtime_experiment_config=tmp_path / "artifacts" / "diagnostics" / "baseline_stimulus_change" / "colab_runtime_experiment.yaml",
        artifact_root=tmp_path / "artifacts" / "diagnostics" / "baseline_stimulus_change",
        decode_type="stimulus_change",
        step_log_every=16,
    )

    assert runtime_config.is_file()
    assert (tmp_path / "artifacts" / "diagnostics" / "baseline_stimulus_change").is_dir()


def test_export_notebook_discovery_artifacts_writes_nested_attempt_without_checkpoint(tmp_path: Path) -> None:
    local_artifact_root = tmp_path / "artifacts"
    checkpoint_dir = local_artifact_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "pcc_best.pt").write_text("checkpoint", encoding="utf-8")
    discovery_artifact = checkpoint_dir / "pcc_best_discovery_discovery.json"
    discovery_artifact.write_text(
        json.dumps({"decoder_summary": {"target_label": "stimulus_change"}}),
        encoding="utf-8",
    )
    decode_coverage = checkpoint_dir / "pcc_best_discovery_discovery_decode_coverage.json"
    decode_coverage.write_text("{}", encoding="utf-8")
    cluster_json = checkpoint_dir / "pcc_best_discovery_discovery_cluster_summary.json"
    cluster_json.write_text("{}", encoding="utf-8")
    cluster_csv = checkpoint_dir / "pcc_best_discovery_discovery_cluster_summary.csv"
    cluster_csv.write_text("cluster_id\n0\n", encoding="utf-8")
    validation_json = checkpoint_dir / "pcc_best_discovery_discovery_validation.json"
    validation_json.write_text("{}", encoding="utf-8")
    validation_csv = checkpoint_dir / "pcc_best_discovery_discovery_validation.csv"
    validation_csv.write_text("metric,value\nprobe_accuracy,1.0\n", encoding="utf-8")
    runtime_experiment_config = tmp_path / "colab_discovery_runtime_experiment.yaml"
    runtime_experiment_config.write_text("dataset_id: allen_visual_behavior_neuropixels\n", encoding="utf-8")

    export_path = export_notebook_discovery_artifacts(
        drive_export_root=tmp_path / "exports",
        local_artifact_root=local_artifact_root,
        run_id="run_20240102_010101",
        decode_type="stimulus_presentations.is_change",
        attempt_timestamp="20240102_030405",
        runtime_experiment_config=runtime_experiment_config,
        checkpoint_path=checkpoint_dir / "pcc_best.pt",
        discovery_run=NotebookDiscoveryRunResult(
            discovery_artifact_path=discovery_artifact,
            decode_coverage_summary_path=decode_coverage,
            cluster_summary_json_path=cluster_json,
            cluster_summary_csv_path=cluster_csv,
        ),
        validation_run=NotebookValidationRunResult(
            validation_summary_json_path=validation_json,
            validation_summary_csv_path=validation_csv,
        ),
    )

    expected_path = build_notebook_discovery_export_path(
        drive_export_root=tmp_path / "exports",
        run_id="run_20240102_010101",
        decode_type="stimulus_presentations.is_change",
        attempt_timestamp="20240102_030405",
    )
    assert export_path == expected_path
    assert (export_path / "checkpoints" / discovery_artifact.name).is_file()
    assert (export_path / "checkpoints" / validation_json.name).is_file()
    assert not (export_path / "checkpoints" / "pcc_best.pt").exists()
    metadata_payload = json.loads((export_path / "discovery_export_metadata.json").read_text(encoding="utf-8"))
    assert metadata_payload["training_run_id"] == "run_20240102_010101"
    assert metadata_payload["decode_type"] == "stimulus_presentations.is_change"
    assert (export_path / "colab_discovery_runtime_experiment.yaml").is_file()


def test_restore_latest_discovery_artifacts_uses_selected_run_and_decode_type(tmp_path: Path) -> None:
    export_root = tmp_path / "exports"
    train_dir = export_root / "run_20240102_010101" / "run_1" / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    older_attempt = export_root / "run_20240102_010101" / "run_1" / "discovery" / "stimulus_change__20240102_010101"
    newer_attempt = export_root / "run_20240102_010101" / "run_1" / "discovery" / "stimulus_change__20240102_020202"
    other_task_attempt = export_root / "run_20240102_010101" / "run_1" / "discovery" / "trials.is_change__20240102_030303"
    for attempt_dir, payload in (
        (older_attempt, {"decoder_summary": {"target_label": "stimulus_change"}, "value": "old"}),
        (newer_attempt, {"decoder_summary": {"target_label": "stimulus_change"}, "value": "new"}),
        (other_task_attempt, {"decoder_summary": {"target_label": "trials.is_change"}, "value": "other"}),
    ):
        checkpoint_dir = attempt_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "pcc_best_discovery_discovery.json").write_text(json.dumps(payload), encoding="utf-8")
        (checkpoint_dir / "pcc_best_discovery_discovery_decode_coverage.json").write_text("{}", encoding="utf-8")
        (checkpoint_dir / "pcc_best_discovery_discovery_cluster_summary.json").write_text("{}", encoding="utf-8")
        (checkpoint_dir / "pcc_best_discovery_discovery_cluster_summary.csv").write_text("cluster_id\n0\n", encoding="utf-8")
        (attempt_dir / "discovery_export_metadata.json").write_text("{}", encoding="utf-8")

    local_artifact_root = tmp_path / "artifacts"
    restored = restore_latest_discovery_artifacts(
        drive_export_root=export_root,
        local_artifact_root=local_artifact_root,
        training_run_id="run_20240102_010101",
        decode_type="stimulus_change",
    )

    assert restored == newer_attempt
    restored_payload = json.loads(
        (local_artifact_root / "checkpoints" / "pcc_best_discovery_discovery.json").read_text(encoding="utf-8")
    )
    assert restored_payload["value"] == "new"
    assert not (local_artifact_root / "discovery_export_metadata.json").exists()


def test_build_notebook_diagnostics_experiment_paths_avoids_name_collisions(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "artifacts" / "checkpoints" / "pcc_best.pt"
    first_paths = build_notebook_diagnostics_experiment_paths(
        local_artifact_root=tmp_path / "artifacts",
        checkpoint_path=checkpoint_path,
        experiment_name="baseline_stimulus_change",
    )
    second_paths = build_notebook_diagnostics_experiment_paths(
        local_artifact_root=tmp_path / "artifacts",
        checkpoint_path=checkpoint_path,
        experiment_name="image_identity_im111",
    )

    assert isinstance(first_paths, NotebookDiagnosticsExperimentPaths)
    assert first_paths.discovery_artifact_path != second_paths.discovery_artifact_path
    assert first_paths.runtime_experiment_config_path != second_paths.runtime_experiment_config_path
    assert first_paths.experiment_root.name == "baseline_stimulus_change"
    assert second_paths.experiment_root.name == "image_identity_im111"
    assert first_paths.alignment_summary_json_path.name == "session_alignment_summary.json"
    assert second_paths.alignment_summary_csv_path.name == "session_alignment_summary.csv"


def test_build_notebook_discovery_comparison_paths_groups_three_arms(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "artifacts" / "checkpoints" / "pcc_best.pt"
    paths = build_notebook_discovery_comparison_paths(
        local_artifact_root=tmp_path / "artifacts",
        checkpoint_path=checkpoint_path,
        decode_type="stimulus_change",
    )

    assert isinstance(paths, NotebookDiscoveryComparisonPaths)
    assert paths.comparison_root == build_notebook_discovery_comparison_local_root(
        local_artifact_root=tmp_path / "artifacts",
        decode_type="stimulus_change",
    )
    assert paths.baseline.arm_root.name == "baseline"
    assert paths.whitening_only.arm_root.name == "whitening_only"
    assert paths.whitening_plus_held_out_alignment.arm_root.name == "whitening_plus_held_out_alignment"
    assert paths.comparison_summary_json_path.name == "comparison_summary.json"
    assert paths.decode_coverage_summary_path.name == "decode_coverage_summary.json"


def test_build_notebook_discovery_comparison_summary_row_reads_primary_and_standard_metrics(tmp_path: Path) -> None:
    discovery_artifact_path = tmp_path / "discovery.json"
    discovery_artifact_path.write_text(
        json.dumps({"decoder_summary": {"target_label": "stimulus_change"}}),
        encoding="utf-8",
    )
    validation_summary_path = tmp_path / "validation.json"
    validation_summary_path.write_text(
        json.dumps(
            {
                "candidate_count": 32,
                "cluster_count": 2,
                "reference_session_id": "session_a",
                "excluded_sessions": [{"session_id": "session_x"}],
                "discovery_fit_metrics": {"probe_accuracy": 0.8, "probe_bce": 0.5},
                "shuffled_fit_metrics": {"probe_accuracy": 0.55, "probe_bce": 0.69},
                "primary_held_out_metrics": {
                    "probe_accuracy": 0.7,
                    "probe_bce": 0.6,
                    "probe_roc_auc": 0.72,
                    "probe_pr_auc": 0.73,
                },
                "primary_held_out_similarity_summary": {
                    "window_roc_auc": 0.61,
                    "window_pr_auc": 0.62,
                },
                "cluster_quality_summary": {
                    "cluster_persistence_mean": 0.4,
                    "silhouette_score": 0.2,
                },
                "standard_test_validation": {
                    "held_out_test_metrics": {"probe_accuracy": 0.58, "probe_bce": 0.67},
                    "held_out_similarity_summary": {"window_roc_auc": 0.57, "window_pr_auc": 0.54},
                },
            }
        ),
        encoding="utf-8",
    )
    cluster_summary_path = tmp_path / "cluster_summary.json"
    cluster_summary_path.write_text(json.dumps({"cluster_count": 2}), encoding="utf-8")
    transform_summary_path = tmp_path / "transform_summary.json"
    transform_summary_path.write_text(json.dumps({"transform_type": "whitening_only"}), encoding="utf-8")

    row = build_notebook_discovery_comparison_summary_row(
        arm_name="whitening_only",
        discovery_artifact_path=discovery_artifact_path,
        validation_summary_path=validation_summary_path,
        cluster_summary_path=cluster_summary_path,
        transform_summary_path=transform_summary_path,
    )

    assert row["arm_name"] == "whitening_only"
    assert row["target_label"] == "stimulus_change"
    assert row["candidate_count"] == 32
    assert row["primary_within_session_held_out_probe_accuracy"] == 0.7
    assert row["standard_test_probe_accuracy"] == 0.58
    assert row["transform_type"] == "whitening_only"


def test_build_notebook_alignment_summary_row_reads_alignment_metrics(tmp_path: Path) -> None:
    alignment_summary_path = tmp_path / "alignment_summary.json"
    alignment_summary_path.write_text(
        json.dumps(
            {
                "reference_session_id": "session_a",
                "session_count": 2,
                "sample_count": 8,
                "aggregate_metrics": {
                    "mean_label_axis_cosine_before": 0.1,
                    "mean_label_axis_cosine_after": 0.9,
                    "mean_positive_centroid_cosine_before": 0.2,
                    "mean_positive_centroid_cosine_after": 0.8,
                    "mean_negative_centroid_cosine_before": 0.3,
                    "mean_negative_centroid_cosine_after": 0.7,
                    "mean_anchor_rmse_after_alignment": 0.05,
                },
                "geometry_original": {
                    "metrics": {
                        "label": {"enrichment_over_base": 1.1},
                        "session_id": {"enrichment_over_base": 3.0},
                        "subject_id": {"enrichment_over_base": 3.0},
                    }
                },
                "geometry_whitened": {
                    "metrics": {
                        "label": {"enrichment_over_base": 1.2},
                        "session_id": {"enrichment_over_base": 2.0},
                        "subject_id": {"enrichment_over_base": 2.0},
                    }
                },
                "geometry_aligned": {
                    "metrics": {
                        "label": {"enrichment_over_base": 1.5},
                        "session_id": {"enrichment_over_base": 1.1},
                        "subject_id": {"enrichment_over_base": 1.1},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    row = build_notebook_alignment_summary_row(
        experiment_name="session_alignment",
        alignment_summary_path=alignment_summary_path,
    )

    assert row["experiment_type"] == "session_alignment"
    assert row["reference_session_id"] == "session_a"
    assert row["label_axis_cosine_before"] == 0.1
    assert row["label_axis_cosine_after"] == 0.9
    assert row["aligned_session_neighbor_enrichment"] == 1.1


def test_export_notebook_discovery_comparison_artifacts_uses_grouped_attempt_layout(tmp_path: Path) -> None:
    local_root = build_notebook_discovery_comparison_local_root(
        local_artifact_root=tmp_path / "artifacts",
        decode_type="stimulus_change",
    )
    (local_root / "baseline").mkdir(parents=True, exist_ok=True)
    (local_root / "baseline" / "summary.json").write_text("{}", encoding="utf-8")
    (local_root / "comparison_summary.json").write_text("{}", encoding="utf-8")

    export_path = export_notebook_discovery_comparison_artifacts(
        drive_export_root=tmp_path / "exports",
        local_artifact_root=tmp_path / "artifacts",
        run_id="run_20240102_010101",
        decode_type="stimulus_change",
        attempt_timestamp="20240102_030405",
    )

    expected_path = build_notebook_discovery_export_path(
        drive_export_root=tmp_path / "exports",
        run_id="run_20240102_010101",
        decode_type="stimulus_change",
        attempt_timestamp="20240102_030405",
    )
    assert export_path == expected_path
    assert (export_path / "baseline" / "summary.json").is_file()
    assert (export_path / "comparison_summary.json").is_file()


def test_discovery_notebook_json_parses() -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "discover_validate_inspect_colab.ipynb"
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert isinstance(payload.get("cells"), list)
    assert len(payload["cells"]) >= 10


def test_export_notebook_diagnostics_artifacts_uses_run_timestamp_layout(tmp_path: Path) -> None:
    local_root = tmp_path / "artifacts" / "diagnostics"
    experiment_root = local_root / "baseline_stimulus_change"
    experiment_root.mkdir(parents=True, exist_ok=True)
    (experiment_root / "summary.json").write_text("{}", encoding="utf-8")
    (local_root / "combined_experiment_summary.json").write_text("{}", encoding="utf-8")

    export_path = export_notebook_diagnostics_artifacts(
        drive_export_root=tmp_path / "exports",
        local_artifact_root=tmp_path / "artifacts",
        run_id="run_20240102_010101",
        diagnostics_timestamp="20240102_040506",
    )

    expected_path = build_notebook_diagnostics_export_path(
        drive_export_root=tmp_path / "exports",
        run_id="run_20240102_010101",
        diagnostics_timestamp="20240102_040506",
    )
    assert export_path == expected_path
    assert (export_path / "baseline_stimulus_change" / "summary.json").is_file()
    assert (export_path / "combined_experiment_summary.json").is_file()


def test_find_existing_discovery_run_filters_target_label(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "artifacts" / "checkpoints" / "pcc_best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("checkpoint", encoding="utf-8")
    runtime_config = tmp_path / "colab_discovery_runtime_experiment.yaml"
    runtime_config.write_text(
        "\n".join(
            [
                "dataset_id: allen_visual_behavior_neuropixels",
                "split_name: train",
                "data_runtime: {}",
                "execution: {}",
                "evaluation:",
                "  max_batches: 16",
                "discovery:",
                "  target_label: stimulus_change",
                "  candidate_session_balance_fraction: 0.2",
                "runtime_subset: {}",
                "splits: {}",
            ]
        ),
        encoding="utf-8",
    )
    discovery_artifact = checkpoint_path.with_name("pcc_best_discovery_discovery.json")
    discovery_artifact.write_text(
        json.dumps(
            {
                "decoder_summary": {"target_label": "stimulus_change"},
                "config_snapshot": yaml.safe_load(runtime_config.read_text(encoding="utf-8")),
            }
        ),
        encoding="utf-8",
    )
    discovery_artifact.with_name("pcc_best_discovery_discovery_decode_coverage.json").write_text("{}", encoding="utf-8")
    discovery_artifact.with_name("pcc_best_discovery_discovery_cluster_summary.json").write_text("{}", encoding="utf-8")
    discovery_artifact.with_name("pcc_best_discovery_discovery_cluster_summary.csv").write_text("cluster_id\n0\n", encoding="utf-8")

    assert find_existing_discovery_run(
        checkpoint_path=checkpoint_path,
        target_label="stimulus_change",
        experiment_config_path=runtime_config,
    ) is not None
    assert find_existing_discovery_run(
        checkpoint_path=checkpoint_path,
        target_label="trials.is_change",
        experiment_config_path=runtime_config,
    ) is None


def test_find_existing_discovery_run_rejects_mismatched_runtime_config(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "artifacts" / "checkpoints" / "pcc_best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("checkpoint", encoding="utf-8")
    discovery_artifact = checkpoint_path.with_name("pcc_best_discovery_discovery.json")
    runtime_config = tmp_path / "colab_discovery_runtime_experiment.yaml"
    runtime_config.write_text(
        "\n".join(
            [
                "dataset_id: allen_visual_behavior_neuropixels",
                "split_name: train",
                "data_runtime: {}",
                "execution: {}",
                "evaluation:",
                "  max_batches: 16",
                "discovery:",
                "  target_label: trials.go",
                "  candidate_session_balance_fraction: 0.2",
                "runtime_subset: {}",
                "splits: {}",
            ]
        ),
        encoding="utf-8",
    )
    discovery_artifact.write_text(
        json.dumps(
            {
                "decoder_summary": {"target_label": "trials.go"},
                "config_snapshot": {
                    "dataset_id": "allen_visual_behavior_neuropixels",
                    "split_name": "train",
                    "data_runtime": {},
                    "execution": {},
                    "evaluation": {"max_batches": 16},
                    "discovery": {
                        "target_label": "trials.go",
                        "candidate_session_balance_fraction": 1.0,
                    },
                    "runtime_subset": {},
                    "splits": {},
                },
            }
        ),
        encoding="utf-8",
    )
    discovery_artifact.with_name("pcc_best_discovery_discovery_decode_coverage.json").write_text("{}", encoding="utf-8")
    discovery_artifact.with_name("pcc_best_discovery_discovery_cluster_summary.json").write_text("{}", encoding="utf-8")
    discovery_artifact.with_name("pcc_best_discovery_discovery_cluster_summary.csv").write_text(
        "cluster_id\n0\n",
        encoding="utf-8",
    )

    assert (
        find_existing_discovery_run(
            checkpoint_path=checkpoint_path,
            target_label="trials.go",
            experiment_config_path=runtime_config,
        )
        is None
    )


def test_find_existing_discovery_run_supports_experiment_specific_output_paths(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "artifacts" / "checkpoints" / "pcc_best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("checkpoint", encoding="utf-8")
    experiment_paths = build_notebook_diagnostics_experiment_paths(
        local_artifact_root=tmp_path / "artifacts",
        checkpoint_path=checkpoint_path,
        experiment_name="baseline_stimulus_change",
    )
    experiment_paths.discovery_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_config = experiment_paths.runtime_experiment_config_path
    runtime_config.parent.mkdir(parents=True, exist_ok=True)
    runtime_config.write_text(
        "\n".join(
            [
                "dataset_id: allen_visual_behavior_neuropixels",
                "split_name: train",
                "data_runtime: {}",
                "execution: {}",
                "evaluation:",
                "  max_batches: 16",
                "discovery:",
                "  target_label: stimulus_change",
                "runtime_subset: {}",
                "splits: {}",
            ]
        ),
        encoding="utf-8",
    )
    experiment_paths.discovery_artifact_path.write_text(
        json.dumps(
            {
                "decoder_summary": {"target_label": "stimulus_change"},
                "config_snapshot": yaml.safe_load(runtime_config.read_text(encoding="utf-8")),
            }
        ),
        encoding="utf-8",
    )
    experiment_paths.decode_coverage_summary_path.write_text("{}", encoding="utf-8")
    experiment_paths.cluster_summary_json_path.write_text("{}", encoding="utf-8")
    experiment_paths.cluster_summary_csv_path.write_text("cluster_id\n0\n", encoding="utf-8")

    assert (
        find_existing_discovery_run(
            checkpoint_path=checkpoint_path,
            target_label="stimulus_change",
            experiment_config_path=runtime_config,
            discovery_output_path=experiment_paths.discovery_artifact_path,
        )
        is not None
    )


def test_describe_notebook_compute_targets_reports_cpu_probe_and_metrics(tmp_path: Path) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset_id: allen_visual_behavior_neuropixels",
                "execution:",
                "  device: cpu",
                "training:",
                "  log_every_steps: 8",
                "data_runtime:",
                "  context_bins: 500",
                "  bin_width_ms: 20.0",
                "  context_duration_s: 10.0",
                "splits:",
                "  train: train",
                "  valid: valid",
                "  discovery: discovery",
                "  test: test",
                "model:",
                "  input_dim: 1",
                "  hidden_dim: 8",
                "  num_layers: 1",
                "  num_heads: 1",
                "  dropout: 0.0",
                "objective:",
                "  continuation_baseline_type: previous_patch",
                "optimization:",
                "  batch_size: 4",
                "  learning_rate: 0.001",
                "discovery:",
                "  target_label: stimulus_change",
                "  sampling_strategy: label_balanced",
                "  probe_epochs: 1",
                "  probe_learning_rate: 0.01",
                "  min_cluster_size: 2",
                "  top_k_candidates: 4",
                "  min_candidate_score: 0.0",
                "  max_batches: 4",
                "  shuffle_seed: 7",
                "evaluation:",
                "  max_batches: 4",
                "  sequential_step_s: 1.0",
                "artifacts:",
                "  checkpoint_dir: artifacts/checkpoints",
                "  summary_path: artifacts/training_summary.json",
            ]
        ),
        encoding="utf-8",
    )

    targets = describe_notebook_compute_targets(experiment_config_path=config_path)
    assert targets["encoder_device"] == "cpu"
    assert targets["probe_device"] == "cpu"
    assert targets["clustering_device"] == "cpu"
    assert targets["metrics_device"] == "cpu"


def test_load_notebook_split_counts_reads_runtime_split_manifest(tmp_path: Path) -> None:
    selected_split_manifest = tmp_path / "artifacts" / "runtime_subset" / "selected_split_manifest.json"
    selected_split_manifest.parent.mkdir(parents=True, exist_ok=True)
    selected_split_manifest.write_text(
        json.dumps(
            {
                "dataset_id": "allen_visual_behavior_neuropixels",
                "seed": 7,
                "primary_axis": "session",
                "assignments": [
                    {"recording_id": "allen_visual_behavior_neuropixels/1", "split": "train", "group_id": "1"},
                    {"recording_id": "allen_visual_behavior_neuropixels/2", "split": "train", "group_id": "2"},
                    {"recording_id": "allen_visual_behavior_neuropixels/3", "split": "valid", "group_id": "3"},
                    {"recording_id": "allen_visual_behavior_neuropixels/4", "split": "discovery", "group_id": "4"},
                ],
            }
        ),
        encoding="utf-8",
    )

    split_counts = load_notebook_split_counts(
        split_manifest_path=selected_split_manifest,
    )

    assert split_counts == {"train": 2, "valid": 1, "discovery": 1}


def test_run_streaming_command_captures_output_on_failure(tmp_path: Path) -> None:
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_streaming_command(
            [
                "python",
                "-c",
                "print(\"Cannot fit additive probe because no positive 'stimulus_change' labels were found in the sampled windows.\"); raise SystemExit(1)",
            ],
            cwd=tmp_path,
            step_log_every=16,
        )

    assert "no positive 'stimulus_change' labels" in (excinfo.value.output or "")


def test_output_indicates_missing_positive_labels_is_target_label_agnostic() -> None:
    assert output_indicates_missing_positive_labels(
        "Cannot fit additive probe because no positive 'behavioral_choice.hit' labels were found in the sampled windows."
    )
    assert not output_indicates_missing_positive_labels("Some unrelated subprocess error")
