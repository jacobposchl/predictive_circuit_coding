from __future__ import annotations

import json
from pathlib import Path

import yaml

from predictive_circuit_coding.utils import (
    NotebookCommandStreamFormatter,
    NotebookDatasetConfig,
    build_notebook_discovery_runtime_config,
    load_notebook_split_counts,
    output_indicates_missing_positive_labels,
    prepare_notebook_runtime_context,
    resolve_notebook_checkpoint,
    restore_latest_exported_artifacts,
    run_streaming_command,
)
import pytest
import subprocess


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
        step_log_every=16,
    )

    runtime_payload = yaml.safe_load(context.experiment_config_path.read_text(encoding="utf-8"))
    assert runtime_payload["training"]["log_every_steps"] == 16
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
    assert profile_payload["selected_session_count"] == 2
    assert profile_payload["selected_sessions_preview"][0]["session_id"] == "101"


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


def test_restore_latest_exported_artifacts_restores_latest_train_run(tmp_path: Path) -> None:
    export_root = tmp_path / "exports"
    old_run = export_root / "train_run_20240101_010101"
    new_run = export_root / "train_run_20240102_010101"
    for run_dir in (old_run, new_run):
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "pcc_best.pt").write_text(run_dir.name, encoding="utf-8")
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
    assert (local_artifact_root / "checkpoints" / "pcc_best.pt").read_text(encoding="utf-8") == new_run.name
    assert runtime_experiment_config.read_text(encoding="utf-8").startswith("dataset_id:")


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
                "  sampling_strategy: sequential",
                "  search_max_batches: 32",
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
        step_log_every=16,
    )

    payload = yaml.safe_load(runtime_config.read_text(encoding="utf-8"))
    assert payload["discovery"]["target_label"] == "behavior.outcome.hit"
    assert payload["discovery"]["sampling_strategy"] == "label_balanced"
    assert "search_max_batches" not in payload["discovery"]
    assert payload["training"]["log_every_steps"] == 16
    assert payload["dataset_selection"] == {}
    assert payload["runtime_subset"]["config_name_prefix"] == "torch_brain_runtime"
    assert payload["artifacts"]["checkpoint_dir"] == str((tmp_path / "artifacts" / "checkpoints").resolve())
    assert payload["artifacts"]["summary_path"] == str((tmp_path / "artifacts" / "training_summary.json").resolve())


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
