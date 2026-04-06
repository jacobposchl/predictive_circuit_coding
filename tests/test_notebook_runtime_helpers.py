from __future__ import annotations

import json
from pathlib import Path

import yaml

from predictive_circuit_coding.utils import (
    NotebookCommandStreamFormatter,
    NotebookDatasetConfig,
    prepare_notebook_runtime_context,
    resolve_notebook_checkpoint,
)


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
    assert runtime_payload["dataset_selection"]["split_primary_axis"] == "session"
    assert runtime_payload["dataset_selection"]["output_name"] == "familiar_2_session_subset"

    session_ids_file = tmp_path / "artifacts" / "familiar_2_session_ids.txt"
    assert session_ids_file.read_text(encoding="utf-8").splitlines() == ["101", "102"]
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
