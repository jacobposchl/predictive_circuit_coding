from __future__ import annotations

import csv
import json
from pathlib import Path

from predictive_circuit_coding.cli.prepare_data import main


def _write_config(tmp_path: Path) -> Path:
    config_dir = tmp_path / "configs" / "pcc"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "test.yaml"
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
                "  primary_axis: subject",
                "  train_fraction: 0.70",
                "  valid_fraction: 0.10",
                "  discovery_fraction: 0.10",
                "  test_fraction: 0.10",
                "runtime:",
                "  local_cpu_only: true",
                "  training_surface: colab_a100",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _write_session_csv(tmp_path: Path) -> Path:
    rows = [
        {
            "session_id": "s1",
            "subject_id": "mouse_a",
            "raw_data_path": "raw/session_s1.nwb",
            "duration_s": "120.0",
            "n_units": "50",
            "brain_regions": "VISp,LGd",
            "trial_count": "80",
        },
        {
            "session_id": "s2",
            "subject_id": "mouse_b",
            "raw_data_path": "raw/session_s2.nwb",
            "duration_s": "100.0",
            "n_units": "40",
            "brain_regions": "VISl",
            "trial_count": "70",
        },
        {
            "session_id": "s3",
            "subject_id": "mouse_c",
            "raw_data_path": "raw/session_s3.nwb",
            "duration_s": "110.0",
            "n_units": "60",
            "brain_regions": "VISp;LP",
            "trial_count": "60",
        },
        {
            "session_id": "s4",
            "subject_id": "mouse_d",
            "raw_data_path": "raw/session_s4.nwb",
            "duration_s": "130.0",
            "n_units": "55",
            "brain_regions": "VISam",
            "trial_count": "90",
        },
    ]
    path = tmp_path / "sessions.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_prepare_cli_creates_workspace_and_manifests(tmp_path: Path):
    config_path = _write_config(tmp_path)
    csv_path = _write_session_csv(tmp_path)

    try:
        main(["init-workspace", "--config", str(config_path)])
    except SystemExit as exc:
        assert exc.code == 0

    try:
        main(["build-session-manifest", "--config", str(config_path), "--input", str(csv_path)])
    except SystemExit as exc:
        assert exc.code == 0

    try:
        main(["plan-splits", "--config", str(config_path)])
    except SystemExit as exc:
        assert exc.code == 0

    workspace_root = tmp_path / "data" / "allen_visual_behavior_neuropixels"
    session_manifest = workspace_root / "manifests" / "session_manifest.json"
    split_manifest = workspace_root / "splits" / "split_manifest.json"
    assert session_manifest.is_file()
    assert split_manifest.is_file()

    with session_manifest.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["dataset_id"] == "allen_visual_behavior_neuropixels"
    assert len(payload["records"]) == 4


def test_prepare_cli_can_build_split_specific_dataset_config(tmp_path: Path):
    config_path = _write_config(tmp_path)
    csv_path = _write_session_csv(tmp_path)

    for args in (
        ["init-workspace", "--config", str(config_path)],
        ["build-session-manifest", "--config", str(config_path), "--input", str(csv_path)],
        ["plan-splits", "--config", str(config_path)],
        ["build-dataset-config", "--config", str(config_path), "--split", "train"],
    ):
        try:
            main(args)
        except SystemExit as exc:
            assert exc.code == 0

    dataset_config = tmp_path / "data" / "allen_visual_behavior_neuropixels" / "splits" / "torch_brain_train.yaml"
    assert dataset_config.is_file()
