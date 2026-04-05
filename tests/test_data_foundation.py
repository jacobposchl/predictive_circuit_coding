from __future__ import annotations

import csv
import json
from pathlib import Path

from predictive_circuit_coding.data import (
    REQUIRED_PROVENANCE_FIELDS,
    build_session_manifest_from_table,
    build_split_manifest,
    create_workspace,
    load_preparation_config,
)


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


def test_config_loader_resolves_workspace_relative_to_repo_root(tmp_path: Path):
    config_path = _write_config(tmp_path)
    config = load_preparation_config(config_path)
    assert config.dataset.workspace_root == tmp_path / "data" / "allen_visual_behavior_neuropixels"


def test_create_workspace_builds_expected_directories(tmp_path: Path):
    config = load_preparation_config(_write_config(tmp_path))
    workspace = create_workspace(config)
    assert workspace.raw.is_dir()
    assert workspace.brainset_prepared_root.is_dir()
    assert workspace.manifests.is_dir()
    assert workspace.splits.is_dir()
    assert workspace.logs.is_dir()


def test_create_workspace_skips_in_repo_raw_dir_when_external_cache_root_is_configured(tmp_path: Path):
    config_path = _write_config(tmp_path)
    external_cache_root = tmp_path / "external_cache"
    text = config_path.read_text(encoding="utf-8")
    text += (
        "\nbrainsets_pipeline:\n"
        "  local_pipeline_path: ../../brainsets_local_pipelines/allen_visual_behavior_neuropixels/pipeline.py\n"
        "  runner_cores: 2\n"
        "  use_active_environment: true\n"
        "  processed_only_upload: true\n"
        "  keep_raw_cache: true\n"
        "  default_session_ids_file:\n"
        "  default_max_sessions:\n"
        f"allen_sdk:\n  cache_root: {external_cache_root.as_posix()}\n  cleanup_raw_after_processing: false\n"
        "unit_filtering:\n"
        "  filter_by_validity: true\n"
        "  filter_out_of_brain_units: true\n"
        "  amplitude_cutoff_maximum: 0.1\n"
        "  presence_ratio_minimum: 0.95\n"
        "  isi_violations_maximum: 0.5\n"
    )
    config_path.write_text(text, encoding="utf-8")

    config = load_preparation_config(config_path)
    workspace = create_workspace(config)

    assert config.allen_sdk.cache_root == external_cache_root.resolve()
    assert not workspace.raw.exists()
    assert workspace.brainset_prepared_root.is_dir()


def test_session_manifest_builds_prepared_paths_and_regions(tmp_path: Path):
    config = load_preparation_config(_write_config(tmp_path))
    workspace = create_workspace(config)
    manifest = build_session_manifest_from_table(
        config,
        input_path=_write_session_csv(tmp_path),
        workspace=workspace,
    )
    assert manifest.dataset_id == "allen_visual_behavior_neuropixels"
    assert len(manifest.records) == 4
    first = manifest.records[0]
    assert first.recording_id == "allen_visual_behavior_neuropixels/s1"
    assert first.brain_regions == ("VISp", "LGd")
    assert first.prepared_session_path.endswith(".h5")


def test_split_manifest_uses_subject_grouping_and_covers_all_records(tmp_path: Path):
    config = load_preparation_config(_write_config(tmp_path))
    workspace = create_workspace(config)
    manifest = build_session_manifest_from_table(
        config,
        input_path=_write_session_csv(tmp_path),
        workspace=workspace,
    )
    split_manifest = build_split_manifest(manifest, config=config.splits)
    assert {item.recording_id for item in split_manifest.assignments} == {
        record.recording_id for record in manifest.records
    }
    assert split_manifest.primary_axis == "subject"
    assert any(item.split == "discovery" for item in split_manifest.assignments)
    assert any(item.split == "test" for item in split_manifest.assignments)


def test_required_provenance_fields_cover_window_and_unit_context():
    assert "recording_id" in REQUIRED_PROVENANCE_FIELDS
    assert "unit_id" in REQUIRED_PROVENANCE_FIELDS
    assert "window_start_s" in REQUIRED_PROVENANCE_FIELDS
    assert "patch_index" in REQUIRED_PROVENANCE_FIELDS
