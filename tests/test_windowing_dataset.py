from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from predictive_circuit_coding.data import (
    build_session_manifest_from_table,
    build_split_intervals_for_assignment,
    build_split_manifest,
    create_workspace,
    load_preparation_config,
    write_prepared_session,
    write_split_manifest,
)
from predictive_circuit_coding.windowing import (
    FixedWindowConfig,
    build_dataset_bundle,
    build_random_fixed_window_sampler,
    build_sequential_fixed_window_sampler,
    describe_sampler_windows,
    summarize_window_sample,
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
        {"session_id": "s1", "subject_id": "mouse_a", "raw_data_path": "raw/s1.nwb", "duration_s": "1.0", "n_units": "2", "brain_regions": "VISp,LGd", "trial_count": "10"},
        {"session_id": "s2", "subject_id": "mouse_b", "raw_data_path": "raw/s2.nwb", "duration_s": "1.0", "n_units": "2", "brain_regions": "VISl", "trial_count": "10"},
        {"session_id": "s3", "subject_id": "mouse_c", "raw_data_path": "raw/s3.nwb", "duration_s": "1.0", "n_units": "2", "brain_regions": "VISam", "trial_count": "10"},
        {"session_id": "s4", "subject_id": "mouse_d", "raw_data_path": "raw/s4.nwb", "duration_s": "1.0", "n_units": "2", "brain_regions": "LP", "trial_count": "10"},
    ]
    path = tmp_path / "sessions.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _build_workspace_with_sessions(tmp_path: Path):
    config = load_preparation_config(_write_config(tmp_path))
    workspace = create_workspace(config)
    manifest = build_session_manifest_from_table(
        config,
        input_path=_write_session_csv(tmp_path),
        workspace=workspace,
    )
    split_manifest = build_split_manifest(manifest, config=config.splits)
    write_split_manifest(split_manifest, workspace.split_manifest_path)
    split_lookup = {assignment.recording_id: assignment.split for assignment in split_manifest.assignments}
    for index, record in enumerate(manifest.records):
        write_prepared_session(
            path=record.prepared_session_path,
            dataset_id=manifest.dataset_id,
            session_id=record.session_id,
            subject_id=record.subject_id,
            duration_s=record.duration_s,
            spike_timestamps_s=np.asarray([0.05, 0.15, 0.55, 0.95], dtype=np.float64),
            spike_unit_index=np.asarray([0, 1, 0, 1], dtype=np.int64),
            unit_ids=np.asarray([f"u{index}_0", f"u{index}_1"], dtype=object),
            unit_brain_regions=np.asarray([record.brain_regions[0], record.brain_regions[-1]], dtype=object),
            unit_probe_depth_um=np.asarray([100.0 + index, 200.0 + index], dtype=np.float64),
            split_intervals=build_split_intervals_for_assignment(
                domain_start_s=0.0,
                domain_end_s=record.duration_s,
                assigned_split=split_lookup[record.recording_id],
            ),
        )
    return workspace, split_manifest


def test_build_dataset_bundle_and_sample_windows(tmp_path: Path):
    workspace, split_manifest = _build_workspace_with_sessions(tmp_path)
    bundle = build_dataset_bundle(
        workspace=workspace,
        split_manifest=split_manifest,
        split="train",
    )
    assert bundle.config_path.is_file()
    assert len(bundle.dataset.get_session_ids()) >= 1

    sampler = build_sequential_fixed_window_sampler(
        bundle.dataset,
        window=FixedWindowConfig(window_length_s=0.25, step_s=0.25),
    )
    rows = describe_sampler_windows(sampler, limit=10)
    assert len(rows) >= 4

    sample = bundle.dataset.get(rows[0].recording_id, rows[0].start_s, rows[0].end_s)
    metadata = summarize_window_sample(sample)
    assert metadata.recording_id == rows[0].recording_id
    assert metadata.n_units == 2
    assert metadata.start_s == 0.0
    assert metadata.end_s == 0.25

    if hasattr(bundle.dataset, "_close_open_files"):
        bundle.dataset._close_open_files()


def test_random_sampler_uses_real_sampling_intervals(tmp_path: Path):
    workspace, split_manifest = _build_workspace_with_sessions(tmp_path)
    bundle = build_dataset_bundle(
        workspace=workspace,
        split_manifest=split_manifest,
        split="discovery",
    )
    sampler = build_random_fixed_window_sampler(
        bundle.dataset,
        window=FixedWindowConfig(window_length_s=0.25, seed=13),
    )
    rows = describe_sampler_windows(sampler, limit=3)
    assert rows
    assert all(row.end_s > row.start_s for row in rows)

    if hasattr(bundle.dataset, "_close_open_files"):
        bundle.dataset._close_open_files()
