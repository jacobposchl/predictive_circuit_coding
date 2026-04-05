from __future__ import annotations

import subprocess
from pathlib import Path

import h5py
import numpy as np

from predictive_circuit_coding.cli.prepare_data import main
from predictive_circuit_coding.data import (
    build_brainsets_runner_command,
    create_workspace,
    load_session_catalog,
    load_preparation_config,
    load_session_manifest,
    load_split_manifest,
    load_temporaldata_session,
    scan_prepared_session,
    write_temporaldata_session,
)
from predictive_circuit_coding.windowing import (
    FixedWindowConfig,
    build_dataset_bundle,
    build_sequential_fixed_window_sampler,
    describe_sampler_windows,
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
                "brainsets_pipeline:",
                "  local_pipeline_path: ../../brainsets_local_pipelines/allen_visual_behavior_neuropixels/pipeline.py",
                "  runner_cores: 2",
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


def _write_unsplit_session(path: Path, *, dataset_id: str, session_id: str, subject_id: str, regions: list[str]) -> None:
    from temporaldata import ArrayDict, Data, Interval, IrregularTimeSeries

    domain = Interval(
        start=np.asarray([0.0], dtype=np.float64),
        end=np.asarray([1.0], dtype=np.float64),
    )
    data = Data(
        brainset=Data(id=dataset_id),
        session=Data(id=session_id),
        subject=Data(id=subject_id),
        units=ArrayDict(
            id=np.asarray([f"{session_id}_u0", f"{session_id}_u1"], dtype=object),
            brain_region=np.asarray(regions, dtype=object),
            probe_depth_um=np.asarray([100.0, 250.0], dtype=np.float64),
        ),
        spikes=IrregularTimeSeries(
            timestamps=np.asarray([0.05, 0.20, 0.60, 0.90], dtype=np.float64),
            unit_index=np.asarray([0, 1, 0, 1], dtype=np.int64),
            domain=domain,
        ),
        stimulus_presentations=Interval(
            start=np.asarray([0.0, 0.5], dtype=np.float64),
            end=np.asarray([0.25, 0.75], dtype=np.float64),
        ),
        domain=domain,
    )
    write_temporaldata_session(data, path=path)


def test_build_brainsets_runner_command_uses_local_pipeline_and_subset_args(tmp_path: Path):
    config = load_preparation_config(_write_config(tmp_path))
    workspace = create_workspace(config)
    session_ids_file = tmp_path / "session_ids.txt"
    session_ids_file.write_text("12345\n67890\n", encoding="utf-8")

    command = build_brainsets_runner_command(
        config,
        workspace=workspace,
        session_ids_file=session_ids_file,
        max_sessions=3,
    )

    assert command[:3] == [command[0], "-m", "brainsets.runner"]
    assert str(config.brainsets_pipeline.local_pipeline_path) in command
    assert "--session-ids-file" in command
    assert "--max-sessions" in command
    assert "--amplitude-cutoff-maximum" in command
    assert "--presence-ratio-minimum" in command
    assert "--isi-violations-maximum" in command
    assert str(session_ids_file.resolve()) in command


def test_prepare_allen_neuropixels_builds_upload_ready_bundle(monkeypatch, tmp_path: Path):
    config_path = _write_config(tmp_path)
    config = load_preparation_config(config_path)
    workspace = create_workspace(config)

    def fake_run(command, check, text):
        assert check is True
        assert text is True
        processed_root = Path(command[command.index("--processed-dir") + 1]) / config.dataset.dataset_id
        processed_root.mkdir(parents=True, exist_ok=True)
        _write_unsplit_session(
            processed_root / "1001.h5",
            dataset_id=config.dataset.dataset_id,
            session_id="1001",
            subject_id="mouse_a",
            regions=["VISp", "LGd"],
        )
        _write_unsplit_session(
            processed_root / "1002.h5",
            dataset_id=config.dataset.dataset_id,
            session_id="1002",
            subject_id="mouse_b",
            regions=["VISl", "LP"],
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("predictive_circuit_coding.data.brainsets_runner.subprocess.run", fake_run)

    try:
        main(["prepare-allen-visual-behavior-neuropixels", "--config", str(config_path), "--max-sessions", "2"])
    except SystemExit as exc:
        assert exc.code == 0

    session_manifest = load_session_manifest(workspace.session_manifest_path)
    session_catalog = load_session_catalog(workspace.session_catalog_path)
    split_manifest = load_split_manifest(workspace.split_manifest_path)
    upload_manifest_path = workspace.manifests / "upload_bundle.json"

    assert len(session_manifest.records) == 2
    assert len(session_catalog.records) == 2
    assert upload_manifest_path.is_file()
    assert all(record.prepared_session_path for record in session_manifest.records)
    assert {item.split for item in split_manifest.assignments} <= {"train", "valid", "discovery", "test"}

    for split_name in ("train", "valid", "discovery", "test"):
        assert (workspace.splits / f"torch_brain_{split_name}.yaml").is_file()


def test_prepare_allen_neuropixels_builds_catalog_and_split_specific_dataset_configs(monkeypatch, tmp_path: Path):
    config_path = _write_config(tmp_path)
    config = load_preparation_config(config_path)
    workspace = create_workspace(config)

    def fake_run(command, check, text):
        processed_root = Path(command[command.index("--processed-dir") + 1]) / config.dataset.dataset_id
        processed_root.mkdir(parents=True, exist_ok=True)
        for index in range(4):
            _write_unsplit_session(
                processed_root / f"200{index}.h5",
                dataset_id=config.dataset.dataset_id,
                session_id=f"200{index}",
                subject_id=f"mouse_{index}",
                regions=["VISp", "VISl"],
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("predictive_circuit_coding.data.brainsets_runner.subprocess.run", fake_run)

    try:
        main(["prepare-allen-visual-behavior-neuropixels", "--config", str(config_path)])
    except SystemExit as exc:
        assert exc.code == 0

    assert workspace.session_catalog_path.is_file()
    split_manifest = load_split_manifest(workspace.split_manifest_path)
    bundle = build_dataset_bundle(
        workspace=workspace,
        split_manifest=split_manifest,
        split="train",
    )
    sampler = build_sequential_fixed_window_sampler(
        bundle.dataset,
        window=FixedWindowConfig(window_length_s=0.25, step_s=0.25),
    )
    rows = describe_sampler_windows(sampler, limit=4)
    assert rows
    session_id = rows[0].recording_id.split("/", 1)[1]
    session = load_temporaldata_session(workspace.brainset_prepared_root / f"{session_id}.h5", lazy=False)
    assert session.session.id == session_id

    if hasattr(bundle.dataset, "_close_open_files"):
        bundle.dataset._close_open_files()


def test_scan_prepared_session_decodes_byte_region_labels(tmp_path: Path):
    dataset_id = "allen_visual_behavior_neuropixels"
    prepared_root = tmp_path / "prepared" / dataset_id
    prepared_root.mkdir(parents=True)
    session_path = prepared_root / "3001.h5"
    _write_unsplit_session(
        session_path,
        dataset_id=dataset_id,
        session_id="3001",
        subject_id="mouse_bytes",
        regions=[b"VISp", b"LP"],
    )

    scan = scan_prepared_session(
        session_path,
        dataset_id=dataset_id,
        raw_root=tmp_path / "raw",
    )

    assert scan.brain_regions == ("LP", "VISp")


def test_scan_prepared_session_resolves_external_allen_cache_session_path(tmp_path: Path):
    dataset_id = "allen_visual_behavior_neuropixels"
    prepared_root = tmp_path / "prepared" / dataset_id
    prepared_root.mkdir(parents=True)
    session_path = prepared_root / "4001.h5"
    _write_unsplit_session(
        session_path,
        dataset_id=dataset_id,
        session_id="4001",
        subject_id="mouse_cache",
        regions=["VISp", "VISl"],
    )
    cache_root = tmp_path / "allen_cache"
    expected_raw_path = (
        cache_root
        / "visual-behavior-neuropixels-0.5.0"
        / "behavior_ecephys_sessions"
        / "4001"
    )
    expected_raw_path.mkdir(parents=True)

    scan = scan_prepared_session(
        session_path,
        dataset_id=dataset_id,
        raw_root=cache_root,
    )

    assert Path(scan.raw_data_path) == expected_raw_path.resolve()
