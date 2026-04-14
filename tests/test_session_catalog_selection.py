from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from predictive_circuit_coding.benchmarks.pipeline import _ensure_local_prepared_sessions
from predictive_circuit_coding.cli.prepare_data import main as prepare_main
from predictive_circuit_coding.cli.train import main as train_main
from predictive_circuit_coding.data import (
    build_session_catalog_from_prepared_sessions,
    build_split_manifest,
    create_workspace,
    load_preparation_config,
    load_session_catalog,
    project_catalog_to_session_manifest,
    resolve_runtime_dataset_view,
    write_session_catalog,
    write_split_manifest,
    write_temporaldata_session,
)


def _write_prep_config(tmp_path: Path, *, cache_root: Path | None = None) -> Path:
    config_dir = tmp_path / "configs" / "pcc"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "prep.yaml"
    lines = [
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
        "  train_fraction: 0.25",
        "  valid_fraction: 0.25",
        "  discovery_fraction: 0.25",
        "  test_fraction: 0.25",
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
        f"  cache_root: {cache_root.as_posix()}" if cache_root is not None else "  cache_root:",
        "  cleanup_raw_after_processing: false",
        "unit_filtering:",
        "  filter_by_validity: true",
        "  filter_out_of_brain_units: true",
        "  amplitude_cutoff_maximum: 0.1",
        "  presence_ratio_minimum: 0.95",
        "  isi_violations_maximum: 0.5",
    ]
    config_path.write_text("\n".join(lines), encoding="utf-8")
    return config_path


def _write_experiment_config(
    tmp_path: Path,
    *,
    dataset_selection_lines: list[str] | None = None,
) -> Path:
    config_dir = tmp_path / "configs" / "pcc"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "experiment.yaml"
    lines = [
        "dataset_id: allen_visual_behavior_neuropixels",
        "split_name: train",
        "seed: 13",
        "splits:",
        "  train: train",
        "  valid: valid",
        "  discovery: discovery",
        "  test: test",
        "dataset_selection:",
        "  output_name: runtime_selection",
    ]
    lines.extend(dataset_selection_lines or [])
    lines.extend(
        [
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
            "model:",
            "  d_model: 16",
            "  num_heads: 4",
            "  temporal_layers: 1",
            "  spatial_layers: 1",
            "  dropout: 0.0",
            "  mlp_ratio: 2.0",
            "  l2_normalize_tokens: true",
            "  norm_eps: 1.0e-5",
            "objective:",
            "  predictive_target_type: delta",
            "  continuation_baseline_type: previous_patch",
            "  predictive_loss: mse",
            "  reconstruction_loss: mse",
            "  reconstruction_weight: 0.1",
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
            "  log_every_steps: 1",
            "execution:",
            "  device: cpu",
            "  mixed_precision: false",
            "evaluation:",
            "  max_batches: 1",
            "  sequential_step_s: 2.0",
            "discovery:",
            "  target_label: stimulus_change",
            "  max_batches: 1",
            "  probe_epochs: 5",
            "  probe_learning_rate: 0.05",
            "  top_k_candidates: 8",
            "  min_candidate_score: -100.0",
            "  min_cluster_size: 2",
            "  stability_rounds: 2",
            "  shuffle_seed: 19",
            "artifacts:",
            "  checkpoint_dir: ../../artifacts/checkpoints",
            "  summary_path: ../../artifacts/training_summary.json",
            "  checkpoint_prefix: pcc_test",
            "  save_config_snapshot: true",
        ]
    )
    config_path.write_text("\n".join(lines), encoding="utf-8")
    return config_path


def _write_session(path: Path, *, dataset_id: str, session_id: str, subject_id: str, regions: list[str]) -> None:
    from temporaldata import ArrayDict, Data, Interval, IrregularTimeSeries

    domain = Interval(start=np.asarray([0.0], dtype=np.float64), end=np.asarray([4.0], dtype=np.float64))
    data = Data(
        brainset=Data(id=dataset_id),
        session=Data(id=session_id),
        subject=Data(id=subject_id),
        units=ArrayDict(
            id=np.asarray([f"{session_id}_u0", f"{session_id}_u1"], dtype=object),
            brain_region=np.asarray(regions, dtype=object),
            probe_depth_um=np.asarray([100.0, 240.0], dtype=np.float64),
        ),
        spikes=IrregularTimeSeries(
            timestamps=np.asarray([0.10, 0.30, 0.60, 1.20, 2.10, 2.30, 2.60, 3.20], dtype=np.float64),
            unit_index=np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64),
            domain=domain,
        ),
        trials=Interval(
            start=np.asarray([0.0, 2.0], dtype=np.float64),
            end=np.asarray([1.0, 3.0], dtype=np.float64),
            go=np.asarray([False, True], dtype=bool),
            hit=np.asarray([False, True], dtype=bool),
        ),
        stimulus_presentations=Interval(
            start=np.asarray([0.20, 1.20], dtype=np.float64),
            end=np.asarray([0.40, 1.40], dtype=np.float64),
            stimulus_name=np.asarray(["images", "images"], dtype=object),
            image_name=np.asarray(["im0", "im1"], dtype=object),
            is_change=np.asarray([False, True], dtype=bool),
        ),
        domain=domain,
    )
    write_temporaldata_session(data, path=path)


def _write_ecephys_metadata(cache_root: Path, rows: list[dict[str, object]]) -> None:
    metadata_dir = cache_root / "visual-behavior-neuropixels-0.5.0" / "project_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ecephys_session_id",
        "behavior_session_id",
        "date_of_acquisition",
        "equipment_name",
        "session_type",
        "mouse_id",
        "genotype",
        "sex",
        "project_code",
        "age_in_days",
        "unit_count",
        "probe_count",
        "channel_count",
        "structure_acronyms",
        "image_set",
        "prior_exposures_to_image_set",
        "session_number",
        "experience_level",
        "prior_exposures_to_omissions",
        "file_id",
        "abnormal_histology",
        "abnormal_activity",
    ]
    with (metadata_dir / "ecephys_sessions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_build_session_catalog_from_prepared_sessions_merges_allen_session_metadata(tmp_path: Path):
    cache_root = tmp_path / "external_cache"
    prep_config_path = _write_prep_config(tmp_path, cache_root=cache_root)
    prep_config = load_preparation_config(prep_config_path)
    workspace = create_workspace(prep_config)
    dataset_id = prep_config.dataset.dataset_id

    _write_ecephys_metadata(
        cache_root,
        [
            {
                "ecephys_session_id": "1001",
                "behavior_session_id": "2001",
                "date_of_acquisition": "2020-08-19 14:47:08.574000+00:00",
                "equipment_name": "NP.1",
                "session_type": "EPHYS_1_images_G_5uL_reward",
                "mouse_id": "524761",
                "genotype": "wt/wt",
                "sex": "F",
                "project_code": "NeuropixelVisualBehavior",
                "age_in_days": "151",
                "unit_count": "2179",
                "probe_count": "5",
                "channel_count": "1920",
                "structure_acronyms": "['VISp', 'LP']",
                "image_set": "G",
                "prior_exposures_to_image_set": "30",
                "session_number": "1",
                "experience_level": "Familiar",
                "prior_exposures_to_omissions": "0",
                "file_id": "870",
                "abnormal_histology": "",
                "abnormal_activity": "",
            }
        ],
    )
    (cache_root / "visual-behavior-neuropixels-0.5.0" / "behavior_ecephys_sessions" / "1001").mkdir(parents=True)
    _write_session(
        workspace.brainset_prepared_root / "1001.h5",
        dataset_id=dataset_id,
        session_id="1001",
        subject_id="524761",
        regions=["VISp", "LP"],
    )

    catalog = build_session_catalog_from_prepared_sessions(prep_config, workspace=workspace)

    assert len(catalog.records) == 1
    record = catalog.records[0]
    assert record.session_type == "EPHYS_1_images_G_5uL_reward"
    assert record.image_set == "G"
    assert record.experience_level == "Familiar"
    assert record.project_code == "NeuropixelVisualBehavior"
    assert record.session_number == 1
    assert record.allen_unit_count == 2179
    assert record.probe_count == 5
    assert record.channel_count == 1920
    assert record.prior_exposures_to_image_set == 30.0
    assert Path(record.raw_data_path).name == "1001"


def test_materialize_runtime_selection_filters_metadata_and_writes_selected_artifacts(tmp_path: Path):
    cache_root = tmp_path / "external_cache"
    prep_config_path = _write_prep_config(tmp_path, cache_root=cache_root)
    prep_config = load_preparation_config(prep_config_path)
    workspace = create_workspace(prep_config)
    dataset_id = prep_config.dataset.dataset_id

    metadata_rows = []
    session_specs = [
        ("1001", "mouse_a", "Familiar", "G"),
        ("1002", "mouse_b", "Familiar", "G"),
        ("1003", "mouse_c", "Familiar", "G"),
        ("1004", "mouse_d", "Familiar", "G"),
        ("1005", "mouse_e", "Novel", "H"),
    ]
    for index, (session_id, subject_id, experience_level, image_set) in enumerate(session_specs, start=1):
        _write_session(
            workspace.brainset_prepared_root / f"{session_id}.h5",
            dataset_id=dataset_id,
            session_id=session_id,
            subject_id=subject_id,
            regions=["VISp", "LP"] if index % 2 else ["VISl", "LP"],
        )
        (cache_root / "visual-behavior-neuropixels-0.5.0" / "behavior_ecephys_sessions" / session_id).mkdir(parents=True, exist_ok=True)
        metadata_rows.append(
            {
                "ecephys_session_id": session_id,
                "behavior_session_id": str(2000 + index),
                "date_of_acquisition": f"2020-08-{18+index:02d} 14:47:08.574000+00:00",
                "equipment_name": "NP.1",
                "session_type": f"EPHYS_1_images_{image_set}_5uL_reward",
                "mouse_id": subject_id,
                "genotype": "wt/wt",
                "sex": "F",
                "project_code": "NeuropixelVisualBehavior",
                "age_in_days": "151",
                "unit_count": str(1000 + index),
                "probe_count": "5",
                "channel_count": "1920",
                "structure_acronyms": "['VISp', 'LP']",
                "image_set": image_set,
                "prior_exposures_to_image_set": str(index),
                "session_number": str(index),
                "experience_level": experience_level,
                "prior_exposures_to_omissions": "0",
                "file_id": str(800 + index),
                "abnormal_histology": "",
                "abnormal_activity": "",
            }
        )
    _write_ecephys_metadata(cache_root, metadata_rows)

    try:
        prepare_main(["build-session-catalog", "--config", str(prep_config_path)])
    except SystemExit as exc:
        assert exc.code == 0

    experiment_config_path = _write_experiment_config(
        tmp_path,
        dataset_selection_lines=[
            "  experience_levels: [Familiar]",
            "  image_sets: [G]",
        ],
    )
    with experiment_config_path.open("r", encoding="utf-8") as handle:
        assert "experience_levels" in handle.read()

    from predictive_circuit_coding.training import load_experiment_config

    experiment_config = load_experiment_config(experiment_config_path)
    view = resolve_runtime_dataset_view(
        experiment_config=experiment_config,
        data_config_path=prep_config_path,
    )

    assert view.selection_active is True
    assert view.selection_summary is not None
    assert view.selection_summary.session_count == 4
    assert Path(view.session_catalog_path).is_file()
    assert Path(view.split_manifest_path).is_file()
    assert (view.config_dir / "torch_brain_selected_train.yaml").is_file()
    assert (view.config_dir / "torch_brain_selected_valid.yaml").is_file()
    assert all(record.experience_level == "Familiar" for record in view.session_catalog.records)
    assert all(record.image_set == "G" for record in view.session_catalog.records)
    assert "1005" not in {record.session_id for record in view.session_catalog.records}


def test_train_cli_records_runtime_selection_sidecar_inputs(tmp_path: Path):
    prep_config_path = _write_prep_config(tmp_path)
    prep_config = load_preparation_config(prep_config_path)
    workspace = create_workspace(prep_config)
    dataset_id = prep_config.dataset.dataset_id
    for session_id, subject_id in (
        ("session_train", "mouse_train"),
        ("session_valid", "mouse_valid"),
        ("session_discovery", "mouse_discovery"),
        ("session_test", "mouse_test"),
    ):
        _write_session(
            workspace.brainset_prepared_root / f"{session_id}.h5",
            dataset_id=dataset_id,
            session_id=session_id,
            subject_id=subject_id,
            regions=["VISp", "LP"],
        )
    catalog = build_session_catalog_from_prepared_sessions(prep_config, workspace=workspace)
    write_session_catalog(catalog, workspace.session_catalog_path)
    manifest = project_catalog_to_session_manifest(catalog)
    write_split_manifest(build_split_manifest(manifest, config=prep_config.splits), workspace.split_manifest_path)

    experiment_config_path = _write_experiment_config(
        tmp_path,
        dataset_selection_lines=[
            "  session_ids: [session_train, session_valid, session_discovery, session_test]",
        ],
    )

    try:
        train_main(["--config", str(experiment_config_path), "--data-config", str(prep_config_path)])
    except SystemExit as exc:
        assert exc.code == 0

    sidecar_path = tmp_path / "artifacts" / "training_summary_train_run_manifest.json"
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["inputs"]["dataset_selection_active"] is True
    assert payload["inputs"]["runtime_split_manifest_path"].endswith("selected_split_manifest.json")


def test_resolve_runtime_dataset_view_rebuilds_support_files_from_prepared_sessions(tmp_path: Path):
    prep_config_path = _write_prep_config(tmp_path)
    prep_config = load_preparation_config(prep_config_path)
    workspace = create_workspace(prep_config)
    dataset_id = prep_config.dataset.dataset_id
    for session_id, subject_id in (
        ("session_train", "mouse_train"),
        ("session_valid", "mouse_valid"),
        ("session_discovery", "mouse_discovery"),
        ("session_test", "mouse_test"),
    ):
        _write_session(
            workspace.brainset_prepared_root / f"{session_id}.h5",
            dataset_id=dataset_id,
            session_id=session_id,
            subject_id=subject_id,
            regions=["VISp", "LP"],
        )

    experiment_config_path = _write_experiment_config(tmp_path)

    from predictive_circuit_coding.training import load_experiment_config

    experiment_config = load_experiment_config(experiment_config_path)
    view = resolve_runtime_dataset_view(
        experiment_config=experiment_config,
        data_config_path=prep_config_path,
    )

    assert Path(view.session_catalog_path).is_file()
    assert Path(view.split_manifest_path).is_file()
    assert workspace.session_manifest_path.is_file()
    assert len(view.session_catalog.records) == 4
    assert {record.session_id for record in view.session_catalog.records} == {
        "session_train",
        "session_valid",
        "session_discovery",
        "session_test",
    }


def test_ensure_local_prepared_sessions_stages_from_source_root(tmp_path: Path):
    prep_config_path = _write_prep_config(tmp_path)
    prep_config = load_preparation_config(prep_config_path)
    workspace = create_workspace(prep_config)
    dataset_id = prep_config.dataset.dataset_id
    source_root = tmp_path / "source_dataset"
    source_prepared_root = source_root / "prepared" / dataset_id
    source_prepared_root.mkdir(parents=True, exist_ok=True)

    for session_id, subject_id in (
        ("session_train", "mouse_train"),
        ("session_valid", "mouse_valid"),
    ):
        _write_session(
            source_prepared_root / f"{session_id}.h5",
            dataset_id=dataset_id,
            session_id=session_id,
            subject_id=subject_id,
            regions=["VISp", "LP"],
        )

    _ensure_local_prepared_sessions(
        data_config_path=prep_config_path,
        source_dataset_root=source_root,
        stage_prepared_sessions_locally=True,
    )

    assert sorted(path.stem for path in workspace.brainset_prepared_root.glob("*.h5")) == [
        "session_train",
        "session_valid",
    ]

