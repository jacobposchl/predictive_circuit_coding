from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from predictive_circuit_coding.cli.discover import main as discover_main
from predictive_circuit_coding.cli.evaluate import main as evaluate_main
from predictive_circuit_coding.cli.train import main as train_main
from predictive_circuit_coding.cli.validate import main as validate_main
from predictive_circuit_coding.data import (
    SessionManifest,
    SessionRecord,
    SplitAssignment,
    SplitManifest,
    build_split_intervals_for_assignment,
    create_workspace,
    load_preparation_config,
    write_session_manifest,
    write_split_manifest,
    write_temporaldata_session,
)
from predictive_circuit_coding.decoding import fit_additive_probe
from predictive_circuit_coding.decoding.extract import (
    DiscoveryWindowPlan,
    DiscoveryWindowPlanRecord,
    EncodedDiscoverySelection,
)
from predictive_circuit_coding.discovery import run_representation_comparison_from_encoded
from predictive_circuit_coding.training import load_experiment_config, train_model
from predictive_circuit_coding.training.contracts import DiscoveryCoverageSummary, EvaluationSummary
from predictive_circuit_coding.utils import collect_notebook_target_value_counts
from predictive_circuit_coding.utils import inspect_notebook_target_field_availability


def _write_preparation_config(tmp_path: Path) -> Path:
    config_dir = tmp_path / "configs" / "pcc"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "prep.yaml"
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


def _write_experiment_config(
    tmp_path: Path,
    *,
    discovery_sampling_strategy: str = "sequential",
    discovery_target_label: str = "stimulus_change",
    discovery_target_label_mode: str = "auto",
    discovery_target_label_match_value: str | None = None,
) -> Path:
    config_dir = tmp_path / "configs" / "pcc"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "experiment.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset_id: allen_visual_behavior_neuropixels",
                "split_name: train",
                "seed: 13",
                "splits:",
                "  train: train",
                "  valid: valid",
                "  discovery: discovery",
                "  test: test",
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
                "  num_epochs: 2",
                "  train_steps_per_epoch: 2",
                "  validation_steps: 2",
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
                "  max_batches: 2",
                "  sequential_step_s: 2.0",
                "discovery:",
                f"  target_label: {discovery_target_label}",
                f"  target_label_mode: {discovery_target_label_mode}",
                f"  target_label_match_value: {'' if discovery_target_label_match_value is None else discovery_target_label_match_value}",
                "  max_batches: 2",
                "  probe_epochs: 20",
                "  probe_learning_rate: 0.05",
                "  top_k_candidates: 8",
                "  min_candidate_score: -100.0",
                "  min_cluster_size: 2",
                "  stability_rounds: 3",
                "  shuffle_seed: 19",
                f"  sampling_strategy: {discovery_sampling_strategy}",
                "artifacts:",
                "  checkpoint_dir: ../../artifacts/checkpoints",
                "  summary_path: ../../artifacts/training_summary.json",
                "  checkpoint_prefix: pcc_test",
                "  save_config_snapshot: true",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _write_synthetic_session(
    *,
    path: Path,
    dataset_id: str,
    session_id: str,
    subject_id: str,
    assigned_split: str,
) -> None:
    from temporaldata import ArrayDict, Data, Interval, IrregularTimeSeries

    duration_s = 4.0
    split_intervals = build_split_intervals_for_assignment(
        domain_start_s=0.0,
        domain_end_s=duration_s,
        assigned_split=assigned_split,
    )
    domain = Interval(
        start=np.asarray([0.0], dtype=np.float64),
        end=np.asarray([duration_s], dtype=np.float64),
    )
    spikes = IrregularTimeSeries(
        timestamps=np.asarray(
            [
                0.10,
                0.30,
                0.60,
                1.20,
                2.10,
                2.30,
                2.60,
                3.20,
                3.40,
                3.70,
            ],
            dtype=np.float64,
        ),
        unit_index=np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64),
        domain=domain,
    )
    data = Data(
        brainset=Data(id=dataset_id),
        session=Data(id=session_id),
        subject=Data(id=subject_id),
        units=ArrayDict(
            id=np.asarray([f"{session_id}_u0", f"{session_id}_u1"], dtype=object),
            brain_region=np.asarray(["VISp", "LP"], dtype=object),
            probe_depth_um=np.asarray([100.0, 220.0], dtype=np.float32),
        ),
        spikes=spikes,
        trials=Interval(
            start=np.asarray([0.0, 2.0], dtype=np.float64),
            end=np.asarray([1.0, 3.0], dtype=np.float64),
            go=np.asarray([False, True], dtype=bool),
            hit=np.asarray([False, True], dtype=bool),
        ),
        stimulus_presentations=Interval(
            start=np.asarray([0.20, 0.90, 2.20, 2.90], dtype=np.float64),
            end=np.asarray([0.40, 1.10, 2.40, 3.10], dtype=np.float64),
            stimulus_name=np.asarray(["images", "images", "images", "images"], dtype=object),
            image_name=np.asarray(["im0", "im1", "im2", "im3"], dtype=object),
            is_change=np.asarray([False, False, True, True], dtype=bool),
        ),
        domain=domain,
        train_domain=Interval(
            start=np.asarray([start for start, _ in split_intervals.train], dtype=np.float64),
            end=np.asarray([end for _, end in split_intervals.train], dtype=np.float64),
        ),
        valid_domain=Interval(
            start=np.asarray([start for start, _ in split_intervals.valid], dtype=np.float64),
            end=np.asarray([end for _, end in split_intervals.valid], dtype=np.float64),
        ),
        discovery_domain=Interval(
            start=np.asarray([start for start, _ in split_intervals.discovery], dtype=np.float64),
            end=np.asarray([end for _, end in split_intervals.discovery], dtype=np.float64),
        ),
        test_domain=Interval(
            start=np.asarray([start for start, _ in split_intervals.test], dtype=np.float64),
            end=np.asarray([end for _, end in split_intervals.test], dtype=np.float64),
        ),
    )
    data.add_split_mask("train", data.train_domain)
    data.add_split_mask("valid", data.valid_domain)
    data.add_split_mask("discovery", data.discovery_domain)
    data.add_split_mask("test", data.test_domain)
    write_temporaldata_session(data, path=path)


def _create_prepared_workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    prep_config_path = _write_preparation_config(tmp_path)
    prep_config = load_preparation_config(prep_config_path)
    workspace = create_workspace(prep_config)
    dataset_id = prep_config.dataset.dataset_id

    assignments = (
        ("session_train", "mouse_train", "train"),
        ("session_valid", "mouse_valid", "valid"),
        ("session_discovery", "mouse_discovery", "discovery"),
        ("session_test", "mouse_test", "test"),
    )
    records: list[SessionRecord] = []
    split_rows: list[SplitAssignment] = []
    for session_id, subject_id, split_name in assignments:
        session_path = workspace.brainset_prepared_root / f"{session_id}.h5"
        _write_synthetic_session(
            path=session_path,
            dataset_id=dataset_id,
            session_id=session_id,
            subject_id=subject_id,
            assigned_split=split_name,
        )
        records.append(
            SessionRecord(
                recording_id=f"{dataset_id}/{session_id}",
                session_id=session_id,
                subject_id=subject_id,
                raw_data_path=f"raw/{session_id}.nwb",
                duration_s=4.0,
                n_units=2,
                brain_regions=("LP", "VISp"),
                trial_count=2,
                prepared_session_path=str(session_path),
            )
        )
        split_rows.append(
            SplitAssignment(
                recording_id=f"{dataset_id}/{session_id}",
                split=split_name,
                group_id=subject_id,
            )
        )

    write_session_manifest(
        SessionManifest(
            dataset_id=dataset_id,
            source_name=dataset_id,
            records=tuple(records),
        ),
        workspace.session_manifest_path,
    )
    write_split_manifest(
        SplitManifest(
            dataset_id=dataset_id,
            seed=7,
            primary_axis="subject",
            assignments=tuple(split_rows),
        ),
        workspace.split_manifest_path,
    )
    experiment_config_path = _write_experiment_config(tmp_path)
    return prep_config_path, experiment_config_path, workspace.root


def test_additive_probe_learns_synthetic_token_label_structure():
    generator = torch.Generator().manual_seed(3)
    tokens = torch.randn((8, 3, 4), generator=generator)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32)
    tokens[labels == 1, :, 0] += 3.0
    token_mask = torch.ones((8, 3), dtype=torch.bool)

    fit = fit_additive_probe(
        tokens=tokens,
        token_mask=token_mask,
        labels=labels,
        epochs=200,
        learning_rate=0.05,
    )

    assert fit.metrics["probe_accuracy"] >= 0.95


def test_stage_5_and_6_cli_workflow_runs_end_to_end(tmp_path: Path):
    prep_config_path, experiment_config_path, workspace_root = _create_prepared_workspace(tmp_path)
    checkpoint_path = tmp_path / "artifacts" / "checkpoints" / "pcc_test_best.pt"
    evaluation_path = tmp_path / "artifacts" / "checkpoints" / "pcc_test_best_test_evaluation.json"
    discovery_path = tmp_path / "artifacts" / "checkpoints" / "pcc_test_best_discovery_discovery.json"
    discovery_coverage_path = tmp_path / "artifacts" / "checkpoints" / "pcc_test_best_discovery_discovery_decode_coverage.json"
    cluster_summary_json_path = tmp_path / "artifacts" / "checkpoints" / "pcc_test_best_discovery_discovery_cluster_summary.json"
    cluster_summary_csv_path = tmp_path / "artifacts" / "checkpoints" / "pcc_test_best_discovery_discovery_cluster_summary.csv"
    validation_json_path = tmp_path / "artifacts" / "checkpoints" / "pcc_test_best_discovery_discovery_validation.json"
    validation_csv_path = tmp_path / "artifacts" / "checkpoints" / "pcc_test_best_discovery_discovery_validation.csv"
    training_sidecar_path = tmp_path / "artifacts" / "training_summary_train_run_manifest.json"
    evaluation_sidecar_path = tmp_path / "artifacts" / "checkpoints" / "pcc_test_best_test_evaluation_evaluate_run_manifest.json"
    discovery_sidecar_path = tmp_path / "artifacts" / "checkpoints" / "pcc_test_best_discovery_discovery_discover_run_manifest.json"
    validation_sidecar_path = tmp_path / "artifacts" / "checkpoints" / "pcc_test_best_discovery_discovery_validation_validate_run_manifest.json"

    for command in (
        [train_main, ["--config", str(experiment_config_path), "--data-config", str(prep_config_path)]],
        [
            evaluate_main,
            [
                "--config",
                str(experiment_config_path),
                "--data-config",
                str(prep_config_path),
                "--checkpoint",
                str(checkpoint_path),
                "--split",
                "test",
                "--output",
                str(evaluation_path),
            ],
        ],
        [
            discover_main,
            [
                "--config",
                str(experiment_config_path),
                "--data-config",
                str(prep_config_path),
                "--checkpoint",
                str(checkpoint_path),
                "--split",
                "discovery",
                "--output",
                str(discovery_path),
            ],
        ],
        [
            validate_main,
            [
                "--config",
                str(experiment_config_path),
                "--data-config",
                str(prep_config_path),
                "--checkpoint",
                str(checkpoint_path),
                "--discovery-artifact",
                str(discovery_path),
                "--output-json",
                str(validation_json_path),
                "--output-csv",
                str(validation_csv_path),
            ],
        ],
    ):
        entrypoint, argv = command
        try:
            entrypoint(argv)
        except SystemExit as exc:
            assert exc.code == 0

    training_summary_path = tmp_path / "artifacts" / "training_summary.json"
    assert checkpoint_path.is_file()
    assert training_summary_path.is_file()
    assert evaluation_path.is_file()
    assert discovery_path.is_file()
    assert discovery_coverage_path.is_file()
    assert cluster_summary_json_path.is_file()
    assert cluster_summary_csv_path.is_file()
    assert validation_json_path.is_file()
    assert validation_csv_path.is_file()
    assert training_sidecar_path.is_file()
    assert evaluation_sidecar_path.is_file()
    assert discovery_sidecar_path.is_file()
    assert validation_sidecar_path.is_file()

    evaluation_payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    discovery_payload = json.loads(discovery_path.read_text(encoding="utf-8"))
    discovery_coverage_payload = json.loads(discovery_coverage_path.read_text(encoding="utf-8"))
    cluster_summary_payload = json.loads(cluster_summary_json_path.read_text(encoding="utf-8"))
    validation_payload = json.loads(validation_json_path.read_text(encoding="utf-8"))
    training_sidecar_payload = json.loads(training_sidecar_path.read_text(encoding="utf-8"))
    evaluation_sidecar_payload = json.loads(evaluation_sidecar_path.read_text(encoding="utf-8"))
    discovery_sidecar_payload = json.loads(discovery_sidecar_path.read_text(encoding="utf-8"))

    assert evaluation_payload["split_name"] == "test"
    assert "predictive_improvement" in evaluation_payload["metrics"]
    assert discovery_payload["split_name"] == "discovery"
    assert "decoder_summary" in discovery_payload
    assert isinstance(discovery_payload["candidates"], list)
    assert discovery_payload["decoder_summary"]["metric_scope"] == "fit_selected_windows"
    assert discovery_coverage_payload["split_name"] == "discovery"
    assert discovery_coverage_payload["target_label"] == "stimulus_change"
    assert discovery_coverage_payload["positive_window_count"] >= 1
    assert discovery_coverage_payload["negative_window_count"] >= 1
    assert discovery_coverage_payload["sampling_strategy"] == "sequential"
    assert "clusters" in cluster_summary_payload
    assert cluster_summary_payload["cluster_count"] >= 1
    assert "cluster_quality_summary" in cluster_summary_payload
    assert validation_payload["candidate_count"] == len(discovery_payload["candidates"])
    assert "real_label_metrics" in validation_payload
    assert "shuffled_label_metrics" in validation_payload
    assert "held_out_test_metrics" in validation_payload
    assert "held_out_similarity_summary" in validation_payload
    assert "cluster_quality_summary" in validation_payload
    assert validation_payload["baseline_sensitivity_summary"]["comparison_available"] is True
    assert "sampling_summary" in validation_payload
    assert 0.0 <= validation_payload["held_out_test_metrics"]["probe_accuracy"] <= 1.0
    assert 0.0 <= validation_payload["held_out_similarity_summary"]["window_roc_auc"] <= 1.0
    assert 0.0 <= validation_payload["held_out_similarity_summary"]["window_pr_auc"] <= 1.0
    assert "recurrence_summary" not in validation_payload
    assert "stability_summary" not in discovery_payload
    assert "cluster_quality_summary" in discovery_payload
    assert training_sidecar_payload["command_name"] == "train"
    assert training_sidecar_payload["dataset_id"] == "allen_visual_behavior_neuropixels"
    assert "outputs" in training_sidecar_payload
    assert evaluation_sidecar_payload["command_name"] == "evaluate"
    assert evaluation_sidecar_payload["outputs"]["evaluation_summary_path"] == str(evaluation_path)
    assert discovery_sidecar_payload["command_name"] == "discover"
    assert discovery_sidecar_payload["outputs"]["decode_coverage_summary_path"] == str(discovery_coverage_path)


def test_image_identity_one_vs_rest_workflow_runs_end_to_end(tmp_path: Path):
    prep_config_path, _, _ = _create_prepared_workspace(tmp_path)
    experiment_config_path = _write_experiment_config(
        tmp_path,
        discovery_target_label="stimulus_presentations.image_name",
        discovery_target_label_mode="onset_within_window",
        discovery_target_label_match_value="im2",
    )
    checkpoint_path = tmp_path / "artifacts" / "checkpoints" / "pcc_test_best.pt"
    discovery_path = tmp_path / "artifacts" / "checkpoints" / "pcc_test_image_discovery.json"
    validation_json_path = tmp_path / "artifacts" / "checkpoints" / "pcc_test_image_validation.json"

    for entrypoint, argv in (
        [train_main, ["--config", str(experiment_config_path), "--data-config", str(prep_config_path)]],
        [
            discover_main,
            [
                "--config",
                str(experiment_config_path),
                "--data-config",
                str(prep_config_path),
                "--checkpoint",
                str(checkpoint_path),
                "--split",
                "discovery",
                "--output",
                str(discovery_path),
            ],
        ],
        [
            validate_main,
            [
                "--config",
                str(experiment_config_path),
                "--data-config",
                str(prep_config_path),
                "--checkpoint",
                str(checkpoint_path),
                "--discovery-artifact",
                str(discovery_path),
                "--output-json",
                str(validation_json_path),
            ],
        ],
    ):
        try:
            entrypoint(argv)
        except SystemExit as exc:
            assert exc.code == 0

    discovery_payload = json.loads(discovery_path.read_text(encoding="utf-8"))
    validation_payload = json.loads(validation_json_path.read_text(encoding="utf-8"))

    assert discovery_payload["decoder_summary"]["target_label"] == "stimulus_presentations.image_name"
    assert discovery_payload["config_snapshot"]["discovery"]["target_label_match_value"] == "im2"
    assert len(discovery_payload["candidates"]) >= 1
    assert "held_out_test_metrics" in validation_payload


def test_collect_notebook_target_value_counts_reads_image_names_from_discovery_sessions(tmp_path: Path):
    prep_config_path, _, _ = _create_prepared_workspace(tmp_path)
    experiment_config_path = _write_experiment_config(
        tmp_path,
        discovery_target_label="stimulus_presentations.image_name",
        discovery_target_label_mode="onset_within_window",
        discovery_target_label_match_value="im2",
    )

    counts = collect_notebook_target_value_counts(
        experiment_config_path=experiment_config_path,
        data_config_path=prep_config_path,
        split_name="discovery",
        target_label="stimulus_presentations.image_name",
        target_label_mode="onset_within_window",
    )

    assert counts
    assert {row["value"] for row in counts} == {"im0", "im1", "im2", "im3"}
    assert all(int(row["count"]) == 1 for row in counts)


def test_inspect_notebook_target_field_availability_reports_session_level_image_support(tmp_path: Path):
    prep_config_path, _, _ = _create_prepared_workspace(tmp_path)
    experiment_config_path = _write_experiment_config(
        tmp_path,
        discovery_target_label="stimulus_presentations.image_name",
        discovery_target_label_mode="onset_within_window",
        discovery_target_label_match_value="im2",
    )

    summary = inspect_notebook_target_field_availability(
        experiment_config_path=experiment_config_path,
        data_config_path=prep_config_path,
        split_name="discovery",
        target_label="stimulus_presentations.image_name",
    )

    assert summary["sessions_scanned"] == 1
    assert summary["sessions_with_namespace"] == 1
    assert summary["sessions_with_field"] == 1
    assert tuple(summary["value_counts"]) == (
        {"value": "im0", "count": 1},
        {"value": "im1", "count": 1},
        {"value": "im2", "count": 1},
        {"value": "im3", "count": 1},
    )
    session_row = summary["session_rows"][0]
    assert session_row["has_namespace"] is True
    assert session_row["has_field"] is True
    assert session_row["preview_values"] == ("im0", "im1", "im2", "im3")


def test_run_representation_comparison_from_encoded_reuses_shared_selected_windows(tmp_path: Path):
    experiment_config_path = _write_experiment_config(tmp_path)
    experiment_config = load_experiment_config(experiment_config_path)
    checkpoint_path = tmp_path / "artifacts" / "checkpoints" / "pcc_test_best.pt"

    def _embed(x: float, y: float) -> tuple[float, ...]:
        values = [0.0] * int(experiment_config.model.d_model)
        values[0] = x
        values[1] = y
        return tuple(values)

    windows = (
        ("session_a", "subject_a", 0.0, 1.0, 1.0, _embed(2.0, 0.0), (_embed(2.1, 0.0), _embed(1.9, 0.1))),
        ("session_a", "subject_a", 1.0, 2.0, 1.0, _embed(1.8, 0.2), (_embed(1.9, 0.1), _embed(1.7, 0.2))),
        ("session_a", "subject_a", 2.0, 3.0, 0.0, _embed(-2.0, 0.0), (_embed(-2.1, 0.0), _embed(-1.9, -0.1))),
        ("session_a", "subject_a", 3.0, 4.0, 0.0, _embed(-1.8, -0.2), (_embed(-1.9, -0.1), _embed(-1.7, -0.2))),
        ("session_b", "subject_b", 0.0, 1.0, 1.0, _embed(0.0, 2.0), (_embed(0.0, 2.1), _embed(0.1, 1.9))),
        ("session_b", "subject_b", 1.0, 2.0, 1.0, _embed(-0.2, 1.8), (_embed(-0.1, 1.9), _embed(-0.2, 1.7))),
        ("session_b", "subject_b", 2.0, 3.0, 0.0, _embed(0.0, -2.0), (_embed(0.0, -2.1), _embed(-0.1, -1.9))),
        ("session_b", "subject_b", 3.0, 4.0, 0.0, _embed(0.2, -1.8), (_embed(0.1, -1.9), _embed(0.2, -1.7))),
    )
    recording_ids = tuple(f"allen_visual_behavior_neuropixels/{session_id}" for session_id, *_ in windows)
    session_ids = tuple(row[0] for row in windows)
    subject_ids = tuple(row[1] for row in windows)
    labels = torch.tensor([row[4] for row in windows], dtype=torch.float32)
    pooled_features = torch.tensor([row[5] for row in windows], dtype=torch.float32)
    token_tensors = torch.tensor([[list(token) for token in row[6]] for row in windows], dtype=torch.float32)
    token_mask = torch.ones((len(windows), 2), dtype=torch.bool)
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / "token_shard_00000.pt"
    shard_embeddings: list[tuple[float, ...]] = []
    shard_recording_ids: list[str] = []
    shard_session_ids: list[str] = []
    shard_subject_ids: list[str] = []
    shard_unit_ids: list[str] = []
    shard_regions: list[str] = []
    shard_depths: list[float] = []
    shard_patch_index: list[int] = []
    shard_patch_start_s: list[float] = []
    shard_patch_end_s: list[float] = []
    shard_window_start_s: list[float] = []
    shard_window_end_s: list[float] = []
    shard_labels: list[int] = []
    for window_index, (session_id, subject_id, start_s, end_s, label, _, tokens) in enumerate(windows):
        recording_id = f"allen_visual_behavior_neuropixels/{session_id}"
        for patch_index, token in enumerate(tokens):
            shard_embeddings.append(token)
            shard_recording_ids.append(recording_id)
            shard_session_ids.append(session_id)
            shard_subject_ids.append(subject_id)
            shard_unit_ids.append(f"{session_id}_unit_{patch_index}")
            shard_regions.append("VISp")
            shard_depths.append(100.0 + float(window_index))
            shard_patch_index.append(patch_index)
            shard_patch_start_s.append(float(start_s + (0.5 * patch_index)))
            shard_patch_end_s.append(float(start_s + (0.5 * (patch_index + 1))))
            shard_window_start_s.append(float(start_s))
            shard_window_end_s.append(float(end_s))
            shard_labels.append(int(label > 0.5))
    torch.save(
        {
            "embeddings": torch.tensor(shard_embeddings, dtype=torch.float32),
            "recording_ids": shard_recording_ids,
            "session_ids": shard_session_ids,
            "subject_ids": shard_subject_ids,
            "unit_ids": shard_unit_ids,
            "unit_regions": shard_regions,
            "unit_depth_um": torch.tensor(shard_depths, dtype=torch.float32),
            "patch_index": torch.tensor(shard_patch_index, dtype=torch.long),
            "patch_start_s": torch.tensor(shard_patch_start_s, dtype=torch.float32),
            "patch_end_s": torch.tensor(shard_patch_end_s, dtype=torch.float32),
            "window_start_s": torch.tensor(shard_window_start_s, dtype=torch.float32),
            "window_end_s": torch.tensor(shard_window_end_s, dtype=torch.float32),
            "labels": torch.tensor(shard_labels, dtype=torch.long),
        },
        shard_path,
    )

    coverage_summary = DiscoveryCoverageSummary(
        split_name="discovery",
        target_label="stimulus_change",
        total_scanned_windows=len(windows),
        positive_window_count=4,
        negative_window_count=4,
        selected_positive_count=4,
        selected_negative_count=4,
        sessions_with_positive_windows=("session_a", "session_b"),
        sampling_strategy="label_balanced",
        scan_max_batches=4,
        selected_window_count=len(windows),
    )
    window_plan = DiscoveryWindowPlan(
        split_name="discovery",
        target_label="stimulus_change",
        windows=tuple(
            DiscoveryWindowPlanRecord(
                recording_id=f"allen_visual_behavior_neuropixels/{session_id}",
                session_id=session_id,
                subject_id=subject_id,
                window_start_s=float(start_s),
                window_end_s=float(end_s),
                label=float(label),
            )
            for session_id, subject_id, start_s, end_s, label, _, _ in windows
        ),
        selected_indices=torch.arange(len(windows), dtype=torch.long),
        coverage_summary=coverage_summary,
    )
    encoded = EncodedDiscoverySelection(
        pooled_features=pooled_features,
        labels=labels,
        token_tensors=token_tensors,
        token_mask=token_mask,
        window_recording_ids=recording_ids,
        window_session_ids=session_ids,
        window_subject_ids=subject_ids,
        window_start_s=tuple(float(row[2]) for row in windows),
        window_end_s=tuple(float(row[3]) for row in windows),
        shard_paths=(shard_path,),
        coverage_summary=coverage_summary,
        encoder_device="cpu",
    )

    comparison = run_representation_comparison_from_encoded(
        experiment_config=experiment_config,
        data_config_path=tmp_path / "unused.yaml",
        checkpoint_path=checkpoint_path,
        split_name="discovery",
        window_plan=window_plan,
        encoded=encoded,
        session_holdout_fraction=0.5,
        session_holdout_seed=23,
        run_standard_test_validation=False,
        temporary_root=tmp_path / "comparison_tmp",
    )

    assert comparison.coverage_summary.selected_window_count == len(windows)
    assert comparison.split_summary["fit_window_count"] == 4
    assert comparison.split_summary["heldout_window_count"] == 4
    assert comparison.split_summary["excluded_sessions"] == []
    assert {result.arm_name for result in comparison.arm_results} == {
        "baseline",
        "whitening_only",
        "whitening_plus_held_out_alignment",
    }
    for arm_result in comparison.arm_results:
        assert arm_result.validation_summary["fit_window_count"] == 4
        assert arm_result.validation_summary["heldout_window_count"] == 4
        assert arm_result.validation_summary["cluster_count"] >= 1
        assert arm_result.validation_summary["candidate_count"] >= 2
        debug = arm_result.validation_summary["candidate_selection_summary"]["arm_shard_debug"]
        assert debug["fit_window_count"] == 4
        assert debug["fit_positive_window_count"] == 2
        assert debug["token_row_count"] == 8
        assert debug["positive_token_row_count"] == 4
        assert debug["window_row_count"] == 4


def test_run_representation_comparison_from_encoded_falls_back_when_threshold_selects_no_candidates(tmp_path: Path):
    experiment_config_path = _write_experiment_config(tmp_path)
    experiment_config = load_experiment_config(experiment_config_path)
    experiment_config = replace(
        experiment_config,
        discovery=replace(
            experiment_config.discovery,
            min_candidate_score=999.0,
        ),
    )

    def _embed(x: float, y: float) -> tuple[float, ...]:
        values = [0.0] * int(experiment_config.model.d_model)
        values[0] = x
        values[1] = y
        return tuple(values)

    windows = (
        ("session_a", "subject_a", 0.0, 1.0, 1.0, _embed(2.0, 0.0), (_embed(2.1, 0.0), _embed(1.9, 0.1))),
        ("session_a", "subject_a", 1.0, 2.0, 1.0, _embed(1.8, 0.2), (_embed(1.9, 0.1), _embed(1.7, 0.2))),
        ("session_a", "subject_a", 2.0, 3.0, 0.0, _embed(-2.0, 0.0), (_embed(-2.1, 0.0), _embed(-1.9, -0.1))),
        ("session_a", "subject_a", 3.0, 4.0, 0.0, _embed(-1.8, -0.2), (_embed(-1.9, -0.1), _embed(-1.7, -0.2))),
        ("session_b", "subject_b", 0.0, 1.0, 1.0, _embed(0.0, 2.0), (_embed(0.0, 2.1), _embed(0.1, 1.9))),
        ("session_b", "subject_b", 1.0, 2.0, 1.0, _embed(-0.2, 1.8), (_embed(-0.1, 1.9), _embed(-0.2, 1.7))),
        ("session_b", "subject_b", 2.0, 3.0, 0.0, _embed(0.0, -2.0), (_embed(0.0, -2.1), _embed(-0.1, -1.9))),
        ("session_b", "subject_b", 3.0, 4.0, 0.0, _embed(0.2, -1.8), (_embed(0.1, -1.9), _embed(0.2, -1.7))),
    )
    recording_ids = tuple(f"allen_visual_behavior_neuropixels/{session_id}" for session_id, *_ in windows)
    session_ids = tuple(row[0] for row in windows)
    subject_ids = tuple(row[1] for row in windows)
    labels = torch.tensor([row[4] for row in windows], dtype=torch.float32)
    pooled_features = torch.tensor([row[5] for row in windows], dtype=torch.float32)
    token_tensors = torch.tensor([[list(token) for token in row[6]] for row in windows], dtype=torch.float32)
    token_mask = torch.ones((len(windows), 2), dtype=torch.bool)
    shard_dir = tmp_path / "shards_degraded"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / "token_shard_00000.pt"
    shard_embeddings: list[tuple[float, ...]] = []
    shard_recording_ids: list[str] = []
    shard_session_ids: list[str] = []
    shard_subject_ids: list[str] = []
    shard_unit_ids: list[str] = []
    shard_regions: list[str] = []
    shard_depths: list[float] = []
    shard_patch_index: list[int] = []
    shard_patch_start_s: list[float] = []
    shard_patch_end_s: list[float] = []
    shard_window_start_s: list[float] = []
    shard_window_end_s: list[float] = []
    shard_labels: list[int] = []
    for window_index, (session_id, subject_id, start_s, end_s, label, _, tokens) in enumerate(windows):
        recording_id = f"allen_visual_behavior_neuropixels/{session_id}"
        for patch_index, token in enumerate(tokens):
            shard_embeddings.append(token)
            shard_recording_ids.append(recording_id)
            shard_session_ids.append(session_id)
            shard_subject_ids.append(subject_id)
            shard_unit_ids.append(f"{session_id}_unit_{patch_index}")
            shard_regions.append("VISp")
            shard_depths.append(100.0 + float(window_index))
            shard_patch_index.append(patch_index)
            shard_patch_start_s.append(float(start_s + (0.5 * patch_index)))
            shard_patch_end_s.append(float(start_s + (0.5 * (patch_index + 1))))
            shard_window_start_s.append(float(start_s))
            shard_window_end_s.append(float(end_s))
            shard_labels.append(int(label > 0.5))
    torch.save(
        {
            "embeddings": torch.tensor(shard_embeddings, dtype=torch.float32),
            "recording_ids": shard_recording_ids,
            "session_ids": shard_session_ids,
            "subject_ids": shard_subject_ids,
            "unit_ids": shard_unit_ids,
            "unit_regions": shard_regions,
            "unit_depth_um": torch.tensor(shard_depths, dtype=torch.float32),
            "patch_index": torch.tensor(shard_patch_index, dtype=torch.long),
            "patch_start_s": torch.tensor(shard_patch_start_s, dtype=torch.float32),
            "patch_end_s": torch.tensor(shard_patch_end_s, dtype=torch.float32),
            "window_start_s": torch.tensor(shard_window_start_s, dtype=torch.float32),
            "window_end_s": torch.tensor(shard_window_end_s, dtype=torch.float32),
            "labels": torch.tensor(shard_labels, dtype=torch.long),
        },
        shard_path,
    )
    coverage_summary = DiscoveryCoverageSummary(
        split_name="discovery",
        target_label="stimulus_change",
        total_scanned_windows=len(windows),
        positive_window_count=4,
        negative_window_count=4,
        selected_positive_count=4,
        selected_negative_count=4,
        sessions_with_positive_windows=("session_a", "session_b"),
        sampling_strategy="label_balanced",
        scan_max_batches=4,
        selected_window_count=len(windows),
    )
    window_plan = DiscoveryWindowPlan(
        split_name="discovery",
        target_label="stimulus_change",
        windows=tuple(
            DiscoveryWindowPlanRecord(
                recording_id=f"allen_visual_behavior_neuropixels/{session_id}",
                session_id=session_id,
                subject_id=subject_id,
                window_start_s=float(start_s),
                window_end_s=float(end_s),
                label=float(label),
            )
            for session_id, subject_id, start_s, end_s, label, _, _ in windows
        ),
        selected_indices=torch.arange(len(windows), dtype=torch.long),
        coverage_summary=coverage_summary,
    )
    encoded = EncodedDiscoverySelection(
        pooled_features=pooled_features,
        labels=labels,
        token_tensors=token_tensors,
        token_mask=token_mask,
        window_recording_ids=recording_ids,
        window_session_ids=session_ids,
        window_subject_ids=subject_ids,
        window_start_s=tuple(float(row[2]) for row in windows),
        window_end_s=tuple(float(row[3]) for row in windows),
        shard_paths=(shard_path,),
        coverage_summary=coverage_summary,
        encoder_device="cpu",
    )

    comparison = run_representation_comparison_from_encoded(
        experiment_config=experiment_config,
        data_config_path=tmp_path / "unused.yaml",
        checkpoint_path=tmp_path / "artifacts" / "checkpoints" / "pcc_test_best.pt",
        split_name="discovery",
        window_plan=window_plan,
        encoded=encoded,
        session_holdout_fraction=0.5,
        session_holdout_seed=23,
        run_standard_test_validation=False,
        temporary_root=tmp_path / "comparison_tmp_degraded",
    )

    for arm_result in comparison.arm_results:
        candidate_selection = arm_result.validation_summary["candidate_selection_summary"]
        assert candidate_selection["fallback_used"] is True
        assert candidate_selection["effective_min_score"] is None
        debug = candidate_selection["arm_shard_debug"]
        assert debug["fit_window_count"] == 4
        assert debug["fit_positive_window_count"] == 2
        assert debug["token_row_count"] == 8
        assert debug["positive_token_row_count"] == 4
        assert arm_result.validation_summary["candidate_count"] >= 2


def test_train_model_writes_fallback_best_checkpoint_when_validation_metric_is_nan(tmp_path: Path, monkeypatch):
    prep_config_path, experiment_config_path, _ = _create_prepared_workspace(tmp_path)
    experiment_config = load_experiment_config(experiment_config_path)

    def _nan_evaluation(**kwargs):
        checkpoint_path = kwargs.get("checkpoint_path", "")
        return EvaluationSummary(
            dataset_id="allen_visual_behavior_neuropixels",
            split_name="valid",
            checkpoint_path=str(checkpoint_path),
            metrics={
                "predictive_loss": 0.0,
                "reconstruction_loss": 0.0,
                "predictive_raw_mse": 0.0,
                "predictive_baseline_mse": 0.0,
                "predictive_improvement": float("nan"),
                "token_coverage": 1.0,
            },
            losses={
                "predictive_loss": 0.0,
                "reconstruction_loss": 0.0,
                "total_loss": 0.0,
            },
            window_count=1,
        )

    monkeypatch.setattr(
        "predictive_circuit_coding.training.loop.evaluate_checkpoint_on_split",
        _nan_evaluation,
    )

    result = train_model(
        experiment_config=experiment_config,
        data_config_path=prep_config_path,
        train_split="train",
        valid_split="valid",
    )

    assert result.checkpoint_path.is_file()
    assert result.checkpoint_path.name == "pcc_test_best.pt"
    summary_payload = json.loads(experiment_config.artifacts.summary_path.read_text(encoding="utf-8"))
    assert summary_payload["selection_reason"] == "fallback_latest_due_to_invalid_metric"
    assert summary_payload["epoch"] == result.best_epoch
    assert summary_payload["best_epoch"] == result.best_epoch


def test_training_summary_tracks_best_checkpoint_metrics(tmp_path: Path, monkeypatch):
    prep_config_path, experiment_config_path, _ = _create_prepared_workspace(tmp_path)
    experiment_config = load_experiment_config(experiment_config_path)
    metric_sequence = [0.8, 0.2]

    def _scripted_evaluation(**kwargs):
        value = metric_sequence.pop(0)
        return EvaluationSummary(
            dataset_id="allen_visual_behavior_neuropixels",
            split_name="valid",
            checkpoint_path=str(kwargs.get("checkpoint_path", "")),
            metrics={
                "predictive_loss": 1.0 - value,
                "reconstruction_loss": 0.0,
                "predictive_raw_mse": 1.0,
                "predictive_baseline_mse": 1.0,
                "predictive_improvement": value,
                "token_coverage": 1.0,
            },
            losses={
                "predictive_loss": 1.0 - value,
                "reconstruction_loss": 0.0,
                "total_loss": 1.0 - value,
            },
            window_count=2,
        )

    monkeypatch.setattr(
        "predictive_circuit_coding.training.loop.evaluate_checkpoint_on_split",
        _scripted_evaluation,
    )

    result = train_model(
        experiment_config=experiment_config,
        data_config_path=prep_config_path,
        train_split="train",
        valid_split="valid",
    )

    summary_payload = json.loads(experiment_config.artifacts.summary_path.read_text(encoding="utf-8"))
    assert result.best_epoch == 1
    assert summary_payload["epoch"] == 1
    assert summary_payload["best_epoch"] == 1
    assert summary_payload["metrics"]["predictive_improvement"] == 0.8
    assert summary_payload["checkpoint_path"] == str(result.checkpoint_path)
    checkpoint_payload = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint_payload["best_epoch"] == 1
    assert checkpoint_payload["best_validation_metrics"]["predictive_improvement"] == 0.8


def test_resume_training_preserves_best_epoch_from_latest_checkpoint(tmp_path: Path, monkeypatch):
    prep_config_path, experiment_config_path, _ = _create_prepared_workspace(tmp_path)
    experiment_config = load_experiment_config(experiment_config_path)
    metric_sequence = [0.9, 0.3, 0.2]

    def _scripted_evaluation(**kwargs):
        value = metric_sequence.pop(0)
        return EvaluationSummary(
            dataset_id="allen_visual_behavior_neuropixels",
            split_name="valid",
            checkpoint_path=str(kwargs.get("checkpoint_path", "")),
            metrics={
                "predictive_loss": 1.0 - value,
                "reconstruction_loss": 0.0,
                "predictive_raw_mse": 1.0,
                "predictive_baseline_mse": 1.0,
                "predictive_improvement": value,
                "token_coverage": 1.0,
            },
            losses={
                "predictive_loss": 1.0 - value,
                "reconstruction_loss": 0.0,
                "total_loss": 1.0 - value,
            },
            window_count=2,
        )

    monkeypatch.setattr(
        "predictive_circuit_coding.training.loop.evaluate_checkpoint_on_split",
        _scripted_evaluation,
    )

    initial_result = train_model(
        experiment_config=experiment_config,
        data_config_path=prep_config_path,
        train_split="train",
        valid_split="valid",
    )
    latest_checkpoint_path = experiment_config.artifacts.checkpoint_dir / "pcc_test_latest.pt"
    resumed_config = load_experiment_config(experiment_config_path)
    resumed_config = replace(
        resumed_config,
        training=replace(
            resumed_config.training,
            num_epochs=3,
            resume_checkpoint=latest_checkpoint_path,
        ),
    )

    resumed_result = train_model(
        experiment_config=resumed_config,
        data_config_path=prep_config_path,
        train_split="train",
        valid_split="valid",
    )

    assert initial_result.best_epoch == 1
    assert resumed_result.best_epoch == 1
    resumed_summary = json.loads(resumed_config.artifacts.summary_path.read_text(encoding="utf-8"))
    assert resumed_summary["best_epoch"] == 1
    resumed_checkpoint = torch.load(resumed_result.checkpoint_path, map_location="cpu", weights_only=False)
    assert resumed_checkpoint["best_epoch"] == 1


def test_validation_caps_discovery_extraction_when_sampling_strategy_is_label_balanced(tmp_path: Path):
    """Regression test: validate_discovery_artifact must respect evaluation.max_batches even when
    the configured discovery sampling_strategy is 'label_balanced'.  Previously, label_balanced
    silently ignored max_batches and scanned the entire split, exhausting RAM on large datasets.
    The fix adds sampling_strategy_override='sequential' to the internal discovery extraction call."""
    prep_config_path, experiment_config_path, _ = _create_prepared_workspace(tmp_path)
    # Overwrite experiment config with label_balanced strategy.
    experiment_config_path = _write_experiment_config(
        tmp_path, discovery_sampling_strategy="label_balanced"
    )

    try:
        train_main(["--config", str(experiment_config_path), "--data-config", str(prep_config_path)])
    except SystemExit as exc:
        assert exc.code == 0

    checkpoint_path = tmp_path / "artifacts" / "checkpoints" / "pcc_test_best.pt"
    discovery_path = tmp_path / "artifacts" / "checkpoints" / "pcc_lb_discovery.json"
    try:
        discover_main(
            [
                "--config",
                str(experiment_config_path),
                "--data-config",
                str(prep_config_path),
                "--checkpoint",
                str(checkpoint_path),
                "--split",
                "discovery",
                "--output",
                str(discovery_path),
            ]
        )
    except SystemExit as exc:
        assert exc.code == 0

    validation_json_path = tmp_path / "artifacts" / "checkpoints" / "pcc_lb_validation.json"
    try:
        validate_main(
            [
                "--config",
                str(experiment_config_path),
                "--data-config",
                str(prep_config_path),
                "--checkpoint",
                str(checkpoint_path),
                "--discovery-artifact",
                str(discovery_path),
                "--output-json",
                str(validation_json_path),
            ]
        )
    except SystemExit as exc:
        assert exc.code == 0

    import json as _json

    payload = _json.loads(validation_json_path.read_text(encoding="utf-8"))
    assert "real_label_metrics" in payload
    assert "shuffled_label_metrics" in payload
    assert "held_out_test_metrics" in payload
    assert "held_out_similarity_summary" in payload
