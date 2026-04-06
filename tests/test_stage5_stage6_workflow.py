from __future__ import annotations

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
from predictive_circuit_coding.training import load_experiment_config, train_model
from predictive_circuit_coding.training.contracts import EvaluationSummary


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


def _write_experiment_config(tmp_path: Path) -> Path:
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
                "  target_label: stimulus_change",
                "  max_batches: 2",
                "  probe_epochs: 20",
                "  probe_learning_rate: 0.05",
                "  top_k_candidates: 8",
                "  min_candidate_score: -100.0",
                "  cluster_similarity_threshold: 0.0",
                "  min_cluster_size: 1",
                "  stability_rounds: 3",
                "  recurrence_similarity_threshold: 0.0",
                "  shuffle_seed: 19",
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
    assert discovery_coverage_payload["split_name"] == "discovery"
    assert discovery_coverage_payload["target_label"] == "stimulus_change"
    assert discovery_coverage_payload["positive_window_count"] >= 1
    assert discovery_coverage_payload["negative_window_count"] >= 1
    assert "clusters" in cluster_summary_payload
    assert cluster_summary_payload["cluster_count"] >= 1
    assert validation_payload["candidate_count"] == len(discovery_payload["candidates"])
    assert "real_label_metrics" in validation_payload
    assert "shuffled_label_metrics" in validation_payload
    assert training_sidecar_payload["command_name"] == "train"
    assert training_sidecar_payload["dataset_id"] == "allen_visual_behavior_neuropixels"
    assert "outputs" in training_sidecar_payload
    assert evaluation_sidecar_payload["command_name"] == "evaluate"
    assert evaluation_sidecar_payload["outputs"]["evaluation_summary_path"] == str(evaluation_path)
    assert discovery_sidecar_payload["command_name"] == "discover"
    assert discovery_sidecar_payload["outputs"]["decode_coverage_summary_path"] == str(discovery_coverage_path)


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
