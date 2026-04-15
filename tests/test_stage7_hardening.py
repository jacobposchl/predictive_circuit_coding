from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
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
from predictive_circuit_coding.decoding.extract import _select_label_balanced_indices
from predictive_circuit_coding.discovery import build_discovery_cluster_report
from predictive_circuit_coding.training.artifacts import save_training_checkpoint
from predictive_circuit_coding.training.contracts import (
    CandidateTokenRecord,
    CheckpointMetadata,
    DecoderSummary,
    DiscoveryArtifact,
    TrainingCheckpoint,
)
from predictive_circuit_coding.utils.notebook_progress import NotebookStageReporter, format_duration, verify_paths_exist


def _write_preparation_config(tmp_path: Path) -> Path:
    config_dir = tmp_path / "configs" / "pcc"
    config_dir.mkdir(parents=True, exist_ok=True)
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
    resume_checkpoint: str | None = None,
    min_candidate_score: float = -100.0,
    discovery_sampling_strategy: str = "sequential",
    discovery_max_batches: int = 1,
    discovery_search_max_batches: int | None = None,
    discovery_min_positive_windows: int = 1,
    discovery_negative_to_positive_ratio: float = 1.0,
    optimization_batch_size: int = 2,
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
                f"  batch_size: {optimization_batch_size}",
                "  scheduler_type: none",
                "  scheduler_warmup_steps: 0",
                "training:",
                "  num_epochs: 1",
                "  train_steps_per_epoch: 1",
                "  validation_steps: 1",
                "  checkpoint_every_epochs: 1",
                "  evaluate_every_epochs: 1",
                f"  resume_checkpoint: {resume_checkpoint or ''}",
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
                f"  max_batches: {discovery_max_batches}",
                f"  sampling_strategy: {discovery_sampling_strategy}",
                f"  search_max_batches: {'' if discovery_search_max_batches is None else discovery_search_max_batches}",
                f"  min_positive_windows: {discovery_min_positive_windows}",
                f"  negative_to_positive_ratio: {discovery_negative_to_positive_ratio}",
                "  probe_epochs: 10",
                "  probe_learning_rate: 0.05",
                "  top_k_candidates: 8",
                f"  min_candidate_score: {min_candidate_score}",
                "  min_cluster_size: 2",
                "  stability_rounds: 2",
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


def _write_session(
    *,
    path: Path,
    dataset_id: str,
    session_id: str,
    subject_id: str,
    split_name: str,
    has_positive_change: bool = True,
    positive_change_start_s: float = 1.20,
) -> None:
    from temporaldata import ArrayDict, Data, Interval, IrregularTimeSeries

    duration_s = 4.0
    split_intervals = build_split_intervals_for_assignment(
        domain_start_s=0.0,
        domain_end_s=duration_s,
        assigned_split=split_name,
    )
    domain = Interval(start=np.asarray([0.0], dtype=np.float64), end=np.asarray([duration_s], dtype=np.float64))
    is_change = np.asarray([False, True] if has_positive_change else [False, False], dtype=bool)
    data = Data(
        brainset=Data(id=dataset_id),
        session=Data(id=session_id),
        subject=Data(id=subject_id),
        units=ArrayDict(
            id=np.asarray([f"{session_id}_u0", f"{session_id}_u1"], dtype=object),
            brain_region=np.asarray(["VISp", "LP"], dtype=object),
            probe_depth_um=np.asarray([100.0, 250.0], dtype=np.float32),
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
            start=np.asarray([0.20, positive_change_start_s], dtype=np.float64),
            end=np.asarray([0.40, positive_change_start_s + 0.20], dtype=np.float64),
            stimulus_name=np.asarray(["images", "images"], dtype=object),
            image_name=np.asarray(["im0", "im1"], dtype=object),
            is_change=is_change,
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


def _build_workspace(
    tmp_path: Path,
    *,
    discovery_has_positive: bool = True,
    include_discovery_assignment: bool = True,
    discovery_positive_change_start_s: float = 1.20,
) -> tuple[Path, Path]:
    prep_config_path = _write_preparation_config(tmp_path)
    prep_config = load_preparation_config(prep_config_path)
    workspace = create_workspace(prep_config)
    dataset_id = prep_config.dataset.dataset_id
    session_specs = [
        ("session_train", "mouse_train", "train", True),
        ("session_valid", "mouse_valid", "valid", True),
        ("session_test", "mouse_test", "test", True),
    ]
    if include_discovery_assignment:
        session_specs.append(("session_discovery", "mouse_discovery", "discovery", discovery_has_positive))

    records: list[SessionRecord] = []
    split_rows: list[SplitAssignment] = []
    for session_id, subject_id, split_name, has_positive in session_specs:
        session_path = workspace.brainset_prepared_root / f"{session_id}.h5"
        _write_session(
            path=session_path,
            dataset_id=dataset_id,
            session_id=session_id,
            subject_id=subject_id,
            split_name=split_name,
            has_positive_change=has_positive,
            positive_change_start_s=discovery_positive_change_start_s if split_name == "discovery" else 1.20,
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
        SessionManifest(dataset_id=dataset_id, source_name=dataset_id, records=tuple(records)),
        workspace.session_manifest_path,
    )
    write_split_manifest(
        SplitManifest(dataset_id=dataset_id, seed=7, primary_axis="subject", assignments=tuple(split_rows)),
        workspace.split_manifest_path,
    )
    experiment_config_path = _write_experiment_config(tmp_path)
    return prep_config_path, experiment_config_path


def test_notebook_helpers_cover_timers_and_preflight(tmp_path: Path):
    existing = tmp_path / "exists.txt"
    existing.write_text("ok", encoding="utf-8")

    assert format_duration(61) == "1m 1s"
    assert verify_paths_exist({"existing": existing, "missing": tmp_path / "missing.txt"}) == {
        "existing": True,
        "missing": False,
    }

    reporter = NotebookStageReporter(name="test", expected_duration="short")
    reporter.banner("Notebook")
    reporter.begin("stage", next_artifact="artifact.json")
    reporter.note_checkpoint(existing)
    reporter.finish("stage")


def test_discovery_cluster_report_summarizes_clusters():
    artifact = DiscoveryArtifact(
        dataset_id="allen_visual_behavior_neuropixels",
        split_name="discovery",
        checkpoint_path="artifacts/checkpoints/pcc_best.pt",
        config_snapshot={"dataset_id": "allen_visual_behavior_neuropixels"},
        decoder_summary=DecoderSummary(
            target_label="stimulus_change",
            epochs=10,
            learning_rate=0.05,
            metrics={"probe_accuracy": 1.0},
        ),
        candidates=(
            CandidateTokenRecord(
                candidate_id="candidate_0001",
                cluster_id=0,
                recording_id="allen_visual_behavior_neuropixels/session_a",
                session_id="session_a",
                subject_id="mouse_a",
                unit_id="u0",
                unit_region="VISp",
                unit_depth_um=100.0,
                patch_index=1,
                patch_start_s=0.0,
                patch_end_s=0.5,
                window_start_s=0.0,
                window_end_s=2.0,
                label=1,
                score=0.8,
                embedding=(1.0, 0.0),
                raw_probe_score=0.9,
                negative_background_score=0.1,
            ),
            CandidateTokenRecord(
                candidate_id="candidate_0002",
                cluster_id=0,
                recording_id="allen_visual_behavior_neuropixels/session_b",
                session_id="session_b",
                subject_id="mouse_b",
                unit_id="u1",
                unit_region="VISp",
                unit_depth_um=200.0,
                patch_index=2,
                patch_start_s=0.5,
                patch_end_s=1.0,
                window_start_s=0.0,
                window_end_s=2.0,
                label=1,
                score=0.6,
                embedding=(0.9, 0.1),
                raw_probe_score=-0.2,
                negative_background_score=-0.8,
            ),
        ),
        cluster_stats={"cluster_count": 1.0},
        cluster_quality_summary={
            "silhouette_score": None,
            "non_noise_fraction": 1.0,
            "cluster_persistence_mean": 0.75,
            "cluster_persistence_min": 0.75,
            "cluster_persistence_max": 0.75,
            "cluster_persistence_by_cluster": {0: 0.75},
        },
    )

    report = build_discovery_cluster_report(artifact)

    assert report["cluster_count"] == 1
    assert report["candidate_count"] == 2
    assert report["cluster_quality_summary"]["cluster_persistence_mean"] == 0.75
    assert report["clusters"][0]["cluster_id"] == 0
    assert report["clusters"][0]["cluster_persistence"] == 0.75
    assert report["clusters"][0]["top_regions"][0]["value"] == "VISp"
    assert report["clusters"][0]["representative_candidate_id"] == "candidate_0001"
    assert report["clusters"][0]["mean_raw_probe_score"] == pytest.approx(0.35)
    assert report["clusters"][0]["positive_raw_probe_fraction"] == pytest.approx(0.5)
    assert report["clusters"][0]["representative_raw_probe_score"] == pytest.approx(0.9)


def test_evaluate_cli_fails_for_empty_split(tmp_path: Path):
    prep_config_path, experiment_config_path = _build_workspace(tmp_path, include_discovery_assignment=False)
    fake_checkpoint = tmp_path / "fake.pt"
    save_training_checkpoint(
        TrainingCheckpoint(
            epoch=1,
            global_step=1,
            best_metric=0.0,
            metadata=CheckpointMetadata(
                dataset_id="allen_visual_behavior_neuropixels",
                split_name="train",
                seed=1,
                config_snapshot={"dataset_id": "allen_visual_behavior_neuropixels"},
                model_hparams={"d_model": 16},
                continuation_baseline_type="previous_patch",
            ),
            model_state={},
            optimizer_state={},
            scheduler_state=None,
        ),
        fake_checkpoint,
    )

    with pytest.raises(ValueError, match="has no prepared sessions"):
        evaluate_main(
            [
                "--config",
                str(experiment_config_path),
                "--data-config",
                str(prep_config_path),
                "--checkpoint",
                str(fake_checkpoint),
                "--split",
                "discovery",
            ]
        )


def test_discover_cli_fails_when_no_positive_labels(tmp_path: Path):
    prep_config_path, _ = _build_workspace(tmp_path, discovery_has_positive=False)
    experiment_config_path = _write_experiment_config(
        tmp_path,
        discovery_sampling_strategy="label_balanced",
    )

    try:
        train_main(["--config", str(experiment_config_path), "--data-config", str(prep_config_path)])
    except SystemExit as exc:
        assert exc.code == 0

    checkpoint_path = tmp_path / "artifacts" / "checkpoints" / "pcc_test_best.pt"
    discovery_output_path = tmp_path / "artifacts" / "checkpoints" / "pcc_test_best_discovery_discovery.json"
    coverage_path = tmp_path / "artifacts" / "checkpoints" / "pcc_test_best_discovery_discovery_decode_coverage.json"
    with pytest.raises(ValueError, match="does not provide both classes"):
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
                str(discovery_output_path),
            ]
        )
    assert coverage_path.is_file()
    coverage_payload = json.loads(coverage_path.read_text(encoding="utf-8"))
    assert coverage_payload["positive_window_count"] == 0
    assert coverage_payload["negative_window_count"] > 0
    assert not discovery_output_path.exists()


def test_discover_cli_label_balanced_scans_full_split_and_finds_late_positive_windows(tmp_path: Path):
    prep_config_path, _ = _build_workspace(
        tmp_path,
        discovery_has_positive=True,
        discovery_positive_change_start_s=2.20,
    )
    experiment_config_path = _write_experiment_config(
        tmp_path,
        discovery_sampling_strategy="label_balanced",
        discovery_max_batches=1,
    )

    try:
        train_main(["--config", str(experiment_config_path), "--data-config", str(prep_config_path)])
    except SystemExit as exc:
        assert exc.code == 0

    checkpoint_path = tmp_path / "artifacts" / "checkpoints" / "pcc_test_best.pt"
    discovery_output_path = tmp_path / "artifacts" / "checkpoints" / "late_positive_discovery.json"
    coverage_path = tmp_path / "artifacts" / "checkpoints" / "late_positive_discovery_decode_coverage.json"

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
                str(discovery_output_path),
            ]
        )
    except SystemExit as exc:
        assert exc.code == 0

    coverage_payload = json.loads(coverage_path.read_text(encoding="utf-8"))
    assert coverage_payload["total_scanned_windows"] == 2
    assert coverage_payload["positive_window_count"] == 1
    assert coverage_payload["negative_window_count"] == 1
    assert coverage_payload["selected_positive_count"] == 1
    assert coverage_payload["selected_negative_count"] == 1


def test_discover_cli_label_balanced_respects_search_max_batches(tmp_path: Path):
    prep_config_path, _ = _build_workspace(
        tmp_path,
        discovery_has_positive=True,
        discovery_positive_change_start_s=2.20,
    )
    experiment_config_path = _write_experiment_config(
        tmp_path,
        discovery_sampling_strategy="label_balanced",
        discovery_max_batches=1,
        discovery_search_max_batches=1,
        optimization_batch_size=1,
    )

    try:
        train_main(["--config", str(experiment_config_path), "--data-config", str(prep_config_path)])
    except SystemExit as exc:
        assert exc.code == 0

    checkpoint_path = tmp_path / "artifacts" / "checkpoints" / "pcc_test_best.pt"
    discovery_output_path = tmp_path / "artifacts" / "checkpoints" / "budgeted_discovery.json"
    coverage_path = tmp_path / "artifacts" / "checkpoints" / "budgeted_discovery_decode_coverage.json"

    with pytest.raises(ValueError, match="does not provide both classes"):
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
                str(discovery_output_path),
            ]
        )

    coverage_payload = json.loads(coverage_path.read_text(encoding="utf-8"))
    assert coverage_payload["sampling_strategy"] == "label_balanced"
    assert coverage_payload["scan_max_batches"] == 1
    assert coverage_payload["total_scanned_windows"] == 1
    assert coverage_payload["positive_window_count"] == 0


def test_label_balanced_selection_uses_negative_to_positive_ratio():
    labels = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.float32)
    session_ids = tuple(f"session_{index}" for index in range(len(labels)))

    selected_indices, selected_positive, selected_negative = _select_label_balanced_indices(
        labels=labels,
        session_ids=session_ids,
        seed=7,
        max_selected_windows=6,
        negative_to_positive_ratio=2.0,
    )

    assert len(selected_indices) == 6
    assert len(selected_positive) == 2
    assert len(selected_negative) == 4


def test_discover_cli_fails_when_no_candidates_are_selected(tmp_path: Path):
    prep_config_path, _ = _build_workspace(tmp_path, discovery_has_positive=True)
    experiment_config_path = _write_experiment_config(tmp_path, min_candidate_score=1.0e9)

    try:
        train_main(["--config", str(experiment_config_path), "--data-config", str(prep_config_path)])
    except SystemExit as exc:
        assert exc.code == 0

    checkpoint_path = tmp_path / "artifacts" / "checkpoints" / "pcc_test_best.pt"
    with pytest.raises(ValueError, match="No candidate tokens were selected"):
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
            ]
        )


def test_evaluate_cli_rejects_checkpoint_dataset_mismatch(tmp_path: Path):
    prep_config_path, experiment_config_path = _build_workspace(tmp_path)
    mismatch_checkpoint = tmp_path / "mismatch.pt"
    save_training_checkpoint(
        TrainingCheckpoint(
            epoch=1,
            global_step=1,
            best_metric=0.0,
            metadata=CheckpointMetadata(
                dataset_id="wrong_dataset",
                split_name="train",
                seed=1,
                config_snapshot={"dataset_id": "wrong_dataset"},
                model_hparams={"d_model": 16},
                continuation_baseline_type="previous_patch",
            ),
            model_state={},
            optimizer_state={},
            scheduler_state=None,
        ),
        mismatch_checkpoint,
    )

    with pytest.raises(ValueError, match="does not match config dataset_id"):
        evaluate_main(
            [
                "--config",
                str(experiment_config_path),
                "--data-config",
                str(prep_config_path),
                "--checkpoint",
                str(mismatch_checkpoint),
                "--split",
                "test",
            ]
        )


def test_train_cli_rejects_missing_resume_checkpoint(tmp_path: Path):
    prep_config_path, _ = _build_workspace(tmp_path)
    missing_resume = tmp_path / "does_not_exist.pt"
    experiment_config_path = _write_experiment_config(tmp_path, resume_checkpoint=str(missing_resume))

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        train_main(["--config", str(experiment_config_path), "--data-config", str(prep_config_path)])


def test_validate_cli_recomputes_real_label_metrics_instead_of_trusting_artifact(tmp_path: Path):
    prep_config_path, experiment_config_path = _build_workspace(tmp_path)

    try:
        train_main(["--config", str(experiment_config_path), "--data-config", str(prep_config_path)])
    except SystemExit as exc:
        assert exc.code == 0

    checkpoint_path = tmp_path / "artifacts" / "checkpoints" / "pcc_test_best.pt"
    discovery_path = tmp_path / "artifacts" / "checkpoints" / "pcc_test_best_discovery_discovery.json"
    validation_path = tmp_path / "artifacts" / "checkpoints" / "tamper_validation.json"
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

    payload = json.loads(discovery_path.read_text(encoding="utf-8"))
    payload["decoder_summary"]["metrics"] = {"probe_accuracy": 0.123456, "probe_bce": 9.99}
    discovery_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

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
                str(validation_path),
            ]
        )
    except SystemExit as exc:
        assert exc.code == 0

    validation_payload = json.loads(validation_path.read_text(encoding="utf-8"))
    assert validation_payload["real_label_metrics"]["probe_accuracy"] != 0.123456
    assert validation_payload["sampling_summary"]["discovery_sampled_window_count"] >= 1


def test_validate_cli_rejects_artifact_checkpoint_mismatch(tmp_path: Path):
    prep_config_path, experiment_config_path = _build_workspace(tmp_path)

    try:
        train_main(["--config", str(experiment_config_path), "--data-config", str(prep_config_path)])
    except SystemExit as exc:
        assert exc.code == 0

    checkpoint_path = tmp_path / "artifacts" / "checkpoints" / "pcc_test_best.pt"
    mismatched_checkpoint_path = tmp_path / "artifacts" / "checkpoints" / "other_best.pt"
    mismatched_checkpoint_path.write_bytes(checkpoint_path.read_bytes())
    discovery_path = tmp_path / "artifacts" / "checkpoints" / "pcc_test_best_discovery_discovery.json"

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

    with pytest.raises(ValueError, match="checkpoint_path does not match"):
        validate_main(
            [
                "--config",
                str(experiment_config_path),
                "--data-config",
                str(prep_config_path),
                "--checkpoint",
                str(mismatched_checkpoint_path),
                "--discovery-artifact",
                str(discovery_path),
            ]
        )


def test_validate_cli_rejects_artifact_target_label_mismatch(tmp_path: Path):
    prep_config_path, experiment_config_path = _build_workspace(tmp_path)

    try:
        train_main(["--config", str(experiment_config_path), "--data-config", str(prep_config_path)])
    except SystemExit as exc:
        assert exc.code == 0

    checkpoint_path = tmp_path / "artifacts" / "checkpoints" / "pcc_test_best.pt"
    discovery_path = tmp_path / "artifacts" / "checkpoints" / "pcc_test_best_discovery_discovery.json"

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

    payload = json.loads(discovery_path.read_text(encoding="utf-8"))
    payload["decoder_summary"]["target_label"] = "trials.go"
    discovery_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="target label does not match"):
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
            ]
        )


def test_validate_cli_accepts_prefixed_candidate_session_and_subject_ids(tmp_path: Path):
    prep_config_path, experiment_config_path = _build_workspace(tmp_path)

    try:
        train_main(["--config", str(experiment_config_path), "--data-config", str(prep_config_path)])
    except SystemExit as exc:
        assert exc.code == 0

    checkpoint_path = tmp_path / "artifacts" / "checkpoints" / "pcc_test_best.pt"
    discovery_path = tmp_path / "artifacts" / "checkpoints" / "pcc_test_best_discovery_discovery.json"
    validation_path = tmp_path / "artifacts" / "checkpoints" / "prefixed_ids_validation.json"

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

    payload = json.loads(discovery_path.read_text(encoding="utf-8"))
    prefix = "allen_visual_behavior_neuropixels/"
    for candidate in payload["candidates"]:
        candidate["session_id"] = f"{prefix}{candidate['session_id']}"
        candidate["subject_id"] = f"{prefix}{candidate['subject_id']}"
    discovery_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

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
                str(validation_path),
            ]
        )
    except SystemExit as exc:
        assert exc.code == 0

    validation_payload = json.loads(validation_path.read_text(encoding="utf-8"))
    assert validation_payload["provenance_issues"] == []
