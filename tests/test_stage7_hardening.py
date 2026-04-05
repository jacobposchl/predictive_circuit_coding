from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from predictive_circuit_coding.cli.discover import main as discover_main
from predictive_circuit_coding.cli.evaluate import main as evaluate_main
from predictive_circuit_coding.cli.train import main as train_main
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
from predictive_circuit_coding.discovery import build_discovery_cluster_report
from predictive_circuit_coding.training.artifacts import save_training_checkpoint
from predictive_circuit_coding.training.contracts import (
    CandidateTokenRecord,
    CheckpointMetadata,
    DecoderSummary,
    DiscoveryArtifact,
    TrainingCheckpoint,
)
from predictive_circuit_coding.utils import NotebookStageReporter, format_duration, verify_paths_exist


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
                "  max_batches: 1",
                "  probe_epochs: 10",
                "  probe_learning_rate: 0.05",
                "  top_k_candidates: 8",
                f"  min_candidate_score: {min_candidate_score}",
                "  cluster_similarity_threshold: 0.0",
                "  min_cluster_size: 1",
                "  stability_rounds: 2",
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


def _write_session(
    *,
    path: Path,
    dataset_id: str,
    session_id: str,
    subject_id: str,
    split_name: str,
    has_positive_change: bool = True,
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
            start=np.asarray([0.20, 1.20], dtype=np.float64),
            end=np.asarray([0.40, 1.40], dtype=np.float64),
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
            ),
        ),
        cluster_stats={"cluster_count": 1.0},
        stability_summary={"mean_cluster_count": 1.0},
    )

    report = build_discovery_cluster_report(artifact)

    assert report["cluster_count"] == 1
    assert report["candidate_count"] == 2
    assert report["clusters"][0]["cluster_id"] == 0
    assert report["clusters"][0]["top_regions"][0]["value"] == "VISp"
    assert report["clusters"][0]["representative_candidate_id"] == "candidate_0001"


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
    prep_config_path, experiment_config_path = _build_workspace(tmp_path, discovery_has_positive=False)

    try:
        train_main(["--config", str(experiment_config_path), "--data-config", str(prep_config_path)])
    except SystemExit as exc:
        assert exc.code == 0

    checkpoint_path = tmp_path / "artifacts" / "checkpoints" / "pcc_test_best.pt"
    with pytest.raises(ValueError, match="no positive 'stimulus_change' labels"):
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
