from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

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
from predictive_circuit_coding.utils import NotebookStageReporter, get_console, verify_paths_exist


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a synthetic end-to-end preview of the notebook UI and artifact flow."
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/notebook_ui_preview",
        help="Directory to write the synthetic workspace, artifacts, and preview outputs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the output directory first if it already exists.",
    )
    return parser.parse_args(argv)


def _write_preparation_config(output_root: Path) -> Path:
    config_dir = output_root / "configs" / "pcc"
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


def _write_experiment_config(output_root: Path) -> Path:
    config_dir = output_root / "configs" / "pcc"
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
                "  min_cluster_size: 2",
                "  stability_rounds: 3",
                "  shuffle_seed: 19",
                "artifacts:",
                "  checkpoint_dir: ../../artifacts/checkpoints",
                "  summary_path: ../../artifacts/training_summary.json",
                "  checkpoint_prefix: pcc_preview",
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


def _create_synthetic_workspace(output_root: Path) -> tuple[Path, Path, Path]:
    prep_config_path = _write_preparation_config(output_root)
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
    experiment_config_path = _write_experiment_config(output_root)
    return prep_config_path, experiment_config_path, output_root


def _invoke(entrypoint, argv: list[str]) -> None:
    try:
        entrypoint(argv)
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise


def _print_metric_block(title: str, payload: dict[str, object], *, keys: list[str]) -> None:
    console = get_console()
    console.print(f"[bold]{title}[/bold]")
    for key in keys:
        value = payload.get(key)
        console.print(f"  {key}: {value}")


def _print_cluster_preview(cluster_summary_csv: Path) -> None:
    console = get_console()
    console.print("[bold]Cluster summary preview[/bold]")
    with cluster_summary_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        console.print("  No cluster rows found.")
        return
    for row in rows[:3]:
        console.print(
            "  "
            + ", ".join(
                [
                    f"cluster={row.get('cluster_id')}",
                    f"size={row.get('candidate_count')}",
                    f"mean_score={row.get('mean_candidate_score')}",
                    f"top_regions={row.get('top_regions')}",
                    f"top_sessions={row.get('top_sessions')}",
                ]
            )
        )


def _run_preview(output_root: Path) -> None:
    console = get_console()
    reporter = NotebookStageReporter(
        name="synthetic-notebook-preview",
        expected_duration="about 10-30 seconds on CPU",
    )
    reporter.banner(
        "Predictive Circuit Coding Notebook Preview",
        subtitle="Synthetic end-to-end run that mirrors the stage flow and artifact summaries shown in the notebooks.",
    )

    reporter.begin("setup", next_artifact="synthetic prepared sessions + configs")
    prep_config_path, experiment_config_path, root = _create_synthetic_workspace(output_root)
    reporter.finish("setup")

    checkpoint_path = root / "artifacts" / "checkpoints" / "pcc_preview_best.pt"
    evaluation_path = root / "artifacts" / "checkpoints" / "pcc_preview_best_test_evaluation.json"
    discovery_path = root / "artifacts" / "checkpoints" / "pcc_preview_best_discovery_discovery.json"
    decode_coverage_json_path = (
        root / "artifacts" / "checkpoints" / "pcc_preview_best_discovery_discovery_decode_coverage.json"
    )
    cluster_summary_json_path = (
        root / "artifacts" / "checkpoints" / "pcc_preview_best_discovery_discovery_cluster_summary.json"
    )
    cluster_summary_csv_path = (
        root / "artifacts" / "checkpoints" / "pcc_preview_best_discovery_discovery_cluster_summary.csv"
    )
    validation_json_path = (
        root / "artifacts" / "checkpoints" / "pcc_preview_best_discovery_discovery_validation.json"
    )
    validation_csv_path = (
        root / "artifacts" / "checkpoints" / "pcc_preview_best_discovery_discovery_validation.csv"
    )

    reporter.begin("preflight", next_artifact="checkpoint + evaluation/discovery/validation summaries")
    paths_ok = verify_paths_exist(
        {
            "experiment_config": experiment_config_path,
            "data_config": prep_config_path,
        }
    )
    console.print(paths_ok)
    reporter.finish("preflight")

    reporter.begin("training", next_artifact="best checkpoint")
    _invoke(
        train_main,
        ["--config", str(experiment_config_path), "--data-config", str(prep_config_path)],
    )
    reporter.note_checkpoint(checkpoint_path)
    reporter.finish("training")

    reporter.begin("evaluation", next_artifact="evaluation summary json")
    _invoke(
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
    )
    reporter.finish("evaluation")

    reporter.begin("discovery", next_artifact="discovery artifact + cluster summary")
    _invoke(
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
    )
    reporter.finish("discovery")

    reporter.begin("validation", next_artifact="validation summary json/csv")
    _invoke(
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
    )
    reporter.finish("validation")

    evaluation_payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    discovery_payload = json.loads(discovery_path.read_text(encoding="utf-8"))
    decode_coverage_payload = json.loads(decode_coverage_json_path.read_text(encoding="utf-8"))
    cluster_summary_payload = json.loads(cluster_summary_json_path.read_text(encoding="utf-8"))
    validation_payload = json.loads(validation_json_path.read_text(encoding="utf-8"))

    console.print()
    _print_metric_block(
        "Evaluation preview",
        evaluation_payload.get("metrics", {}),
        keys=[
            "predictive_loss",
            "reconstruction_loss",
            "predictive_raw_mse",
            "predictive_baseline_mse",
            "predictive_improvement",
        ],
    )
    _print_metric_block(
        "Discovery preview",
        {
            "scanned_windows": decode_coverage_payload.get("total_scanned_windows"),
            "selected_positive": decode_coverage_payload.get("selected_positive_count"),
            "selected_negative": decode_coverage_payload.get("selected_negative_count"),
            "candidate_count": len(discovery_payload.get("candidates", [])),
            "cluster_count": cluster_summary_payload.get("cluster_count"),
            "probe_accuracy": discovery_payload.get("decoder_summary", {}).get("metrics", {}).get("probe_accuracy"),
        },
        keys=["scanned_windows", "selected_positive", "selected_negative", "candidate_count", "cluster_count", "probe_accuracy"],
    )
    _print_metric_block(
        "Validation preview",
        {
            "candidate_count": validation_payload.get("candidate_count"),
            "cluster_count": validation_payload.get("cluster_count"),
            "real_probe_accuracy": validation_payload.get("real_label_metrics", {}).get("probe_accuracy"),
            "shuffled_probe_accuracy": validation_payload.get("shuffled_label_metrics", {}).get("probe_accuracy"),
            "held_out_test_probe_accuracy": validation_payload.get("held_out_test_metrics", {}).get("probe_accuracy"),
            "held_out_similarity_roc_auc": validation_payload.get("held_out_similarity_summary", {}).get("window_roc_auc"),
            "held_out_similarity_pr_auc": validation_payload.get("held_out_similarity_summary", {}).get("window_pr_auc"),
        },
        keys=[
            "candidate_count",
            "cluster_count",
            "real_probe_accuracy",
            "shuffled_probe_accuracy",
            "held_out_test_probe_accuracy",
            "held_out_similarity_roc_auc",
            "held_out_similarity_pr_auc",
        ],
    )
    _print_cluster_preview(cluster_summary_csv_path)

    console.print()
    console.print("[bold]Preview artifacts[/bold]")
    console.print(f"  workspace_root: {root}")
    console.print(f"  checkpoint: {checkpoint_path}")
    console.print(f"  evaluation_summary: {evaluation_path}")
    console.print(f"  discovery_artifact: {discovery_path}")
    console.print(f"  decode_coverage_json: {decode_coverage_json_path}")
    console.print(f"  cluster_summary_json: {cluster_summary_json_path}")
    console.print(f"  cluster_summary_csv: {cluster_summary_csv_path}")
    console.print(f"  validation_json: {validation_json_path}")
    console.print(f"  validation_csv: {validation_csv_path}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        if not args.force:
            raise SystemExit(
                f"Output root already exists: {output_root}. Re-run with --force to replace the previous preview."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _run_preview(output_root)


if __name__ == "__main__":
    main()
