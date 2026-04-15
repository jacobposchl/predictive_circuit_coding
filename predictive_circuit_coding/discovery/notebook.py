from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from predictive_circuit_coding.benchmarks.reports import write_single_row_csv, write_summary_rows
from predictive_circuit_coding.discovery.reporting import (
    discovery_cluster_report_paths,
    discovery_coverage_summary_path,
)
from predictive_circuit_coding.training.contracts import write_json_payload
from predictive_circuit_coding.validation.notebook import (
    NotebookValidationRunResult,
    default_validation_output_paths as _default_validation_output_paths,
    flatten_comparison_validation_summary as _flatten_comparison_validation_summary,
)
from predictive_circuit_coding.workflows.notebook_runtime import resolve_notebook_training_export_path


def _sanitize_notebook_export_segment(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    sanitized = sanitized.strip(".-_")
    return sanitized or "default"


def build_notebook_discovery_export_path(
    *,
    drive_export_root: str | Path,
    run_id: str,
    decode_type: str,
    attempt_timestamp: str,
    run_name: str = "run_1",
) -> Path:
    attempt_name = f"{_sanitize_notebook_export_segment(decode_type)}__{attempt_timestamp}"
    return Path(drive_export_root) / str(run_id) / run_name / "discovery" / attempt_name


def export_notebook_discovery_artifacts(
    *,
    drive_export_root: str | Path,
    local_artifact_root: str | Path,
    run_id: str,
    decode_type: str,
    attempt_timestamp: str,
    runtime_experiment_config: str | Path,
    checkpoint_path: str | Path,
    discovery_run: "NotebookDiscoveryRunResult",
    validation_run: "NotebookValidationRunResult | None" = None,
    run_name: str = "run_1",
) -> Path:
    artifact_root = Path(local_artifact_root).resolve()
    target = build_notebook_discovery_export_path(
        drive_export_root=drive_export_root,
        run_id=run_id,
        decode_type=decode_type,
        attempt_timestamp=attempt_timestamp,
        run_name=run_name,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    export_candidates = [
        discovery_run.discovery_artifact_path,
        discovery_run.decode_coverage_summary_path,
        discovery_run.cluster_summary_json_path,
        discovery_run.cluster_summary_csv_path,
    ]
    if validation_run is not None:
        export_candidates.extend(
            [
                validation_run.validation_summary_json_path,
                validation_run.validation_summary_csv_path,
            ]
        )

    for src_candidate in export_candidates:
        src = Path(src_candidate)
        if not src.exists():
            continue
        try:
            relative_path = src.resolve().relative_to(artifact_root)
        except ValueError:
            relative_path = Path(src.name)
        destination = target / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, destination)

    runtime_copy = target / Path(runtime_experiment_config).name
    runtime_copy.write_text(Path(runtime_experiment_config).read_text(encoding="utf-8"), encoding="utf-8")
    metadata_path = target / "discovery_export_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "training_run_id": str(run_id),
                "run_name": str(run_name),
                "decode_type": str(decode_type),
                "checkpoint_name": Path(checkpoint_path).name,
                "local_discovery_runtime_config_path": str(Path(runtime_experiment_config).resolve()),
                "exported_discovery_runtime_config_path": str(runtime_copy),
                "attempt_timestamp": str(attempt_timestamp),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return target


def export_notebook_discovery_comparison_artifacts(
    *,
    drive_export_root: str | Path,
    local_artifact_root: str | Path,
    run_id: str,
    decode_type: str,
    attempt_timestamp: str,
    run_name: str = "run_1",
) -> Path:
    source_root = build_notebook_discovery_comparison_local_root(
        local_artifact_root=local_artifact_root,
        decode_type=decode_type,
    ).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Local discovery comparison root not found: {source_root}")
    target = build_notebook_discovery_export_path(
        drive_export_root=drive_export_root,
        run_id=run_id,
        decode_type=decode_type,
        attempt_timestamp=attempt_timestamp,
        run_name=run_name,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source_root, target)
    return target


def restore_latest_discovery_artifacts(
    *,
    drive_export_root: str | Path,
    local_artifact_root: str | Path,
    training_run_id: str | None = None,
    decode_type: str | None = None,
    run_name: str = "run_1",
) -> Path | None:
    train_export_path = resolve_notebook_training_export_path(
        drive_export_root=drive_export_root,
        training_run_id=training_run_id,
        run_name=run_name,
    )
    if train_export_path is None:
        return None
    discovery_root = train_export_path.parent / "discovery"
    if not discovery_root.is_dir():
        return None

    decode_prefix = None
    if decode_type is not None:
        decode_prefix = f"{_sanitize_notebook_export_segment(decode_type)}__"

    run_candidates = sorted(
        [
            path
            for path in discovery_root.iterdir()
            if path.is_dir() and (decode_prefix is None or path.name.startswith(decode_prefix))
        ],
        key=lambda path: path.name,
    )
    if not run_candidates:
        return None

    latest_run = run_candidates[-1]
    artifact_root = Path(local_artifact_root)
    for src_file in latest_run.rglob("*"):
        if not src_file.is_file():
            continue
        relative_path = src_file.relative_to(latest_run)
        if not relative_path.parts or relative_path.parts[0] != "checkpoints":
            continue
        dst = artifact_root / relative_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst)
    return latest_run


@dataclass(frozen=True)
class NotebookDiscoveryRunResult:
    discovery_artifact_path: Path
    decode_coverage_summary_path: Path
    cluster_summary_json_path: Path
    cluster_summary_csv_path: Path


@dataclass(frozen=True)
class NotebookDiscoveryComparisonArmPaths:
    arm_name: str
    arm_root: Path
    runtime_config_copy_path: Path
    discovery_artifact_path: Path
    cluster_summary_json_path: Path
    cluster_summary_csv_path: Path
    validation_summary_json_path: Path
    validation_summary_csv_path: Path
    transform_summary_json_path: Path
    transform_summary_csv_path: Path
    arm_metadata_json_path: Path


@dataclass(frozen=True)
class NotebookDiscoveryComparisonPaths:
    decode_type: str
    comparison_root: Path
    shared_runtime_experiment_config_path: Path
    decode_coverage_summary_path: Path
    comparison_summary_json_path: Path
    comparison_summary_csv_path: Path
    comparison_metadata_json_path: Path
    baseline: NotebookDiscoveryComparisonArmPaths
    whitening_only: NotebookDiscoveryComparisonArmPaths
    whitening_plus_held_out_alignment: NotebookDiscoveryComparisonArmPaths


@dataclass(frozen=True)
class NotebookDiscoveryComparisonArmResult:
    arm_name: str
    discovery_run: NotebookDiscoveryRunResult
    validation_run: NotebookValidationRunResult
    transform_summary_json_path: Path
    transform_summary_csv_path: Path
    runtime_config_copy_path: Path
    arm_metadata_json_path: Path


@dataclass(frozen=True)
class NotebookDiscoveryComparisonRunResult:
    decode_coverage_summary_path: Path
    comparison_summary_json_path: Path
    comparison_summary_csv_path: Path
    comparison_metadata_json_path: Path
    baseline: NotebookDiscoveryComparisonArmResult
    whitening_only: NotebookDiscoveryComparisonArmResult
    whitening_plus_held_out_alignment: NotebookDiscoveryComparisonArmResult


def build_notebook_discovery_comparison_local_root(
    *,
    local_artifact_root: str | Path,
    decode_type: str,
) -> Path:
    return Path(local_artifact_root) / "discovery_compare" / _sanitize_notebook_export_segment(decode_type)


def _build_notebook_discovery_comparison_arm_paths(
    *,
    comparison_root: Path,
    checkpoint_path: str | Path,
    arm_name: str,
    split_name: str = "discovery",
) -> NotebookDiscoveryComparisonArmPaths:
    arm_root = comparison_root / arm_name
    checkpoints_root = arm_root / "checkpoints"
    checkpoint_stem = Path(checkpoint_path).stem
    discovery_artifact_path = checkpoints_root / f"{checkpoint_stem}_{split_name}_discovery.json"
    validation_summary_json_path, validation_summary_csv_path = _default_validation_output_paths(discovery_artifact_path)
    cluster_summary_json_path, cluster_summary_csv_path = discovery_cluster_report_paths(discovery_artifact_path)
    return NotebookDiscoveryComparisonArmPaths(
        arm_name=arm_name,
        arm_root=arm_root,
        runtime_config_copy_path=arm_root / "colab_runtime_experiment.yaml",
        discovery_artifact_path=discovery_artifact_path,
        cluster_summary_json_path=cluster_summary_json_path,
        cluster_summary_csv_path=cluster_summary_csv_path,
        validation_summary_json_path=validation_summary_json_path,
        validation_summary_csv_path=validation_summary_csv_path,
        transform_summary_json_path=arm_root / "transform_summary.json",
        transform_summary_csv_path=arm_root / "transform_summary.csv",
        arm_metadata_json_path=arm_root / "comparison_arm_metadata.json",
    )


def build_notebook_discovery_comparison_paths(
    *,
    local_artifact_root: str | Path,
    checkpoint_path: str | Path,
    decode_type: str,
    split_name: str = "discovery",
) -> NotebookDiscoveryComparisonPaths:
    comparison_root = build_notebook_discovery_comparison_local_root(
        local_artifact_root=local_artifact_root,
        decode_type=decode_type,
    )
    return NotebookDiscoveryComparisonPaths(
        decode_type=str(decode_type),
        comparison_root=comparison_root,
        shared_runtime_experiment_config_path=comparison_root / "colab_runtime_experiment.yaml",
        decode_coverage_summary_path=comparison_root / "decode_coverage_summary.json",
        comparison_summary_json_path=comparison_root / "comparison_summary.json",
        comparison_summary_csv_path=comparison_root / "comparison_summary.csv",
        comparison_metadata_json_path=comparison_root / "comparison_metadata.json",
        baseline=_build_notebook_discovery_comparison_arm_paths(
            comparison_root=comparison_root,
            checkpoint_path=checkpoint_path,
            arm_name="baseline",
            split_name=split_name,
        ),
        whitening_only=_build_notebook_discovery_comparison_arm_paths(
            comparison_root=comparison_root,
            checkpoint_path=checkpoint_path,
            arm_name="whitening_only",
            split_name=split_name,
        ),
        whitening_plus_held_out_alignment=_build_notebook_discovery_comparison_arm_paths(
            comparison_root=comparison_root,
            checkpoint_path=checkpoint_path,
            arm_name="whitening_plus_held_out_alignment",
            split_name=split_name,
        ),
    )


def _default_discovery_output_path(checkpoint_path: str | Path, split_name: str) -> Path:
    checkpoint = Path(checkpoint_path)
    return checkpoint.with_name(f"{checkpoint.stem}_{split_name}_discovery.json")


def _load_discovery_target_label(discovery_artifact_path: str | Path) -> str | None:
    artifact_path = Path(discovery_artifact_path)
    if not artifact_path.exists():
        return None
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    decoder_summary = payload.get("decoder_summary") or {}
    target_label = decoder_summary.get("target_label")
    if isinstance(target_label, str) and target_label.strip():
        return target_label
    config_snapshot = payload.get("config_snapshot") or {}
    discovery_config = config_snapshot.get("discovery") or {}
    target_label = discovery_config.get("target_label")
    if isinstance(target_label, str) and target_label.strip():
        return target_label
    return None


def _load_discovery_config_snapshot(discovery_artifact_path: str | Path) -> dict | None:
    artifact_path = Path(discovery_artifact_path)
    if not artifact_path.exists():
        return None
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    config_snapshot = payload.get("config_snapshot")
    return config_snapshot if isinstance(config_snapshot, dict) else None


def _normalize_discovery_reuse_config(payload: dict) -> dict:
    return {
        "dataset_id": payload.get("dataset_id"),
        "split_name": payload.get("split_name"),
        "data_runtime": payload.get("data_runtime") or {},
        "execution": payload.get("execution") or {},
        "evaluation": payload.get("evaluation") or {},
        "discovery": payload.get("discovery") or {},
        "runtime_subset": payload.get("runtime_subset") or {},
        "splits": payload.get("splits") or {},
    }


def _discovery_artifact_matches_runtime_config(
    *,
    discovery_artifact_path: str | Path,
    experiment_config_path: str | Path,
) -> bool:
    artifact_config_snapshot = _load_discovery_config_snapshot(discovery_artifact_path)
    if artifact_config_snapshot is None:
        return False
    runtime_payload = yaml.safe_load(Path(experiment_config_path).read_text(encoding="utf-8")) or {}
    return _normalize_discovery_reuse_config(artifact_config_snapshot) == _normalize_discovery_reuse_config(
        runtime_payload
    )


def find_existing_discovery_run(
    *,
    checkpoint_path: str | Path,
    split_name: str = "discovery",
    target_label: str | None = None,
    experiment_config_path: str | Path | None = None,
    discovery_output_path: str | Path | None = None,
) -> "NotebookDiscoveryRunResult | None":
    """Return a NotebookDiscoveryRunResult if all expected discovery artifact files exist locally,
    otherwise return None so the caller can decide whether to re-run discovery."""
    discovery_path = (
        Path(discovery_output_path)
        if discovery_output_path is not None
        else _default_discovery_output_path(checkpoint_path, split_name)
    )
    if not discovery_path.exists():
        return None
    coverage_path = discovery_coverage_summary_path(discovery_path)
    cluster_json, cluster_csv = discovery_cluster_report_paths(discovery_path)
    if not all(p.exists() for p in (coverage_path, cluster_json, cluster_csv)):
        return None
    if target_label is not None:
        existing_target_label = _load_discovery_target_label(discovery_path)
        if existing_target_label != str(target_label):
            return None
    if experiment_config_path is not None and not _discovery_artifact_matches_runtime_config(
        discovery_artifact_path=discovery_path,
        experiment_config_path=experiment_config_path,
    ):
        return None
    return NotebookDiscoveryRunResult(
        discovery_artifact_path=discovery_path,
        decode_coverage_summary_path=coverage_path,
        cluster_summary_json_path=cluster_json,
        cluster_summary_csv_path=cluster_csv,
    )


def run_notebook_discovery(
    *,
    experiment_config_path: str | Path,
    data_config_path: str | Path,
    checkpoint_path: str | Path,
    split_name: str = "discovery",
    output_path: str | Path | None = None,
    progress_ui: bool = True,
) -> NotebookDiscoveryRunResult:
    from tqdm.auto import tqdm

    from predictive_circuit_coding.cli.common import (
        require_checkpoint_matches_dataset,
        require_non_empty_split,
        require_runtime_view,
    )
    from predictive_circuit_coding.discovery import (
        build_discovery_cluster_report,
        discover_motifs_from_plan,
        prepare_discovery_collection,
        write_discovery_artifact,
        write_discovery_cluster_report_csv,
        write_discovery_cluster_report_json,
        write_discovery_coverage_summary,
    )
    from predictive_circuit_coding.training import load_experiment_config

    config = load_experiment_config(experiment_config_path)
    dataset_view = require_runtime_view(experiment_config=config, data_config_path=data_config_path)
    require_non_empty_split(dataset_view=dataset_view, split_name=split_name)
    checkpoint = require_checkpoint_matches_dataset(checkpoint_path=checkpoint_path, dataset_id=config.dataset_id)
    discovery_output_path = Path(output_path) if output_path is not None else _default_discovery_output_path(checkpoint, split_name)
    coverage_path = discovery_coverage_summary_path(discovery_output_path)
    cluster_json_path, cluster_csv_path = discovery_cluster_report_paths(discovery_output_path)

    plan_bar = tqdm(total=0, desc="Discovery coverage scan", unit="window", leave=False, disable=not progress_ui)

    def _plan_progress(current: int, total: int | None) -> None:
        if total is not None and plan_bar.total != total:
            plan_bar.total = total
        plan_bar.n = current
        plan_bar.refresh()

    window_plan = prepare_discovery_collection(
        experiment_config=config,
        data_config_path=data_config_path,
        split_name=split_name,
        dataset_view=dataset_view,
        progress_callback=_plan_progress if progress_ui else None,
    )
    plan_bar.close()
    write_discovery_coverage_summary(window_plan.coverage_summary, coverage_path)

    encode_bar = tqdm(
        total=int(window_plan.coverage_summary.selected_positive_count + window_plan.coverage_summary.selected_negative_count),
        desc="Selected-window discovery",
        unit="window",
        leave=False,
        disable=not progress_ui,
    )

    def _encode_progress(current: int, total: int | None) -> None:
        if total is not None and encode_bar.total != total:
            encode_bar.total = total
        encode_bar.n = current
        encode_bar.refresh()

    result = discover_motifs_from_plan(
        experiment_config=config,
        data_config_path=data_config_path,
        checkpoint_path=checkpoint,
        split_name=split_name,
        window_plan=window_plan,
        dataset_view=dataset_view,
        progress_callback=_encode_progress if progress_ui else None,
    )
    encode_bar.close()
    artifact = result.artifact
    write_discovery_artifact(artifact, discovery_output_path)
    cluster_report = build_discovery_cluster_report(artifact)
    write_discovery_cluster_report_json(cluster_report, cluster_json_path)
    write_discovery_cluster_report_csv(cluster_report, cluster_csv_path)

    return NotebookDiscoveryRunResult(
        discovery_artifact_path=discovery_output_path,
        decode_coverage_summary_path=coverage_path,
        cluster_summary_json_path=cluster_json_path,
        cluster_summary_csv_path=cluster_csv_path,
    )


def _flatten_transform_summary(summary: dict[str, object]) -> dict[str, object]:
    aggregate_metrics = summary.get("aggregate_metrics") or {}
    return {
        "transform_type": summary.get("transform_type"),
        "session_count": summary.get("session_count"),
        "reference_session_id": summary.get("reference_session_id"),
        "mean_label_axis_cosine_before": aggregate_metrics.get("mean_label_axis_cosine_before"),
        "mean_label_axis_cosine_after": aggregate_metrics.get("mean_label_axis_cosine_after"),
        "mean_positive_centroid_cosine_before": aggregate_metrics.get("mean_positive_centroid_cosine_before"),
        "mean_positive_centroid_cosine_after": aggregate_metrics.get("mean_positive_centroid_cosine_after"),
        "mean_negative_centroid_cosine_before": aggregate_metrics.get("mean_negative_centroid_cosine_before"),
        "mean_negative_centroid_cosine_after": aggregate_metrics.get("mean_negative_centroid_cosine_after"),
        "mean_anchor_rmse_after_alignment": aggregate_metrics.get("mean_anchor_rmse_after_alignment"),
        "excluded_session_count": sum(1 for row in summary.get("sessions") or () if not bool(row.get("included", True))),
    }


def _comparison_arm_metadata_payload(
    *,
    arm_name: str,
    transform_mode: str,
    session_holdout_fraction: float,
    session_holdout_seed: int,
    run_standard_test_validation: bool,
    experiment_config_path: str | Path,
) -> dict[str, object]:
    runtime_payload = yaml.safe_load(Path(experiment_config_path).read_text(encoding="utf-8")) or {}
    return {
        "comparison_protocol": "within_session_held_out_v1",
        "arm_name": str(arm_name),
        "transform_mode": str(transform_mode),
        "session_holdout_fraction": float(session_holdout_fraction),
        "session_holdout_seed": int(session_holdout_seed),
        "run_standard_test_validation": bool(run_standard_test_validation),
        "runtime_reuse_key": _normalize_discovery_reuse_config(runtime_payload),
    }


def _comparison_arm_paths_by_name(
    paths: NotebookDiscoveryComparisonPaths,
) -> dict[str, NotebookDiscoveryComparisonArmPaths]:
    return {
        "baseline": paths.baseline,
        "whitening_only": paths.whitening_only,
        "whitening_plus_held_out_alignment": paths.whitening_plus_held_out_alignment,
    }


def _comparison_arm_transform_mode(arm_name: str) -> str:
    return {
        "baseline": "baseline",
        "whitening_only": "whitening_only",
        "whitening_plus_held_out_alignment": "whitening_plus_held_out_alignment",
    }[str(arm_name)]


def _existing_comparison_arm_is_usable(
    *,
    arm_paths: NotebookDiscoveryComparisonArmPaths,
    experiment_config_path: str | Path,
    target_label: str,
    expected_metadata: dict[str, object],
) -> bool:
    discovery_run = find_existing_discovery_run(
        checkpoint_path=arm_paths.discovery_artifact_path,
        target_label=target_label,
        experiment_config_path=experiment_config_path,
        discovery_output_path=arm_paths.discovery_artifact_path,
    )
    if discovery_run is None:
        return False
    expected_paths = (
        arm_paths.validation_summary_json_path,
        arm_paths.validation_summary_csv_path,
        arm_paths.transform_summary_json_path,
        arm_paths.transform_summary_csv_path,
        arm_paths.arm_metadata_json_path,
        arm_paths.runtime_config_copy_path,
    )
    if not all(path.is_file() for path in expected_paths):
        return False
    metadata_payload = json.loads(arm_paths.arm_metadata_json_path.read_text(encoding="utf-8"))
    return metadata_payload == expected_metadata


def build_notebook_discovery_comparison_summary_row(
    *,
    arm_name: str,
    discovery_artifact_path: str | Path,
    validation_summary_path: str | Path,
    cluster_summary_path: str | Path,
    transform_summary_path: str | Path,
) -> dict[str, object]:
    discovery_payload = json.loads(Path(discovery_artifact_path).read_text(encoding="utf-8"))
    validation_payload = json.loads(Path(validation_summary_path).read_text(encoding="utf-8"))
    cluster_payload = json.loads(Path(cluster_summary_path).read_text(encoding="utf-8"))
    transform_payload = json.loads(Path(transform_summary_path).read_text(encoding="utf-8"))
    discovery_fit = validation_payload.get("discovery_fit_metrics") or {}
    shuffled_fit = validation_payload.get("shuffled_fit_metrics") or {}
    primary_held_out = validation_payload.get("primary_held_out_metrics") or {}
    primary_similarity = validation_payload.get("primary_held_out_similarity_summary") or {}
    cluster_quality = validation_payload.get("cluster_quality_summary") or {}
    standard_validation = validation_payload.get("standard_test_validation") or {}
    standard_test_metrics = standard_validation.get("held_out_test_metrics") or {}
    standard_test_similarity = standard_validation.get("held_out_similarity_summary") or {}
    candidate_selection = validation_payload.get("candidate_selection_summary") or {}
    arm_shard_debug = candidate_selection.get("arm_shard_debug") or {}
    return {
        "arm_name": str(arm_name),
        "target_label": discovery_payload.get("decoder_summary", {}).get("target_label"),
        "comparison_status": validation_payload.get("comparison_status"),
        "failure_reason": validation_payload.get("failure_reason"),
        "candidate_count": validation_payload.get("candidate_count"),
        "cluster_count": validation_payload.get("cluster_count"),
        "candidate_selection_fallback_used": candidate_selection.get("fallback_used"),
        "candidate_selection_effective_min_score": candidate_selection.get("effective_min_score"),
        "candidate_selection_precluster_candidate_count": candidate_selection.get("precluster_candidate_count"),
        "debug_allowed_fit_window_key_count": arm_shard_debug.get("allowed_fit_window_key_count"),
        "debug_fit_positive_window_count": arm_shard_debug.get("fit_positive_window_count"),
        "debug_fit_negative_window_count": arm_shard_debug.get("fit_negative_window_count"),
        "debug_shard_file_count": arm_shard_debug.get("shard_file_count"),
        "debug_shard_token_row_count": arm_shard_debug.get("token_row_count"),
        "debug_shard_positive_token_row_count": arm_shard_debug.get("positive_token_row_count"),
        "debug_shard_negative_token_row_count": arm_shard_debug.get("negative_token_row_count"),
        "debug_shard_window_row_count": arm_shard_debug.get("window_row_count"),
        "debug_shard_positive_window_row_count": arm_shard_debug.get("positive_window_row_count"),
        "discovery_probe_accuracy": discovery_fit.get("probe_accuracy"),
        "discovery_probe_bce": discovery_fit.get("probe_bce"),
        "shuffled_probe_accuracy": shuffled_fit.get("probe_accuracy"),
        "shuffled_probe_bce": shuffled_fit.get("probe_bce"),
        "primary_within_session_held_out_probe_accuracy": primary_held_out.get("probe_accuracy"),
        "primary_within_session_held_out_probe_bce": primary_held_out.get("probe_bce"),
        "primary_within_session_held_out_probe_roc_auc": primary_held_out.get("probe_roc_auc"),
        "primary_within_session_held_out_probe_pr_auc": primary_held_out.get("probe_pr_auc"),
        "primary_within_session_held_out_similarity_roc_auc": primary_similarity.get("window_roc_auc"),
        "primary_within_session_held_out_similarity_pr_auc": primary_similarity.get("window_pr_auc"),
        "cluster_persistence_mean": cluster_quality.get("cluster_persistence_mean"),
        "silhouette_score": cluster_quality.get("silhouette_score"),
        "excluded_session_count": len(validation_payload.get("excluded_sessions") or ()),
        "standard_test_probe_accuracy": standard_test_metrics.get("probe_accuracy"),
        "standard_test_probe_bce": standard_test_metrics.get("probe_bce"),
        "standard_test_similarity_roc_auc": standard_test_similarity.get("window_roc_auc"),
        "standard_test_similarity_pr_auc": standard_test_similarity.get("window_pr_auc"),
        "reference_session_id": validation_payload.get("reference_session_id"),
        "transform_type": transform_payload.get("transform_type"),
        "cluster_summary_path": str(Path(cluster_summary_path)),
        "validation_summary_path": str(Path(validation_summary_path)),
    }


def run_notebook_discovery_representation_comparison(
    *,
    experiment_config_path: str | Path,
    data_config_path: str | Path,
    checkpoint_path: str | Path,
    local_artifact_root: str | Path,
    decode_type: str,
    session_holdout_fraction: float,
    session_holdout_seed: int,
    run_standard_test_validation: bool,
    split_name: str = "discovery",
    progress_ui: bool = True,
) -> NotebookDiscoveryComparisonRunResult:
    from tqdm.auto import tqdm

    from predictive_circuit_coding.cli.common import (
        require_checkpoint_matches_dataset,
        require_non_empty_split,
        require_runtime_view,
    )
    from predictive_circuit_coding.decoding.extract import extract_selected_discovery_windows
    from predictive_circuit_coding.discovery import (
        run_representation_comparison_from_encoded,
        write_discovery_artifact,
        write_discovery_cluster_report_csv,
        write_discovery_cluster_report_json,
        write_discovery_coverage_summary,
    )
    from predictive_circuit_coding.training import load_experiment_config

    config = load_experiment_config(experiment_config_path)
    dataset_view = require_runtime_view(experiment_config=config, data_config_path=data_config_path)
    require_non_empty_split(dataset_view=dataset_view, split_name=split_name)
    checkpoint = require_checkpoint_matches_dataset(checkpoint_path=checkpoint_path, dataset_id=config.dataset_id)
    comparison_paths = build_notebook_discovery_comparison_paths(
        local_artifact_root=local_artifact_root,
        checkpoint_path=checkpoint,
        decode_type=decode_type,
        split_name=split_name,
    )
    comparison_paths.comparison_root.mkdir(parents=True, exist_ok=True)
    comparison_paths.shared_runtime_experiment_config_path.write_text(
        Path(experiment_config_path).read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    arm_paths_by_name = _comparison_arm_paths_by_name(comparison_paths)
    expected_metadata_by_arm = {
        arm_name: _comparison_arm_metadata_payload(
            arm_name=arm_name,
            transform_mode=_comparison_arm_transform_mode(arm_name),
            session_holdout_fraction=session_holdout_fraction,
            session_holdout_seed=session_holdout_seed,
            run_standard_test_validation=run_standard_test_validation,
            experiment_config_path=comparison_paths.shared_runtime_experiment_config_path,
        )
        for arm_name in arm_paths_by_name
    }
    arms_to_run = [
        arm_name
        for arm_name, arm_paths in arm_paths_by_name.items()
        if not _existing_comparison_arm_is_usable(
            arm_paths=arm_paths,
            experiment_config_path=comparison_paths.shared_runtime_experiment_config_path,
            target_label=str(config.discovery.target_label),
            expected_metadata=expected_metadata_by_arm[arm_name],
        )
    ]

    metadata_payload: dict[str, object] = {}
    if arms_to_run:
        from predictive_circuit_coding.discovery import prepare_discovery_collection

        plan_bar = tqdm(total=0, desc="Discovery comparison scan", unit="window", leave=False, disable=not progress_ui)

        def _plan_progress(current: int, total: int | None) -> None:
            if total is not None and plan_bar.total != total:
                plan_bar.total = total
            plan_bar.n = current
            plan_bar.refresh()

        window_plan = prepare_discovery_collection(
            experiment_config=config,
            data_config_path=data_config_path,
            split_name=split_name,
            dataset_view=dataset_view,
            progress_callback=_plan_progress if progress_ui else None,
        )
        plan_bar.close()
        write_discovery_coverage_summary(window_plan.coverage_summary, comparison_paths.decode_coverage_summary_path)

        shard_dir = comparison_paths.comparison_root / "_tmp_shared_token_shards"
        encode_bar = tqdm(
            total=int(window_plan.selected_indices.numel()),
            desc="Discovery comparison encoding",
            unit="window",
            leave=False,
            disable=not progress_ui,
        )

        def _encode_progress(current: int, total: int | None) -> None:
            if total is not None and encode_bar.total != total:
                encode_bar.total = total
            encode_bar.n = current
            encode_bar.refresh()

        encoded = extract_selected_discovery_windows(
            experiment_config=config,
            data_config_path=data_config_path,
            checkpoint_path=checkpoint,
            window_plan=window_plan,
            dataset_view=dataset_view,
            shard_dir=shard_dir,
            progress_callback=_encode_progress if progress_ui else None,
        )
        encode_bar.close()

        comparison_run = run_representation_comparison_from_encoded(
            experiment_config=config,
            data_config_path=data_config_path,
            checkpoint_path=checkpoint,
            split_name=split_name,
            window_plan=window_plan,
            encoded=encoded,
            session_holdout_fraction=session_holdout_fraction,
            session_holdout_seed=session_holdout_seed,
            run_standard_test_validation=run_standard_test_validation,
            temporary_root=comparison_paths.comparison_root / "_tmp_arm_shards",
            arm_names=tuple(arms_to_run),
        )
        if shard_dir.exists():
            shutil.rmtree(shard_dir)
        temp_arm_root = comparison_paths.comparison_root / "_tmp_arm_shards"
        if temp_arm_root.exists():
            shutil.rmtree(temp_arm_root)

        metadata_payload = {
            "comparison_protocol": "within_session_held_out_v1",
            "decode_type": str(decode_type),
            "session_holdout_fraction": float(session_holdout_fraction),
            "session_holdout_seed": int(session_holdout_seed),
            "run_standard_test_validation": bool(run_standard_test_validation),
            "coverage_summary": comparison_run.coverage_summary.to_dict(),
            "split_summary": comparison_run.split_summary,
        }
        write_json_payload(metadata_payload, comparison_paths.comparison_metadata_json_path)

        for arm_result in comparison_run.arm_results:
            arm_paths = arm_paths_by_name[arm_result.arm_name]
            arm_paths.arm_root.mkdir(parents=True, exist_ok=True)
            write_discovery_artifact(arm_result.artifact, arm_paths.discovery_artifact_path)
            write_discovery_cluster_report_json(arm_result.cluster_report, arm_paths.cluster_summary_json_path)
            write_discovery_cluster_report_csv(arm_result.cluster_report, arm_paths.cluster_summary_csv_path)
            write_json_payload(arm_result.validation_summary, arm_paths.validation_summary_json_path)
            write_single_row_csv(
                _flatten_comparison_validation_summary(arm_result.validation_summary),
                arm_paths.validation_summary_csv_path,
            )
            write_json_payload(arm_result.transform_summary or {}, arm_paths.transform_summary_json_path)
            write_single_row_csv(
                _flatten_transform_summary(arm_result.transform_summary or {}),
                arm_paths.transform_summary_csv_path,
            )
            arm_paths.runtime_config_copy_path.write_text(
                comparison_paths.shared_runtime_experiment_config_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            write_json_payload(expected_metadata_by_arm[arm_result.arm_name], arm_paths.arm_metadata_json_path)
    else:
        if comparison_paths.comparison_metadata_json_path.is_file():
            metadata_payload = json.loads(comparison_paths.comparison_metadata_json_path.read_text(encoding="utf-8"))

    if not comparison_paths.decode_coverage_summary_path.is_file():
        coverage_payload = metadata_payload.get("coverage_summary") if metadata_payload else None
        if isinstance(coverage_payload, dict):
            write_json_payload(coverage_payload, comparison_paths.decode_coverage_summary_path)

    summary_rows = [
        build_notebook_discovery_comparison_summary_row(
            arm_name=arm_name,
            discovery_artifact_path=arm_paths.discovery_artifact_path,
            validation_summary_path=arm_paths.validation_summary_json_path,
            cluster_summary_path=arm_paths.cluster_summary_json_path,
            transform_summary_path=arm_paths.transform_summary_json_path,
        )
        for arm_name, arm_paths in arm_paths_by_name.items()
    ]
    write_summary_rows(
        summary_rows,
        output_json_path=comparison_paths.comparison_summary_json_path,
        output_csv_path=comparison_paths.comparison_summary_csv_path,
        root_key="experiments",
    )

    def _arm_result(arm_name: str) -> NotebookDiscoveryComparisonArmResult:
        arm_paths = arm_paths_by_name[arm_name]
        return NotebookDiscoveryComparisonArmResult(
            arm_name=arm_name,
            discovery_run=NotebookDiscoveryRunResult(
                discovery_artifact_path=arm_paths.discovery_artifact_path,
                decode_coverage_summary_path=comparison_paths.decode_coverage_summary_path,
                cluster_summary_json_path=arm_paths.cluster_summary_json_path,
                cluster_summary_csv_path=arm_paths.cluster_summary_csv_path,
            ),
            validation_run=NotebookValidationRunResult(
                validation_summary_json_path=arm_paths.validation_summary_json_path,
                validation_summary_csv_path=arm_paths.validation_summary_csv_path,
            ),
            transform_summary_json_path=arm_paths.transform_summary_json_path,
            transform_summary_csv_path=arm_paths.transform_summary_csv_path,
            runtime_config_copy_path=arm_paths.runtime_config_copy_path,
            arm_metadata_json_path=arm_paths.arm_metadata_json_path,
        )

    return NotebookDiscoveryComparisonRunResult(
        decode_coverage_summary_path=comparison_paths.decode_coverage_summary_path,
        comparison_summary_json_path=comparison_paths.comparison_summary_json_path,
        comparison_summary_csv_path=comparison_paths.comparison_summary_csv_path,
        comparison_metadata_json_path=comparison_paths.comparison_metadata_json_path,
        baseline=_arm_result("baseline"),
        whitening_only=_arm_result("whitening_only"),
        whitening_plus_held_out_alignment=_arm_result("whitening_plus_held_out_alignment"),
    )


def collect_notebook_target_value_counts(
    *,
    experiment_config_path: str | Path,
    data_config_path: str | Path,
    split_name: str,
    target_label: str,
    target_label_mode: str = "auto",
) -> tuple[dict[str, object], ...]:
    import numpy as np

    from predictive_circuit_coding.cli.common import require_non_empty_split, require_runtime_view
    from predictive_circuit_coding.data import load_temporaldata_session
    from predictive_circuit_coding.decoding.labels import extract_matching_values_from_annotations
    from predictive_circuit_coding.tokenization import extract_sample_event_annotations
    from predictive_circuit_coding.training import load_experiment_config
    from predictive_circuit_coding.windowing import (
        FixedWindowConfig,
        build_dataset_bundle,
        build_sequential_fixed_window_sampler,
    )
    from predictive_circuit_coding.windowing.dataset import split_session_ids

    def _normalize_value(value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, bytes):
            normalized = value.decode("utf-8", errors="replace").strip()
        else:
            normalized = str(value).strip()
        if not normalized or normalized.lower() == "nan":
            return None
        return normalized

    config = load_experiment_config(experiment_config_path)
    dataset_view = require_runtime_view(experiment_config=config, data_config_path=data_config_path)
    require_non_empty_split(dataset_view=dataset_view, split_name=split_name)

    direct_counts: dict[str, int] = {}
    namespace, dot, field = str(target_label).partition(".")
    if dot and namespace and field and "." not in field:
        for session_id in split_session_ids(dataset_view.split_manifest, split_name):
            session_path = dataset_view.workspace.brainset_prepared_root / f"{session_id}.h5"
            if not session_path.is_file():
                continue
            session = load_temporaldata_session(session_path, lazy=False)
            interval = getattr(session, namespace, None)
            raw_values = getattr(interval, field, None) if interval is not None else None
            if raw_values is None:
                continue
            for raw_value in np.asarray(raw_values, dtype=object).reshape(-1):
                normalized = _normalize_value(raw_value)
                if normalized is None:
                    continue
                direct_counts[normalized] = direct_counts.get(normalized, 0) + 1
    if direct_counts:
        return tuple(
            {"value": value, "count": count}
            for value, count in sorted(direct_counts.items(), key=lambda item: (-item[1], item[0]))
        )

    bundle = build_dataset_bundle(
        workspace=dataset_view.workspace,
        split_manifest=dataset_view.split_manifest,
        split=split_name,
        config_dir=dataset_view.config_dir,
        config_name_prefix=dataset_view.config_name_prefix,
        dataset_split=dataset_view.dataset_split,
    )
    sampler = build_sequential_fixed_window_sampler(
        bundle.dataset,
        window=FixedWindowConfig(
            window_length_s=config.data_runtime.context_duration_s,
            step_s=config.evaluation.sequential_step_s,
        ),
    )
    counts: dict[str, int] = {}
    try:
        for item in sampler:
            sample = bundle.dataset.get(item.recording_id, item.start, item.end)
            annotations = extract_sample_event_annotations(
                sample,
                config.data_runtime,
                window_start_s=float(item.start),
                window_end_s=float(item.end),
            )
            values = extract_matching_values_from_annotations(
                annotations,
                target_label=target_label,
                target_label_mode=target_label_mode,
                window_duration_s=float(item.end) - float(item.start),
            )
            for value in values:
                counts[value] = counts.get(value, 0) + 1
    finally:
        if hasattr(bundle.dataset, "_close_open_files"):
            bundle.dataset._close_open_files()
    return tuple(
        {"value": value, "count": count}
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )


def inspect_notebook_target_field_availability(
    *,
    experiment_config_path: str | Path,
    data_config_path: str | Path,
    split_name: str,
    target_label: str,
    preview_limit: int = 5,
) -> dict[str, object]:
    import numpy as np

    from predictive_circuit_coding.cli.common import require_non_empty_split, require_runtime_view
    from predictive_circuit_coding.data import load_temporaldata_session
    from predictive_circuit_coding.training import load_experiment_config
    from predictive_circuit_coding.windowing.dataset import split_session_ids

    def _normalize_value(value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, bytes):
            normalized = value.decode("utf-8", errors="replace").strip()
        else:
            normalized = str(value).strip()
        if not normalized or normalized.lower() == "nan":
            return None
        return normalized

    namespace, dot, field = str(target_label).partition(".")
    if not dot or not namespace or not field or "." in field:
        raise ValueError(
            "inspect_notebook_target_field_availability requires a direct interval field target like "
            "'stimulus_presentations.image_name'."
        )

    config = load_experiment_config(experiment_config_path)
    dataset_view = require_runtime_view(experiment_config=config, data_config_path=data_config_path)
    require_non_empty_split(dataset_view=dataset_view, split_name=split_name)

    value_counts: dict[str, int] = {}
    session_rows: list[dict[str, object]] = []
    session_ids = split_session_ids(dataset_view.split_manifest, split_name)
    for session_id in session_ids:
        session_path = dataset_view.workspace.brainset_prepared_root / f"{session_id}.h5"
        has_session_file = session_path.is_file()
        has_namespace = False
        has_field = False
        non_null_value_count = 0
        preview_values: list[str] = []
        if has_session_file:
            session = load_temporaldata_session(session_path, lazy=False)
            interval = getattr(session, namespace, None)
            has_namespace = interval is not None
            raw_values = getattr(interval, field, None) if interval is not None else None
            has_field = raw_values is not None
            if raw_values is not None:
                for raw_value in np.asarray(raw_values, dtype=object).reshape(-1):
                    normalized = _normalize_value(raw_value)
                    if normalized is None:
                        continue
                    non_null_value_count += 1
                    if len(preview_values) < preview_limit:
                        preview_values.append(normalized)
                    value_counts[normalized] = value_counts.get(normalized, 0) + 1
        session_rows.append(
            {
                "session_id": str(session_id),
                "session_path": str(session_path),
                "has_session_file": has_session_file,
                "has_namespace": has_namespace,
                "has_field": has_field,
                "non_null_value_count": non_null_value_count,
                "preview_values": tuple(preview_values),
            }
        )

    return {
        "split_name": str(split_name),
        "target_label": str(target_label),
        "sessions_scanned": len(session_rows),
        "sessions_with_namespace": sum(1 for row in session_rows if row["has_namespace"]),
        "sessions_with_field": sum(1 for row in session_rows if row["has_field"]),
        "session_rows": tuple(session_rows),
        "value_counts": tuple(
            {"value": value, "count": count}
            for value, count in sorted(value_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
    }
