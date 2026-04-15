from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from predictive_circuit_coding.benchmarks.reports import write_summary_rows
from predictive_circuit_coding.discovery.reporting import (
    discovery_cluster_report_paths,
    discovery_coverage_summary_path,
)
from predictive_circuit_coding.validation.notebook import default_validation_output_paths as _default_validation_output_paths


@dataclass(frozen=True)
class NotebookGeometryDiagnosticsRunResult:
    window_geometry_summary_json_path: Path
    window_geometry_summary_csv_path: Path
    candidate_geometry_summary_json_path: Path
    candidate_geometry_summary_csv_path: Path


@dataclass(frozen=True)
class NotebookAlignmentDiagnosticsRunResult:
    alignment_summary_json_path: Path
    alignment_summary_csv_path: Path


@dataclass(frozen=True)
class NotebookDiagnosticsExperimentPaths:
    experiment_name: str
    experiment_root: Path
    runtime_experiment_config_path: Path
    discovery_artifact_path: Path
    decode_coverage_summary_path: Path
    cluster_summary_json_path: Path
    cluster_summary_csv_path: Path
    validation_summary_json_path: Path
    validation_summary_csv_path: Path
    window_geometry_summary_json_path: Path
    window_geometry_summary_csv_path: Path
    candidate_geometry_summary_json_path: Path
    candidate_geometry_summary_csv_path: Path
    alignment_summary_json_path: Path
    alignment_summary_csv_path: Path


def build_notebook_diagnostics_local_root(*, local_artifact_root: str | Path) -> Path:
    return Path(local_artifact_root) / "diagnostics"


def build_notebook_diagnostics_experiment_paths(
    *,
    local_artifact_root: str | Path,
    checkpoint_path: str | Path,
    experiment_name: str,
    split_name: str = "discovery",
) -> NotebookDiagnosticsExperimentPaths:
    experiment_root = build_notebook_diagnostics_local_root(local_artifact_root=local_artifact_root) / str(
        experiment_name
    )
    checkpoints_root = experiment_root / "checkpoints"
    checkpoint_stem = Path(checkpoint_path).stem
    discovery_artifact_path = checkpoints_root / f"{checkpoint_stem}_{split_name}_discovery.json"
    decode_coverage_summary_path = discovery_coverage_summary_path(discovery_artifact_path)
    cluster_summary_json_path, cluster_summary_csv_path = discovery_cluster_report_paths(discovery_artifact_path)
    validation_summary_json_path, validation_summary_csv_path = _default_validation_output_paths(discovery_artifact_path)
    return NotebookDiagnosticsExperimentPaths(
        experiment_name=str(experiment_name),
        experiment_root=experiment_root,
        runtime_experiment_config_path=experiment_root / "colab_runtime_experiment.yaml",
        discovery_artifact_path=discovery_artifact_path,
        decode_coverage_summary_path=decode_coverage_summary_path,
        cluster_summary_json_path=cluster_summary_json_path,
        cluster_summary_csv_path=cluster_summary_csv_path,
        validation_summary_json_path=validation_summary_json_path,
        validation_summary_csv_path=validation_summary_csv_path,
        window_geometry_summary_json_path=experiment_root / "window_geometry_summary.json",
        window_geometry_summary_csv_path=experiment_root / "window_geometry_summary.csv",
        candidate_geometry_summary_json_path=experiment_root / "candidate_geometry_summary.json",
        candidate_geometry_summary_csv_path=experiment_root / "candidate_geometry_summary.csv",
        alignment_summary_json_path=experiment_root / "session_alignment_summary.json",
        alignment_summary_csv_path=experiment_root / "session_alignment_summary.csv",
    )


def build_notebook_diagnostics_export_path(
    *,
    drive_export_root: str | Path,
    run_id: str,
    diagnostics_timestamp: str,
    run_name: str = "run_1",
) -> Path:
    return Path(drive_export_root) / str(run_id) / run_name / "diagnostics" / str(diagnostics_timestamp)


def export_notebook_diagnostics_artifacts(
    *,
    drive_export_root: str | Path,
    local_artifact_root: str | Path,
    run_id: str,
    diagnostics_timestamp: str,
    run_name: str = "run_1",
) -> Path:
    source_root = build_notebook_diagnostics_local_root(local_artifact_root=local_artifact_root).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Local diagnostics root not found: {source_root}")
    target = build_notebook_diagnostics_export_path(
        drive_export_root=drive_export_root,
        run_id=run_id,
        diagnostics_timestamp=diagnostics_timestamp,
        run_name=run_name,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source_root, target)
    return target


def run_notebook_session_alignment_diagnostics(
    *,
    experiment_config_path: str | Path,
    data_config_path: str | Path,
    checkpoint_path: str | Path,
    neighbor_k: int,
    output_json_path: str | Path,
    output_csv_path: str | Path,
    split_name: str = "discovery",
    progress_ui: bool = True,
) -> NotebookAlignmentDiagnosticsRunResult:
    from tqdm.auto import tqdm

    from predictive_circuit_coding.cli.common import (
        require_checkpoint_matches_dataset,
        require_non_empty_split,
        require_runtime_view,
    )
    from predictive_circuit_coding.decoding.extract import extract_selected_discovery_windows
    from predictive_circuit_coding.decoding.geometry import (
        summarize_session_alignment_geometry,
        write_session_alignment_csv,
        write_session_alignment_json,
    )
    from predictive_circuit_coding.discovery import prepare_discovery_collection
    from predictive_circuit_coding.training import load_experiment_config

    config = load_experiment_config(experiment_config_path)
    dataset_view = require_runtime_view(experiment_config=config, data_config_path=data_config_path)
    require_non_empty_split(dataset_view=dataset_view, split_name=split_name)
    checkpoint = require_checkpoint_matches_dataset(checkpoint_path=checkpoint_path, dataset_id=config.dataset_id)
    shard_dir = Path(output_json_path).parent / "_tmp_alignment_token_shards"

    plan_bar = tqdm(total=0, desc="Alignment window scan", unit="window", leave=False, disable=not progress_ui)

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

    encode_bar = tqdm(
        total=int(window_plan.selected_indices.numel()),
        desc="Alignment encoding",
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

    summary = summarize_session_alignment_geometry(
        features=encoded.pooled_features,
        labels=encoded.labels,
        session_ids=encoded.window_session_ids,
        subject_ids=encoded.window_subject_ids,
        neighbor_k=neighbor_k,
    )
    write_session_alignment_json(summary, output_json_path)
    write_session_alignment_csv(summary, output_csv_path)
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    return NotebookAlignmentDiagnosticsRunResult(
        alignment_summary_json_path=Path(output_json_path),
        alignment_summary_csv_path=Path(output_csv_path),
    )


def run_notebook_geometry_diagnostics(
    *,
    experiment_config_path: str | Path,
    data_config_path: str | Path,
    checkpoint_path: str | Path,
    discovery_artifact_path: str | Path,
    neighbor_k: int,
    output_json_path: str | Path,
    output_csv_path: str | Path,
    candidate_output_json_path: str | Path,
    candidate_output_csv_path: str | Path,
    split_name: str = "discovery",
    progress_ui: bool = True,
) -> NotebookGeometryDiagnosticsRunResult:
    from tqdm.auto import tqdm

    from predictive_circuit_coding.cli.common import (
        require_checkpoint_matches_dataset,
        require_discovery_artifact_matches_dataset,
        require_non_empty_split,
        require_runtime_view,
    )
    from predictive_circuit_coding.decoding.extract import extract_selected_discovery_windows
    from predictive_circuit_coding.decoding.geometry import (
        summarize_candidate_neighbor_geometry,
        summarize_neighbor_geometry,
        write_neighbor_geometry_csv,
        write_neighbor_geometry_json,
    )
    from predictive_circuit_coding.discovery import prepare_discovery_collection
    from predictive_circuit_coding.training import load_experiment_config
    from predictive_circuit_coding.training.contracts import CandidateTokenRecord

    config = load_experiment_config(experiment_config_path)
    dataset_view = require_runtime_view(experiment_config=config, data_config_path=data_config_path)
    require_non_empty_split(dataset_view=dataset_view, split_name=split_name)
    checkpoint = require_checkpoint_matches_dataset(checkpoint_path=checkpoint_path, dataset_id=config.dataset_id)
    artifact_path = require_discovery_artifact_matches_dataset(
        artifact_path=discovery_artifact_path,
        dataset_id=config.dataset_id,
    )

    plan_bar = tqdm(total=0, desc="Geometry window scan", unit="window", leave=False, disable=not progress_ui)

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

    encode_bar = tqdm(
        total=int(window_plan.coverage_summary.selected_positive_count + window_plan.coverage_summary.selected_negative_count),
        desc="Geometry selected-window extraction",
        unit="window",
        leave=False,
        disable=not progress_ui,
    )

    def _encode_progress(current: int, total: int | None) -> None:
        if total is not None and encode_bar.total != total:
            encode_bar.total = total
        encode_bar.n = current
        encode_bar.refresh()

    shard_dir = Path(output_json_path).parent / "geometry_tmp"
    try:
        encoded = extract_selected_discovery_windows(
            experiment_config=config,
            data_config_path=data_config_path,
            checkpoint_path=checkpoint,
            window_plan=window_plan,
            dataset_view=dataset_view,
            shard_dir=shard_dir,
            progress_callback=_encode_progress if progress_ui else None,
        )
    finally:
        encode_bar.close()
        if shard_dir.exists():
            shutil.rmtree(shard_dir)

    window_summary = summarize_neighbor_geometry(
        features=encoded.pooled_features,
        attributes={
            "label": tuple("positive" if float(value) > 0.0 else "negative" for value in encoded.labels.tolist()),
            "session_id": encoded.window_session_ids,
            "subject_id": encoded.window_subject_ids,
        },
        neighbor_k=neighbor_k,
    )
    artifact_payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    candidates = tuple(
        CandidateTokenRecord(**candidate_payload)
        for candidate_payload in artifact_payload.get("candidates", [])
    )
    candidate_summary = summarize_candidate_neighbor_geometry(candidates=candidates, neighbor_k=neighbor_k)
    write_neighbor_geometry_json(window_summary, output_json_path)
    write_neighbor_geometry_csv(window_summary, output_csv_path)
    write_neighbor_geometry_json(candidate_summary, candidate_output_json_path)
    write_neighbor_geometry_csv(candidate_summary, candidate_output_csv_path)
    return NotebookGeometryDiagnosticsRunResult(
        window_geometry_summary_json_path=Path(output_json_path),
        window_geometry_summary_csv_path=Path(output_csv_path),
        candidate_geometry_summary_json_path=Path(candidate_output_json_path),
        candidate_geometry_summary_csv_path=Path(candidate_output_csv_path),
    )


def build_notebook_diagnostics_summary_row(
    *,
    experiment_name: str,
    discovery_artifact_path: str | Path,
    validation_summary_path: str | Path,
    cluster_summary_path: str | Path,
    window_geometry_summary_path: str | Path | None = None,
    candidate_geometry_summary_path: str | Path | None = None,
) -> dict[str, object]:
    discovery_payload = json.loads(Path(discovery_artifact_path).read_text(encoding="utf-8"))
    validation_payload = json.loads(Path(validation_summary_path).read_text(encoding="utf-8"))
    cluster_payload = json.loads(Path(cluster_summary_path).read_text(encoding="utf-8"))
    window_geometry_payload = (
        json.loads(Path(window_geometry_summary_path).read_text(encoding="utf-8"))
        if window_geometry_summary_path is not None and Path(window_geometry_summary_path).is_file()
        else {}
    )
    candidate_geometry_payload = (
        json.loads(Path(candidate_geometry_summary_path).read_text(encoding="utf-8"))
        if candidate_geometry_summary_path is not None and Path(candidate_geometry_summary_path).is_file()
        else {}
    )
    discovery_config = (discovery_payload.get("config_snapshot") or {}).get("discovery") or {}
    return {
        "experiment_name": str(experiment_name),
        "target_label": discovery_payload.get("decoder_summary", {}).get("target_label"),
        "target_label_match_value": discovery_config.get("target_label_match_value"),
        "candidate_count": validation_payload.get("candidate_count"),
        "cluster_count": validation_payload.get("cluster_count"),
        "real_probe_accuracy": validation_payload.get("real_label_metrics", {}).get("probe_accuracy"),
        "shuffled_probe_accuracy": validation_payload.get("shuffled_label_metrics", {}).get("probe_accuracy"),
        "held_out_test_probe_accuracy": validation_payload.get("held_out_test_metrics", {}).get("probe_accuracy"),
        "held_out_similarity_roc_auc": validation_payload.get("held_out_similarity_summary", {}).get("window_roc_auc"),
        "held_out_similarity_pr_auc": validation_payload.get("held_out_similarity_summary", {}).get("window_pr_auc"),
        "cluster_persistence_mean": validation_payload.get("cluster_quality_summary", {}).get("cluster_persistence_mean"),
        "silhouette_score": validation_payload.get("cluster_quality_summary", {}).get("silhouette_score"),
        "window_label_neighbor_enrichment": (((window_geometry_payload.get("metrics") or {}).get("label") or {}).get("enrichment_over_base")),
        "window_session_neighbor_enrichment": (((window_geometry_payload.get("metrics") or {}).get("session_id") or {}).get("enrichment_over_base")),
        "window_subject_neighbor_enrichment": (((window_geometry_payload.get("metrics") or {}).get("subject_id") or {}).get("enrichment_over_base")),
        "candidate_session_neighbor_enrichment": (((candidate_geometry_payload.get("metrics") or {}).get("session_id") or {}).get("enrichment_over_base")),
        "candidate_subject_neighbor_enrichment": (((candidate_geometry_payload.get("metrics") or {}).get("subject_id") or {}).get("enrichment_over_base")),
        "candidate_unit_region_neighbor_enrichment": (((candidate_geometry_payload.get("metrics") or {}).get("unit_region") or {}).get("enrichment_over_base")),
        "cluster_summary_path": str(Path(cluster_summary_path)),
    }


def build_notebook_alignment_summary_row(
    *,
    experiment_name: str,
    alignment_summary_path: str | Path,
) -> dict[str, object]:
    payload = json.loads(Path(alignment_summary_path).read_text(encoding="utf-8"))
    original_geometry = payload.get("geometry_original") or {}
    whitened_geometry = payload.get("geometry_whitened") or {}
    aligned_geometry = payload.get("geometry_aligned") or {}
    aggregate = payload.get("aggregate_metrics") or {}

    def _metric(summary: dict[str, object], attribute: str) -> float | None:
        return (((summary.get("metrics") or {}).get(attribute) or {}).get("enrichment_over_base"))

    return {
        "experiment_name": str(experiment_name),
        "experiment_type": "session_alignment",
        "reference_session_id": payload.get("reference_session_id"),
        "session_count": payload.get("session_count"),
        "sample_count": payload.get("sample_count"),
        "label_axis_cosine_before": aggregate.get("mean_label_axis_cosine_before"),
        "label_axis_cosine_after": aggregate.get("mean_label_axis_cosine_after"),
        "positive_centroid_cosine_before": aggregate.get("mean_positive_centroid_cosine_before"),
        "positive_centroid_cosine_after": aggregate.get("mean_positive_centroid_cosine_after"),
        "negative_centroid_cosine_before": aggregate.get("mean_negative_centroid_cosine_before"),
        "negative_centroid_cosine_after": aggregate.get("mean_negative_centroid_cosine_after"),
        "anchor_rmse_after_alignment": aggregate.get("mean_anchor_rmse_after_alignment"),
        "raw_label_neighbor_enrichment": _metric(original_geometry, "label"),
        "raw_session_neighbor_enrichment": _metric(original_geometry, "session_id"),
        "raw_subject_neighbor_enrichment": _metric(original_geometry, "subject_id"),
        "whitened_label_neighbor_enrichment": _metric(whitened_geometry, "label"),
        "whitened_session_neighbor_enrichment": _metric(whitened_geometry, "session_id"),
        "whitened_subject_neighbor_enrichment": _metric(whitened_geometry, "subject_id"),
        "aligned_label_neighbor_enrichment": _metric(aligned_geometry, "label"),
        "aligned_session_neighbor_enrichment": _metric(aligned_geometry, "session_id"),
        "aligned_subject_neighbor_enrichment": _metric(aligned_geometry, "subject_id"),
        "alignment_summary_path": str(Path(alignment_summary_path)),
    }


def write_notebook_diagnostics_summary(
    rows: list[dict[str, object]],
    *,
    output_json_path: str | Path,
    output_csv_path: str | Path,
) -> tuple[Path, Path]:
    return write_summary_rows(
        rows,
        output_json_path=output_json_path,
        output_csv_path=output_csv_path,
        root_key="experiments",
    )
