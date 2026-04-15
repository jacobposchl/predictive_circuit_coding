from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NotebookValidationRunResult:
    validation_summary_json_path: Path
    validation_summary_csv_path: Path


def default_validation_output_paths(discovery_artifact_path: str | Path) -> tuple[Path, Path]:
    artifact = Path(discovery_artifact_path)
    return (
        artifact.with_name(f"{artifact.stem}_validation.json"),
        artifact.with_name(f"{artifact.stem}_validation.csv"),
    )


def run_notebook_validation(
    *,
    experiment_config_path: str | Path,
    data_config_path: str | Path,
    checkpoint_path: str | Path,
    discovery_artifact_path: str | Path,
    output_json_path: str | Path | None = None,
    output_csv_path: str | Path | None = None,
    progress_ui: bool = True,
) -> NotebookValidationRunResult:
    from tqdm.auto import tqdm

    from predictive_circuit_coding.cli.common import (
        require_checkpoint_matches_dataset,
        require_discovery_artifact_matches_validation_inputs,
        require_non_empty_split,
        require_runtime_view,
    )
    from predictive_circuit_coding.training import (
        load_experiment_config,
        write_validation_summary,
        write_validation_summary_csv,
    )
    from predictive_circuit_coding.validation.run import validate_discovery_artifact

    config = load_experiment_config(experiment_config_path)
    dataset_view = require_runtime_view(experiment_config=config, data_config_path=data_config_path)
    require_non_empty_split(dataset_view=dataset_view, split_name=config.splits.discovery)
    require_non_empty_split(dataset_view=dataset_view, split_name=config.splits.test)
    checkpoint = require_checkpoint_matches_dataset(checkpoint_path=checkpoint_path, dataset_id=config.dataset_id)
    artifact_path = require_discovery_artifact_matches_validation_inputs(
        artifact_path=discovery_artifact_path,
        dataset_id=config.dataset_id,
        checkpoint_path=checkpoint,
        target_label=config.discovery.target_label,
    )
    default_json, default_csv = default_validation_output_paths(artifact_path)
    target_json = Path(output_json_path) if output_json_path is not None else default_json
    target_csv = Path(output_csv_path) if output_csv_path is not None else default_csv
    validation_bar = tqdm(
        total=int(config.evaluation.max_batches),
        desc="Held-out extraction / validation",
        unit="batch",
        leave=False,
        disable=not progress_ui,
    )

    def _validation_progress(current: int, total: int | None) -> None:
        if total is not None and validation_bar.total != total:
            validation_bar.total = total
        validation_bar.n = current
        validation_bar.refresh()

    try:
        summary = validate_discovery_artifact(
            experiment_config=config,
            data_config_path=data_config_path,
            checkpoint_path=checkpoint,
            discovery_artifact_path=artifact_path,
            dataset_view=dataset_view,
            progress_callback=_validation_progress if progress_ui else None,
        )
    finally:
        validation_bar.close()
    write_validation_summary(summary, target_json)
    write_validation_summary_csv(summary, target_csv)
    return NotebookValidationRunResult(
        validation_summary_json_path=target_json,
        validation_summary_csv_path=target_csv,
    )


def flatten_comparison_validation_summary(summary: dict[str, object]) -> dict[str, object]:
    discovery_fit = summary.get("discovery_fit_metrics") or {}
    shuffled_fit = summary.get("shuffled_fit_metrics") or {}
    primary_held_out = summary.get("primary_held_out_metrics") or {}
    primary_similarity = summary.get("primary_held_out_similarity_summary") or {}
    standard_validation = summary.get("standard_test_validation") or {}
    standard_test_metrics = standard_validation.get("held_out_test_metrics") or {}
    standard_test_similarity = standard_validation.get("held_out_similarity_summary") or {}
    cluster_quality = summary.get("cluster_quality_summary") or {}
    candidate_selection = summary.get("candidate_selection_summary") or {}
    arm_shard_debug = candidate_selection.get("arm_shard_debug") or {}
    return {
        "arm_name": summary.get("arm_name"),
        "transform_mode": summary.get("transform_mode"),
        "comparison_status": summary.get("comparison_status"),
        "failure_reason": summary.get("failure_reason"),
        "fit_window_count": summary.get("fit_window_count"),
        "heldout_window_count": summary.get("heldout_window_count"),
        "candidate_count": summary.get("candidate_count"),
        "cluster_count": summary.get("cluster_count"),
        "excluded_session_count": len(summary.get("excluded_sessions") or ()),
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
        "primary_held_out_probe_accuracy": primary_held_out.get("probe_accuracy"),
        "primary_held_out_probe_bce": primary_held_out.get("probe_bce"),
        "primary_held_out_probe_roc_auc": primary_held_out.get("probe_roc_auc"),
        "primary_held_out_probe_pr_auc": primary_held_out.get("probe_pr_auc"),
        "primary_similarity_roc_auc": primary_similarity.get("window_roc_auc"),
        "primary_similarity_pr_auc": primary_similarity.get("window_pr_auc"),
        "cluster_persistence_mean": cluster_quality.get("cluster_persistence_mean"),
        "silhouette_score": cluster_quality.get("silhouette_score"),
        "standard_test_probe_accuracy": standard_test_metrics.get("probe_accuracy"),
        "standard_test_probe_bce": standard_test_metrics.get("probe_bce"),
        "standard_test_similarity_roc_auc": standard_test_similarity.get("window_roc_auc"),
        "standard_test_similarity_pr_auc": standard_test_similarity.get("window_pr_auc"),
    }


__all__ = [
    "NotebookValidationRunResult",
    "default_validation_output_paths",
    "flatten_comparison_validation_summary",
    "run_notebook_validation",
]
