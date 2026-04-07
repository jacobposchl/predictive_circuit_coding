from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Callable

from predictive_circuit_coding.decoding import fit_additive_probe_features
from predictive_circuit_coding.decoding.extract import (
    DiscoveryWindowPlan,
    build_discovery_window_plan,
    extract_selected_discovery_windows,
)
from predictive_circuit_coding.decoding.scoring import select_candidate_tokens_from_shards
from predictive_circuit_coding.discovery.clustering import cluster_candidate_tokens
from predictive_circuit_coding.discovery.stability import estimate_clustering_stability
from predictive_circuit_coding.training.config import ExperimentConfig
from predictive_circuit_coding.training.contracts import (
    DecoderSummary,
    DiscoveryArtifact,
    DiscoveryCoverageSummary,
    write_json_payload,
)


@dataclass(frozen=True)
class DiscoveryRunResult:
    artifact: DiscoveryArtifact
    coverage_summary: DiscoveryCoverageSummary


ProgressCallback = Callable[[int, int | None], None]


def _ensure_binary_label_coverage(summary: DiscoveryCoverageSummary) -> None:
    if summary.positive_window_count > 0 and summary.negative_window_count > 0:
        return
    raise ValueError(
        "Cannot build discovery probe inputs because the requested split does not provide both classes for the "
        f"target label '{summary.target_label}'. Split='{summary.split_name}', scanned_windows="
        f"{summary.total_scanned_windows}, positive_windows={summary.positive_window_count}, "
        f"negative_windows={summary.negative_window_count}, positive_sessions={list(summary.sessions_with_positive_windows)}."
    )


def prepare_discovery_collection(
    *,
    experiment_config: ExperimentConfig,
    data_config_path: str | Path,
    split_name: str,
    dataset_view=None,
    progress_callback: ProgressCallback | None = None,
) -> DiscoveryWindowPlan:
    return build_discovery_window_plan(
        experiment_config=experiment_config,
        data_config_path=data_config_path,
        split_name=split_name,
        dataset_view=dataset_view,
        progress_callback=progress_callback,
    )


def discover_motifs_from_plan(
    *,
    experiment_config: ExperimentConfig,
    data_config_path: str | Path,
    checkpoint_path: str | Path,
    split_name: str,
    window_plan: DiscoveryWindowPlan,
    dataset_view=None,
    progress_callback: ProgressCallback | None = None,
) -> DiscoveryRunResult:
    _ensure_binary_label_coverage(window_plan.coverage_summary)
    shard_root = Path(checkpoint_path).with_name(f"{Path(checkpoint_path).stem}_{split_name}_discovery_tmp")
    try:
        encoded = extract_selected_discovery_windows(
            experiment_config=experiment_config,
            data_config_path=data_config_path,
            checkpoint_path=checkpoint_path,
            window_plan=window_plan,
            dataset_view=dataset_view,
            shard_dir=shard_root,
            progress_callback=progress_callback,
        )
        probe_fit = fit_additive_probe_features(
            features=encoded.pooled_features,
            labels=encoded.labels,
            epochs=experiment_config.discovery.probe_epochs,
            learning_rate=experiment_config.discovery.probe_learning_rate,
            label_name=experiment_config.discovery.target_label,
        )
        candidates = select_candidate_tokens_from_shards(
            shard_paths=encoded.shard_paths,
            probe_state_dict=probe_fit.state_dict,
            top_k=experiment_config.discovery.top_k_candidates,
            min_score=experiment_config.discovery.min_candidate_score,
        )
    finally:
        if shard_root.exists():
            shutil.rmtree(shard_root)
    if not candidates:
        raise ValueError(
            f"No candidate tokens were selected from the discovery split for target label "
            f"'{experiment_config.discovery.target_label}'. Lower the candidate threshold, "
            "increase discovery coverage, or confirm that positive target-label windows exist in the sampled windows."
        )
    clustered_candidates, cluster_stats = cluster_candidate_tokens(
        candidates=candidates,
        min_cluster_size=experiment_config.discovery.min_cluster_size,
    )
    if int(cluster_stats.get("cluster_count", 0)) <= 0:
        raise ValueError(
            "Discovery selected candidate tokens but clustering produced no non-noise motif clusters. "
            "Relax the clustering threshold or reduce the minimum cluster size."
        )
    cluster_quality_summary = estimate_clustering_stability(cluster_stats=cluster_stats)
    artifact = DiscoveryArtifact(
        dataset_id=experiment_config.dataset_id,
        split_name=split_name,
        checkpoint_path=str(checkpoint_path),
        config_snapshot=experiment_config.to_dict(),
        decoder_summary=DecoderSummary(
            target_label=experiment_config.discovery.target_label,
            epochs=experiment_config.discovery.probe_epochs,
            learning_rate=experiment_config.discovery.probe_learning_rate,
            metrics=probe_fit.metrics,
            probe_state=probe_fit.state_dict,
        ),
        candidates=clustered_candidates,
        cluster_stats=cluster_stats,
        cluster_quality_summary=cluster_quality_summary,
    )
    return DiscoveryRunResult(
        artifact=artifact,
        coverage_summary=window_plan.coverage_summary,
    )


def discover_motifs(
    *,
    experiment_config: ExperimentConfig,
    data_config_path: str | Path,
    checkpoint_path: str | Path,
    split_name: str,
    dataset_view=None,
) -> DiscoveryRunResult:
    window_plan = prepare_discovery_collection(
        experiment_config=experiment_config,
        data_config_path=data_config_path,
        split_name=split_name,
        dataset_view=dataset_view,
    )
    return discover_motifs_from_plan(
        experiment_config=experiment_config,
        data_config_path=data_config_path,
        checkpoint_path=checkpoint_path,
        split_name=split_name,
        window_plan=window_plan,
        dataset_view=dataset_view,
    )


def write_discovery_artifact(artifact: DiscoveryArtifact, path: str | Path) -> Path:
    return write_json_payload(artifact.to_dict(), path)


def write_discovery_coverage_summary(summary: DiscoveryCoverageSummary, path: str | Path) -> Path:
    return write_json_payload(summary.to_dict(), path)
