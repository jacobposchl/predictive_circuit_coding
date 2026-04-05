from __future__ import annotations

from pathlib import Path

from predictive_circuit_coding.decoding import extract_frozen_tokens, fit_additive_probe, score_token_records
from predictive_circuit_coding.discovery.candidates import select_candidate_tokens
from predictive_circuit_coding.discovery.clustering import cluster_candidate_tokens
from predictive_circuit_coding.discovery.stability import estimate_clustering_stability
from predictive_circuit_coding.training.config import ExperimentConfig
from predictive_circuit_coding.training.contracts import DecoderSummary, DiscoveryArtifact, write_json_payload


def discover_motifs(
    *,
    experiment_config: ExperimentConfig,
    data_config_path: str | Path,
    checkpoint_path: str | Path,
    split_name: str,
    dataset_view=None,
) -> DiscoveryArtifact:
    collection = extract_frozen_tokens(
        experiment_config=experiment_config,
        data_config_path=data_config_path,
        checkpoint_path=checkpoint_path,
        split_name=split_name,
        max_batches=experiment_config.discovery.max_batches,
        dataset_view=dataset_view,
    )
    probe_fit = fit_additive_probe(
        tokens=collection.tokens,
        token_mask=collection.token_mask,
        labels=collection.labels,
        epochs=experiment_config.discovery.probe_epochs,
        learning_rate=experiment_config.discovery.probe_learning_rate,
    )
    scored_records = score_token_records(
        records=collection.records,
        probe_state_dict=probe_fit.state_dict,
    )
    candidates = select_candidate_tokens(
        scored_records=scored_records,
        top_k=experiment_config.discovery.top_k_candidates,
        min_score=experiment_config.discovery.min_candidate_score,
    )
    if not candidates:
        raise ValueError(
            "No candidate tokens were selected from the discovery split. Lower the candidate threshold, "
            "increase discovery coverage, or confirm that positive 'stimulus_change' labels exist in the sampled windows."
        )
    clustered_candidates, cluster_stats = cluster_candidate_tokens(
        candidates=candidates,
        similarity_threshold=experiment_config.discovery.cluster_similarity_threshold,
        min_cluster_size=experiment_config.discovery.min_cluster_size,
    )
    if int(cluster_stats.get("cluster_count", 0)) <= 0:
        raise ValueError(
            "Discovery selected candidate tokens but clustering produced no non-noise motif clusters. "
            "Relax the clustering threshold or reduce the minimum cluster size."
        )
    stability_summary = estimate_clustering_stability(
        candidates=clustered_candidates,
        similarity_threshold=experiment_config.discovery.cluster_similarity_threshold,
        min_cluster_size=experiment_config.discovery.min_cluster_size,
        rounds=experiment_config.discovery.stability_rounds,
        seed=experiment_config.discovery.shuffle_seed,
    )
    return DiscoveryArtifact(
        dataset_id=experiment_config.dataset_id,
        split_name=split_name,
        checkpoint_path=str(checkpoint_path),
        config_snapshot=experiment_config.to_dict(),
        decoder_summary=DecoderSummary(
            target_label=experiment_config.discovery.target_label,
            epochs=experiment_config.discovery.probe_epochs,
            learning_rate=experiment_config.discovery.probe_learning_rate,
            metrics=probe_fit.metrics,
        ),
        candidates=clustered_candidates,
        cluster_stats=cluster_stats,
        stability_summary=stability_summary,
    )


def write_discovery_artifact(artifact: DiscoveryArtifact, path: str | Path) -> Path:
    return write_json_payload(artifact.to_dict(), path)
