from __future__ import annotations

import argparse
import sys
from pathlib import Path

from predictive_circuit_coding.cli.common import (
    emit_run_manifest,
    get_cli_console,
    print_artifact,
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
from predictive_circuit_coding.discovery.reporting import (
    discovery_cluster_report_paths,
    discovery_coverage_summary_path,
)
from predictive_circuit_coding.training import load_experiment_config
from predictive_circuit_coding.training.logging import StageLogger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover candidate motifs from frozen encoder tokens.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    parser.add_argument("--data-config", required=True, help="Path to a preparation YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to a model checkpoint.")
    parser.add_argument("--split", default="discovery", choices=["discovery", "train", "valid", "test"], help="Split to use for token discovery.")
    parser.add_argument("--output", default=None, help="Optional explicit path for the discovery artifact JSON.")
    return parser.parse_args(argv)


def _default_output_path(checkpoint_path: str | Path, split_name: str) -> Path:
    checkpoint = Path(checkpoint_path)
    return checkpoint.with_name(f"{checkpoint.stem}_{split_name}_discovery.json")


def _run(args: argparse.Namespace) -> int:
    console = get_cli_console()
    config = load_experiment_config(args.config)
    dataset_view = require_runtime_view(experiment_config=config, data_config_path=args.data_config)
    logger = StageLogger(name="pcc-discover")
    logger.log_stage("preflight", expected_next="decode coverage summary + discovery artifact")
    require_non_empty_split(dataset_view=dataset_view, split_name=args.split)
    checkpoint_path = require_checkpoint_matches_dataset(
        checkpoint_path=args.checkpoint,
        dataset_id=config.dataset_id,
    )
    output_path = Path(args.output) if args.output else _default_output_path(checkpoint_path, args.split)
    coverage_summary_path = discovery_coverage_summary_path(output_path)
    collection = prepare_discovery_collection(
        experiment_config=config,
        data_config_path=args.data_config,
        split_name=args.split,
        dataset_view=dataset_view,
    )
    write_discovery_coverage_summary(collection.coverage_summary, coverage_summary_path)
    logger.log(
        "Decode coverage: "
        f"split={collection.coverage_summary.split_name}, "
        f"target_label={collection.coverage_summary.target_label}, "
        f"scanned_windows={collection.coverage_summary.total_scanned_windows}, "
        f"positive_windows={collection.coverage_summary.positive_window_count}, "
        f"negative_windows={collection.coverage_summary.negative_window_count}, "
        f"selected_positive={collection.coverage_summary.selected_positive_count}, "
        f"selected_negative={collection.coverage_summary.selected_negative_count}"
    )
    logger.log(
        f"Positive-window sessions: {list(collection.coverage_summary.sessions_with_positive_windows)}"
    )
    logger.log_stage("discovery", expected_next="discovery artifact json")
    result = discover_motifs_from_plan(
        experiment_config=config,
        data_config_path=args.data_config,
        checkpoint_path=checkpoint_path,
        split_name=args.split,
        window_plan=collection,
        dataset_view=dataset_view,
    )
    artifact = result.artifact
    write_discovery_artifact(artifact, output_path)
    cluster_report = build_discovery_cluster_report(artifact)
    cluster_report_json, cluster_report_csv = discovery_cluster_report_paths(output_path)
    write_discovery_cluster_report_json(cluster_report, cluster_report_json)
    write_discovery_cluster_report_csv(cluster_report, cluster_report_csv)
    print_artifact(console, label="Decode coverage summary", path=coverage_summary_path)
    print_artifact(console, label="Discovery artifact", path=output_path)
    print_artifact(console, label="Cluster summary JSON", path=cluster_report_json)
    print_artifact(console, label="Cluster summary CSV", path=cluster_report_csv)
    sidecar_path = emit_run_manifest(
        command_name="discover",
        dataset_id=config.dataset_id,
        output_path=output_path,
        inputs={
            "config_path": str(Path(args.config).resolve()),
            "data_config_path": str(Path(args.data_config).resolve()),
            "checkpoint_path": str(checkpoint_path),
            "split": args.split,
            "runtime_split_manifest_path": str(dataset_view.split_manifest_path),
            "runtime_session_catalog_path": str(dataset_view.session_catalog_path),
            "runtime_subset_active": config.runtime_subset is not None,
            "dataset_selection_active": dataset_view.selection_active,
        },
        outputs={
            "decode_coverage_summary_path": str(coverage_summary_path),
            "discovery_artifact_path": str(output_path),
            "cluster_summary_json": str(cluster_report_json),
            "cluster_summary_csv": str(cluster_report_csv),
        },
    )
    print_artifact(console, label="Run manifest", path=sidecar_path)
    console.print(f"Candidates: {len(artifact.candidates)}")
    console.print(f"Clusters: {artifact.cluster_stats.get('cluster_count', 0)}")
    console.print(f"Silhouette score: {artifact.cluster_quality_summary.get('silhouette_score')}")
    return 0


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(_run(parse_args(argv)))


if __name__ == "__main__":
    main(sys.argv[1:])
