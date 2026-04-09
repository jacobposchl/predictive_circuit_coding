from __future__ import annotations

import argparse
import sys
from pathlib import Path

from predictive_circuit_coding.benchmarks.reports import build_final_project_summary, write_single_row_summary, write_summary_rows
from predictive_circuit_coding.benchmarks.run import (
    default_benchmark_task_specs,
    default_motif_arm_specs,
    default_representation_arm_specs,
    run_motif_benchmark_matrix,
    run_representation_benchmark_matrix,
)
from predictive_circuit_coding.cli.common import (
    emit_run_manifest,
    get_cli_console,
    print_artifact,
    require_checkpoint_matches_dataset,
    require_runtime_view,
)
from predictive_circuit_coding.training import load_experiment_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the final representation and motif benchmark matrix.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    parser.add_argument("--data-config", required=True, help="Path to a preparation YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to a model checkpoint.")
    parser.add_argument("--output-root", required=True, help="Benchmark output root directory.")
    parser.add_argument("--skip-representation", action="store_true", help="Skip the representation benchmark stage.")
    parser.add_argument("--skip-motifs", action="store_true", help="Skip the motif benchmark stage.")
    parser.add_argument("--include-image-identity", action="store_true", help="Include the optional image identity one-vs-rest task.")
    parser.add_argument("--image-target-name", default=None, help="Optional image_name value for one-vs-rest image identity.")
    parser.add_argument("--pca-components", type=int, default=64, help="PCA component cap for PCA benchmark arms.")
    parser.add_argument("--session-holdout-fraction", type=float, default=0.5, help="Within-session held-out fraction for discovery windows.")
    parser.add_argument("--session-holdout-seed", type=int, default=None, help="Optional held-out split seed.")
    parser.add_argument("--neighbor-k", type=int, default=5, help="k for geometry neighbor enrichment metrics.")
    parser.add_argument("--retain-debug-intermediates", action="store_true", help="Keep temporary benchmark shard intermediates.")
    return parser.parse_args(argv)


def _run(args: argparse.Namespace) -> int:
    console = get_cli_console()
    config = load_experiment_config(args.config)
    require_runtime_view(experiment_config=config, data_config_path=args.data_config)
    checkpoint_path = require_checkpoint_matches_dataset(
        checkpoint_path=args.checkpoint,
        dataset_id=config.dataset_id,
    )
    output_root = Path(args.output_root)
    representation_root = output_root / "representation"
    motif_root = output_root / "motifs"
    reports_root = output_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    task_specs = default_benchmark_task_specs(
        include_image_identity=bool(args.include_image_identity),
        image_target_name=args.image_target_name,
    )
    representation_rows: list[dict[str, object]] = []
    motif_rows: list[dict[str, object]] = []

    if not args.skip_representation:
        representation_results = run_representation_benchmark_matrix(
            experiment_config=config,
            data_config_path=args.data_config,
            checkpoint_path=checkpoint_path,
            output_root=representation_root,
            task_specs=task_specs,
            arm_specs=default_representation_arm_specs(pca_components=int(args.pca_components)),
            session_holdout_fraction=float(args.session_holdout_fraction),
            session_holdout_seed=args.session_holdout_seed,
            neighbor_k=int(args.neighbor_k),
        )
        representation_rows = [result.summary for result in representation_results]
    representation_json_path, representation_csv_path = write_summary_rows(
        representation_rows,
        output_json_path=reports_root / "representation_benchmark_summary.json",
        output_csv_path=reports_root / "representation_benchmark_summary.csv",
        root_key="representation_benchmark",
    )

    if not args.skip_motifs:
        motif_results = run_motif_benchmark_matrix(
            experiment_config=config,
            data_config_path=args.data_config,
            checkpoint_path=checkpoint_path,
            output_root=motif_root,
            task_specs=task_specs,
            arm_specs=default_motif_arm_specs(pca_components=int(args.pca_components)),
            session_holdout_fraction=float(args.session_holdout_fraction),
            session_holdout_seed=args.session_holdout_seed,
            debug_retain_intermediates=bool(args.retain_debug_intermediates),
        )
        motif_rows = [result.summary for result in motif_results]
    motif_json_path, motif_csv_path = write_summary_rows(
        motif_rows,
        output_json_path=reports_root / "motif_benchmark_summary.json",
        output_csv_path=reports_root / "motif_benchmark_summary.csv",
        root_key="motif_benchmark",
    )
    final_summary = build_final_project_summary(
        representation_rows=representation_rows,
        motif_rows=motif_rows,
    )
    final_json_path, final_csv_path = write_single_row_summary(
        final_summary,
        output_json_path=reports_root / "final_project_summary.json",
        output_csv_path=reports_root / "final_project_summary.csv",
        root_key="final_project_summary",
    )

    print_artifact(console, label="Representation benchmark summary", path=representation_json_path)
    print_artifact(console, label="Motif benchmark summary", path=motif_json_path)
    print_artifact(console, label="Final project summary", path=final_json_path)
    sidecar_path = emit_run_manifest(
        command_name="benchmark",
        dataset_id=config.dataset_id,
        output_path=final_json_path,
        inputs={
            "config_path": str(Path(args.config).resolve()),
            "data_config_path": str(Path(args.data_config).resolve()),
            "checkpoint_path": str(checkpoint_path),
            "output_root": str(output_root.resolve()),
            "include_image_identity": bool(args.include_image_identity),
            "image_target_name": args.image_target_name,
            "pca_components": int(args.pca_components),
            "session_holdout_fraction": float(args.session_holdout_fraction),
            "session_holdout_seed": args.session_holdout_seed,
            "neighbor_k": int(args.neighbor_k),
        },
        outputs={
            "representation_summary_json_path": str(representation_json_path),
            "representation_summary_csv_path": str(representation_csv_path),
            "motif_summary_json_path": str(motif_json_path),
            "motif_summary_csv_path": str(motif_csv_path),
            "final_summary_json_path": str(final_json_path),
            "final_summary_csv_path": str(final_csv_path),
        },
    )
    print_artifact(console, label="Run manifest", path=sidecar_path)
    return 0


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(_run(parse_args(argv)))


if __name__ == "__main__":
    main(sys.argv[1:])
