from __future__ import annotations

import argparse
import sys
from pathlib import Path

from predictive_circuit_coding.benchmarks.reports import build_final_project_summary, write_single_row_summary, write_summary_rows
from predictive_circuit_coding.benchmarks.run import default_benchmark_task_specs, default_motif_arm_specs, run_motif_benchmark_matrix
from predictive_circuit_coding.cli.common import (
    emit_run_manifest,
    get_cli_console,
    print_artifact,
    require_checkpoint_matches_dataset,
    require_runtime_view,
)
from predictive_circuit_coding.training import load_experiment_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trained-encoder refinement discovery comparisons.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    parser.add_argument("--data-config", required=True, help="Path to a preparation YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained model checkpoint.")
    parser.add_argument("--output-root", required=True, help="Refinement output root directory.")
    parser.add_argument("--session-holdout-fraction", type=float, default=0.5, help="Within-session held-out fraction for discovery windows.")
    parser.add_argument("--session-holdout-seed", type=int, default=None, help="Optional held-out split seed.")
    parser.add_argument("--retain-debug-intermediates", action="store_true", help="Keep temporary refinement shard intermediates.")
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
    refinement_root = output_root / "refinement"
    reports_root = output_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    results = run_motif_benchmark_matrix(
        experiment_config=config,
        data_config_path=args.data_config,
        checkpoint_path=checkpoint_path,
        output_root=refinement_root,
        task_specs=default_benchmark_task_specs(),
        arm_specs=default_motif_arm_specs(),
        session_holdout_fraction=float(args.session_holdout_fraction),
        session_holdout_seed=args.session_holdout_seed,
        debug_retain_intermediates=bool(args.retain_debug_intermediates),
    )
    rows = [result.summary for result in results]
    refinement_json_path, refinement_csv_path = write_summary_rows(
        rows,
        output_json_path=reports_root / "refinement_summary.json",
        output_csv_path=reports_root / "refinement_summary.csv",
        root_key="refinement",
    )
    final_json_path, final_csv_path = write_single_row_summary(
        build_final_project_summary(motif_rows=rows),
        output_json_path=reports_root / "final_project_summary.json",
        output_csv_path=reports_root / "final_project_summary.csv",
        root_key="final_project_summary",
    )

    print_artifact(console, label="Refinement summary", path=refinement_json_path)
    print_artifact(console, label="Final project summary", path=final_json_path)
    sidecar_path = emit_run_manifest(
        command_name="refine",
        dataset_id=config.dataset_id,
        output_path=final_json_path,
        inputs={
            "config_path": str(Path(args.config).resolve()),
            "data_config_path": str(Path(args.data_config).resolve()),
            "checkpoint_path": str(checkpoint_path),
            "output_root": str(output_root.resolve()),
            "session_holdout_fraction": float(args.session_holdout_fraction),
            "session_holdout_seed": args.session_holdout_seed,
        },
        outputs={
            "refinement_summary_json_path": str(refinement_json_path),
            "refinement_summary_csv_path": str(refinement_csv_path),
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
