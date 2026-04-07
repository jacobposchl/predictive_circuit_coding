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
from predictive_circuit_coding.evaluation import evaluate_checkpoint_on_split
from predictive_circuit_coding.training import load_experiment_config, write_evaluation_summary
from predictive_circuit_coding.training.logging import StageLogger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on a prepared split.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    parser.add_argument("--data-config", required=True, help="Path to a preparation YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to a model checkpoint.")
    parser.add_argument("--split", required=True, choices=["valid", "test", "train", "discovery"], help="Split to evaluate.")
    parser.add_argument("--output", default=None, help="Optional explicit path for the evaluation summary JSON.")
    return parser.parse_args(argv)


def _default_output_path(checkpoint_path: str | Path, split_name: str) -> Path:
    checkpoint = Path(checkpoint_path)
    return checkpoint.with_name(f"{checkpoint.stem}_{split_name}_evaluation.json")


def _run(args: argparse.Namespace) -> int:
    console = get_cli_console()
    config = load_experiment_config(args.config)
    dataset_view = require_runtime_view(experiment_config=config, data_config_path=args.data_config)
    logger = StageLogger(name="pcc-evaluate")
    logger.log_stage("preflight", expected_next="evaluation summary")
    require_non_empty_split(dataset_view=dataset_view, split_name=args.split)
    checkpoint_path = require_checkpoint_matches_dataset(
        checkpoint_path=args.checkpoint,
        dataset_id=config.dataset_id,
    )
    logger.log_stage("evaluation", expected_next="evaluation summary json")
    summary = evaluate_checkpoint_on_split(
        experiment_config=config,
        data_config_path=args.data_config,
        checkpoint_path=checkpoint_path,
        split_name=args.split,
        dataset_view=dataset_view,
    )
    output_path = Path(args.output) if args.output else _default_output_path(checkpoint_path, args.split)
    write_evaluation_summary(summary, output_path)
    print_artifact(console, label="Evaluation summary", path=output_path)
    sidecar_path = emit_run_manifest(
        command_name="evaluate",
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
            "evaluation_summary_path": str(output_path),
        },
    )
    print_artifact(console, label="Run manifest", path=sidecar_path)
    console.print(f"Predictive improvement: {summary.metrics.get('predictive_improvement', 0.0):.6f}")
    return 0


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(_run(parse_args(argv)))


if __name__ == "__main__":
    main(sys.argv[1:])
