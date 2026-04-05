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
from predictive_circuit_coding.training import load_experiment_config, train_model
from predictive_circuit_coding.training.logging import StageLogger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the predictive circuit coding model on a prepared split.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    parser.add_argument("--data-config", required=True, help="Path to a preparation YAML config.")
    parser.add_argument("--split", default=None, help="Optional override for the training split name.")
    parser.add_argument("--valid-split", default=None, help="Optional override for the validation split name.")
    return parser.parse_args(argv)


def _run(args: argparse.Namespace) -> int:
    console = get_cli_console()
    config = load_experiment_config(args.config)
    train_split = args.split or config.splits.train
    valid_split = args.valid_split or config.splits.valid
    dataset_view = require_runtime_view(experiment_config=config, data_config_path=args.data_config)
    logger = StageLogger(name="pcc-train")
    logger.log_stage("preflight", expected_next="checkpoint + training summary")
    require_non_empty_split(dataset_view=dataset_view, split_name=train_split)
    require_non_empty_split(dataset_view=dataset_view, split_name=valid_split)
    if config.training.resume_checkpoint is not None:
        require_checkpoint_matches_dataset(
            checkpoint_path=config.training.resume_checkpoint,
            dataset_id=config.dataset_id,
        )
        logger.log(f"Resume checkpoint verified: {config.training.resume_checkpoint}")
    logger.log_stage("training", expected_next="best checkpoint")
    result = train_model(
        experiment_config=config,
        data_config_path=args.data_config,
        train_split=train_split,
        valid_split=valid_split,
        dataset_view=dataset_view,
    )
    print_artifact(console, label="Best checkpoint", path=result.checkpoint_path)
    print_artifact(console, label="Training summary", path=result.summary_path)
    sidecar_path = emit_run_manifest(
        command_name="train",
        dataset_id=config.dataset_id,
        output_path=result.summary_path,
        inputs={
            "config_path": str(Path(args.config).resolve()),
            "data_config_path": str(Path(args.data_config).resolve()),
            "train_split": train_split,
            "valid_split": valid_split,
            "runtime_split_manifest_path": str(dataset_view.split_manifest_path),
            "runtime_session_catalog_path": str(dataset_view.session_catalog_path),
            "dataset_selection_active": dataset_view.selection_active,
            "resume_checkpoint": str(config.training.resume_checkpoint) if config.training.resume_checkpoint else None,
        },
        outputs={
            "checkpoint_path": str(result.checkpoint_path),
            "summary_path": str(result.summary_path),
            "best_epoch": result.best_epoch,
            "best_metric": result.best_metric,
        },
    )
    print_artifact(console, label="Run manifest", path=sidecar_path)
    console.print(f"Best epoch: {result.best_epoch}")
    console.print(f"Best predictive improvement: {result.best_metric:.6f}")
    return 0


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(_run(parse_args(argv)))


if __name__ == "__main__":
    main(sys.argv[1:])
