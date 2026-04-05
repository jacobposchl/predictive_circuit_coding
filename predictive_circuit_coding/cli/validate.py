from __future__ import annotations

import argparse
import sys
from pathlib import Path

from predictive_circuit_coding.cli.common import (
    emit_run_manifest,
    get_cli_console,
    print_artifact,
    require_checkpoint_matches_dataset,
    require_discovery_artifact_matches_dataset,
    require_non_empty_split,
    warn,
)
from predictive_circuit_coding.training import (
    load_experiment_config,
    write_validation_summary,
    write_validation_summary_csv,
)
from predictive_circuit_coding.training.logging import StageLogger
from predictive_circuit_coding.validation import validate_discovery_artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run conservative validation checks for a discovery artifact.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    parser.add_argument("--data-config", required=True, help="Path to a preparation YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to a model checkpoint.")
    parser.add_argument("--discovery-artifact", required=True, help="Path to a discovery artifact JSON.")
    parser.add_argument("--output-json", default=None, help="Optional explicit path for the validation summary JSON.")
    parser.add_argument("--output-csv", default=None, help="Optional explicit path for the validation summary CSV.")
    return parser.parse_args(argv)


def _default_output_paths(discovery_artifact_path: str | Path) -> tuple[Path, Path]:
    artifact = Path(discovery_artifact_path)
    return (
        artifact.with_name(f"{artifact.stem}_validation.json"),
        artifact.with_name(f"{artifact.stem}_validation.csv"),
    )


def _run(args: argparse.Namespace) -> int:
    console = get_cli_console()
    config = load_experiment_config(args.config)
    logger = StageLogger(name="pcc-validate")
    logger.log_stage("preflight", expected_next="validation summary json/csv")
    require_non_empty_split(data_config_path=args.data_config, split_name=config.splits.discovery)
    require_non_empty_split(data_config_path=args.data_config, split_name=config.splits.test)
    checkpoint_path = require_checkpoint_matches_dataset(
        checkpoint_path=args.checkpoint,
        dataset_id=config.dataset_id,
    )
    discovery_artifact_path = require_discovery_artifact_matches_dataset(
        artifact_path=args.discovery_artifact,
        dataset_id=config.dataset_id,
    )
    logger.log_stage("validation", expected_next="validation summary json/csv")
    summary = validate_discovery_artifact(
        experiment_config=config,
        data_config_path=args.data_config,
        checkpoint_path=checkpoint_path,
        discovery_artifact_path=discovery_artifact_path,
    )
    default_json, default_csv = _default_output_paths(discovery_artifact_path)
    output_json = Path(args.output_json) if args.output_json else default_json
    output_csv = Path(args.output_csv) if args.output_csv else default_csv
    write_validation_summary(summary, output_json)
    write_validation_summary_csv(summary, output_csv)
    print_artifact(console, label="Validation summary", path=output_json)
    print_artifact(console, label="Validation CSV", path=output_csv)
    sidecar_path = emit_run_manifest(
        command_name="validate",
        dataset_id=config.dataset_id,
        output_path=output_json,
        inputs={
            "config_path": str(Path(args.config).resolve()),
            "data_config_path": str(Path(args.data_config).resolve()),
            "checkpoint_path": str(checkpoint_path),
            "discovery_artifact_path": str(discovery_artifact_path),
            "discovery_split": config.splits.discovery,
            "test_split": config.splits.test,
        },
        outputs={
            "validation_summary_json": str(output_json),
            "validation_summary_csv": str(output_csv),
        },
    )
    print_artifact(console, label="Run manifest", path=sidecar_path)
    console.print(f"Real probe accuracy: {summary.real_label_metrics.get('probe_accuracy', 0.0):.6f}")
    console.print(f"Shuffled probe accuracy: {summary.shuffled_label_metrics.get('probe_accuracy', 0.0):.6f}")
    if summary.recurrence_summary.get("recurrence_hit_count", 0) == 0:
        warn(
            console,
            "Validation found no held-out recurrence hits. This is not a crash condition, but it weakens the motif-stability story for this run.",
        )
    return 0


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(_run(parse_args(argv)))


if __name__ == "__main__":
    main(sys.argv[1:])
