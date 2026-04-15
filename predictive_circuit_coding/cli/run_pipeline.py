from __future__ import annotations

import argparse
import sys

from predictive_circuit_coding.cli.common import get_cli_console, print_artifact
from predictive_circuit_coding.workflows import run_pipeline_from_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the end-to-end predictive circuit coding pipeline from a pipeline YAML config.")
    parser.add_argument("--pipeline-config", required=True, help="Path to a pipeline YAML config.")
    parser.add_argument("--run-id", default=None, help="Optional explicit pipeline run id.")
    return parser.parse_args(argv)


def _run(args: argparse.Namespace) -> int:
    console = get_cli_console()
    result = run_pipeline_from_config(
        pipeline_config_path=args.pipeline_config,
        pipeline_run_id=args.run_id,
    )
    print_artifact(console, label="Pipeline manifest", path=result.pipeline_manifest_path)
    print_artifact(console, label="Pipeline state", path=result.pipeline_state_path)
    print_artifact(console, label="Checkpoint", path=result.checkpoint_path)
    print_artifact(console, label="Refinement summary", path=result.refinement_summary_json_path)
    print_artifact(console, label="Final summary", path=result.final_summary_json_path)
    return 0


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(_run(parse_args(argv)))


if __name__ == "__main__":
    main(sys.argv[1:])
