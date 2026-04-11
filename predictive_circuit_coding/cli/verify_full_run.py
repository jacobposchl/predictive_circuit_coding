from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.table import Table

from predictive_circuit_coding.benchmarks.verification import verify_full_run_readiness
from predictive_circuit_coding.cli.common import get_cli_console, print_artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that a full notebook pipeline config is safe to launch before spending Colab time."
    )
    parser.add_argument(
        "--pipeline-config",
        required=True,
        help="Path to the full pipeline YAML, e.g. configs/pcc/pipeline_cross_session_aug_full.yaml.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/full_run_verification",
        help="Directory for verification JSON/CSV outputs.",
    )
    return parser.parse_args(argv)


def _run(args: argparse.Namespace) -> int:
    console = get_cli_console()
    result = verify_full_run_readiness(
        pipeline_config_path=args.pipeline_config,
        output_root=args.output_root,
    )
    console.print(f"[bold]Full-run verification status:[/bold] {result.status}")
    console.print(f"Training variant: {result.training_variant_name}")
    console.print(f"Training epochs: {result.training_num_epochs}")
    console.print(f"Split counts: {result.split_counts}")

    coverage_table = Table(title="Task Coverage Gate")
    for column in (
        "task",
        "split",
        "status",
        "scanned",
        "positive",
        "negative",
        "selected_positive",
        "selected_negative",
        "positive_sessions",
    ):
        coverage_table.add_column(column)
    for row in result.coverage_rows:
        coverage_table.add_row(
            row.task_name,
            row.split_name,
            row.status,
            str(row.total_scanned_windows),
            str(row.positive_window_count),
            str(row.negative_window_count),
            str(row.selected_positive_count),
            str(row.selected_negative_count),
            str(row.positive_session_count),
        )
    console.print(coverage_table)

    if result.issues:
        issue_table = Table(title="Blocking Issues")
        issue_table.add_column("gate")
        issue_table.add_column("severity")
        issue_table.add_column("message")
        for issue in result.issues:
            issue_table.add_row(issue.gate, issue.severity, issue.message)
        console.print(issue_table)

    print_artifact(console, label="Verification summary", path=result.summary_json_path)
    print_artifact(console, label="Task coverage CSV", path=result.coverage_csv_path)
    return 0 if result.ok else 2


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(_run(parse_args(argv)))


if __name__ == "__main__":
    main(sys.argv[1:])
