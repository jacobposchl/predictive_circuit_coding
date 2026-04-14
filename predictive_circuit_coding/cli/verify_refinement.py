from __future__ import annotations

import argparse
import sys

from rich.table import Table

from predictive_circuit_coding.benchmarks.verification import verify_refinement_readiness
from predictive_circuit_coding.cli.common import get_cli_console, print_artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify that a refinement pipeline config is ready to launch.")
    parser.add_argument("--pipeline-config", required=True, help="Path to the refinement pipeline YAML.")
    parser.add_argument("--output-root", default="artifacts/refinement_verification", help="Directory for verification JSON/CSV outputs.")
    return parser.parse_args(argv)


def _run(args: argparse.Namespace) -> int:
    console = get_cli_console()
    result = verify_refinement_readiness(
        pipeline_config_path=args.pipeline_config,
        output_root=args.output_root,
    )
    console.print(f"[bold]Refinement verification status:[/bold] {result.status}")
    console.print(f"Variant: {result.variant_name}")
    console.print(f"Split counts: {result.split_counts}")

    coverage_table = Table(title="Task Coverage")
    for column in ("task", "split", "status", "scanned", "positive", "negative", "positive_sessions"):
        coverage_table.add_column(column)
    for row in result.coverage_rows:
        coverage_table.add_row(
            row.task_name,
            row.split_name,
            row.status,
            str(row.total_scanned_windows),
            str(row.positive_window_count),
            str(row.negative_window_count),
            str(row.positive_session_count),
        )
    console.print(coverage_table)

    if result.issues:
        issue_table = Table(title="Issues")
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
