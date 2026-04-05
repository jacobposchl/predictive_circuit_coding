from __future__ import annotations

import argparse
import sys
from pathlib import Path

from predictive_circuit_coding.data import (
    apply_split_assignments_to_prepared_sessions,
    build_brainsets_runner_command,
    build_session_manifest_from_prepared_sessions,
    build_session_manifest_from_table,
    build_split_manifest,
    create_workspace,
    load_split_manifest,
    load_preparation_config,
    load_session_manifest,
    run_brainsets_pipeline,
    write_session_manifest,
    write_split_manifest,
    write_upload_bundle_manifest,
)
from predictive_circuit_coding.data.layout import build_workspace
from predictive_circuit_coding.utils import build_dependency_table, collect_dependency_status, get_console


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local CPU-first data preparation commands for predictive circuit coding.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_env = subparsers.add_parser("check-env", help="Show dependency and Python compatibility status.")
    check_env.set_defaults(func=_run_check_env)

    init_workspace = subparsers.add_parser("init-workspace", help="Create the canonical local data workspace.")
    init_workspace.add_argument("--config", required=True, help="Path to a YAML preparation config.")
    init_workspace.set_defaults(func=_run_init_workspace)

    build_manifest = subparsers.add_parser("build-session-manifest", help="Build a standardized session manifest from a CSV table.")
    build_manifest.add_argument("--config", required=True, help="Path to a YAML preparation config.")
    build_manifest.add_argument("--input", required=True, help="CSV table describing sessions.")
    build_manifest.set_defaults(func=_run_build_session_manifest)

    plan_splits = subparsers.add_parser("plan-splits", help="Build the deterministic split manifest from the session manifest.")
    plan_splits.add_argument("--config", required=True, help="Path to a YAML preparation config.")
    plan_splits.add_argument("--manifest", default=None, help="Optional explicit path to the session manifest JSON.")
    plan_splits.set_defaults(func=_run_plan_splits)

    build_dataset = subparsers.add_parser("build-dataset-config", help="Build a split-specific torch_brain dataset config from the split manifest.")
    build_dataset.add_argument("--config", required=True, help="Path to a YAML preparation config.")
    build_dataset.add_argument("--split", required=True, choices=["train", "valid", "discovery", "test"], help="Split to materialize.")
    build_dataset.add_argument("--manifest", default=None, help="Optional explicit path to the split manifest JSON.")
    build_dataset.set_defaults(func=_run_build_dataset_config)

    inspect_windows = subparsers.add_parser("inspect-windows", help="Inspect fixed windows from a split-specific torch_brain dataset.")
    inspect_windows.add_argument("--config", required=True, help="Path to a YAML preparation config.")
    inspect_windows.add_argument("--split", required=True, choices=["train", "valid", "discovery", "test"], help="Split to inspect.")
    inspect_windows.add_argument("--window-length", required=True, type=float, help="Window length in seconds.")
    inspect_windows.add_argument("--step", type=float, default=None, help="Optional sequential step size in seconds.")
    inspect_windows.add_argument("--limit", type=int, default=5, help="Maximum number of windows to display.")
    inspect_windows.add_argument("--random", action="store_true", help="Use the random fixed-window sampler instead of sequential.")
    inspect_windows.set_defaults(func=_run_inspect_windows)

    prepare_allen = subparsers.add_parser(
        "prepare-allen-visual-behavior-neuropixels",
        help="Run the local brainsets Allen Visual Behavior Neuropixels pipeline and build an upload-ready processed bundle.",
    )
    prepare_allen.add_argument("--config", required=True, help="Path to a YAML preparation config.")
    prepare_allen.add_argument("--session-ids-file", default=None, help="Optional text file containing one session id per line.")
    prepare_allen.add_argument("--max-sessions", type=int, default=None, help="Optional cap for subset runs.")
    prepare_allen.add_argument("--cleanup-raw", action="store_true", help="Delete the Allen raw cache after successful processing.")
    prepare_allen.add_argument("--reprocess", action="store_true", help="Rebuild processed session files even if they already exist.")
    prepare_allen.add_argument("--redownload", action="store_true", help="Force AllenSDK to redownload source session files when supported.")
    prepare_allen.set_defaults(func=_run_prepare_allen_visual_behavior_neuropixels)

    prepare_allen_legacy = subparsers.add_parser(
        "prepare-allen-neuropixels",
        help="Deprecated alias for prepare-allen-visual-behavior-neuropixels.",
    )
    prepare_allen_legacy.add_argument("--config", required=True, help="Path to a YAML preparation config.")
    prepare_allen_legacy.add_argument("--session-ids-file", default=None, help="Optional text file containing one session id per line.")
    prepare_allen_legacy.add_argument("--max-sessions", type=int, default=None, help="Optional cap for subset runs.")
    prepare_allen_legacy.add_argument("--cleanup-raw", action="store_true", help="Delete the Allen raw cache after successful processing.")
    prepare_allen_legacy.add_argument("--reprocess", action="store_true", help="Rebuild processed session files even if they already exist.")
    prepare_allen_legacy.add_argument("--redownload", action="store_true", help="Force AllenSDK to redownload source session files when supported.")
    prepare_allen_legacy.set_defaults(func=_run_prepare_allen_visual_behavior_neuropixels)
    return parser.parse_args(argv)


def _run_check_env(_: argparse.Namespace) -> int:
    console = get_console()
    console.print(build_dependency_table(collect_dependency_status()))
    return 0


def _run_init_workspace(args: argparse.Namespace) -> int:
    console = get_console()
    config = load_preparation_config(args.config)
    workspace = create_workspace(config)
    raw_dir = config.allen_sdk.cache_root or workspace.raw
    console.print(f"[green]Workspace ready[/green] at {workspace.root}")
    console.print(f"Raw data dir      : {raw_dir}")
    console.print(f"Prepared data dir : {workspace.brainset_prepared_root}")
    console.print(f"Manifest dir      : {workspace.manifests}")
    console.print(f"Split dir         : {workspace.splits}")
    return 0


def _run_build_session_manifest(args: argparse.Namespace) -> int:
    console = get_console()
    config = load_preparation_config(args.config)
    workspace = create_workspace(config)
    manifest = build_session_manifest_from_table(config, input_path=args.input, workspace=workspace)
    write_session_manifest(manifest, workspace.session_manifest_path)
    console.print(f"[green]Wrote session manifest[/green] to {workspace.session_manifest_path}")
    console.print(f"Sessions: {len(manifest.records)}")
    return 0


def _run_plan_splits(args: argparse.Namespace) -> int:
    console = get_console()
    config = load_preparation_config(args.config)
    workspace = build_workspace(config)
    manifest_path = args.manifest or workspace.session_manifest_path
    session_manifest = load_session_manifest(manifest_path)
    split_manifest = build_split_manifest(session_manifest, config=config.splits)
    write_split_manifest(split_manifest, workspace.split_manifest_path)
    counts: dict[str, int] = {}
    for item in split_manifest.assignments:
        counts[item.split] = counts.get(item.split, 0) + 1
    console.print(f"[green]Wrote split manifest[/green] to {workspace.split_manifest_path}")
    for split_name in ("train", "valid", "discovery", "test"):
        console.print(f"{split_name:>10}: {counts.get(split_name, 0)} recordings")
    return 0


def _run_build_dataset_config(args: argparse.Namespace) -> int:
    from predictive_circuit_coding.windowing import build_torch_brain_config

    console = get_console()
    config = load_preparation_config(args.config)
    workspace = create_workspace(config)
    split_manifest_path = args.manifest or workspace.split_manifest_path
    split_manifest = load_split_manifest(split_manifest_path)
    config_path = build_torch_brain_config(
        workspace=workspace,
        dataset_id=split_manifest.dataset_id,
        session_ids=[
            assignment.recording_id.split("/", 1)[1]
            for assignment in split_manifest.assignments
            if assignment.split == args.split
        ],
        split=args.split,
    )
    console.print(f"[green]Wrote torch_brain config[/green] to {config_path}")
    return 0


def _run_inspect_windows(args: argparse.Namespace) -> int:
    from predictive_circuit_coding.windowing import (
        FixedWindowConfig,
        build_dataset_bundle,
        describe_sampler_windows,
    )

    console = get_console()
    config = load_preparation_config(args.config)
    workspace = build_workspace(config)
    split_manifest = load_split_manifest(workspace.split_manifest_path)
    bundle = build_dataset_bundle(
        workspace=workspace,
        split_manifest=split_manifest,
        split=args.split,
    )
    window = FixedWindowConfig(
        window_length_s=float(args.window_length),
        step_s=args.step,
        seed=0,
        drop_short=False,
    )
    if args.random:
        from predictive_circuit_coding.windowing import build_random_fixed_window_sampler

        sampler = build_random_fixed_window_sampler(bundle.dataset, window=window)
    else:
        from predictive_circuit_coding.windowing import build_sequential_fixed_window_sampler

        sampler = build_sequential_fixed_window_sampler(bundle.dataset, window=window)
    rows = describe_sampler_windows(sampler, limit=args.limit)
    console.print(f"Split          : {args.split}")
    console.print(f"Dataset config : {bundle.config_path}")
    console.print(f"Window count   : showing first {len(rows)}")
    for item in rows:
        console.print(f"{item.recording_id} [{item.start_s:.3f}, {item.end_s:.3f}]")
    return 0


def _build_all_split_configs(*, workspace, split_manifest) -> list[Path]:
    from predictive_circuit_coding.windowing import build_torch_brain_config

    paths: list[Path] = []
    for split_name in ("train", "valid", "discovery", "test"):
        paths.append(
            build_torch_brain_config(
                workspace=workspace,
                dataset_id=split_manifest.dataset_id,
                session_ids=[
                    assignment.recording_id.split("/", 1)[1]
                    for assignment in split_manifest.assignments
                    if assignment.split == split_name
                ],
                split=split_name,
            )
        )
    return paths


def _run_prepare_allen_visual_behavior_neuropixels(args: argparse.Namespace) -> int:
    console = get_console()
    config = load_preparation_config(args.config)
    workspace = create_workspace(config)
    command = build_brainsets_runner_command(
        config,
        workspace=workspace,
        session_ids_file=args.session_ids_file,
        max_sessions=args.max_sessions,
        reprocess=args.reprocess,
        redownload=args.redownload,
    )
    console.print("[bold]Brainsets runner[/bold]")
    console.print(" ".join(command))
    run_brainsets_pipeline(
        config,
        workspace=workspace,
        session_ids_file=args.session_ids_file,
        max_sessions=args.max_sessions,
        reprocess=args.reprocess,
        redownload=args.redownload,
    )

    manifest = build_session_manifest_from_prepared_sessions(config, workspace=workspace)
    if not manifest.records:
        raise RuntimeError(
            "The Allen Visual Behavior Neuropixels pipeline completed without producing any processed sessions."
        )
    write_session_manifest(manifest, workspace.session_manifest_path)
    split_manifest = build_split_manifest(manifest, config=config.splits)
    write_split_manifest(split_manifest, workspace.split_manifest_path)
    apply_split_assignments_to_prepared_sessions(workspace=workspace, split_manifest=split_manifest)
    config_paths = _build_all_split_configs(workspace=workspace, split_manifest=split_manifest)
    upload_manifest = write_upload_bundle_manifest(
        workspace=workspace,
        dataset_id=config.dataset.dataset_id,
        upload_processed_only=config.brainsets_pipeline.processed_only_upload,
    )

    if args.cleanup_raw or config.allen_sdk.cleanup_raw_after_processing:
        raw_root = config.allen_sdk.cache_root or workspace.raw
        target = raw_root / config.dataset.dataset_id
        if target.exists():
            import shutil

            shutil.rmtree(target)
            console.print(f"[yellow]Removed raw cache[/yellow] at {target}")

    console.print(f"[green]Wrote session manifest[/green] to {workspace.session_manifest_path}")
    console.print(f"[green]Wrote split manifest[/green] to {workspace.split_manifest_path}")
    for config_path in config_paths:
        console.print(f"[green]Wrote dataset config[/green] to {config_path}")
    console.print(f"[green]Wrote upload manifest[/green] to {upload_manifest}")
    console.print(f"Prepared sessions: {len(manifest.records)}")
    return 0


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main(sys.argv[1:])
