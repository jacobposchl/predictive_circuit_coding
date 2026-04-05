from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from predictive_circuit_coding.data import build_workspace, load_preparation_config, load_split_manifest
from predictive_circuit_coding.training.artifacts import load_training_checkpoint, write_run_manifest
from predictive_circuit_coding.utils import get_console
from predictive_circuit_coding.windowing.dataset import split_session_ids


def require_existing_file(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def require_split_manifest(data_config_path: str | Path):
    prep_config = load_preparation_config(data_config_path)
    workspace = build_workspace(prep_config)
    if not workspace.split_manifest_path.is_file():
        raise FileNotFoundError(
            "Split manifest not found. Run local data preparation first so "
            f"{workspace.split_manifest_path} exists."
        )
    split_manifest = load_split_manifest(workspace.split_manifest_path)
    return prep_config, workspace, split_manifest


def require_non_empty_split(*, data_config_path: str | Path, split_name: str) -> tuple[object, object, object]:
    prep_config, workspace, split_manifest = require_split_manifest(data_config_path)
    session_ids = split_session_ids(split_manifest, split_name)
    if not session_ids:
        raise ValueError(
            f"Split '{split_name}' has no prepared sessions in {workspace.split_manifest_path}. "
            "Prepare data and plan splits before running this command."
        )
    return prep_config, workspace, split_manifest


def require_checkpoint_matches_dataset(*, checkpoint_path: str | Path, dataset_id: str) -> Path:
    checkpoint = require_existing_file(checkpoint_path, label="Checkpoint")
    payload = load_training_checkpoint(checkpoint, map_location="cpu")
    metadata = payload.get("metadata", {})
    checkpoint_dataset_id = metadata.get("dataset_id")
    if checkpoint_dataset_id and checkpoint_dataset_id != dataset_id:
        raise ValueError(
            f"Checkpoint dataset_id '{checkpoint_dataset_id}' does not match config dataset_id '{dataset_id}'."
        )
    return checkpoint


def require_discovery_artifact_matches_dataset(*, artifact_path: str | Path, dataset_id: str) -> Path:
    import json

    artifact = require_existing_file(artifact_path, label="Discovery artifact")
    with artifact.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    artifact_dataset_id = payload.get("dataset_id")
    if artifact_dataset_id and artifact_dataset_id != dataset_id:
        raise ValueError(
            f"Discovery artifact dataset_id '{artifact_dataset_id}' does not match config dataset_id '{dataset_id}'."
        )
    return artifact


def emit_run_manifest(
    *,
    command_name: str,
    dataset_id: str,
    output_path: str | Path,
    inputs: dict[str, object],
    outputs: dict[str, object],
) -> Path:
    target = Path(output_path)
    sidecar = target.with_name(f"{target.stem}_{command_name}_run_manifest.json")
    return write_run_manifest(
        {
            "command_name": command_name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_id": dataset_id,
            "inputs": inputs,
            "outputs": outputs,
        },
        sidecar,
    )


def print_artifact(console, *, label: str, path: str | Path) -> None:
    console.print(f"[green]{label}[/green] {Path(path)}")


def warn(console, message: str) -> None:
    console.print(f"[yellow]{message}[/yellow]")


def get_cli_console():
    return get_console()
