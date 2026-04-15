from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from predictive_circuit_coding.data import resolve_runtime_dataset_view
from predictive_circuit_coding.training.artifacts import load_training_checkpoint, write_run_manifest
from predictive_circuit_coding.utils import get_console
from predictive_circuit_coding.validation.artifact_checks import (
    load_discovery_artifact,
    validate_discovery_artifact_identity,
)
from predictive_circuit_coding.windowing.dataset import split_session_ids


def require_existing_file(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def require_runtime_view(*, experiment_config, data_config_path: str | Path):
    return resolve_runtime_dataset_view(
        experiment_config=experiment_config,
        data_config_path=data_config_path,
    )


def require_non_empty_split(*, dataset_view, split_name: str):
    session_ids = split_session_ids(dataset_view.split_manifest, split_name)
    if not session_ids:
        raise ValueError(
            f"Split '{split_name}' has no prepared sessions in {dataset_view.split_manifest_path}. "
            "Prepare data and materialize the requested dataset view before running this command."
        )
    return dataset_view


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
    artifact = require_existing_file(artifact_path, label="Discovery artifact")
    payload = load_discovery_artifact(artifact)
    validate_discovery_artifact_identity(
        artifact=payload,
        dataset_id=dataset_id,
        require_fields=False,
    )
    return artifact


def require_discovery_artifact_matches_validation_inputs(
    *,
    artifact_path: str | Path,
    dataset_id: str,
    checkpoint_path: str | Path,
    target_label: str,
) -> Path:
    artifact = require_existing_file(artifact_path, label="Discovery artifact")
    payload = load_discovery_artifact(artifact)
    validate_discovery_artifact_identity(
        artifact=payload,
        dataset_id=dataset_id,
        checkpoint_path=checkpoint_path,
        target_label=target_label,
        require_fields=False,
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
