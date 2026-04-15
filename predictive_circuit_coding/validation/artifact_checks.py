from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_discovery_artifact(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Discovery artifact {path} must contain a JSON object.")
    return payload


def validate_discovery_artifact_identity(
    *,
    artifact: dict[str, Any],
    dataset_id: str,
    split_name: str | None = None,
    checkpoint_path: str | Path | None = None,
    target_label: str | None = None,
    require_fields: bool,
) -> None:
    artifact_dataset_id = artifact.get("dataset_id")
    if artifact_dataset_id is None:
        if require_fields:
            raise ValueError(
                "Discovery artifact dataset_id does not match the validation config dataset_id. "
                f"artifact='{artifact_dataset_id}', config='{dataset_id}'."
            )
    elif str(artifact_dataset_id) != str(dataset_id):
        raise ValueError(
            "Discovery artifact dataset_id does not match the validation config dataset_id. "
            f"artifact='{artifact_dataset_id}', config='{dataset_id}'."
        )

    if split_name is not None:
        artifact_split_name = artifact.get("split_name")
        if artifact_split_name is None:
            if require_fields:
                raise ValueError(
                    "Discovery artifact split_name does not match the validation discovery split. "
                    f"artifact='{artifact_split_name}', config='{split_name}'."
                )
        elif str(artifact_split_name) != str(split_name):
            raise ValueError(
                "Discovery artifact split_name does not match the validation discovery split. "
                f"artifact='{artifact_split_name}', config='{split_name}'."
            )

    if checkpoint_path is not None:
        artifact_checkpoint = artifact.get("checkpoint_path")
        artifact_checkpoint_resolved = (
            Path(artifact_checkpoint).resolve()
            if isinstance(artifact_checkpoint, (str, Path))
            else None
        )
        checkpoint_resolved = Path(checkpoint_path).resolve()
        if artifact_checkpoint_resolved is None:
            if require_fields:
                raise ValueError(
                    "Discovery artifact checkpoint_path does not match the checkpoint selected for validation. "
                    f"artifact={artifact_checkpoint_resolved}, checkpoint={checkpoint_resolved}."
                )
        elif artifact_checkpoint_resolved != checkpoint_resolved:
            raise ValueError(
                "Discovery artifact checkpoint_path does not match the checkpoint selected for validation. "
                f"artifact={artifact_checkpoint_resolved}, checkpoint={checkpoint_resolved}."
            )

    if target_label is not None:
        decoder_summary = artifact.get("decoder_summary")
        if not isinstance(decoder_summary, dict):
            if require_fields:
                raise ValueError("Discovery artifact decoder_summary must be present and must be an object.")
            return
        artifact_target_label = decoder_summary.get("target_label")
        if artifact_target_label is None:
            if require_fields:
                raise ValueError(
                    "Discovery artifact target label does not match the validation config target label. "
                    f"artifact='{artifact_target_label}', config='{target_label}'."
                )
        elif str(artifact_target_label) != str(target_label):
            raise ValueError(
                "Discovery artifact target label does not match the validation config target label. "
                f"artifact='{artifact_target_label}', config='{target_label}'."
            )


__all__ = [
    "load_discovery_artifact",
    "validate_discovery_artifact_identity",
]
