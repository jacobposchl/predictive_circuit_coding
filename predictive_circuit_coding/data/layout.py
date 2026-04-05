from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from predictive_circuit_coding.data.config import DataPreparationConfig


@dataclass(frozen=True)
class PreparationWorkspace:
    root: Path
    raw: Path
    prepared: Path
    brainset_prepared_root: Path
    manifests: Path
    splits: Path
    logs: Path
    session_manifest_path: Path
    split_manifest_path: Path


def build_workspace(config: DataPreparationConfig) -> PreparationWorkspace:
    root = config.dataset.workspace_root
    prepared = root / config.dataset.prepared_subdir
    manifests = root / config.dataset.manifests_subdir
    splits = root / config.dataset.splits_subdir
    brainset_prepared_root = prepared / config.dataset.dataset_id
    return PreparationWorkspace(
        root=root,
        raw=root / config.dataset.raw_subdir,
        prepared=prepared,
        brainset_prepared_root=brainset_prepared_root,
        manifests=manifests,
        splits=splits,
        logs=root / config.dataset.logs_subdir,
        session_manifest_path=manifests / config.dataset.session_manifest_name,
        split_manifest_path=splits / config.dataset.split_manifest_name,
    )


def create_workspace(config: DataPreparationConfig) -> PreparationWorkspace:
    workspace = build_workspace(config)
    paths = [
        workspace.root,
        workspace.prepared,
        workspace.brainset_prepared_root,
        workspace.manifests,
        workspace.splits,
        workspace.logs,
    ]
    if config.allen_sdk.cache_root is None:
        paths.append(workspace.raw)
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return workspace
