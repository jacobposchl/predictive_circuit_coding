from __future__ import annotations

from pathlib import Path

import yaml

from predictive_circuit_coding.data import build_workspace, load_preparation_config, load_session_catalog
from predictive_circuit_coding.training import load_experiment_config
from predictive_circuit_coding.workflows.notebook_runtime import materialize_notebook_prepared_sessions


def write_runtime_experiment_config(
    *,
    base_experiment_config: str | Path,
    runtime_experiment_config_path: Path,
    step_log_every: int,
    artifact_root: Path | None = None,
) -> Path:
    source_config = load_experiment_config(base_experiment_config)
    payload = source_config.to_dict()
    payload.pop("config_path", None)
    payload.setdefault("training", {})
    payload["training"]["log_every_steps"] = int(step_log_every)
    resolved_artifact_root = (artifact_root or runtime_experiment_config_path.parent).resolve()
    payload.setdefault("artifacts", {})
    payload["artifacts"]["checkpoint_dir"] = str((resolved_artifact_root / "checkpoints").resolve())
    payload["artifacts"]["summary_path"] = str((resolved_artifact_root / "training_summary.json").resolve())
    runtime_experiment_config_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_experiment_config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return runtime_experiment_config_path


def resolve_source_session_ids(*, source_dataset_root: str | Path, dataset_id: str) -> list[str]:
    source_root = Path(source_dataset_root).resolve()
    catalog_path = source_root / "manifests" / "session_catalog.json"
    if catalog_path.is_file():
        catalog = load_session_catalog(catalog_path)
        session_ids = sorted({record.session_id for record in catalog.records})
        if session_ids:
            return session_ids
    prepared_root = source_root / "prepared" / str(dataset_id)
    session_ids = sorted(path.stem for path in prepared_root.glob("*.h5"))
    if session_ids:
        return session_ids
    raise FileNotFoundError(
        f"No prepared sessions found under {prepared_root}. Populate the source dataset root or disable local staging."
    )


def ensure_local_prepared_sessions(
    *,
    data_config_path: str | Path,
    source_dataset_root: str | Path | None,
    stage_prepared_sessions_locally: bool,
) -> None:
    prep_config = load_preparation_config(data_config_path)
    workspace = build_workspace(prep_config)
    if any(workspace.brainset_prepared_root.glob("*.h5")):
        return
    if not stage_prepared_sessions_locally:
        return
    if source_dataset_root is None:
        raise FileNotFoundError(
            "No prepared sessions were found under the local workspace and paths.source_dataset_root is not set. "
            "Either stage prepared sessions locally or run local data preparation before training."
        )
    if Path(source_dataset_root).resolve() == workspace.root.resolve():
        return
    session_ids = resolve_source_session_ids(
        source_dataset_root=source_dataset_root,
        dataset_id=prep_config.dataset.dataset_id,
    )
    materialize_notebook_prepared_sessions(
        source_dataset_root=source_dataset_root,
        target_dataset_root=workspace.root,
        session_ids=session_ids,
        dataset_id=prep_config.dataset.dataset_id,
        reset_target=True,
    )
