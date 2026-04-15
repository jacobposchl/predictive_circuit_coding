from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

from predictive_circuit_coding.utils.notebook_progress import NotebookProgressUI, NotebookStageSummary
from predictive_circuit_coding.workflows.contracts import (
    PIPELINE_STAGE_ORDER,
    PipelinePaths,
    PipelineRunManifest,
    PipelineStageState,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def path_identity(path: str | Path | None) -> str | None:
    return str(Path(path).resolve()) if path is not None else None


def resolve_latest_run_id(drive_export_root: str | Path) -> str | None:
    root = Path(drive_export_root)
    if not root.is_dir():
        return None
    candidates = sorted(path.name for path in root.iterdir() if path.is_dir() and path.name.startswith("run_"))
    return candidates[-1] if candidates else None


def build_pipeline_paths(
    *,
    local_artifact_root: str | Path,
    drive_export_root: str | Path | None,
    run_id: str,
    run_name: str = "run_1",
) -> PipelinePaths:
    local_run_root = Path(local_artifact_root) / "pipeline_runs" / str(run_id) / run_name
    drive_run_root = Path(drive_export_root) / str(run_id) / run_name if drive_export_root is not None else None
    pipeline_root = local_run_root / "pipeline"
    return PipelinePaths(
        run_id=str(run_id),
        local_run_root=local_run_root,
        drive_run_root=drive_run_root,
        train_root=local_run_root / "train",
        evaluation_root=local_run_root / "evaluation",
        refinement_root=local_run_root / "refinement",
        diagnostics_root=local_run_root / "diagnostics",
        reports_root=local_run_root / "reports",
        pipeline_root=pipeline_root,
        runtime_experiment_config_path=local_run_root / "train" / "colab_runtime_experiment.yaml",
        pipeline_config_snapshot_path=pipeline_root / "pipeline_config_snapshot.yaml",
        pipeline_manifest_path=pipeline_root / "pipeline_manifest.json",
        pipeline_state_path=pipeline_root / "pipeline_state.json",
    )


def sync_path(source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)


def sync_run_relatives(paths: PipelinePaths, relatives: Iterable[str]) -> None:
    if paths.drive_run_root is None:
        return
    for relative in relatives:
        sync_path(paths.local_run_root / relative, paths.drive_run_root / relative)


def load_pipeline_state(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    stages = payload.get("stages", payload)
    return {str(key): dict(value) for key, value in stages.items() if isinstance(value, dict)}


def write_pipeline_state(path: Path, states: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"stages": states}, indent=2), encoding="utf-8")


def set_stage_state(
    *,
    states: dict[str, dict[str, Any]],
    paths: PipelinePaths,
    stage_name: str,
    status: str,
    config_hash: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> None:
    previous = states.get(stage_name, {})
    states[stage_name] = PipelineStageState(
        stage_name=stage_name,
        status=status,
        config_hash=config_hash,
        inputs=inputs,
        outputs=outputs or {},
        created_at_utc=str(previous.get("created_at_utc") or utc_now()),
        updated_at_utc=utc_now(),
        error_message=error_message,
    ).to_dict()
    write_pipeline_state(paths.pipeline_state_path, states)


def stage_is_reusable(
    *,
    states: dict[str, dict[str, Any]],
    stage_name: str,
    config_hash: str,
    inputs: dict[str, Any],
) -> bool:
    state = states.get(stage_name) or {}
    return (
        state.get("status") in {"complete", "reused"}
        and state.get("config_hash") == config_hash
        and dict(state.get("inputs") or {}) == inputs
    )


def write_pipeline_manifest(paths: PipelinePaths, *, dataset_id: str, created_at_utc: str) -> None:
    manifest = PipelineRunManifest(
        run_id=paths.run_id,
        dataset_id=dataset_id,
        stage_order=PIPELINE_STAGE_ORDER,
        local_run_root=str(paths.local_run_root),
        drive_run_root=str(paths.drive_run_root) if paths.drive_run_root is not None else None,
        config_snapshot_path=str(paths.pipeline_config_snapshot_path),
        created_at_utc=created_at_utc,
        updated_at_utc=utc_now(),
    )
    paths.pipeline_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    paths.pipeline_manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")


def summary_rows_from_json(path: Path, root_key: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get(root_key, [])
    return [dict(row) for row in rows if isinstance(row, dict)]


def stage_summary(stage_name: str, outputs: dict[str, Any], status: str) -> NotebookStageSummary:
    return NotebookStageSummary(
        stage_name=stage_name,
        status=status,
        headline=f"{stage_name}: {status}",
        artifact_paths={key: str(value) for key, value in outputs.items() if value},
    )


def mark_stage_reused(
    *,
    paths: PipelinePaths,
    states: dict[str, dict[str, Any]],
    stage_name: str,
    config_hash: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    progress_ui: NotebookProgressUI | None = None,
) -> dict[str, Any]:
    set_stage_state(
        states=states,
        paths=paths,
        stage_name=stage_name,
        status="reused",
        config_hash=config_hash,
        inputs=inputs,
        outputs=outputs,
    )
    sync_run_relatives(paths, ("pipeline",))
    if progress_ui is not None:
        progress_ui.finish_stage(stage_summary(stage_name, outputs, "reused"))
        progress_ui.advance_pipeline()
    return outputs


def mark_stage_skipped(
    *,
    paths: PipelinePaths,
    states: dict[str, dict[str, Any]],
    stage_name: str,
    config_hash: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
) -> dict[str, Any]:
    set_stage_state(
        states=states,
        paths=paths,
        stage_name=stage_name,
        status="skipped",
        config_hash=config_hash,
        inputs=inputs,
        outputs=outputs,
    )
    sync_run_relatives(paths, ("pipeline",))
    return outputs


def mark_stage_running(
    *,
    paths: PipelinePaths,
    states: dict[str, dict[str, Any]],
    stage_name: str,
    config_hash: str,
    inputs: dict[str, Any],
) -> None:
    set_stage_state(
        states=states,
        paths=paths,
        stage_name=stage_name,
        status="running",
        config_hash=config_hash,
        inputs=inputs,
        outputs={},
    )
    sync_run_relatives(paths, ("pipeline",))


def mark_stage_complete(
    *,
    paths: PipelinePaths,
    states: dict[str, dict[str, Any]],
    stage_name: str,
    config_hash: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    synced_relatives: Iterable[str] = (),
    progress_ui: NotebookProgressUI | None = None,
) -> dict[str, Any]:
    set_stage_state(
        states=states,
        paths=paths,
        stage_name=stage_name,
        status="complete",
        config_hash=config_hash,
        inputs=inputs,
        outputs=outputs,
    )
    sync_run_relatives(paths, tuple(synced_relatives) + ("pipeline",))
    if progress_ui is not None:
        progress_ui.finish_stage(stage_summary(stage_name, outputs, "complete"))
        progress_ui.advance_pipeline()
    return outputs


def mark_stage_failed(
    *,
    paths: PipelinePaths,
    states: dict[str, dict[str, Any]],
    stage_name: str,
    config_hash: str,
    inputs: dict[str, Any],
    error: Exception,
    progress_ui: NotebookProgressUI | None = None,
) -> None:
    set_stage_state(
        states=states,
        paths=paths,
        stage_name=stage_name,
        status="failed",
        config_hash=config_hash,
        inputs=inputs,
        outputs={},
        error_message=str(error),
    )
    sync_run_relatives(paths, ("pipeline",))
    if progress_ui is not None:
        progress_ui.fail_stage(stage_name=stage_name, error_message=str(error), debug_log_path=None, tail_lines=())
