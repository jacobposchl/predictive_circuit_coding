from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from predictive_circuit_coding.data.config import DataPreparationConfig
from predictive_circuit_coding.data.layout import PreparationWorkspace


def build_brainsets_runner_command(
    config: DataPreparationConfig,
    *,
    workspace: PreparationWorkspace,
    session_ids_file: str | Path | None = None,
    max_sessions: int | None = None,
    reprocess: bool = False,
    redownload: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "brainsets.runner",
        str(config.brainsets_pipeline.local_pipeline_path),
        "--raw-dir",
        str(config.allen_sdk.cache_root or workspace.raw),
        "--processed-dir",
        str(workspace.prepared),
        "--cores",
        str(config.brainsets_pipeline.runner_cores),
    ]
    resolved_session_ids = Path(session_ids_file).resolve() if session_ids_file else config.brainsets_pipeline.default_session_ids_file
    resolved_max_sessions = max_sessions if max_sessions is not None else config.brainsets_pipeline.default_max_sessions
    if resolved_session_ids is not None:
        command.extend(["--session-ids-file", str(resolved_session_ids)])
    if resolved_max_sessions is not None:
        command.extend(["--max-sessions", str(int(resolved_max_sessions))])
    if not config.unit_filtering.filter_by_validity:
        command.append("--no-filter-by-validity")
    if not config.unit_filtering.filter_out_of_brain_units:
        command.append("--no-filter-out-of-brain-units")
    if config.unit_filtering.amplitude_cutoff_maximum is not None:
        command.extend(["--amplitude-cutoff-maximum", str(float(config.unit_filtering.amplitude_cutoff_maximum))])
    if config.unit_filtering.presence_ratio_minimum is not None:
        command.extend(["--presence-ratio-minimum", str(float(config.unit_filtering.presence_ratio_minimum))])
    if config.unit_filtering.isi_violations_maximum is not None:
        command.extend(["--isi-violations-maximum", str(float(config.unit_filtering.isi_violations_maximum))])
    if reprocess:
        command.append("--reprocess")
    if redownload:
        command.append("--redownload")
    return command


def run_brainsets_pipeline(
    config: DataPreparationConfig,
    *,
    workspace: PreparationWorkspace,
    session_ids_file: str | Path | None = None,
    max_sessions: int | None = None,
    reprocess: bool = False,
    redownload: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = build_brainsets_runner_command(
        config,
        workspace=workspace,
        session_ids_file=session_ids_file,
        max_sessions=max_sessions,
        reprocess=reprocess,
        redownload=redownload,
    )
    return subprocess.run(command, check=True, text=True)
