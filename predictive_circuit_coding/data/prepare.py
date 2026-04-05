from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path

from predictive_circuit_coding.data.config import DataPreparationConfig
from predictive_circuit_coding.data.layout import PreparationWorkspace, create_workspace
from predictive_circuit_coding.data.manifest import SessionManifest, SessionRecord, default_prepared_session_path, write_session_manifest
from predictive_circuit_coding.data.splits import SplitManifest, build_split_manifest, write_split_manifest


@dataclass(frozen=True)
class PreparationSummary:
    workspace: PreparationWorkspace
    session_manifest: SessionManifest | None
    split_manifest: SplitManifest | None


def _parse_brain_regions(raw_value: str) -> tuple[str, ...]:
    items = [item.strip() for item in raw_value.replace(";", ",").split(",")]
    return tuple(item for item in items if item)


def build_session_manifest_from_table(
    config: DataPreparationConfig,
    *,
    input_path: str | Path,
    workspace: PreparationWorkspace,
) -> SessionManifest:
    path = Path(input_path)
    if config.preparation.session_table_format != "csv":
        raise ValueError("Stage 1 currently supports only CSV session tables")

    records: list[SessionRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            session_id = str(row[config.preparation.session_id_field]).strip()
            subject_id = str(row[config.preparation.subject_id_field]).strip()
            raw_data_path = str(row[config.preparation.raw_path_field]).strip()
            recording_id = config.preparation.recording_id_template.format(
                dataset_id=config.dataset.dataset_id,
                session_id=session_id,
                subject_id=subject_id,
            )
            prepared_session_path = default_prepared_session_path(workspace, recording_id)
            records.append(
                SessionRecord(
                    recording_id=recording_id,
                    session_id=session_id,
                    subject_id=subject_id,
                    raw_data_path=raw_data_path,
                    duration_s=float(row[config.preparation.duration_field]),
                    n_units=int(row[config.preparation.n_units_field]),
                    brain_regions=_parse_brain_regions(str(row[config.preparation.brain_regions_field])),
                    trial_count=int(row[config.preparation.trial_count_field]),
                    prepared_session_path=str(prepared_session_path),
                )
            )
    if not records:
        raise ValueError(f"No session records found in {path}")
    return SessionManifest(
        dataset_id=config.dataset.dataset_id,
        source_name=config.dataset.source_name,
        records=tuple(records),
    )


def prepare_workspace(
    config: DataPreparationConfig,
    *,
    input_path: str | Path | None = None,
    build_splits: bool = False,
) -> PreparationSummary:
    workspace = create_workspace(config)
    session_manifest = None
    split_manifest = None
    if input_path is not None:
        session_manifest = build_session_manifest_from_table(
            config,
            input_path=input_path,
            workspace=workspace,
        )
        write_session_manifest(session_manifest, workspace.session_manifest_path)
        if build_splits:
            split_manifest = build_split_manifest(session_manifest, config=config.splits)
            write_split_manifest(split_manifest, workspace.split_manifest_path)
    return PreparationSummary(
        workspace=workspace,
        session_manifest=session_manifest,
        split_manifest=split_manifest,
    )
