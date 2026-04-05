from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import json
from pathlib import Path
from typing import Any

from predictive_circuit_coding.data.config import DataPreparationConfig
from predictive_circuit_coding.data.layout import PreparationWorkspace
from predictive_circuit_coding.data.manifest import SessionManifest, SessionRecord
from predictive_circuit_coding.data.processed_sessions import PreparedSessionScan, scan_prepared_session


@dataclass(frozen=True)
class SessionCatalogRecord:
    recording_id: str
    session_id: str
    subject_id: str
    raw_data_path: str
    duration_s: float
    n_units: int
    brain_regions: tuple[str, ...]
    trial_count: int
    prepared_session_path: str | None = None
    behavior_session_id: str | None = None
    session_type: str | None = None
    image_set: str | None = None
    experience_level: str | None = None
    session_number: int | None = None
    project_code: str | None = None
    date_of_acquisition: str | None = None
    allen_unit_count: int | None = None
    probe_count: int | None = None
    channel_count: int | None = None
    prior_exposures_to_image_set: float | None = None
    prior_exposures_to_omissions: float | None = None
    abnormal_histology: str | None = None
    abnormal_activity: str | None = None
    genotype: str | None = None
    sex: str | None = None
    equipment_name: str | None = None
    allen_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["brain_regions"] = list(self.brain_regions)
        return payload


@dataclass(frozen=True)
class SessionCatalog:
    dataset_id: str
    source_name: str
    records: tuple[SessionCatalogRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "source_name": self.source_name,
            "records": [record.to_dict() for record in self.records],
        }


def _normalize_scalar(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_session_id(value: Any) -> str | None:
    text = _normalize_scalar(value)
    if text is None:
        return None
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def _optional_int(value: Any) -> int | None:
    text = _normalize_scalar(value)
    if text is None:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _optional_float(value: Any) -> float | None:
    text = _normalize_scalar(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _resolve_project_metadata_file(raw_root: str | Path, filename: str) -> Path | None:
    root = Path(raw_root).resolve()
    candidates = [
        root / "project_metadata" / filename,
        root / filename,
    ]
    for dataset_dir in sorted(root.glob("visual-behavior-neuropixels-*")):
        candidates.append(dataset_dir / "project_metadata" / filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _load_ecephys_session_metadata(raw_root: str | Path) -> dict[str, dict[str, Any]]:
    path = _resolve_project_metadata_file(raw_root, "ecephys_sessions.csv")
    if path is None:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            session_id = _normalize_session_id(row.get("ecephys_session_id"))
            if session_id is None:
                continue
            normalized = {str(key): _normalize_scalar(value) for key, value in row.items()}
            rows[session_id] = normalized
    return rows


def _promote_allen_metadata(scan: PreparedSessionScan, row: dict[str, Any] | None) -> SessionCatalogRecord:
    metadata = dict(row or {})
    promoted_keys = {
        "ecephys_session_id",
        "behavior_session_id",
        "session_type",
        "image_set",
        "experience_level",
        "session_number",
        "project_code",
        "date_of_acquisition",
        "unit_count",
        "probe_count",
        "channel_count",
        "prior_exposures_to_image_set",
        "prior_exposures_to_omissions",
        "abnormal_histology",
        "abnormal_activity",
        "genotype",
        "sex",
        "equipment_name",
        "mouse_id",
        "structure_acronyms",
        "file_id",
        "age_in_days",
    }
    return SessionCatalogRecord(
        recording_id=scan.recording_id,
        session_id=scan.session_id,
        subject_id=scan.subject_id,
        raw_data_path=scan.raw_data_path,
        duration_s=scan.duration_s,
        n_units=scan.n_units,
        brain_regions=scan.brain_regions,
        trial_count=scan.trial_count,
        prepared_session_path=scan.prepared_session_path,
        behavior_session_id=_normalize_session_id(metadata.get("behavior_session_id")),
        session_type=_normalize_scalar(metadata.get("session_type")),
        image_set=_normalize_scalar(metadata.get("image_set")),
        experience_level=_normalize_scalar(metadata.get("experience_level")),
        session_number=_optional_int(metadata.get("session_number")),
        project_code=_normalize_scalar(metadata.get("project_code")),
        date_of_acquisition=_normalize_scalar(metadata.get("date_of_acquisition")),
        allen_unit_count=_optional_int(metadata.get("unit_count")),
        probe_count=_optional_int(metadata.get("probe_count")),
        channel_count=_optional_int(metadata.get("channel_count")),
        prior_exposures_to_image_set=_optional_float(metadata.get("prior_exposures_to_image_set")),
        prior_exposures_to_omissions=_optional_float(metadata.get("prior_exposures_to_omissions")),
        abnormal_histology=_normalize_scalar(metadata.get("abnormal_histology")),
        abnormal_activity=_normalize_scalar(metadata.get("abnormal_activity")),
        genotype=_normalize_scalar(metadata.get("genotype")),
        sex=_normalize_scalar(metadata.get("sex")),
        equipment_name=_normalize_scalar(metadata.get("equipment_name")),
        allen_metadata={
            key: value
            for key, value in metadata.items()
            if key not in promoted_keys and value not in (None, "")
        },
    )


def build_session_catalog_from_prepared_sessions(
    config: DataPreparationConfig,
    *,
    workspace: PreparationWorkspace,
) -> SessionCatalog:
    raw_root = config.allen_sdk.cache_root or workspace.raw
    allen_metadata = _load_ecephys_session_metadata(raw_root)
    records: list[SessionCatalogRecord] = []
    for path in sorted(workspace.brainset_prepared_root.glob("*.h5")):
        scan = scan_prepared_session(
            path,
            dataset_id=config.dataset.dataset_id,
            raw_root=raw_root,
        )
        records.append(_promote_allen_metadata(scan, allen_metadata.get(scan.session_id)))
    return SessionCatalog(
        dataset_id=config.dataset.dataset_id,
        source_name=config.dataset.source_name,
        records=tuple(records),
    )


def project_catalog_to_session_manifest(catalog: SessionCatalog) -> SessionManifest:
    return SessionManifest(
        dataset_id=catalog.dataset_id,
        source_name=catalog.source_name,
        records=tuple(
            SessionRecord(
                recording_id=record.recording_id,
                session_id=record.session_id,
                subject_id=record.subject_id,
                raw_data_path=record.raw_data_path,
                duration_s=record.duration_s,
                n_units=record.n_units,
                brain_regions=record.brain_regions,
                trial_count=record.trial_count,
                prepared_session_path=record.prepared_session_path,
            )
            for record in catalog.records
        ),
    )


def build_session_catalog_from_manifest(manifest: SessionManifest) -> SessionCatalog:
    return SessionCatalog(
        dataset_id=manifest.dataset_id,
        source_name=manifest.source_name,
        records=tuple(
            SessionCatalogRecord(
                recording_id=record.recording_id,
                session_id=record.session_id,
                subject_id=record.subject_id,
                raw_data_path=record.raw_data_path,
                duration_s=record.duration_s,
                n_units=record.n_units,
                brain_regions=record.brain_regions,
                trial_count=record.trial_count,
                prepared_session_path=record.prepared_session_path,
            )
            for record in manifest.records
        ),
    )


def write_session_catalog(catalog: SessionCatalog, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(catalog.to_dict(), handle, indent=2)
    return target


def load_session_catalog(path: str | Path) -> SessionCatalog:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = tuple(
        SessionCatalogRecord(
            recording_id=str(item["recording_id"]),
            session_id=str(item["session_id"]),
            subject_id=str(item["subject_id"]),
            raw_data_path=str(item["raw_data_path"]),
            duration_s=float(item["duration_s"]),
            n_units=int(item["n_units"]),
            brain_regions=tuple(str(region) for region in item.get("brain_regions", [])),
            trial_count=int(item["trial_count"]),
            prepared_session_path=str(item["prepared_session_path"]) if item.get("prepared_session_path") else None,
            behavior_session_id=str(item["behavior_session_id"]) if item.get("behavior_session_id") else None,
            session_type=str(item["session_type"]) if item.get("session_type") else None,
            image_set=str(item["image_set"]) if item.get("image_set") else None,
            experience_level=str(item["experience_level"]) if item.get("experience_level") else None,
            session_number=int(item["session_number"]) if item.get("session_number") is not None else None,
            project_code=str(item["project_code"]) if item.get("project_code") else None,
            date_of_acquisition=str(item["date_of_acquisition"]) if item.get("date_of_acquisition") else None,
            allen_unit_count=int(item["allen_unit_count"]) if item.get("allen_unit_count") is not None else None,
            probe_count=int(item["probe_count"]) if item.get("probe_count") is not None else None,
            channel_count=int(item["channel_count"]) if item.get("channel_count") is not None else None,
            prior_exposures_to_image_set=(
                float(item["prior_exposures_to_image_set"])
                if item.get("prior_exposures_to_image_set") is not None
                else None
            ),
            prior_exposures_to_omissions=(
                float(item["prior_exposures_to_omissions"])
                if item.get("prior_exposures_to_omissions") is not None
                else None
            ),
            abnormal_histology=str(item["abnormal_histology"]) if item.get("abnormal_histology") else None,
            abnormal_activity=str(item["abnormal_activity"]) if item.get("abnormal_activity") else None,
            genotype=str(item["genotype"]) if item.get("genotype") else None,
            sex=str(item["sex"]) if item.get("sex") else None,
            equipment_name=str(item["equipment_name"]) if item.get("equipment_name") else None,
            allen_metadata=dict(item.get("allen_metadata", {})),
        )
        for item in payload["records"]
    )
    return SessionCatalog(
        dataset_id=str(payload["dataset_id"]),
        source_name=str(payload["source_name"]),
        records=records,
    )


def write_session_catalog_csv(catalog: SessionCatalog, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "recording_id",
        "session_id",
        "subject_id",
        "raw_data_path",
        "duration_s",
        "n_units",
        "brain_regions",
        "trial_count",
        "prepared_session_path",
        "behavior_session_id",
        "session_type",
        "image_set",
        "experience_level",
        "session_number",
        "project_code",
        "date_of_acquisition",
        "allen_unit_count",
        "probe_count",
        "channel_count",
        "prior_exposures_to_image_set",
        "prior_exposures_to_omissions",
        "abnormal_histology",
        "abnormal_activity",
        "genotype",
        "sex",
        "equipment_name",
        "allen_metadata_json",
    ]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in catalog.records:
            writer.writerow(
                {
                    "recording_id": record.recording_id,
                    "session_id": record.session_id,
                    "subject_id": record.subject_id,
                    "raw_data_path": record.raw_data_path,
                    "duration_s": f"{record.duration_s:.6f}",
                    "n_units": record.n_units,
                    "brain_regions": ",".join(record.brain_regions),
                    "trial_count": record.trial_count,
                    "prepared_session_path": record.prepared_session_path or "",
                    "behavior_session_id": record.behavior_session_id or "",
                    "session_type": record.session_type or "",
                    "image_set": record.image_set or "",
                    "experience_level": record.experience_level or "",
                    "session_number": record.session_number if record.session_number is not None else "",
                    "project_code": record.project_code or "",
                    "date_of_acquisition": record.date_of_acquisition or "",
                    "allen_unit_count": record.allen_unit_count if record.allen_unit_count is not None else "",
                    "probe_count": record.probe_count if record.probe_count is not None else "",
                    "channel_count": record.channel_count if record.channel_count is not None else "",
                    "prior_exposures_to_image_set": (
                        record.prior_exposures_to_image_set
                        if record.prior_exposures_to_image_set is not None
                        else ""
                    ),
                    "prior_exposures_to_omissions": (
                        record.prior_exposures_to_omissions
                        if record.prior_exposures_to_omissions is not None
                        else ""
                    ),
                    "abnormal_histology": record.abnormal_histology or "",
                    "abnormal_activity": record.abnormal_activity or "",
                    "genotype": record.genotype or "",
                    "sex": record.sex or "",
                    "equipment_name": record.equipment_name or "",
                    "allen_metadata_json": json.dumps(record.allen_metadata, sort_keys=True),
                }
            )
    return target
