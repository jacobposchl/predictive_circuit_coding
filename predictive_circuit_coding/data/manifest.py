from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from predictive_circuit_coding.data.contracts import PREPARED_SESSION_FILE_SUFFIX
from predictive_circuit_coding.data.layout import PreparationWorkspace
from predictive_circuit_coding.utils.dependencies import ensure_optional_dependency


@dataclass(frozen=True)
class SessionRecord:
    recording_id: str
    session_id: str
    subject_id: str
    raw_data_path: str
    duration_s: float
    n_units: int
    brain_regions: tuple[str, ...]
    trial_count: int
    prepared_session_path: str | None = None


@dataclass(frozen=True)
class SessionManifest:
    dataset_id: str
    source_name: str
    records: tuple[SessionRecord, ...]


def write_session_manifest(manifest: SessionManifest, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_id": manifest.dataset_id,
        "source_name": manifest.source_name,
        "records": [asdict(record) for record in manifest.records],
    }
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return target


def load_session_manifest(path: str | Path) -> SessionManifest:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = tuple(
        SessionRecord(
            recording_id=str(item["recording_id"]),
            session_id=str(item["session_id"]),
            subject_id=str(item["subject_id"]),
            raw_data_path=str(item["raw_data_path"]),
            duration_s=float(item["duration_s"]),
            n_units=int(item["n_units"]),
            brain_regions=tuple(str(region) for region in item.get("brain_regions", [])),
            trial_count=int(item["trial_count"]),
            prepared_session_path=str(item["prepared_session_path"]) if item.get("prepared_session_path") else None,
        )
        for item in payload["records"]
    )
    return SessionManifest(
        dataset_id=str(payload["dataset_id"]),
        source_name=str(payload["source_name"]),
        records=records,
    )


def default_prepared_session_path(workspace: PreparationWorkspace, recording_id: str) -> Path:
    session_id = recording_id.split("/")[-1]
    return workspace.brainset_prepared_root / f"{session_id}{PREPARED_SESSION_FILE_SUFFIX}"


def write_temporaldata_session(data: Any, *, path: str | Path) -> Path:
    ensure_optional_dependency("temporaldata", package_name="temporaldata")
    ensure_optional_dependency("h5py", package_name="h5py")
    import h5py
    serialize_fn_map = None
    try:
        from brainsets import serialize_fn_map as brainsets_serialize_fn_map

        serialize_fn_map = brainsets_serialize_fn_map
    except RuntimeError:
        serialize_fn_map = None
    except ImportError:
        serialize_fn_map = None

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(target, "w") as handle:
        if serialize_fn_map is None:
            data.to_hdf5(handle)
        else:
            data.to_hdf5(handle, serialize_fn_map=serialize_fn_map)
    return target
