from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from predictive_circuit_coding.data.config import DataPreparationConfig
from predictive_circuit_coding.data.layout import PreparationWorkspace
from predictive_circuit_coding.data.manifest import SessionManifest, SessionRecord, write_temporaldata_session
from predictive_circuit_coding.data.splits import SPLIT_NAMES, SplitManifest
from predictive_circuit_coding.utils.dependencies import ensure_optional_dependency


@dataclass(frozen=True)
class PreparedSessionScan:
    recording_id: str
    session_id: str
    subject_id: str
    raw_data_path: str
    duration_s: float
    n_units: int
    brain_regions: tuple[str, ...]
    trial_count: int
    prepared_session_path: str


@dataclass(frozen=True)
class UploadBundleManifest:
    dataset_id: str
    workspace_root: str
    upload_processed_only: bool
    files: tuple[str, ...]


def _ensure_temporaldata() -> None:
    ensure_optional_dependency("temporaldata", package_name="temporaldata")
    ensure_optional_dependency("h5py", package_name="h5py")


def load_temporaldata_session(path: str | Path, *, lazy: bool = False):
    _ensure_temporaldata()
    import h5py
    from temporaldata import Data

    with h5py.File(Path(path), "r") as handle:
        return Data.from_hdf5(handle, lazy=lazy)


def _safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default) if obj is not None else default


def _normalize_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _extract_domain_duration_s(data) -> float:
    domain = _safe_attr(data, "domain")
    if domain is None or len(domain) == 0:
        return 0.0
    return float(np.max(domain.end) - np.min(domain.start))


def _extract_unit_count(data) -> int:
    units = _safe_attr(data, "units")
    unit_ids = _safe_attr(units, "id")
    if unit_ids is None:
        return 0
    return int(len(unit_ids))


def _extract_brain_regions(data) -> tuple[str, ...]:
    units = _safe_attr(data, "units")
    if units is None:
        return ()
    candidates = (
        _safe_attr(units, "brain_region"),
        _safe_attr(units, "structure_acronym"),
        _safe_attr(units, "ecephys_structure_acronym"),
        _safe_attr(units, "location"),
    )
    for values in candidates:
        if values is None:
            continue
        normalized = sorted(
            {
                _normalize_string(value)
                for value in np.asarray(values).tolist()
                if _normalize_string(value) and _normalize_string(value) != "nan"
            }
        )
        if normalized:
            return tuple(normalized)
    return ()


def _extract_trial_count(data) -> int:
    for key in ("trials", "stimulus_presentations"):
        value = _safe_attr(data, key)
        starts = _safe_attr(value, "start")
        if starts is not None:
            return int(len(starts))
    return 0


def _resolve_raw_session_path(raw_root: str | Path, session_id: str) -> Path:
    root = Path(raw_root).resolve()
    candidates: list[Path] = []

    # AllenSDK cache roots may point at:
    # 1. the top-level parent containing the versioned dataset directory
    # 2. the versioned dataset directory itself
    # 3. the behavior_ecephys_sessions directory directly
    candidates.append(root / "behavior_ecephys_sessions" / session_id)
    for dataset_dir in sorted(root.glob("visual-behavior-neuropixels-*")):
        candidates.append(dataset_dir / "behavior_ecephys_sessions" / session_id)
    candidates.append(root / session_id)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve() if candidates else (root / session_id).resolve()


def scan_prepared_session(
    path: str | Path,
    *,
    dataset_id: str,
    raw_root: str | Path,
) -> PreparedSessionScan:
    session_path = Path(path).resolve()
    data = load_temporaldata_session(session_path, lazy=False)
    session_id = _normalize_string(_safe_attr(_safe_attr(data, "session"), "id", session_path.stem))
    subject_id = _normalize_string(_safe_attr(_safe_attr(data, "subject"), "id", "unknown_subject"))
    return PreparedSessionScan(
        recording_id=f"{dataset_id}/{session_id}",
        session_id=session_id,
        subject_id=subject_id,
        raw_data_path=str(_resolve_raw_session_path(raw_root, session_id)),
        duration_s=_extract_domain_duration_s(data),
        n_units=_extract_unit_count(data),
        brain_regions=_extract_brain_regions(data),
        trial_count=_extract_trial_count(data),
        prepared_session_path=str(session_path),
    )


def build_session_manifest_from_prepared_sessions(
    config: DataPreparationConfig,
    *,
    workspace: PreparationWorkspace,
) -> SessionManifest:
    records: list[SessionRecord] = []
    for path in sorted(workspace.brainset_prepared_root.glob("*.h5")):
        scan = scan_prepared_session(
            path,
            dataset_id=config.dataset.dataset_id,
            raw_root=config.allen_sdk.cache_root or workspace.raw,
        )
        records.append(
            SessionRecord(
                recording_id=scan.recording_id,
                session_id=scan.session_id,
                subject_id=scan.subject_id,
                raw_data_path=scan.raw_data_path,
                duration_s=scan.duration_s,
                n_units=scan.n_units,
                brain_regions=scan.brain_regions,
                trial_count=scan.trial_count,
                prepared_session_path=scan.prepared_session_path,
            )
        )
    return SessionManifest(
        dataset_id=config.dataset.dataset_id,
        source_name=config.dataset.source_name,
        records=tuple(records),
    )


def _empty_interval():
    from temporaldata import Interval

    return Interval(
        start=np.asarray([], dtype=np.float64),
        end=np.asarray([], dtype=np.float64),
    )


def _full_domain_interval(data):
    from temporaldata import Interval

    domain = data.domain
    return Interval(
        start=np.asarray(domain.start, dtype=np.float64),
        end=np.asarray(domain.end, dtype=np.float64),
    )


def _strip_split_state(obj: Any) -> None:
    keys = list(obj.keys()) if hasattr(obj, "keys") else list(getattr(obj, "__dict__", {}))
    for key in keys:
        if key.endswith("_mask"):
            delattr(obj, key)
            continue
        if key in {f"{name}_domain" for name in SPLIT_NAMES}:
            delattr(obj, key)
            continue
        value = getattr(obj, key)
        if hasattr(value, "keys"):
            _strip_split_state(value)


def _set_split_domains(data, *, assigned_split: str) -> None:
    empty = _empty_interval()
    full = _full_domain_interval(data)
    data.train_domain = empty
    data.valid_domain = empty
    data.discovery_domain = empty
    data.test_domain = empty
    if assigned_split == "train":
        data.set_train_domain(full)
    elif assigned_split == "valid":
        data.set_valid_domain(full)
    elif assigned_split == "discovery":
        data.discovery_domain = full
        data.add_split_mask("discovery", full)
    elif assigned_split == "test":
        data.set_test_domain(full)
    else:
        raise ValueError(f"Unsupported split: {assigned_split}")


def apply_split_assignments_to_prepared_sessions(
    *,
    workspace: PreparationWorkspace,
    split_manifest: SplitManifest,
) -> None:
    assignments = {item.recording_id: item.split for item in split_manifest.assignments}
    for recording_id, split_name in assignments.items():
        session_id = recording_id.split("/", 1)[1]
        path = workspace.brainset_prepared_root / f"{session_id}.h5"
        data = load_temporaldata_session(path, lazy=False)
        _strip_split_state(data)
        _set_split_domains(data, assigned_split=split_name)
        write_temporaldata_session(data, path=path)


def write_upload_bundle_manifest(
    *,
    workspace: PreparationWorkspace,
    dataset_id: str,
    upload_processed_only: bool,
) -> Path:
    files = [
        path.relative_to(workspace.root).as_posix()
        for path in sorted(workspace.brainset_prepared_root.glob("*.h5"))
    ]
    files.extend(
        [
            workspace.session_manifest_path.relative_to(workspace.root).as_posix(),
            workspace.split_manifest_path.relative_to(workspace.root).as_posix(),
        ]
    )
    for split_name in SPLIT_NAMES:
        config_path = workspace.splits / f"torch_brain_{split_name}.yaml"
        if config_path.is_file():
            files.append(config_path.relative_to(workspace.root).as_posix())
    manifest = UploadBundleManifest(
        dataset_id=dataset_id,
        workspace_root=str(workspace.root),
        upload_processed_only=upload_processed_only,
        files=tuple(files),
    )
    target = workspace.manifests / "upload_bundle.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(asdict(manifest), handle, indent=2)
    return target
