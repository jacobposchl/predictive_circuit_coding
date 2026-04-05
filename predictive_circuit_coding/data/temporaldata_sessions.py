from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from predictive_circuit_coding.data.contracts import REQUIRED_PROVENANCE_FIELDS
from predictive_circuit_coding.data.manifest import write_temporaldata_session
from predictive_circuit_coding.utils.dependencies import ensure_optional_dependency

SPLIT_NAMES = ("train", "valid", "discovery", "test")


@dataclass(frozen=True)
class SessionSplitIntervals:
    train: tuple[tuple[float, float], ...]
    valid: tuple[tuple[float, float], ...]
    discovery: tuple[tuple[float, float], ...]
    test: tuple[tuple[float, float], ...]


def _ensure_dependencies() -> None:
    ensure_optional_dependency("temporaldata", package_name="temporaldata")


def _to_interval(segments: Iterable[tuple[float, float]]):
    from temporaldata import Interval

    pairs = list(segments)
    starts = np.asarray([start for start, _ in pairs], dtype=np.float64)
    ends = np.asarray([end for _, end in pairs], dtype=np.float64)
    return Interval(start=starts, end=ends)


def empty_interval():
    return _to_interval(())


def full_interval(start_s: float, end_s: float):
    return _to_interval(((float(start_s), float(end_s)),))


def build_split_intervals_for_assignment(
    *,
    domain_start_s: float,
    domain_end_s: float,
    assigned_split: str,
) -> SessionSplitIntervals:
    if assigned_split not in SPLIT_NAMES:
        raise ValueError(f"Unknown split: {assigned_split}")
    intervals = {name: () for name in SPLIT_NAMES}
    intervals[assigned_split] = ((float(domain_start_s), float(domain_end_s)),)
    return SessionSplitIntervals(
        train=tuple(intervals["train"]),
        valid=tuple(intervals["valid"]),
        discovery=tuple(intervals["discovery"]),
        test=tuple(intervals["test"]),
    )


def build_temporaldata_session(
    *,
    dataset_id: str,
    session_id: str,
    subject_id: str,
    duration_s: float,
    spike_timestamps_s: np.ndarray,
    spike_unit_index: np.ndarray,
    unit_ids: np.ndarray,
    unit_brain_regions: np.ndarray,
    unit_probe_depth_um: np.ndarray,
    split_intervals: SessionSplitIntervals,
):
    _ensure_dependencies()
    from temporaldata import ArrayDict, Data, IrregularTimeSeries

    domain = full_interval(0.0, float(duration_s))
    units = ArrayDict(
        id=np.asarray(unit_ids, dtype=object),
        brain_region=np.asarray(unit_brain_regions, dtype=object),
        probe_depth_um=np.asarray(unit_probe_depth_um, dtype=np.float64),
    )
    spikes = IrregularTimeSeries(
        timestamps=np.asarray(spike_timestamps_s, dtype=np.float64),
        unit_index=np.asarray(spike_unit_index, dtype=np.int64),
        domain=domain,
    )
    data = Data(
        brainset=Data(id=str(dataset_id)),
        session=Data(id=str(session_id)),
        subject=Data(id=str(subject_id)),
        spikes=spikes,
        units=units,
        provenance_fields=np.asarray(REQUIRED_PROVENANCE_FIELDS, dtype=object),
        domain=domain,
        train_domain=_to_interval(split_intervals.train),
        valid_domain=_to_interval(split_intervals.valid),
        discovery_domain=_to_interval(split_intervals.discovery),
        test_domain=_to_interval(split_intervals.test),
    )
    data.add_split_mask("train", data.train_domain)
    data.add_split_mask("valid", data.valid_domain)
    data.add_split_mask("discovery", data.discovery_domain)
    data.add_split_mask("test", data.test_domain)
    return data


def write_prepared_session(
    *,
    path: str | Path,
    dataset_id: str,
    session_id: str,
    subject_id: str,
    duration_s: float,
    spike_timestamps_s: np.ndarray,
    spike_unit_index: np.ndarray,
    unit_ids: np.ndarray,
    unit_brain_regions: np.ndarray,
    unit_probe_depth_um: np.ndarray,
    split_intervals: SessionSplitIntervals,
) -> Path:
    data = build_temporaldata_session(
        dataset_id=dataset_id,
        session_id=session_id,
        subject_id=subject_id,
        duration_s=duration_s,
        spike_timestamps_s=spike_timestamps_s,
        spike_unit_index=spike_unit_index,
        unit_ids=unit_ids,
        unit_brain_regions=unit_brain_regions,
        unit_probe_depth_um=unit_probe_depth_um,
        split_intervals=split_intervals,
    )
    return write_temporaldata_session(data, path=path)
