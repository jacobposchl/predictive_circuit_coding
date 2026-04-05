# /// brainset-pipeline
# python-version = "3.10"
# dependencies = [
#   "allensdk==2.16.2",
#   "brainsets==0.2.0",
#   "temporaldata==0.1.1",
#   "numpy==1.23.5",
#   "pandas==1.5.3",
#   "scipy==1.10.1",
#   "h5py==3.8.0",
#   "pynwb==2.2.0",
#   "psycopg2-binary==2.9.10",
# ]
# ///

from __future__ import annotations

from argparse import ArgumentParser
import datetime as dt
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from temporaldata import ArrayDict, Data, Interval, IrregularTimeSeries

from brainsets import serialize_fn_map
from brainsets.descriptions import (
    BrainsetDescription,
    DeviceDescription,
    SessionDescription,
    SubjectDescription,
)
from brainsets.pipeline import BrainsetPipeline
from brainsets.taxonomy import RecordingTech, Sex, Species
from brainsets.taxonomy.mice import BrainRegion


parser = ArgumentParser()
parser.add_argument("--session-ids-file", type=Path, default=None)
parser.add_argument("--max-sessions", type=int, default=None)
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")
parser.set_defaults(
    filter_by_validity=True,
    filter_out_of_brain_units=True,
)
parser.add_argument("--filter-by-validity", dest="filter_by_validity", action="store_true")
parser.add_argument("--no-filter-by-validity", dest="filter_by_validity", action="store_false")
parser.add_argument("--filter-out-of-brain-units", dest="filter_out_of_brain_units", action="store_true")
parser.add_argument("--no-filter-out-of-brain-units", dest="filter_out_of_brain_units", action="store_false")
parser.add_argument("--amplitude-cutoff-maximum", type=float, default=0.1)
parser.add_argument("--presence-ratio-minimum", type=float, default=0.95)
parser.add_argument("--isi-violations-maximum", type=float, default=0.5)


def _import_cache_cls():
    try:
        from allensdk.brain_observatory.behavior.behavior_project_cache.behavior_neuropixels_project_cache import (
            VisualBehaviorNeuropixelsProjectCache,
        )
    except ImportError as exc:
        raise RuntimeError(
            "AllenSDK is required for Allen Visual Behavior Neuropixels preparation. "
            "Install the dedicated prep environment before running this pipeline."
        ) from exc
    return VisualBehaviorNeuropixelsProjectCache


def _build_cache(raw_dir: Path):
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_cls = _import_cache_cls()
    return cache_cls.from_s3_cache(cache_dir=raw_dir)


def _load_requested_session_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def _safe_string(value: Any, *, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    return str(value)


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    return float(value)


def _safe_datetime(value: Any) -> dt.datetime:
    if value is None:
        return dt.datetime(1970, 1, 1)
    try:
        if pd.isna(value):
            return dt.datetime(1970, 1, 1)
    except TypeError:
        pass
    return pd.Timestamp(value).to_pydatetime()


def _normalize_session_table(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.reset_index()
    if "ecephys_session_id" not in frame.columns:
        for candidate in ("session_id", "id", "index"):
            if candidate in frame.columns:
                frame = frame.rename(columns={candidate: "ecephys_session_id"})
                break
    if "ecephys_session_id" not in frame.columns:
        raise ValueError(f"Could not find ecephys session ids in columns: {list(frame.columns)}")
    frame["session_id"] = frame["ecephys_session_id"].astype(str)
    frame["id"] = frame["session_id"]
    return frame


def _merge_units_and_channels(
    session,
    *,
    filter_by_validity: bool,
    filter_out_of_brain_units: bool,
    amplitude_cutoff_maximum: float | None,
    presence_ratio_minimum: float | None,
    isi_violations_maximum: float | None,
) -> pd.DataFrame:
    units = session.get_units(
        filter_by_validity=filter_by_validity,
        filter_out_of_brain_units=filter_out_of_brain_units,
        amplitude_cutoff_maximum=amplitude_cutoff_maximum,
        presence_ratio_minimum=presence_ratio_minimum,
        isi_violations_maximum=isi_violations_maximum,
    ).copy()
    channels = session.get_channels(filter_by_validity=filter_by_validity).copy()
    units.index = units.index.astype(int)
    channels.index = channels.index.astype(int)
    merged = units.merge(channels, left_on="peak_channel_id", right_index=True, how="left")
    merged.index = units.index
    return merged.sort_index()


def _flatten_spike_times(spike_times: dict[int, np.ndarray], unit_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    timestamps: list[np.ndarray] = []
    unit_index: list[np.ndarray] = []
    retained_unit_ids: list[int] = []
    for position, unit_id in enumerate(unit_ids):
        unit_spikes = np.asarray(spike_times.get(unit_id, []), dtype=np.float64)
        if unit_spikes.size == 0:
            continue
        timestamps.append(unit_spikes)
        unit_index.append(np.full(unit_spikes.shape, len(retained_unit_ids), dtype=np.int64))
        retained_unit_ids.append(unit_id)
    if not timestamps:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.int64)
    flat_timestamps = np.concatenate(timestamps)
    flat_unit_index = np.concatenate(unit_index)
    order = np.argsort(flat_timestamps, kind="stable")
    return flat_timestamps[order], flat_unit_index[order]


def _retained_units(unit_table: pd.DataFrame, spike_times: dict[int, np.ndarray]) -> pd.DataFrame:
    retained_ids = [int(unit_id) for unit_id in unit_table.index if len(np.asarray(spike_times.get(int(unit_id), []))) > 0]
    if not retained_ids:
        return unit_table.iloc[0:0].copy()
    return unit_table.loc[retained_ids].copy()


def _to_interval(frame: pd.DataFrame, *, start_col: str = "start_time", stop_col: str = "stop_time") -> Interval | None:
    if frame is None or frame.empty or start_col not in frame.columns or stop_col not in frame.columns:
        return None
    frame = frame.dropna(subset=[start_col, stop_col]).copy()
    if frame.empty:
        return None
    columns = [start_col, stop_col]
    for candidate in (
        "active",
        "is_change",
        "omitted",
        "stimulus_block",
        "stimulus_block_name",
        "stimulus_name",
        "image_name",
        "rewarded",
        "flashes_since_change",
        "trials_id",
        "position_x",
        "position_y",
        "color",
        "contrast",
        "orientation",
        "duration",
        "level",
        "condition",
        "response_time",
        "change_frame",
        "go",
        "catch",
        "hit",
        "miss",
        "false_alarm",
        "correct_reject",
        "aborted",
        "auto_rewarded",
    ):
        if candidate in frame.columns:
            columns.append(candidate)
    interval_frame = frame.loc[:, columns].rename(columns={start_col: "start", stop_col: "end"})
    interval = Interval.from_dataframe(interval_frame)
    interval.timestamps = (interval.start + interval.end) / 2.0
    return interval


def _dominant_visual_region(structures: pd.Series) -> BrainRegion | None:
    if structures is None:
        return None
    counts = structures.value_counts()
    for structure_name in counts.index.tolist():
        try:
            return BrainRegion.from_string(str(structure_name))
        except Exception:
            continue
    return None


def _build_units_array(unit_table: pd.DataFrame) -> ArrayDict:
    quality = unit_table.get("quality", pd.Series(["good"] * len(unit_table), index=unit_table.index))
    amplitude_cutoff = unit_table.get("amplitude_cutoff", pd.Series([np.inf] * len(unit_table), index=unit_table.index))
    presence_ratio = unit_table.get("presence_ratio", pd.Series([0.0] * len(unit_table), index=unit_table.index))
    snr = unit_table.get("snr", pd.Series([0.0] * len(unit_table), index=unit_table.index))
    firing_rate = unit_table.get("firing_rate", pd.Series([0.0] * len(unit_table), index=unit_table.index))
    isi_violations = unit_table.get("isi_violations", pd.Series([0.0] * len(unit_table), index=unit_table.index))
    return ArrayDict(
        id=unit_table.index.astype(str).to_numpy(dtype=object),
        quality=quality.fillna("unknown").astype(str).to_numpy(dtype=object),
        structure_acronym=unit_table["structure_acronym"].fillna("unknown").astype(str).to_numpy(dtype=object),
        peak_channel_id=unit_table["peak_channel_id"].fillna(-1).astype(int).astype(str).to_numpy(dtype=object),
        probe_id=unit_table["probe_id"].fillna(-1).astype(int).astype(str).to_numpy(dtype=object),
        probe_vertical_position=unit_table["probe_vertical_position"].fillna(0.0).astype(float).to_numpy(dtype=np.float64),
        probe_horizontal_position=unit_table["probe_horizontal_position"].fillna(0.0).astype(float).to_numpy(dtype=np.float64),
        amplitude_cutoff=amplitude_cutoff.fillna(np.inf).astype(float).to_numpy(dtype=np.float64),
        presence_ratio=presence_ratio.fillna(0.0).astype(float).to_numpy(dtype=np.float64),
        snr=snr.fillna(0.0).astype(float).to_numpy(dtype=np.float64),
        firing_rate=firing_rate.fillna(0.0).astype(float).to_numpy(dtype=np.float64),
        isi_violations=isi_violations.fillna(0.0).astype(float).to_numpy(dtype=np.float64),
    )


def _max_interval_end(interval: Interval | None) -> float:
    if interval is None or len(interval) == 0:
        return 0.0
    return float(np.max(interval.end))


class Pipeline(BrainsetPipeline):
    brainset_id = "allen_visual_behavior_neuropixels"
    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir: Path, args) -> pd.DataFrame:
        cache = _build_cache(raw_dir)
        manifest = _normalize_session_table(cache.get_ecephys_session_table())
        requested_session_ids = _load_requested_session_ids(args.session_ids_file)
        if requested_session_ids is not None:
            manifest = manifest.loc[manifest["session_id"].isin(requested_session_ids)]
        manifest = manifest.sort_values("ecephys_session_id")
        if args.max_sessions is not None:
            manifest = manifest.head(int(args.max_sessions))
        return manifest.set_index("id")

    def download(self, manifest_item):
        self.update_status("DOWNLOADING")
        cache = _build_cache(self.raw_dir)
        session_id = int(manifest_item.ecephys_session_id)
        session = cache.get_ecephys_session(ecephys_session_id=session_id)
        return session, manifest_item._asdict()

    def process(self, download_output):
        session, manifest_item = download_output
        session_id = str(manifest_item["session_id"])
        store_path = self.processed_dir / f"{session_id}.h5"
        self.processed_dir.mkdir(exist_ok=True, parents=True)
        if store_path.exists() and not self.args.reprocess:
            self.update_status("Skipped Processing")
            return

        self.update_status("Extracting Metadata")
        metadata = dict(session.metadata)
        unit_table = _merge_units_and_channels(
            session,
            filter_by_validity=bool(self.args.filter_by_validity),
            filter_out_of_brain_units=bool(self.args.filter_out_of_brain_units),
            amplitude_cutoff_maximum=self.args.amplitude_cutoff_maximum,
            presence_ratio_minimum=self.args.presence_ratio_minimum,
            isi_violations_maximum=self.args.isi_violations_maximum,
        )
        unit_table = _retained_units(unit_table, session.spike_times)
        spike_timestamps_s, spike_unit_index = _flatten_spike_times(session.spike_times, unit_table.index.astype(int).tolist())

        stimulus_presentations = _to_interval(session.stimulus_presentations)
        trials = _to_interval(getattr(session, "trials", None))
        optotagging = _to_interval(getattr(session, "optotagging_table", None))

        domain_end_s = max(
            float(spike_timestamps_s.max()) if spike_timestamps_s.size else 0.0,
            _max_interval_end(stimulus_presentations),
            _max_interval_end(trials),
            _max_interval_end(optotagging),
        )
        domain = Interval(
            start=np.asarray([0.0], dtype=np.float64),
            end=np.asarray([domain_end_s], dtype=np.float64),
        )
        spikes = IrregularTimeSeries(
            timestamps=spike_timestamps_s,
            unit_index=spike_unit_index,
            domain=domain,
        )

        dominant_region = None
        if not unit_table.empty and "structure_acronym" in unit_table.columns:
            dominant_region = _dominant_visual_region(unit_table["structure_acronym"])

        subject = SubjectDescription(
            id=_safe_string(metadata.get("mouse_id"), default=f"mouse_{session_id}"),
            species=Species.MUS_MUSCULUS,
            age=_safe_float(metadata.get("age_in_days"), default=0.0),
            sex=Sex.from_string(_safe_string(metadata.get("sex"), default="U")),
            genotype=_safe_string(metadata.get("full_genotype"), default="unknown"),
        )
        session_description = SessionDescription(
            id=session_id,
            recording_date=_safe_datetime(metadata.get("date_of_acquisition")),
        )
        device = DeviceDescription(
            id=_safe_string(metadata.get("equipment_name"), default=f"neuropixels_{session_id}"),
            recording_tech=RecordingTech.NEUROPIXELS_SPIKES,
            target_area=dominant_region,
        )
        brainset = BrainsetDescription(
            id=self.brainset_id,
            origin_version="allen_visual_behavior_neuropixels",
            derived_version="1.0.0",
            source="https://allensdk.readthedocs.io/en/stable/visual_behavior_neuropixels_quickstart.html",
            description="Allen Institute Visual Behavior Neuropixels sessions converted to temporaldata for predictive circuit coding.",
        )

        payload = {
            "brainset": brainset,
            "subject": subject,
            "session": session_description,
            "device": device,
            "units": _build_units_array(unit_table),
            "spikes": spikes,
            "domain": domain,
        }
        if stimulus_presentations is not None:
            payload["stimulus_presentations"] = stimulus_presentations
        if trials is not None:
            payload["trials"] = trials
        if optotagging is not None:
            payload["optotagging"] = optotagging

        data = Data(**payload)

        self.update_status("Storing")
        with h5py.File(store_path, "w") as handle:
            data.to_hdf5(handle, serialize_fn_map=serialize_fn_map)
