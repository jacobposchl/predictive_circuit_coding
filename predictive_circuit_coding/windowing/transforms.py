from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowMetadata:
    recording_id: str
    brainset_id: str
    session_id: str
    subject_id: str
    start_s: float
    end_s: float
    n_spikes: int
    n_units: int


def summarize_window_sample(sample) -> WindowMetadata:
    spikes = getattr(sample, "spikes", None)
    units = getattr(sample, "units", None)
    session_id = getattr(getattr(sample, "session", None), "id", "")
    brainset_id = getattr(getattr(sample, "brainset", None), "id", "")
    subject_id = getattr(getattr(sample, "subject", None), "id", "")
    start_values = getattr(sample.domain, "start", [])
    end_values = getattr(sample.domain, "end", [])
    start_s = float(start_values[0]) if len(start_values) else 0.0
    end_s = float(end_values[-1]) if len(end_values) else 0.0
    recording_id = f"{brainset_id}/{session_id.split('/')[-1]}" if brainset_id else session_id
    return WindowMetadata(
        recording_id=recording_id,
        brainset_id=brainset_id,
        session_id=session_id,
        subject_id=subject_id,
        start_s=start_s,
        end_s=end_s,
        n_spikes=int(len(getattr(spikes, "timestamps", []))) if spikes is not None else 0,
        n_units=int(len(getattr(units, "id", []))) if units is not None else 0,
    )
