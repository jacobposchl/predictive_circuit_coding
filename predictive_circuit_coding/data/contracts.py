from __future__ import annotations

PREPARED_SESSION_FILE_SUFFIX = ".h5"

REQUIRED_PROVENANCE_FIELDS = (
    "dataset_id",
    "recording_id",
    "session_id",
    "subject_id",
    "unit_id",
    "brain_region",
    "probe_depth_um",
    "trial_id",
    "window_start_s",
    "window_end_s",
    "patch_index",
    "event_alignment_label",
)
