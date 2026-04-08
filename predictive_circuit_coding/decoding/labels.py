from __future__ import annotations

from typing import Any

import torch

from predictive_circuit_coding.training.contracts import PopulationWindowBatch


LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    # Allen sampled windows can surface change labels either on stimulus presentations
    # or on trials, depending on which interval payload survives into the sampled view.
    "stimulus_change": ("stimulus_presentations.is_change", "trials.is_change"),
}

EVENT_LOCAL_LABELS: frozenset[str] = frozenset(
    {
        "stimulus_change",
        "stimulus_presentations.is_change",
        "stimulus_presentations.omitted",
        "trials.is_change",
        "trials.go",
        "trials.catch",
        "trials.hit",
        "trials.miss",
        "trials.false_alarm",
        "trials.correct_reject",
        "trials.aborted",
        "trials.auto_rewarded",
    }
)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _resolve_target_label_paths(target_label: str) -> tuple[tuple[str, ...], ...]:
    candidates = LABEL_ALIASES.get(target_label, (target_label,))
    paths = tuple(tuple(part for part in candidate.split(".") if part) for candidate in candidates)
    if not paths or any(not path for path in paths):
        raise ValueError("target_label must not be empty")
    return paths


def _lookup_nested_value(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for part in path:
        if not isinstance(current, dict):
            return ()
        current = current.get(part)
        if current is None:
            return ()
    return current


def _window_label_mode(*, target_label: str, target_label_mode: str) -> str:
    if target_label_mode != "auto":
        return target_label_mode
    alias_targets = LABEL_ALIASES.get(target_label, ())
    if target_label in EVENT_LOCAL_LABELS or any(alias in EVENT_LOCAL_LABELS for alias in alias_targets):
        return "onset_within_window"
    return "overlap"


def _normalize_sequence(values: Any) -> tuple[Any, ...]:
    if isinstance(values, (tuple, list)):
        return tuple(values)
    if values in (None, ()):
        return tuple()
    return (values,)


def _timed_matches(annotation: dict[str, Any], path: tuple[str, ...]) -> tuple[tuple[Any, float | None, float | None], ...]:
    values = _normalize_sequence(_lookup_nested_value(annotation, path))
    if not values:
        return tuple()
    root_payload = annotation.get(path[0], {})
    if not isinstance(root_payload, dict):
        return tuple((value, None, None) for value in values)
    starts = _normalize_sequence(root_payload.get("start_s", ()))
    ends = _normalize_sequence(root_payload.get("end_s", ()))
    matches: list[tuple[Any, float | None, float | None]] = []
    for index, value in enumerate(values):
        start_s = float(starts[index]) if index < len(starts) and starts[index] is not None else None
        end_s = float(ends[index]) if index < len(ends) and ends[index] is not None else None
        matches.append((value, start_s, end_s))
    return tuple(matches)


def extract_binary_label_from_annotations(
    annotation: dict[str, Any],
    *,
    target_label: str,
    target_label_mode: str = "auto",
    window_duration_s: float | None = None,
) -> float:
    paths = _resolve_target_label_paths(target_label)
    resolved_mode = _window_label_mode(target_label=target_label, target_label_mode=target_label_mode)
    for path in paths:
        timed_matches = _timed_matches(annotation, path)
        if resolved_mode == "overlap":
            if any(_coerce_bool(value) for value, _, _ in timed_matches):
                return 1.0
            continue
        if resolved_mode == "onset_within_window":
            has_timing = any(start_s is not None for _, start_s, _ in timed_matches)
            for value, start_s, _ in timed_matches:
                if start_s is None:
                    continue
                if start_s >= 0.0 and _coerce_bool(value):
                    return 1.0
            if target_label_mode == "auto" and not has_timing and any(_coerce_bool(value) for value, _, _ in timed_matches):
                return 1.0
            continue
        if resolved_mode == "centered_onset":
            if window_duration_s is None or window_duration_s <= 0.0:
                raise ValueError("window_duration_s is required when target_label_mode='centered_onset'")
            lower_bound = 0.25 * float(window_duration_s)
            upper_bound = 0.75 * float(window_duration_s)
            has_timing = any(start_s is not None for _, start_s, _ in timed_matches)
            for value, start_s, _ in timed_matches:
                if start_s is None:
                    continue
                if lower_bound <= start_s <= upper_bound and _coerce_bool(value):
                    return 1.0
            if target_label_mode == "auto" and not has_timing and any(_coerce_bool(value) for value, _, _ in timed_matches):
                return 1.0
            continue
        raise ValueError(f"Unsupported target_label_mode: {resolved_mode}")
    return 0.0


def extract_binary_labels(
    batch: PopulationWindowBatch,
    *,
    target_label: str,
    target_label_mode: str = "auto",
) -> torch.Tensor:
    labels = []
    for index, annotation in enumerate(batch.provenance.event_annotations):
        window_duration_s = float(
            batch.provenance.window_end_s[index].item() - batch.provenance.window_start_s[index].item()
        )
        labels.append(
            extract_binary_label_from_annotations(
                annotation,
                target_label=target_label,
                target_label_mode=target_label_mode,
                window_duration_s=window_duration_s,
            )
        )
    return torch.tensor(labels, dtype=torch.float32)


def extract_stimulus_change_labels(batch: PopulationWindowBatch) -> torch.Tensor:
    return extract_binary_labels(batch, target_label="stimulus_change")
