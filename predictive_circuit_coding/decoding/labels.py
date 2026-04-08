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


def _coerce_match_text(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value).strip()


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


def _filter_timed_matches(
    timed_matches: tuple[tuple[Any, float | None, float | None], ...],
    *,
    resolved_mode: str,
    target_label_mode: str,
    window_duration_s: float | None,
) -> tuple[tuple[Any, float | None, float | None], ...]:
    if resolved_mode == "overlap":
        return timed_matches
    if resolved_mode == "onset_within_window":
        filtered = tuple(
            (value, start_s, end_s)
            for value, start_s, end_s in timed_matches
            if start_s is not None and start_s >= 0.0
        )
        has_timing = any(start_s is not None for _, start_s, _ in timed_matches)
        if target_label_mode == "auto" and not has_timing:
            return timed_matches
        return filtered
    if resolved_mode == "centered_onset":
        if window_duration_s is None or window_duration_s <= 0.0:
            raise ValueError("window_duration_s is required when target_label_mode='centered_onset'")
        lower_bound = 0.25 * float(window_duration_s)
        upper_bound = 0.75 * float(window_duration_s)
        filtered = tuple(
            (value, start_s, end_s)
            for value, start_s, end_s in timed_matches
            if start_s is not None and lower_bound <= start_s <= upper_bound
        )
        has_timing = any(start_s is not None for _, start_s, _ in timed_matches)
        if target_label_mode == "auto" and not has_timing:
            return timed_matches
        return filtered
    raise ValueError(f"Unsupported target_label_mode: {resolved_mode}")


def extract_matching_values_from_annotations(
    annotation: dict[str, Any],
    *,
    target_label: str,
    target_label_mode: str = "auto",
    window_duration_s: float | None = None,
) -> tuple[str, ...]:
    paths = _resolve_target_label_paths(target_label)
    resolved_mode = _window_label_mode(target_label=target_label, target_label_mode=target_label_mode)
    resolved_values: list[str] = []
    seen: set[str] = set()
    for path in paths:
        timed_matches = _timed_matches(annotation, path)
        filtered_matches = _filter_timed_matches(
            timed_matches,
            resolved_mode=resolved_mode,
            target_label_mode=target_label_mode,
            window_duration_s=window_duration_s,
        )
        for value, _, _ in filtered_matches:
            text = _coerce_match_text(value)
            if not text or text in seen:
                continue
            seen.add(text)
            resolved_values.append(text)
    return tuple(resolved_values)


def extract_binary_label_from_annotations(
    annotation: dict[str, Any],
    *,
    target_label: str,
    target_label_mode: str = "auto",
    target_label_match_value: str | None = None,
    window_duration_s: float | None = None,
) -> float:
    paths = _resolve_target_label_paths(target_label)
    resolved_mode = _window_label_mode(target_label=target_label, target_label_mode=target_label_mode)
    match_value = _coerce_match_text(target_label_match_value) if target_label_match_value is not None else None
    for path in paths:
        timed_matches = _timed_matches(annotation, path)
        filtered_matches = _filter_timed_matches(
            timed_matches,
            resolved_mode=resolved_mode,
            target_label_mode=target_label_mode,
            window_duration_s=window_duration_s,
        )
        if match_value is None:
            if any(_coerce_bool(value) for value, _, _ in filtered_matches):
                return 1.0
            continue
        if any(_coerce_match_text(value) == match_value for value, _, _ in filtered_matches):
            return 1.0
    return 0.0


def extract_binary_labels(
    batch: PopulationWindowBatch,
    *,
    target_label: str,
    target_label_mode: str = "auto",
    target_label_match_value: str | None = None,
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
                target_label_match_value=target_label_match_value,
                window_duration_s=window_duration_s,
            )
        )
    return torch.tensor(labels, dtype=torch.float32)


def extract_stimulus_change_labels(batch: PopulationWindowBatch) -> torch.Tensor:
    return extract_binary_labels(batch, target_label="stimulus_change")
