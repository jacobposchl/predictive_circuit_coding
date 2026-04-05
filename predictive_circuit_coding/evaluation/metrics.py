from __future__ import annotations


def aggregate_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    if not metric_dicts:
        return {}
    keys = sorted({key for item in metric_dicts for key in item})
    return {
        key: float(sum(item.get(key, 0.0) for item in metric_dicts) / len(metric_dicts))
        for key in keys
    }
