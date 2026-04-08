from __future__ import annotations


def aggregate_metric_dicts(
    metric_dicts: list[dict[str, float]],
    *,
    weights: list[float] | tuple[float, ...] | None = None,
) -> dict[str, float]:
    if not metric_dicts:
        return {}
    if weights is None:
        weights = [1.0] * len(metric_dicts)
    if len(weights) != len(metric_dicts):
        raise ValueError("weights must align with metric_dicts")
    total_weight = float(sum(float(weight) for weight in weights))
    if total_weight <= 0.0:
        raise ValueError("aggregate weights must sum to a positive value")
    keys = sorted({key for item in metric_dicts for key in item})
    return {
        key: float(
            sum(item.get(key, 0.0) * float(weight) for item, weight in zip(metric_dicts, weights, strict=True))
            / total_weight
        )
        for key in keys
    }
