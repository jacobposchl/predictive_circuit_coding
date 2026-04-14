from __future__ import annotations

from pathlib import Path
import json

import torch

from predictive_circuit_coding.tokenization import CountNormalizationStats, build_population_window_collator
from predictive_circuit_coding.training.config import ExperimentConfig
from predictive_circuit_coding.training.runtime import iter_sampler_batches
from predictive_circuit_coding.windowing import FixedWindowConfig, build_dataset_bundle, build_sequential_fixed_window_sampler


def fit_count_normalization_stats(
    *,
    experiment_config: ExperimentConfig,
    dataset_view,
    split_name: str,
    output_path: str | Path,
) -> CountNormalizationStats:
    if experiment_config.count_normalization.mode == "none":
        return CountNormalizationStats(mode="none", mean=0.0, std=1.0, count=0)
    if experiment_config.count_normalization.mode != "log1p_train_zscore":
        raise ValueError(f"Unsupported count normalization mode: {experiment_config.count_normalization.mode}")

    bundle = build_dataset_bundle(
        workspace=dataset_view.workspace,
        split_manifest=dataset_view.split_manifest,
        split=split_name,
        config_dir=dataset_view.config_dir,
        config_name_prefix=dataset_view.config_name_prefix,
        dataset_split=dataset_view.dataset_split,
    )
    sampler = build_sequential_fixed_window_sampler(
        bundle.dataset,
        window=FixedWindowConfig(
            window_length_s=experiment_config.data_runtime.context_duration_s,
            step_s=experiment_config.evaluation.sequential_step_s,
        ),
    )
    raw_collator = build_population_window_collator(experiment_config.data_runtime)

    total_count = 0
    total_sum = 0.0
    total_sq_sum = 0.0
    for batch in iter_sampler_batches(
        dataset=bundle.dataset,
        sampler=sampler,
        collator=raw_collator,
        batch_size=experiment_config.optimization.batch_size,
        max_batches=None,
    ):
        valid_mask = batch.unit_mask.unsqueeze(-1).expand_as(batch.counts)
        if not bool(valid_mask.any().item()):
            continue
        values = torch.log1p(batch.counts[valid_mask].to(dtype=torch.float64).clamp_min(0.0))
        total_count += int(values.numel())
        total_sum += float(values.sum().item())
        total_sq_sum += float((values * values).sum().item())

    if hasattr(bundle.dataset, "_close_open_files"):
        bundle.dataset._close_open_files()

    if total_count <= 0:
        stats = CountNormalizationStats(
            mode=experiment_config.count_normalization.mode,
            mean=0.0,
            std=1.0,
            count=0,
        )
    else:
        mean = total_sum / float(total_count)
        variance = max((total_sq_sum / float(total_count)) - (mean * mean), 0.0)
        stats = CountNormalizationStats(
            mode=experiment_config.count_normalization.mode,
            mean=mean,
            std=max(variance**0.5, 1.0e-6),
            count=total_count,
        )

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")
    return stats
