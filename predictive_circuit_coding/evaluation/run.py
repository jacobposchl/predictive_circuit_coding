from __future__ import annotations

from pathlib import Path

import torch

from predictive_circuit_coding.data import resolve_runtime_dataset_view
from predictive_circuit_coding.training.artifacts import load_training_checkpoint
from predictive_circuit_coding.training.config import ExperimentConfig
from predictive_circuit_coding.training.contracts import EvaluationSummary
from predictive_circuit_coding.training.factories import (
    build_model_from_config,
    build_objective_from_config,
    build_tokenizer_from_config,
)
from predictive_circuit_coding.training.runtime import iter_sampler_batches, resolve_device
from predictive_circuit_coding.training.step import run_training_step
from predictive_circuit_coding.windowing import (
    FixedWindowConfig,
    build_dataset_bundle,
    build_sequential_fixed_window_sampler,
)
from predictive_circuit_coding.evaluation.metrics import aggregate_metric_dicts


def evaluate_checkpoint_on_split(
    *,
    experiment_config: ExperimentConfig,
    data_config_path: str | Path,
    checkpoint_path: str,
    split_name: str,
    model=None,
    max_batches: int | None = None,
    dataset_view=None,
) -> EvaluationSummary:
    dataset_view = dataset_view or resolve_runtime_dataset_view(
        experiment_config=experiment_config,
        data_config_path=data_config_path,
    )
    workspace = dataset_view.workspace
    split_manifest = dataset_view.split_manifest
    tokenizer = build_tokenizer_from_config(experiment_config)
    objective = build_objective_from_config(experiment_config)
    device = resolve_device(experiment_config.execution.device)
    if model is None:
        model = build_model_from_config(experiment_config)
        state = load_training_checkpoint(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state"])
    model = model.to(device)
    model.eval()

    bundle = build_dataset_bundle(
        workspace=workspace,
        split_manifest=split_manifest,
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
    metric_rows: list[dict[str, float]] = []
    loss_rows: list[dict[str, float]] = []
    batch_weights: list[float] = []
    window_count = 0
    with torch.no_grad():
        for batch in iter_sampler_batches(
            dataset=bundle.dataset,
            sampler=sampler,
            collator=tokenizer,
            batch_size=experiment_config.optimization.batch_size,
            max_batches=max_batches or experiment_config.evaluation.max_batches,
        ):
            batch = batch.to(device)
            output = run_training_step(model, objective, batch)
            metric_rows.append(output.metrics)
            loss_rows.append(output.losses)
            batch_weights.append(float(output.batch_size))
            window_count += int(output.batch_size)
    if hasattr(bundle.dataset, "_close_open_files"):
        bundle.dataset._close_open_files()
    return EvaluationSummary(
        dataset_id=experiment_config.dataset_id,
        split_name=split_name,
        checkpoint_path=checkpoint_path,
        metrics=aggregate_metric_dicts(metric_rows, weights=batch_weights),
        losses=aggregate_metric_dicts(loss_rows, weights=batch_weights),
        window_count=window_count,
    )
