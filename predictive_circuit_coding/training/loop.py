from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import math

import torch

from predictive_circuit_coding.data import resolve_runtime_dataset_view
from predictive_circuit_coding.evaluation.metrics import aggregate_metric_dicts
from predictive_circuit_coding.evaluation.run import evaluate_checkpoint_on_split
from predictive_circuit_coding.training.artifacts import (
    load_training_checkpoint,
    save_training_checkpoint,
    write_training_summary,
)
from predictive_circuit_coding.training.config import ExperimentConfig
from predictive_circuit_coding.training.contracts import (
    CheckpointMetadata,
    TrainingCheckpoint,
    TrainingSummary,
    write_json_payload,
)
from predictive_circuit_coding.training.factories import (
    build_model_from_config,
    build_objective_from_config,
    build_tokenizer_from_config,
)
from predictive_circuit_coding.training.logging import StageLogger
from predictive_circuit_coding.training.runtime import iter_sampler_batches, resolve_device
from predictive_circuit_coding.training.step import run_training_step
from predictive_circuit_coding.windowing import (
    FixedWindowConfig,
    build_dataset_bundle,
    build_random_fixed_window_sampler,
)

def build_optimizer(config: ExperimentConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )


def build_scheduler(config: ExperimentConfig, optimizer: torch.optim.Optimizer):
    if config.optimization.scheduler_type == "none":
        return None
    if config.optimization.scheduler_type == "cosine":
        total_steps = config.training.num_epochs * config.training.train_steps_per_epoch
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))
    raise ValueError(f"Unsupported scheduler type: {config.optimization.scheduler_type}")


def build_checkpoint_metadata(config: ExperimentConfig, *, split_name: str) -> CheckpointMetadata:
    return CheckpointMetadata(
        dataset_id=config.dataset_id,
        split_name=split_name,
        seed=config.seed,
        config_snapshot=config.to_dict(),
        model_hparams={
            "d_model": config.model.d_model,
            "num_heads": config.model.num_heads,
            "temporal_layers": config.model.temporal_layers,
            "spatial_layers": config.model.spatial_layers,
            "dropout": config.model.dropout,
        },
        continuation_baseline_type=config.objective.continuation_baseline_type,
    )
@dataclass(frozen=True)
class TrainingRunResult:
    checkpoint_path: Path
    summary_path: Path
    best_epoch: int
    best_metric: float


def train_model(
    *,
    experiment_config: ExperimentConfig,
    data_config_path: str | Path,
    train_split: str,
    valid_split: str,
    dataset_view=None,
) -> TrainingRunResult:
    dataset_view = dataset_view or resolve_runtime_dataset_view(
        experiment_config=experiment_config,
        data_config_path=data_config_path,
    )
    workspace = dataset_view.workspace
    split_manifest = dataset_view.split_manifest

    tokenizer = build_tokenizer_from_config(experiment_config)
    model = build_model_from_config(experiment_config)
    objective = build_objective_from_config(experiment_config)
    optimizer = build_optimizer(experiment_config, model)
    scheduler = build_scheduler(experiment_config, optimizer)
    device = resolve_device(experiment_config.execution.device)
    model = model.to(device)
    scaler_enabled = bool(experiment_config.execution.mixed_precision and device.type == "cuda")
    logger = StageLogger(name="train")
    if experiment_config.artifacts.save_config_snapshot:
        snapshot_path = (
            experiment_config.artifacts.checkpoint_dir
            / f"{experiment_config.artifacts.checkpoint_prefix}_config_snapshot.json"
        )
        write_json_payload(experiment_config.to_dict(), snapshot_path)
        logger.log_artifact(label="config snapshot", path=snapshot_path)

    start_epoch = 1
    global_step = 0
    best_metric = float("-inf")
    best_epoch = 0
    best_checkpoint_path = experiment_config.artifacts.checkpoint_dir / f"{experiment_config.artifacts.checkpoint_prefix}_best.pt"
    latest_epoch_checkpoint_path: Path | None = None
    latest_epoch_checkpoint: TrainingCheckpoint | None = None

    if experiment_config.training.resume_checkpoint is not None:
        state = load_training_checkpoint(experiment_config.training.resume_checkpoint, map_location=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        if scheduler is not None and state.get("scheduler_state") is not None:
            scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        best_metric = float(state["best_metric"])
        best_epoch = int(state["epoch"])
        logger.log(f"Resumed from checkpoint at epoch {state['epoch']}")

    for epoch in range(start_epoch, experiment_config.training.num_epochs + 1):
        logger.log_stage(
            f"epoch {epoch}/{experiment_config.training.num_epochs}",
            expected_next="checkpoint or training summary",
        )
        model.train()
        train_bundle = build_dataset_bundle(
            workspace=workspace,
            split_manifest=split_manifest,
            split=train_split,
            config_dir=dataset_view.config_dir,
            config_name_prefix=dataset_view.config_name_prefix,
            dataset_split=dataset_view.dataset_split,
        )
        train_sampler = build_random_fixed_window_sampler(
            train_bundle.dataset,
            window=FixedWindowConfig(
                window_length_s=experiment_config.data_runtime.context_duration_s,
                seed=experiment_config.training.train_window_seed + epoch,
            ),
        )
        train_metrics: list[dict[str, float]] = []
        for step_index, batch in enumerate(
            iter_sampler_batches(
                dataset=train_bundle.dataset,
                sampler=train_sampler,
                collator=tokenizer,
                batch_size=experiment_config.optimization.batch_size,
                max_batches=experiment_config.training.train_steps_per_epoch,
            ),
            start=1,
        ):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            autocast_context = (
                torch.autocast(device_type=device.type, dtype=torch.float16) if scaler_enabled else nullcontext()
            )
            with autocast_context:
                step_output = run_training_step(model, objective, batch)
            step_output.loss.backward()
            if experiment_config.optimization.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), experiment_config.optimization.grad_clip_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            global_step += 1
            train_metrics.append({**step_output.metrics, **step_output.losses})
            if step_index % experiment_config.training.log_every_steps == 0:
                logger.log_metrics(prefix=f"epoch={epoch} step={step_index}", metrics={**step_output.metrics, **step_output.losses})

        if hasattr(train_bundle.dataset, "_close_open_files"):
            train_bundle.dataset._close_open_files()

        if epoch % experiment_config.training.evaluate_every_epochs == 0:
            evaluation = evaluate_checkpoint_on_split(
                experiment_config=experiment_config,
                data_config_path=data_config_path,
                model=model,
                split_name=valid_split,
                checkpoint_path=str(best_checkpoint_path if best_checkpoint_path.exists() else ""),
                max_batches=experiment_config.training.validation_steps,
                dataset_view=dataset_view,
            )
            valid_metric = float(evaluation.metrics["predictive_improvement"])
            if valid_metric >= best_metric:
                best_metric = valid_metric
                best_epoch = epoch
                checkpoint = TrainingCheckpoint(
                    epoch=epoch,
                    global_step=global_step,
                    best_metric=best_metric,
                    metadata=build_checkpoint_metadata(experiment_config, split_name=train_split),
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict() if scheduler is not None else None,
                )
                save_training_checkpoint(checkpoint, best_checkpoint_path)
                logger.log_artifact(label="best checkpoint", path=best_checkpoint_path)

            summary = TrainingSummary(
                dataset_id=experiment_config.dataset_id,
                split_name=train_split,
                epoch=epoch,
                best_epoch=best_epoch,
                metrics=aggregate_metric_dicts([evaluation.metrics]),
                losses=aggregate_metric_dicts(train_metrics),
                checkpoint_path=str(best_checkpoint_path),
            )
            write_training_summary(summary, experiment_config.artifacts.summary_path)
            logger.log_artifact(label="training summary", path=experiment_config.artifacts.summary_path)

        if epoch % experiment_config.training.checkpoint_every_epochs == 0:
            checkpoint = TrainingCheckpoint(
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                metadata=build_checkpoint_metadata(experiment_config, split_name=train_split),
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict() if scheduler is not None else None,
            )
            epoch_path = experiment_config.artifacts.checkpoint_dir / f"{experiment_config.artifacts.checkpoint_prefix}_latest.pt"
            save_training_checkpoint(checkpoint, epoch_path)
            latest_epoch_checkpoint_path = epoch_path
            latest_epoch_checkpoint = checkpoint
            logger.log_artifact(label="latest checkpoint", path=epoch_path)

    if not best_checkpoint_path.exists() and latest_epoch_checkpoint is not None:
        save_training_checkpoint(latest_epoch_checkpoint, best_checkpoint_path)
        logger.log(
            "best checkpoint metric was not set cleanly; using the latest epoch checkpoint as a fallback"
        )
        logger.log_artifact(label="fallback best checkpoint", path=best_checkpoint_path)
        if best_epoch == 0:
            best_epoch = latest_epoch_checkpoint.epoch
        if not math.isfinite(best_metric):
            best_metric = 0.0

    return TrainingRunResult(
        checkpoint_path=best_checkpoint_path,
        summary_path=experiment_config.artifacts.summary_path,
        best_epoch=best_epoch,
        best_metric=best_metric,
    )
