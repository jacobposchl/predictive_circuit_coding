from __future__ import annotations

import csv
from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
import math
import random
import shutil
from typing import Callable

import numpy as np
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
from predictive_circuit_coding.training.normalization import fit_count_normalization_stats
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
from predictive_circuit_coding.training.runtime import iter_sampler_batches, resolve_device, run_training_step
from predictive_circuit_coding.utils.notebook_progress import TrainingProgressEvent
from predictive_circuit_coding.windowing import (
    FixedWindowConfig,
    build_dataset_bundle,
    build_random_fixed_window_sampler,
)

def build_optimizer(config: ExperimentConfig, parameters) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        parameters,
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )


def build_scheduler(config: ExperimentConfig, optimizer: torch.optim.Optimizer):
    if config.optimization.scheduler_type == "none":
        return None
    if config.optimization.scheduler_type == "cosine":
        total_steps = config.training.num_epochs * config.training.train_steps_per_epoch
        warmup_steps = min(config.optimization.scheduler_warmup_steps, max(0, total_steps - 1))
        cosine_steps = max(1, total_steps - warmup_steps)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_steps)
        if warmup_steps <= 0:
            return cosine
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / float(max(1, warmup_steps + 1)),
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
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
        variant_name=config.experiment.variant_name,
        reconstruction_target_mode=config.objective.reconstruction_target_mode,
        count_normalization_mode=config.count_normalization.mode,
        count_normalization_stats_path=(
            str(config.count_normalization.stats_path)
            if config.count_normalization.stats_path is not None
            else None
        ),
    )


@dataclass(frozen=True)
class TrainingRunResult:
    checkpoint_path: Path
    summary_path: Path
    history_json_path: Path
    history_csv_path: Path
    best_epoch: int
    best_metric: float

TrainingProgressCallback = Callable[[TrainingProgressEvent], None]


def _maybe_emit_training_progress(
    progress_callback: TrainingProgressCallback | None,
    event: TrainingProgressEvent,
) -> None:
    if progress_callback is not None:
        progress_callback(event)


def _set_training_seed(seed: int, *, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _write_training_history_rows(
    rows: list[dict[str, float | int | str | bool | None]],
    *,
    output_json_path: Path,
    output_csv_path: Path,
) -> tuple[Path, Path]:
    write_json_payload({"training_history": rows}, output_json_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
    with output_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_json_path, output_csv_path


def _load_prior_training_history_rows(
    *,
    history_json_path: Path,
    start_epoch: int,
) -> list[dict[str, float | int | str | bool | None]]:
    if not history_json_path.is_file() or start_epoch <= 1:
        return []
    payload = json.loads(history_json_path.read_text(encoding="utf-8"))
    rows = payload.get("training_history") or []
    prior_rows: list[dict[str, float | int | str | bool | None]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            epoch = int(row.get("epoch", 0))
        except (TypeError, ValueError):
            continue
        if 0 < epoch < start_epoch:
            prior_rows.append(dict(row))
    return prior_rows


def _prefixed_metrics(prefix: str, metrics: dict[str, float] | None) -> dict[str, float | None]:
    if not metrics:
        return {}
    return {f"{prefix}_{str(key)}": float(value) for key, value in metrics.items() if value is not None}


def _float_metric_dict(payload: object) -> dict[str, float] | None:
    if not isinstance(payload, dict):
        return None
    parsed: dict[str, float] = {}
    for key, value in payload.items():
        if value is None:
            continue
        parsed[str(key)] = float(value)
    return parsed


def _load_resume_summary_payload(
    *,
    summary_path: Path,
    resume_checkpoint_path: Path,
) -> dict | None:
    candidate_paths = [summary_path]
    inferred_source_summary = resume_checkpoint_path.parent.parent / summary_path.name
    if inferred_source_summary not in candidate_paths:
        candidate_paths.append(inferred_source_summary)
    for candidate in candidate_paths:
        if not candidate.is_file():
            continue
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    return None


def _copy_checkpoint(source: Path, target: Path) -> None:
    if source.resolve() == target.resolve():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _restore_best_checkpoint_from_resume(
    *,
    resume_checkpoint_path: Path,
    best_checkpoint_path: Path,
    checkpoint_prefix: str,
    state: dict,
    logger: StageLogger,
) -> bool:
    if best_checkpoint_path.is_file():
        return True
    source_best_checkpoint_path = resume_checkpoint_path.with_name(f"{checkpoint_prefix}_best.pt")
    if source_best_checkpoint_path.is_file():
        _copy_checkpoint(source_best_checkpoint_path, best_checkpoint_path)
        logger.log_artifact(label="resumed best checkpoint", path=best_checkpoint_path)
        return True
    state_epoch = int(state["epoch"])
    best_epoch = int(state.get("best_epoch", state_epoch))
    if best_epoch == state_epoch:
        _copy_checkpoint(resume_checkpoint_path, best_checkpoint_path)
        logger.log_artifact(label="resumed best checkpoint", path=best_checkpoint_path)
        return True
    return False


def _write_summary_if_missing(
    *,
    summary_path: Path,
    experiment_config: ExperimentConfig,
    train_split: str,
    checkpoint_path: Path,
    best_epoch: int,
    metrics: dict[str, float] | None,
    losses: dict[str, float] | None,
    selection_reason: str,
    logger: StageLogger,
) -> None:
    if summary_path.is_file() or best_epoch <= 0:
        return
    summary = _build_training_summary(
        experiment_config=experiment_config,
        train_split=train_split,
        checkpoint_path=checkpoint_path,
        epoch=best_epoch,
        best_epoch=best_epoch,
        metrics=metrics or {},
        losses=losses or {},
        selection_reason=selection_reason,
    )
    write_training_summary(summary, summary_path)
    logger.log_artifact(label="training summary", path=summary_path)


def _build_training_summary(
    *,
    experiment_config: ExperimentConfig,
    train_split: str,
    checkpoint_path: Path,
    epoch: int,
    best_epoch: int,
    metrics: dict[str, float],
    losses: dict[str, float],
    selection_reason: str,
) -> TrainingSummary:
    return TrainingSummary(
        dataset_id=experiment_config.dataset_id,
        split_name=train_split,
        epoch=epoch,
        best_epoch=best_epoch,
        metrics=dict(metrics),
        losses=dict(losses),
        checkpoint_path=str(checkpoint_path),
        variant_name=experiment_config.experiment.variant_name,
        reconstruction_target_mode=experiment_config.objective.reconstruction_target_mode,
        count_normalization_mode=experiment_config.count_normalization.mode,
        count_normalization_stats_path=(
            str(experiment_config.count_normalization.stats_path)
            if experiment_config.count_normalization.stats_path is not None
            else None
        ),
        selection_reason=selection_reason,
    )


def train_model(
    *,
    experiment_config: ExperimentConfig,
    data_config_path: str | Path,
    train_split: str,
    valid_split: str,
    dataset_view=None,
    progress_callback: TrainingProgressCallback | None = None,
    log_sink: Callable[[str], None] | None = None,
    emit_logs: bool = True,
) -> TrainingRunResult:
    dataset_view = dataset_view or resolve_runtime_dataset_view(
        experiment_config=experiment_config,
        data_config_path=data_config_path,
    )
    workspace = dataset_view.workspace
    split_manifest = dataset_view.split_manifest

    if (
        experiment_config.count_normalization.mode != "none"
        and experiment_config.count_normalization.stats_path is not None
        and not experiment_config.count_normalization.stats_path.is_file()
    ):
        stats = fit_count_normalization_stats(
            experiment_config=experiment_config,
            dataset_view=dataset_view,
            split_name=train_split,
            output_path=experiment_config.count_normalization.stats_path,
        )
        if log_sink is not None:
            log_sink(
                "count normalization stats fitted "
                f"| mode={stats.mode} mean={stats.mean:.6g} std={stats.std:.6g} count={stats.count}"
            )
    tokenizer = build_tokenizer_from_config(experiment_config)
    device = resolve_device(experiment_config.execution.device)
    _set_training_seed(experiment_config.seed, device=device)
    model = build_model_from_config(experiment_config)
    objective = build_objective_from_config(experiment_config)
    training_history_json_path = experiment_config.artifacts.summary_path.with_name("training_history.json")
    training_history_csv_path = experiment_config.artifacts.summary_path.with_name("training_history.csv")
    training_history_rows: list[dict[str, float | int | str | bool | None]] = []
    optimizer = build_optimizer(
        experiment_config,
        model.parameters(),
    )
    scheduler = build_scheduler(experiment_config, optimizer)
    model = model.to(device)
    scaler_enabled = bool(experiment_config.execution.mixed_precision and device.type == "cuda")
    grad_scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    logger = StageLogger(name="train", emit_console=emit_logs, log_sink=log_sink)
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
    best_validation_metrics: dict[str, float] | None = None
    best_training_losses: dict[str, float] | None = None
    best_selection_reason = "validated_best"
    latest_evaluation_metrics: dict[str, float] | None = None
    latest_training_losses: dict[str, float] | None = None
    best_checkpoint_path = experiment_config.artifacts.checkpoint_dir / f"{experiment_config.artifacts.checkpoint_prefix}_best.pt"
    latest_epoch_checkpoint: TrainingCheckpoint | None = None

    if experiment_config.training.resume_checkpoint is not None:
        resume_checkpoint_path = experiment_config.training.resume_checkpoint
        state = load_training_checkpoint(resume_checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        if scheduler is not None and state.get("scheduler_state") is not None:
            scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        best_metric = float(state["best_metric"])
        best_epoch = int(state.get("best_epoch", state["epoch"]))
        best_validation_metrics = (
            {str(key): float(value) for key, value in state.get("best_validation_metrics", {}).items()}
            if state.get("best_validation_metrics") is not None
            else None
        )
        summary_payload = _load_resume_summary_payload(
            summary_path=experiment_config.artifacts.summary_path,
            resume_checkpoint_path=resume_checkpoint_path,
        )
        if summary_payload is not None and int(summary_payload.get("best_epoch", 0)) == best_epoch:
            best_training_losses = _float_metric_dict(summary_payload.get("losses"))
            best_validation_metrics = best_validation_metrics or _float_metric_dict(summary_payload.get("metrics"))
            best_selection_reason = str(summary_payload.get("selection_reason", best_selection_reason))
        _restore_best_checkpoint_from_resume(
            resume_checkpoint_path=resume_checkpoint_path,
            best_checkpoint_path=best_checkpoint_path,
            checkpoint_prefix=experiment_config.artifacts.checkpoint_prefix,
            state=state,
            logger=logger,
        )
        training_history_rows = _load_prior_training_history_rows(
            history_json_path=training_history_json_path,
            start_epoch=start_epoch,
        )
        logger.log(f"Resumed from checkpoint at epoch {state['epoch']}")
        _maybe_emit_training_progress(
            progress_callback,
            TrainingProgressEvent(
                event_type="resume",
                epoch=int(state["epoch"]),
                epoch_total=experiment_config.training.num_epochs,
                message=f"Resumed from checkpoint at epoch {state['epoch']}",
            ),
        )

    for epoch in range(start_epoch, experiment_config.training.num_epochs + 1):
        logger.log_stage(
            f"epoch {epoch}/{experiment_config.training.num_epochs}",
            expected_next="checkpoint or training summary",
        )
        _maybe_emit_training_progress(
            progress_callback,
            TrainingProgressEvent(
                event_type="epoch_start",
                epoch=epoch,
                epoch_total=experiment_config.training.num_epochs,
                step_total=experiment_config.training.train_steps_per_epoch,
                message=f"epoch {epoch}/{experiment_config.training.num_epochs}",
            ),
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
        train_metric_weights: list[float] = []
        try:
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
                if scaler_enabled:
                    grad_scaler.scale(step_output.loss).backward()
                    if experiment_config.optimization.grad_clip_norm is not None:
                        grad_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            experiment_config.optimization.grad_clip_norm,
                        )
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    step_output.loss.backward()
                    if experiment_config.optimization.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), experiment_config.optimization.grad_clip_norm)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
                train_metrics.append({**step_output.metrics, **step_output.losses})
                train_metric_weights.append(float(step_output.batch_size))
                _maybe_emit_training_progress(
                    progress_callback,
                    TrainingProgressEvent(
                        event_type="step",
                        epoch=epoch,
                        epoch_total=experiment_config.training.num_epochs,
                        step=step_index,
                        step_total=experiment_config.training.train_steps_per_epoch,
                        metrics={**step_output.metrics, **step_output.losses},
                    ),
                )
                if step_index % experiment_config.training.log_every_steps == 0:
                    logger.log_metrics(prefix=f"epoch={epoch} step={step_index}", metrics={**step_output.metrics, **step_output.losses})
        finally:
            if hasattr(train_bundle.dataset, "_close_open_files"):
                train_bundle.dataset._close_open_files()

        if not train_metrics:
            raise RuntimeError(f"No training batches were sampled from split '{train_split}' for epoch {epoch}.")

        current_training_losses = aggregate_metric_dicts(train_metrics, weights=train_metric_weights)
        latest_training_losses = current_training_losses
        epoch_evaluation_metrics: dict[str, float] | None = None
        epoch_became_best = False

        if epoch % experiment_config.training.evaluate_every_epochs == 0:
            _maybe_emit_training_progress(
                progress_callback,
                TrainingProgressEvent(
                    event_type="validation_start",
                    epoch=epoch,
                    epoch_total=experiment_config.training.num_epochs,
                    message=f"validation {valid_split}",
                ),
            )
            evaluation = evaluate_checkpoint_on_split(
                experiment_config=experiment_config,
                data_config_path=data_config_path,
                model=model,
                split_name=valid_split,
                checkpoint_path=str(best_checkpoint_path),
                max_batches=experiment_config.training.validation_steps,
                dataset_view=dataset_view,
            )
            valid_metric = float(evaluation.metrics["predictive_improvement"])
            latest_evaluation_metrics = dict(evaluation.metrics)
            epoch_evaluation_metrics = dict(evaluation.metrics)
            _maybe_emit_training_progress(
                progress_callback,
                TrainingProgressEvent(
                    event_type="validation_end",
                    epoch=epoch,
                    epoch_total=experiment_config.training.num_epochs,
                    metrics=dict(evaluation.metrics),
                    message=f"validation {valid_split}",
                ),
            )
            if math.isfinite(valid_metric) and valid_metric >= best_metric:
                best_metric = valid_metric
                best_epoch = epoch
                best_validation_metrics = dict(evaluation.metrics)
                best_training_losses = dict(current_training_losses)
                best_selection_reason = "validated_best"
                epoch_became_best = True
                checkpoint = TrainingCheckpoint(
                    epoch=epoch,
                    global_step=global_step,
                    best_metric=best_metric,
                    metadata=build_checkpoint_metadata(experiment_config, split_name=train_split),
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict() if scheduler is not None else None,
                    best_epoch=best_epoch,
                    best_validation_metrics=best_validation_metrics,
                    auxiliary_state=None,
                )
                save_training_checkpoint(checkpoint, best_checkpoint_path)
                logger.log_artifact(label="best checkpoint", path=best_checkpoint_path)
                _maybe_emit_training_progress(
                    progress_callback,
                    TrainingProgressEvent(
                        event_type="checkpoint_saved",
                        epoch=epoch,
                        epoch_total=experiment_config.training.num_epochs,
                        checkpoint_path=str(best_checkpoint_path),
                        message="best checkpoint",
                    ),
                )

            if best_epoch > 0 and best_validation_metrics is not None and best_training_losses is not None:
                summary = _build_training_summary(
                    experiment_config=experiment_config,
                    train_split=train_split,
                    checkpoint_path=best_checkpoint_path,
                    epoch=best_epoch,
                    best_epoch=best_epoch,
                    metrics=best_validation_metrics,
                    losses=best_training_losses,
                    selection_reason=best_selection_reason,
                )
                write_training_summary(summary, experiment_config.artifacts.summary_path)
                logger.log_artifact(label="training summary", path=experiment_config.artifacts.summary_path)

        training_history_rows.append(
            {
                "epoch": int(epoch),
                "global_step": int(global_step),
                "variant_name": experiment_config.experiment.variant_name,
                "reconstruction_target_mode": experiment_config.objective.reconstruction_target_mode,
                "count_normalization_mode": experiment_config.count_normalization.mode,
                "train_split": str(train_split),
                "valid_split": str(valid_split),
                "learning_rate": float(optimizer.param_groups[0].get("lr", 0.0)),
                "evaluated": epoch_evaluation_metrics is not None,
                "became_best": bool(epoch_became_best),
                "best_epoch_so_far": int(best_epoch),
                "best_predictive_improvement_so_far": (
                    float(best_metric)
                    if math.isfinite(float(best_metric))
                    else None
                ),
                **_prefixed_metrics("train", current_training_losses),
                **_prefixed_metrics("valid", epoch_evaluation_metrics),
            }
        )
        _write_training_history_rows(
            training_history_rows,
            output_json_path=training_history_json_path,
            output_csv_path=training_history_csv_path,
        )

        if epoch % experiment_config.training.checkpoint_every_epochs == 0:
            checkpoint = TrainingCheckpoint(
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                metadata=build_checkpoint_metadata(experiment_config, split_name=train_split),
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict() if scheduler is not None else None,
                best_epoch=best_epoch,
                best_validation_metrics=best_validation_metrics,
                auxiliary_state=None,
            )
            epoch_path = experiment_config.artifacts.checkpoint_dir / f"{experiment_config.artifacts.checkpoint_prefix}_latest.pt"
            save_training_checkpoint(checkpoint, epoch_path)
            latest_epoch_checkpoint = checkpoint
            logger.log_artifact(label="latest checkpoint", path=epoch_path)
            _maybe_emit_training_progress(
                progress_callback,
                TrainingProgressEvent(
                    event_type="checkpoint_saved",
                    epoch=epoch,
                    epoch_total=experiment_config.training.num_epochs,
                    checkpoint_path=str(epoch_path),
                    message="latest checkpoint",
                ),
            )

        _maybe_emit_training_progress(
            progress_callback,
            TrainingProgressEvent(
                event_type="epoch_end",
                epoch=epoch,
                epoch_total=experiment_config.training.num_epochs,
                step=experiment_config.training.train_steps_per_epoch,
                step_total=experiment_config.training.train_steps_per_epoch,
                metrics=dict(current_training_losses),
                message=f"epoch {epoch} complete",
            ),
        )

    if not best_checkpoint_path.exists() and latest_epoch_checkpoint is not None and (
        best_epoch in {0, latest_epoch_checkpoint.epoch} or not math.isfinite(best_metric)
    ):
        save_training_checkpoint(latest_epoch_checkpoint, best_checkpoint_path)
        logger.log(
            "best checkpoint metric was not set cleanly; using the latest epoch checkpoint as a fallback"
        )
        logger.log_artifact(label="fallback best checkpoint", path=best_checkpoint_path)
        if best_epoch == 0:
            best_epoch = latest_epoch_checkpoint.epoch
        if not math.isfinite(best_metric):
            if latest_evaluation_metrics is not None:
                fallback_metric = float(latest_evaluation_metrics.get("predictive_improvement", 0.0))
                best_metric = fallback_metric if math.isfinite(fallback_metric) else 0.0
            else:
                best_metric = 0.0
        best_selection_reason = "fallback_latest_due_to_invalid_metric"
        fallback_summary = _build_training_summary(
            experiment_config=experiment_config,
            train_split=train_split,
            checkpoint_path=best_checkpoint_path,
            epoch=best_epoch,
            best_epoch=best_epoch,
            metrics=latest_evaluation_metrics or {},
            losses=latest_training_losses or {},
            selection_reason=best_selection_reason,
        )
        write_training_summary(fallback_summary, experiment_config.artifacts.summary_path)
        logger.log_artifact(label="training summary", path=experiment_config.artifacts.summary_path)

    if not best_checkpoint_path.exists():
        raise FileNotFoundError(
            "Best checkpoint was not available after training. "
            f"Expected {best_checkpoint_path}. If resuming from a latest checkpoint whose best_epoch is "
            "different from the latest epoch, keep the corresponding best checkpoint beside the resume checkpoint."
        )
    _write_summary_if_missing(
        summary_path=experiment_config.artifacts.summary_path,
        experiment_config=experiment_config,
        train_split=train_split,
        checkpoint_path=best_checkpoint_path,
        best_epoch=best_epoch,
        metrics=best_validation_metrics or latest_evaluation_metrics,
        losses=best_training_losses or latest_training_losses,
        selection_reason=best_selection_reason,
        logger=logger,
    )

    _maybe_emit_training_progress(
        progress_callback,
        TrainingProgressEvent(
            event_type="training_complete",
            epoch=best_epoch,
            epoch_total=experiment_config.training.num_epochs,
            checkpoint_path=str(best_checkpoint_path),
            metrics={"predictive_improvement": float(best_metric)},
            message="training complete",
        ),
    )

    return TrainingRunResult(
        checkpoint_path=best_checkpoint_path,
        summary_path=experiment_config.artifacts.summary_path,
        history_json_path=training_history_json_path,
        history_csv_path=training_history_csv_path,
        best_epoch=best_epoch,
        best_metric=best_metric,
    )
