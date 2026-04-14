from __future__ import annotations

import csv
from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
import math
import random
from typing import Callable

import numpy as np
import torch

from predictive_circuit_coding.data import resolve_runtime_dataset_view
from predictive_circuit_coding.decoding import summarize_neighbor_geometry
from predictive_circuit_coding.decoding.labels import extract_binary_labels
from predictive_circuit_coding.evaluation.metrics import aggregate_metric_dicts
from predictive_circuit_coding.evaluation.run import evaluate_checkpoint_on_split
from predictive_circuit_coding.models import RegionRatePredictiveHead
from predictive_circuit_coding.objectives import CrossSessionRegionLoss, RegionRateDonorCache, RegionRateTargetBuilder
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
    TrainingStepOutput,
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
from predictive_circuit_coding.utils.notebook import TrainingProgressEvent
from predictive_circuit_coding.windowing import (
    FixedWindowConfig,
    build_dataset_bundle,
    build_random_fixed_window_sampler,
    build_sequential_fixed_window_sampler,
)
from predictive_circuit_coding.windowing.dataset import split_session_ids

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
        training_variant_name=config.objective.cross_session_aug.training_variant_name,
        cross_session_aug_enabled=bool(config.objective.cross_session_aug.enabled),
    )


@dataclass(frozen=True)
class TrainingRunResult:
    checkpoint_path: Path
    summary_path: Path
    history_json_path: Path
    history_csv_path: Path
    best_epoch: int
    best_metric: float
    geometry_monitor_json_path: Path | None = None
    geometry_monitor_csv_path: Path | None = None


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


def _scheduled_value(*, start: float, end: float, epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return float(end)
    progress = min(max(epoch - 1, 0), warmup_epochs) / float(warmup_epochs)
    return float(start) + (float(end) - float(start)) * float(progress)


def _derive_canonical_regions(*, dataset_view, split_name: str, explicit_regions: tuple[str, ...]) -> tuple[str, ...]:
    if explicit_regions:
        return tuple(str(region) for region in explicit_regions)
    train_session_ids = set(split_session_ids(dataset_view.split_manifest, split_name))
    regions = {
        str(region)
        for record in dataset_view.session_catalog.records
        if str(record.session_id) in train_session_ids
        for region in record.brain_regions
        if str(region).strip()
    }
    return tuple(sorted(regions))


def _pooled_window_features(tokens: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
    flat_tokens = tokens.detach().cpu().reshape(tokens.shape[0], -1, tokens.shape[-1]).to(dtype=torch.float32)
    flat_mask = patch_mask.detach().cpu().reshape(patch_mask.shape[0], -1)
    expanded_mask = flat_mask.unsqueeze(-1).to(dtype=flat_tokens.dtype)
    counts = expanded_mask.sum(dim=1).clamp_min(1.0)
    return (flat_tokens * expanded_mask).sum(dim=1) / counts


def _write_geometry_monitor_rows(
    rows: list[dict[str, float | int | str | bool | None]],
    *,
    output_json_path: Path,
    output_csv_path: Path,
) -> tuple[Path, Path]:
    write_json_payload({"cross_session_geometry_monitor": rows}, output_json_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
    with output_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_json_path, output_csv_path


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


def _run_cross_session_geometry_monitor(
    *,
    experiment_config: ExperimentConfig,
    data_config_path: str | Path,
    dataset_view,
    model: torch.nn.Module,
    device: torch.device,
    split_name: str,
    max_batches: int,
    neighbor_k: int,
    target_label: str,
    target_label_mode: str,
    target_label_match_value: str | None,
) -> dict[str, float | int | None]:
    tokenizer = build_tokenizer_from_config(experiment_config)
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

    features: list[torch.Tensor] = []
    labels: list[str] = []
    session_ids: list[str] = []
    subject_ids: list[str] = []
    scaler_enabled = bool(experiment_config.execution.mixed_precision and device.type == "cuda")

    model_was_training = bool(model.training)
    model.eval()
    with torch.no_grad():
        for batch in iter_sampler_batches(
            dataset=bundle.dataset,
            sampler=sampler,
            collator=tokenizer,
            batch_size=experiment_config.optimization.batch_size,
            max_batches=max_batches,
        ):
            label_tensor = extract_binary_labels(
                batch,
                target_label=target_label,
                target_label_mode=target_label_mode,
                target_label_match_value=target_label_match_value,
            )
            device_batch = batch.to(device)
            autocast_context = (
                torch.autocast(device_type=device.type, dtype=torch.float16) if scaler_enabled else nullcontext()
            )
            with autocast_context:
                forward_output = model(device_batch)
            features.append(_pooled_window_features(forward_output.tokens, forward_output.patch_mask))
            labels.extend(str(int(float(value.item()) > 0.5)) for value in label_tensor)
            session_ids.extend(str(value) for value in batch.provenance.session_ids)
            subject_ids.extend(str(value) for value in batch.provenance.subject_ids)

    if hasattr(bundle.dataset, "_close_open_files"):
        bundle.dataset._close_open_files()
    if model_was_training:
        model.train()

    if not features:
        return {
            "sample_count": 0,
            "neighbor_k": 0,
            "label_neighbor_enrichment": None,
            "session_neighbor_enrichment": None,
            "subject_neighbor_enrichment": None,
        }

    feature_tensor = torch.cat(features, dim=0)
    geometry_summary = summarize_neighbor_geometry(
        features=feature_tensor,
        attributes={
            "label": tuple(labels),
            "session_id": tuple(session_ids),
            "subject_id": tuple(subject_ids),
        },
        neighbor_k=neighbor_k,
    )
    metrics = geometry_summary.get("metrics") or {}
    return {
        "sample_count": int(geometry_summary.get("sample_count", 0)),
        "neighbor_k": int(geometry_summary.get("neighbor_k", 0)),
        "label_neighbor_enrichment": ((metrics.get("label") or {}).get("enrichment_over_base")),
        "session_neighbor_enrichment": ((metrics.get("session_id") or {}).get("enrichment_over_base")),
        "subject_neighbor_enrichment": ((metrics.get("subject_id") or {}).get("enrichment_over_base")),
    }


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
        training_variant_name=experiment_config.objective.cross_session_aug.training_variant_name,
        cross_session_aug_enabled=bool(experiment_config.objective.cross_session_aug.enabled),
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

    tokenizer = build_tokenizer_from_config(experiment_config)
    device = resolve_device(experiment_config.execution.device)
    _set_training_seed(experiment_config.seed, device=device)
    model = build_model_from_config(experiment_config)
    objective = build_objective_from_config(experiment_config)
    aug_config = experiment_config.objective.cross_session_aug
    canonical_regions = _derive_canonical_regions(
        dataset_view=dataset_view,
        split_name=train_split,
        explicit_regions=aug_config.canonical_regions,
    )
    region_head: RegionRatePredictiveHead | None = None
    donor_cache: RegionRateDonorCache | None = None
    region_target_builder: RegionRateTargetBuilder | None = None
    region_loss = CrossSessionRegionLoss() if aug_config.enabled else None
    geometry_monitor_rows: list[dict[str, float | int | str | bool | None]] = []
    geometry_monitor_json_path: Path | None = None
    geometry_monitor_csv_path: Path | None = None
    training_history_json_path = experiment_config.artifacts.summary_path.with_name("training_history.json")
    training_history_csv_path = experiment_config.artifacts.summary_path.with_name("training_history.csv")
    training_history_rows: list[dict[str, float | int | str | bool | None]] = []
    if aug_config.enabled:
        if not canonical_regions:
            raise ValueError(
                "objective.cross_session_aug.enabled requires at least one canonical region in the training split."
            )
        region_head = RegionRatePredictiveHead(
            model.encoder.config.d_model,
            num_regions=len(canonical_regions),
        )
        donor_cache = RegionRateDonorCache(
            max_size_per_label_session=aug_config.donor_cache_size_per_label_session
        )
        region_target_builder = RegionRateTargetBuilder(
            canonical_regions=canonical_regions,
            donor_cache=donor_cache,
            min_shared_regions=aug_config.min_shared_regions,
            exclude_final_prediction_patch=experiment_config.objective.exclude_final_prediction_patch,
            rng=random.Random(experiment_config.seed + 137),
        )
    modules_for_optimization = [model]
    if region_head is not None:
        modules_for_optimization.append(region_head)
    optimizer = build_optimizer(
        experiment_config,
        [parameter for module in modules_for_optimization for parameter in module.parameters()],
    )
    scheduler = build_scheduler(experiment_config, optimizer)
    model = model.to(device)
    if region_head is not None:
        region_head = region_head.to(device)
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

    def _auxiliary_checkpoint_state() -> dict[str, object] | None:
        if not aug_config.enabled or region_head is None or donor_cache is None:
            return None
        return {
            "region_head_state": region_head.state_dict(),
            "donor_cache_state": donor_cache.state_dict(),
            "canonical_regions": list(canonical_regions),
        }

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
    latest_epoch_checkpoint_path: Path | None = None
    latest_epoch_checkpoint: TrainingCheckpoint | None = None

    if experiment_config.training.resume_checkpoint is not None:
        state = load_training_checkpoint(experiment_config.training.resume_checkpoint, map_location=device)
        model.load_state_dict(state["model_state"])
        auxiliary_state = state.get("auxiliary_state")
        if aug_config.enabled:
            if region_head is None or donor_cache is None:
                raise RuntimeError("auxiliary modules were not initialized for an augmented resume")
            if auxiliary_state is None:
                raise ValueError(
                    "Resume checkpoint is missing auxiliary_state, but objective.cross_session_aug.enabled is true."
                )
            region_head_state = auxiliary_state.get("region_head_state")
            if region_head_state is None:
                raise ValueError("Resume checkpoint auxiliary_state is missing region_head_state.")
            region_head.load_state_dict(region_head_state)
            donor_cache.load_state_dict(auxiliary_state.get("donor_cache_state"))
        elif auxiliary_state is not None:
            raise ValueError(
                "Resume checkpoint contains auxiliary_state, but objective.cross_session_aug.enabled is false. "
                "Use a matching augmented config to resume this run."
            )
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
            if not aug_config.enabled:
                autocast_context = (
                    torch.autocast(device_type=device.type, dtype=torch.float16) if scaler_enabled else nullcontext()
                )
                with autocast_context:
                    step_output = run_training_step(model, objective, batch)
            else:
                if region_head is None or region_target_builder is None or region_loss is None:
                    raise RuntimeError("Augmented training is enabled, but auxiliary modules were not initialized.")
                current_aug_prob = _scheduled_value(
                    start=aug_config.aug_prob_start,
                    end=aug_config.aug_prob_end,
                    epoch=epoch,
                    warmup_epochs=aug_config.warmup_epochs,
                )
                current_region_loss_weight = _scheduled_value(
                    start=aug_config.region_loss_weight_start,
                    end=aug_config.region_loss_weight_end,
                    epoch=epoch,
                    warmup_epochs=aug_config.warmup_epochs,
                )
                label_tensor = extract_binary_labels(
                    batch,
                    target_label=aug_config.target_label,
                    target_label_mode=aug_config.target_label_mode,
                    target_label_match_value=aug_config.target_label_match_value,
                )
                region_targets = region_target_builder(
                    batch,
                    labels=label_tensor,
                    aug_prob=current_aug_prob,
                )
                autocast_context = (
                    torch.autocast(device_type=device.type, dtype=torch.float16) if scaler_enabled else nullcontext()
                )
                with autocast_context:
                    forward_output = model(batch)
                    objective_output = objective(batch, forward_output)
                    predicted_region_rates = region_head(forward_output.tokens, batch.unit_mask)
                    auxiliary_loss, auxiliary_metrics = region_loss.evaluate(
                        predicted_region_rates=predicted_region_rates,
                        region_targets=region_targets,
                        region_loss_weight=current_region_loss_weight,
                    )
                    total_loss = objective_output.loss + auxiliary_loss
                losses = dict(objective_output.losses)
                losses["total_loss"] = float(total_loss.detach().item())
                metrics = dict(objective_output.metrics)
                metrics.update(auxiliary_metrics)
                metrics["cross_session_aug_prob"] = float(current_aug_prob)
                metrics["cross_session_region_loss_weight"] = float(current_region_loss_weight)
                step_output = TrainingStepOutput(
                    loss=total_loss,
                    losses=losses,
                    metrics=metrics,
                    batch_size=batch.batch_size,
                    token_count=batch.token_count,
                )
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

        if hasattr(train_bundle.dataset, "_close_open_files"):
            train_bundle.dataset._close_open_files()

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
                checkpoint_path=str(best_checkpoint_path if best_checkpoint_path.exists() else ""),
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
                    auxiliary_state=_auxiliary_checkpoint_state(),
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

            if aug_config.enabled and aug_config.geometry_monitor_every_epochs > 0:
                if epoch % aug_config.geometry_monitor_every_epochs == 0:
                    geometry_metrics = _run_cross_session_geometry_monitor(
                        experiment_config=experiment_config,
                        data_config_path=data_config_path,
                        dataset_view=dataset_view,
                        model=model,
                        device=device,
                        split_name=aug_config.geometry_monitor_split,
                        max_batches=aug_config.geometry_monitor_max_batches,
                        neighbor_k=aug_config.geometry_monitor_neighbor_k,
                        target_label=aug_config.target_label,
                        target_label_mode=aug_config.target_label_mode,
                        target_label_match_value=aug_config.target_label_match_value,
                    )
                    geometry_monitor_rows.append(
                        {
                            "epoch": int(epoch),
                            "training_variant_name": aug_config.training_variant_name,
                            "cross_session_aug_enabled": True,
                            "cross_session_aug_prob": _scheduled_value(
                                start=aug_config.aug_prob_start,
                                end=aug_config.aug_prob_end,
                                epoch=epoch,
                                warmup_epochs=aug_config.warmup_epochs,
                            ),
                            "cross_session_region_loss_weight": _scheduled_value(
                                start=aug_config.region_loss_weight_start,
                                end=aug_config.region_loss_weight_end,
                                epoch=epoch,
                                warmup_epochs=aug_config.warmup_epochs,
                            ),
                            "split_name": aug_config.geometry_monitor_split,
                            **geometry_metrics,
                        }
                    )
                    geometry_monitor_json_path = experiment_config.artifacts.summary_path.with_name(
                        "cross_session_geometry_monitor.json"
                    )
                    geometry_monitor_csv_path = experiment_config.artifacts.summary_path.with_name(
                        "cross_session_geometry_monitor.csv"
                    )
                    _write_geometry_monitor_rows(
                        geometry_monitor_rows,
                        output_json_path=geometry_monitor_json_path,
                        output_csv_path=geometry_monitor_csv_path,
                    )
                    logger.log_artifact(label="cross-session geometry monitor", path=geometry_monitor_json_path)
                    _maybe_emit_training_progress(
                        progress_callback,
                        TrainingProgressEvent(
                            event_type="geometry_monitor",
                            epoch=epoch,
                            epoch_total=experiment_config.training.num_epochs,
                            metrics={
                                key: value
                                for key, value in geometry_monitor_rows[-1].items()
                                if isinstance(value, (int, float))
                            },
                            message="cross-session geometry monitor",
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
                "training_variant_name": aug_config.training_variant_name,
                "cross_session_aug_enabled": bool(aug_config.enabled),
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
                auxiliary_state=_auxiliary_checkpoint_state(),
            )
            epoch_path = experiment_config.artifacts.checkpoint_dir / f"{experiment_config.artifacts.checkpoint_prefix}_latest.pt"
            save_training_checkpoint(checkpoint, epoch_path)
            latest_epoch_checkpoint_path = epoch_path
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

    if not best_checkpoint_path.exists() and latest_epoch_checkpoint is not None:
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
        geometry_monitor_json_path=geometry_monitor_json_path,
        geometry_monitor_csv_path=geometry_monitor_csv_path,
    )
