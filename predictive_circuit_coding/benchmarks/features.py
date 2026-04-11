from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Callable

import torch
from sklearn.decomposition import PCA

from predictive_circuit_coding.data import resolve_runtime_dataset_view
from predictive_circuit_coding.decoding import apply_session_linear_transforms_to_features
from predictive_circuit_coding.decoding.extract import DiscoveryWindowPlan
from predictive_circuit_coding.decoding.labels import extract_binary_labels
from predictive_circuit_coding.training.artifacts import load_training_checkpoint
from predictive_circuit_coding.training.config import ExperimentConfig
from predictive_circuit_coding.training.contracts import DiscoveryCoverageSummary
from predictive_circuit_coding.training.factories import build_model_from_config, build_tokenizer_from_config
from predictive_circuit_coding.training.runtime import iter_sampler_batches, resolve_device
from predictive_circuit_coding.windowing import (
    FixedWindowConfig,
    build_dataset_bundle,
    build_sequential_fixed_window_sampler,
)


ProgressCallback = Callable[[int, int | None], None]


@dataclass(frozen=True)
class BenchmarkWindowCollection:
    feature_family: str
    pooled_features: torch.Tensor
    token_tensors: torch.Tensor
    token_mask: torch.Tensor
    labels: torch.Tensor
    window_recording_ids: tuple[str, ...]
    window_session_ids: tuple[str, ...]
    window_subject_ids: tuple[str, ...]
    window_start_s: tuple[float, ...]
    window_end_s: tuple[float, ...]
    shard_paths: tuple[Path, ...]
    coverage_summary: DiscoveryCoverageSummary | None = None


@dataclass(frozen=True)
class GlobalLinearTransform:
    mean: torch.Tensor
    linear: torch.Tensor
    transform_type: str


def _maybe_update_progress(progress_callback: ProgressCallback | None, current: int, total: int | None) -> None:
    if progress_callback is not None:
        progress_callback(current, total)


def _pooled_features_from_tokens(tokens: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    mask = token_mask.to(dtype=tokens.dtype).unsqueeze(-1)
    token_counts = mask.sum(dim=1).clamp_min(1.0)
    return (tokens * mask).sum(dim=1) / token_counts


def _build_count_token_embeddings(batch) -> tuple[torch.Tensor, torch.Tensor]:
    patch_counts = batch.patch_counts.detach().cpu().to(dtype=torch.float32)
    patch_mask = batch.patch_mask.detach().cpu()
    batch_size, unit_count, patch_count, patch_bins = patch_counts.shape
    feature_dim = patch_count * patch_bins
    tokens = torch.zeros((batch_size, unit_count, patch_count, feature_dim), dtype=torch.float32)
    for patch_index in range(patch_count):
        start = patch_index * patch_bins
        end = start + patch_bins
        tokens[:, :, patch_index, start:end] = patch_counts[:, :, patch_index, :]
    flat_tokens = tokens.reshape(batch_size, unit_count * patch_count, feature_dim)
    flat_mask = patch_mask.reshape(batch_size, unit_count * patch_count)
    return flat_tokens, flat_mask


def _write_benchmark_token_shard(
    *,
    shard_dir: Path,
    shard_index: int,
    batch,
    tokens: torch.Tensor,
    token_mask: torch.Tensor,
    labels: torch.Tensor,
    planned_windows: list | None = None,
) -> Path | None:
    embeddings: list[torch.Tensor] = []
    recording_ids: list[str] = []
    session_ids: list[str] = []
    subject_ids: list[str] = []
    unit_ids: list[str] = []
    unit_regions: list[str] = []
    unit_depth_um: list[float] = []
    patch_index: list[int] = []
    patch_start_s: list[float] = []
    patch_end_s: list[float] = []
    window_start_s: list[float] = []
    window_end_s: list[float] = []
    window_labels: list[int] = []

    patch_count = int(batch.patch_mask.shape[2])
    for batch_index in range(tokens.shape[0]):
        sample_tokens = tokens[batch_index]
        sample_mask = token_mask[batch_index]
        sample_label = int(labels[batch_index].item() > 0.0)
        window = planned_windows[batch_index] if planned_windows is not None else None
        flat_position = 0
        for unit_index, unit_id in enumerate(batch.provenance.unit_ids[batch_index]):
            for local_patch_index in range(patch_count):
                if not bool(batch.patch_mask[batch_index, unit_index, local_patch_index].item()):
                    flat_position += 1
                    continue
                if bool(sample_mask[flat_position].item()):
                    embeddings.append(sample_tokens[flat_position].detach().cpu())
                    if window is not None:
                        recording_ids.append(str(window.recording_id))
                        session_ids.append(str(window.session_id))
                        subject_ids.append(str(window.subject_id))
                        window_start_s.append(float(window.window_start_s))
                        window_end_s.append(float(window.window_end_s))
                    else:
                        recording_ids.append(str(batch.provenance.recording_ids[batch_index]))
                        session_ids.append(str(batch.provenance.session_ids[batch_index]))
                        subject_ids.append(str(batch.provenance.subject_ids[batch_index]))
                        window_start_s.append(float(batch.provenance.window_start_s[batch_index].item()))
                        window_end_s.append(float(batch.provenance.window_end_s[batch_index].item()))
                    unit_ids.append(unit_id)
                    unit_regions.append(batch.provenance.unit_regions[batch_index][unit_index])
                    unit_depth_um.append(float(batch.provenance.unit_depth_um[batch_index, unit_index].item()))
                    patch_index.append(int(local_patch_index))
                    patch_start_s.append(float(batch.provenance.patch_start_s[batch_index, local_patch_index].item()))
                    patch_end_s.append(float(batch.provenance.patch_end_s[batch_index, local_patch_index].item()))
                    window_labels.append(sample_label)
                flat_position += 1

    if not embeddings:
        return None

    shard_path = shard_dir / f"token_shard_{shard_index:05d}.pt"
    torch.save(
        {
            "embeddings": torch.stack(embeddings, dim=0),
            "recording_ids": recording_ids,
            "session_ids": session_ids,
            "subject_ids": subject_ids,
            "unit_ids": unit_ids,
            "unit_regions": unit_regions,
            "unit_depth_um": torch.tensor(unit_depth_um, dtype=torch.float32),
            "patch_index": torch.tensor(patch_index, dtype=torch.long),
            "patch_start_s": torch.tensor(patch_start_s, dtype=torch.float32),
            "patch_end_s": torch.tensor(patch_end_s, dtype=torch.float32),
            "window_start_s": torch.tensor(window_start_s, dtype=torch.float32),
            "window_end_s": torch.tensor(window_end_s, dtype=torch.float32),
            "labels": torch.tensor(window_labels, dtype=torch.long),
        },
        shard_path,
    )
    return shard_path


def _pad_token_chunks(
    token_chunks: list[torch.Tensor],
    mask_chunks: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not token_chunks:
        return torch.empty((0, 0, 0), dtype=torch.float32), torch.empty((0, 0), dtype=torch.bool)
    max_seq_len = max(chunk.shape[1] for chunk in token_chunks)
    total_windows = sum(chunk.shape[0] for chunk in token_chunks)
    token_dim = token_chunks[0].shape[2]
    token_tensors = torch.zeros((total_windows, max_seq_len, token_dim), dtype=torch.float32)
    token_mask = torch.zeros((total_windows, max_seq_len), dtype=torch.bool)
    offset = 0
    for token_chunk, mask_chunk in zip(token_chunks, mask_chunks, strict=False):
        count = int(token_chunk.shape[0])
        seq_len = int(token_chunk.shape[1])
        token_tensors[offset : offset + count, :seq_len] = token_chunk
        token_mask[offset : offset + count, :seq_len] = mask_chunk
        offset += count
    return token_tensors, token_mask


def fit_global_pca_transform(
    *,
    features: torch.Tensor,
    fit_indices: torch.Tensor,
    max_components: int | None,
) -> tuple[GlobalLinearTransform | None, dict[str, float | int | None]]:
    feature_tensor = features.detach().cpu().to(dtype=torch.float32)
    if feature_tensor.numel() == 0:
        return None, {
            "transform_type": "pca",
            "applied": False,
            "components": None,
            "input_dim": 0,
            "explained_variance_ratio_sum": None,
        }
    fit_tensor = feature_tensor.index_select(0, fit_indices.detach().cpu().to(dtype=torch.long))
    input_dim = int(feature_tensor.shape[1])
    component_limit = int(max_components) if max_components is not None else input_dim
    component_count = max(1, min(component_limit, input_dim, int(fit_tensor.shape[0])))
    if component_count >= input_dim:
        return None, {
            "transform_type": "pca",
            "applied": False,
            "components": input_dim,
            "input_dim": input_dim,
            "explained_variance_ratio_sum": 1.0,
        }
    model = PCA(n_components=component_count, svd_solver="auto", random_state=0)
    model.fit(fit_tensor.numpy())
    transform = GlobalLinearTransform(
        mean=torch.tensor(model.mean_, dtype=torch.float32),
        linear=torch.tensor(model.components_.T, dtype=torch.float32),
        transform_type="pca",
    )
    return transform, {
        "transform_type": "pca",
        "applied": True,
        "components": component_count,
        "input_dim": input_dim,
        "explained_variance_ratio_sum": float(model.explained_variance_ratio_.sum()),
    }


def apply_global_linear_transform_to_features(
    *,
    features: torch.Tensor,
    transform: GlobalLinearTransform | None,
) -> torch.Tensor:
    if transform is None or features.numel() == 0:
        return features.detach().cpu().to(dtype=torch.float32)
    feature_tensor = features.detach().cpu().to(dtype=torch.float32)
    return (feature_tensor - transform.mean) @ transform.linear


def apply_global_linear_transform_to_tokens(
    *,
    tokens: torch.Tensor,
    transform: GlobalLinearTransform | None,
) -> torch.Tensor:
    if transform is None or tokens.numel() == 0:
        return tokens.detach().cpu().to(dtype=torch.float32)
    token_tensor = tokens.detach().cpu().to(dtype=torch.float32)
    original_shape = token_tensor.shape
    flat = token_tensor.reshape(-1, original_shape[-1])
    transformed = (flat - transform.mean) @ transform.linear
    return transformed.reshape(original_shape[0], original_shape[1], -1)


def _build_encoder_tokens(
    *,
    experiment_config: ExperimentConfig,
    checkpoint_path: str | Path | None,
    use_checkpoint: bool,
):
    device = resolve_device(experiment_config.execution.device)
    if use_checkpoint:
        model = build_model_from_config(experiment_config).to(device)
    else:
        cuda_devices: list[int] = []
        if device.type == "cuda" and torch.cuda.is_available():
            cuda_devices = [torch.cuda.current_device() if device.index is None else int(device.index)]
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            torch.manual_seed(int(experiment_config.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(experiment_config.seed))
            model = build_model_from_config(experiment_config).to(device)
    if use_checkpoint:
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required when use_checkpoint=True")
        checkpoint = load_training_checkpoint(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    model.eval()
    scaler_enabled = bool(experiment_config.execution.mixed_precision and device.type == "cuda")
    return model, device, scaler_enabled


def extract_benchmark_selected_windows(
    *,
    experiment_config: ExperimentConfig,
    data_config_path: str | Path,
    feature_family: str,
    window_plan: DiscoveryWindowPlan,
    checkpoint_path: str | Path | None = None,
    dataset_view=None,
    shard_dir: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> BenchmarkWindowCollection:
    dataset_view = dataset_view or resolve_runtime_dataset_view(
        experiment_config=experiment_config,
        data_config_path=data_config_path,
    )
    bundle = build_dataset_bundle(
        workspace=dataset_view.workspace,
        split_manifest=dataset_view.split_manifest,
        split=window_plan.split_name,
        config_dir=dataset_view.config_dir,
        config_name_prefix=dataset_view.config_name_prefix,
        dataset_split=dataset_view.dataset_split,
    )
    tokenizer = build_tokenizer_from_config(experiment_config)
    selected_windows = [window_plan.windows[index] for index in window_plan.selected_indices.tolist()]
    total_selected = len(selected_windows)

    model = None
    device = torch.device("cpu")
    scaler_enabled = False
    if feature_family in {"encoder", "untrained_encoder"}:
        model, device, scaler_enabled = _build_encoder_tokens(
            experiment_config=experiment_config,
            checkpoint_path=checkpoint_path,
            use_checkpoint=feature_family == "encoder",
        )

    shard_paths: list[Path] = []
    if shard_dir is not None:
        shard_root = Path(shard_dir)
        if shard_root.exists():
            shutil.rmtree(shard_root)
        shard_root.mkdir(parents=True, exist_ok=True)
    else:
        shard_root = None

    pooled_chunks: list[torch.Tensor] = []
    token_chunks: list[torch.Tensor] = []
    mask_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    recording_ids: list[str] = []
    session_ids: list[str] = []
    subject_ids: list[str] = []
    window_start_s: list[float] = []
    window_end_s: list[float] = []
    shard_index = 0
    processed = 0

    with torch.no_grad():
        for start in range(0, total_selected, experiment_config.optimization.batch_size):
            window_batch = selected_windows[start : start + experiment_config.optimization.batch_size]
            samples = [
                bundle.dataset.get(window.recording_id, window.window_start_s, window.window_end_s)
                for window in window_batch
            ]
            batch = tokenizer(samples)
            labels = torch.tensor([float(window.label) for window in window_batch], dtype=torch.float32)
            if feature_family in {"encoder", "untrained_encoder"}:
                device_batch = batch.to(device)
                autocast_context = (
                    torch.autocast(device_type=device.type, dtype=torch.float16) if scaler_enabled else nullcontext()
                )
                with autocast_context:
                    output = model(device_batch)  # type: ignore[operator]
                flat_tokens = output.tokens.detach().cpu().reshape(output.tokens.shape[0], -1, output.tokens.shape[-1])
                flat_mask = output.patch_mask.detach().cpu().reshape(output.patch_mask.shape[0], -1)
            elif feature_family == "count_patch_mean":
                flat_tokens, flat_mask = _build_count_token_embeddings(batch)
            else:
                raise ValueError(f"Unsupported feature_family: {feature_family}")

            pooled_chunks.append(_pooled_features_from_tokens(flat_tokens, flat_mask))
            token_chunks.append(flat_tokens)
            mask_chunks.append(flat_mask)
            label_chunks.append(labels)
            recording_ids.extend(str(window.recording_id) for window in window_batch)
            session_ids.extend(str(window.session_id) for window in window_batch)
            subject_ids.extend(str(window.subject_id) for window in window_batch)
            window_start_s.extend(float(window.window_start_s) for window in window_batch)
            window_end_s.extend(float(window.window_end_s) for window in window_batch)
            if shard_root is not None:
                shard_path = _write_benchmark_token_shard(
                    shard_dir=shard_root,
                    shard_index=shard_index,
                    batch=batch,
                    tokens=flat_tokens,
                    token_mask=flat_mask,
                    labels=labels,
                    planned_windows=window_batch,
                )
                if shard_path is not None:
                    shard_paths.append(shard_path)
                    shard_index += 1
            processed += len(window_batch)
            _maybe_update_progress(progress_callback, processed, total_selected)

    if hasattr(bundle.dataset, "_close_open_files"):
        bundle.dataset._close_open_files()

    token_tensors, token_mask = _pad_token_chunks(token_chunks, mask_chunks)
    return BenchmarkWindowCollection(
        feature_family=feature_family,
        pooled_features=torch.cat(pooled_chunks, dim=0) if pooled_chunks else torch.empty((0, 0), dtype=torch.float32),
        token_tensors=token_tensors,
        token_mask=token_mask,
        labels=torch.cat(label_chunks, dim=0) if label_chunks else torch.empty((0,), dtype=torch.float32),
        window_recording_ids=tuple(recording_ids),
        window_session_ids=tuple(session_ids),
        window_subject_ids=tuple(subject_ids),
        window_start_s=tuple(window_start_s),
        window_end_s=tuple(window_end_s),
        shard_paths=tuple(shard_paths),
        coverage_summary=window_plan.coverage_summary,
    )


def extract_benchmark_split_collection(
    *,
    experiment_config: ExperimentConfig,
    data_config_path: str | Path,
    feature_family: str,
    split_name: str,
    target_label: str,
    target_label_mode: str = "auto",
    target_label_match_value: str | None = None,
    checkpoint_path: str | Path | None = None,
    dataset_view=None,
    max_batches: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> BenchmarkWindowCollection:
    dataset_view = dataset_view or resolve_runtime_dataset_view(
        experiment_config=experiment_config,
        data_config_path=data_config_path,
    )
    bundle = build_dataset_bundle(
        workspace=dataset_view.workspace,
        split_manifest=dataset_view.split_manifest,
        split=split_name,
        config_dir=dataset_view.config_dir,
        config_name_prefix=dataset_view.config_name_prefix,
        dataset_split=dataset_view.dataset_split,
    )
    tokenizer = build_tokenizer_from_config(experiment_config)
    sampler = build_sequential_fixed_window_sampler(
        bundle.dataset,
        window=FixedWindowConfig(
            window_length_s=experiment_config.data_runtime.context_duration_s,
            step_s=experiment_config.evaluation.sequential_step_s,
        ),
    )

    model = None
    device = torch.device("cpu")
    scaler_enabled = False
    if feature_family in {"encoder", "untrained_encoder"}:
        model, device, scaler_enabled = _build_encoder_tokens(
            experiment_config=experiment_config,
            checkpoint_path=checkpoint_path,
            use_checkpoint=feature_family == "encoder",
        )

    pooled_chunks: list[torch.Tensor] = []
    token_chunks: list[torch.Tensor] = []
    mask_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    recording_ids: list[str] = []
    session_ids: list[str] = []
    subject_ids: list[str] = []
    window_start_s: list[float] = []
    window_end_s: list[float] = []
    processed_batches = 0

    with torch.no_grad():
        for batch in iter_sampler_batches(
            dataset=bundle.dataset,
            sampler=sampler,
            collator=tokenizer,
            batch_size=experiment_config.optimization.batch_size,
            max_batches=max_batches,
        ):
            labels = extract_binary_labels(
                batch,
                target_label=target_label,
                target_label_mode=target_label_mode,
                target_label_match_value=target_label_match_value,
            )
            if feature_family in {"encoder", "untrained_encoder"}:
                device_batch = batch.to(device)
                autocast_context = (
                    torch.autocast(device_type=device.type, dtype=torch.float16) if scaler_enabled else nullcontext()
                )
                with autocast_context:
                    output = model(device_batch)  # type: ignore[operator]
                flat_tokens = output.tokens.detach().cpu().reshape(output.tokens.shape[0], -1, output.tokens.shape[-1])
                flat_mask = output.patch_mask.detach().cpu().reshape(output.patch_mask.shape[0], -1)
            elif feature_family == "count_patch_mean":
                flat_tokens, flat_mask = _build_count_token_embeddings(batch)
            else:
                raise ValueError(f"Unsupported feature_family: {feature_family}")

            pooled_chunks.append(_pooled_features_from_tokens(flat_tokens, flat_mask))
            token_chunks.append(flat_tokens)
            mask_chunks.append(flat_mask)
            label_chunks.append(labels.detach().cpu())
            recording_ids.extend(str(value) for value in batch.provenance.recording_ids)
            session_ids.extend(str(value) for value in batch.provenance.session_ids)
            subject_ids.extend(str(value) for value in batch.provenance.subject_ids)
            window_start_s.extend(float(value.item()) for value in batch.provenance.window_start_s)
            window_end_s.extend(float(value.item()) for value in batch.provenance.window_end_s)
            processed_batches += 1
            _maybe_update_progress(progress_callback, processed_batches, max_batches)

    if hasattr(bundle.dataset, "_close_open_files"):
        bundle.dataset._close_open_files()

    token_tensors, token_mask = _pad_token_chunks(token_chunks, mask_chunks)
    return BenchmarkWindowCollection(
        feature_family=feature_family,
        pooled_features=torch.cat(pooled_chunks, dim=0) if pooled_chunks else torch.empty((0, 0), dtype=torch.float32),
        token_tensors=token_tensors,
        token_mask=token_mask,
        labels=torch.cat(label_chunks, dim=0) if label_chunks else torch.empty((0,), dtype=torch.float32),
        window_recording_ids=tuple(recording_ids),
        window_session_ids=tuple(session_ids),
        window_subject_ids=tuple(subject_ids),
        window_start_s=tuple(window_start_s),
        window_end_s=tuple(window_end_s),
        shard_paths=tuple(),
        coverage_summary=None,
    )


def apply_collection_transform(
    *,
    collection: BenchmarkWindowCollection,
    transform: GlobalLinearTransform | None,
) -> BenchmarkWindowCollection:
    return BenchmarkWindowCollection(
        feature_family=collection.feature_family,
        pooled_features=apply_global_linear_transform_to_features(
            features=collection.pooled_features,
            transform=transform,
        ),
        token_tensors=apply_global_linear_transform_to_tokens(
            tokens=collection.token_tensors,
            transform=transform,
        ),
        token_mask=collection.token_mask.detach().cpu(),
        labels=collection.labels.detach().cpu(),
        window_recording_ids=collection.window_recording_ids,
        window_session_ids=collection.window_session_ids,
        window_subject_ids=collection.window_subject_ids,
        window_start_s=collection.window_start_s,
        window_end_s=collection.window_end_s,
        shard_paths=collection.shard_paths,
        coverage_summary=collection.coverage_summary,
    )


def write_collection_token_shards(
    *,
    collection: BenchmarkWindowCollection,
    output_dir: str | Path,
    global_transform: GlobalLinearTransform | None = None,
    session_transforms: dict[str, object] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[Path, ...]:
    target_dir = Path(output_dir)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    total_shards = len(collection.shard_paths)
    for shard_index, shard_path in enumerate(collection.shard_paths):
        payload = torch.load(Path(shard_path), map_location="cpu", weights_only=False)
        embeddings = payload["embeddings"].detach().cpu().to(dtype=torch.float32)
        if global_transform is not None:
            embeddings = apply_global_linear_transform_to_features(
                features=embeddings,
                transform=global_transform,
            )
        if session_transforms is not None:
            embeddings = apply_session_linear_transforms_to_features(
                features=embeddings,
                session_ids=tuple(str(value) for value in payload["session_ids"]),
                transforms=session_transforms,
            )
        output_payload = dict(payload)
        output_payload["embeddings"] = embeddings
        output_path = target_dir / f"token_shard_{shard_index:05d}.pt"
        torch.save(output_payload, output_path)
        output_paths.append(output_path)
        _maybe_update_progress(progress_callback, shard_index + 1, total_shards)
    return tuple(output_paths)
