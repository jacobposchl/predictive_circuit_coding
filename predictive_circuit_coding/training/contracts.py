from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

import torch


def _jsonify_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonify_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify_value(item) for item in value]
    return value


@dataclass(frozen=True)
class TokenProvenanceBatch:
    recording_ids: tuple[str, ...]
    session_ids: tuple[str, ...]
    subject_ids: tuple[str, ...]
    unit_ids: tuple[tuple[str, ...], ...]
    unit_regions: tuple[tuple[str, ...], ...]
    unit_depth_um: torch.Tensor
    patch_start_s: torch.Tensor
    patch_end_s: torch.Tensor
    window_start_s: torch.Tensor
    window_end_s: torch.Tensor
    event_annotations: tuple[dict[str, Any], ...]

    def to(self, device: torch.device | str) -> TokenProvenanceBatch:
        return TokenProvenanceBatch(
            recording_ids=self.recording_ids,
            session_ids=self.session_ids,
            subject_ids=self.subject_ids,
            unit_ids=self.unit_ids,
            unit_regions=self.unit_regions,
            unit_depth_um=self.unit_depth_um.to(device),
            patch_start_s=self.patch_start_s.to(device),
            patch_end_s=self.patch_end_s.to(device),
            window_start_s=self.window_start_s.to(device),
            window_end_s=self.window_end_s.to(device),
            event_annotations=self.event_annotations,
        )


@dataclass(frozen=True)
class PopulationWindowBatch:
    counts: torch.Tensor
    patch_counts: torch.Tensor
    unit_mask: torch.Tensor
    patch_mask: torch.Tensor
    bin_width_s: float
    provenance: TokenProvenanceBatch

    @property
    def batch_size(self) -> int:
        return int(self.counts.shape[0])

    @property
    def context_bins(self) -> int:
        return int(self.counts.shape[-1])

    @property
    def num_units(self) -> int:
        return int(self.counts.shape[1])

    @property
    def num_patches(self) -> int:
        return int(self.patch_counts.shape[2])

    @property
    def patch_bins(self) -> int:
        return int(self.patch_counts.shape[-1])

    @property
    def token_count(self) -> int:
        return int(self.patch_mask.sum().item())

    def to(self, device: torch.device | str) -> PopulationWindowBatch:
        return PopulationWindowBatch(
            counts=self.counts.to(device),
            patch_counts=self.patch_counts.to(device),
            unit_mask=self.unit_mask.to(device),
            patch_mask=self.patch_mask.to(device),
            bin_width_s=self.bin_width_s,
            provenance=self.provenance.to(device),
        )


@dataclass(frozen=True)
class ModelForwardOutput:
    tokens: torch.Tensor
    predictive_outputs: torch.Tensor
    reconstruction_outputs: torch.Tensor
    unit_mask: torch.Tensor
    patch_mask: torch.Tensor


@dataclass(frozen=True)
class TrainingStepOutput:
    loss: torch.Tensor
    losses: dict[str, float]
    metrics: dict[str, float]
    batch_size: int
    token_count: int


@dataclass(frozen=True)
class CheckpointMetadata:
    dataset_id: str
    split_name: str
    seed: int
    config_snapshot: dict[str, Any]
    model_hparams: dict[str, Any]
    continuation_baseline_type: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingCheckpoint:
    epoch: int
    global_step: int
    best_metric: float
    metadata: CheckpointMetadata
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any] | None
    best_epoch: int = 0
    best_validation_metrics: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "best_validation_metrics": self.best_validation_metrics,
            "metadata": self.metadata.to_dict(),
            "model_state": self.model_state,
            "optimizer_state": self.optimizer_state,
            "scheduler_state": self.scheduler_state,
        }


@dataclass(frozen=True)
class TrainingSummary:
    dataset_id: str
    split_name: str
    epoch: int
    best_epoch: int
    metrics: dict[str, float]
    losses: dict[str, float]
    checkpoint_path: str
    selection_reason: str = "validated_best"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationSummary:
    dataset_id: str
    split_name: str
    checkpoint_path: str
    metrics: dict[str, float]
    losses: dict[str, float]
    window_count: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checkpoint_path"] = str(self.checkpoint_path)
        return payload


@dataclass(frozen=True)
class FrozenTokenRecord:
    recording_id: str
    session_id: str
    subject_id: str
    unit_id: str
    unit_region: str
    unit_depth_um: float
    patch_index: int
    patch_start_s: float
    patch_end_s: float
    window_start_s: float
    window_end_s: float
    label: int
    score: float
    embedding: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecoderSummary:
    target_label: str
    epochs: int
    learning_rate: float
    metrics: dict[str, float]
    probe_state: dict[str, Any] | None = None
    metric_scope: str = "fit_selected_windows"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["probe_state"] = _jsonify_value(self.probe_state)
        return payload


@dataclass(frozen=True)
class DiscoveryCoverageSummary:
    split_name: str
    target_label: str
    total_scanned_windows: int
    positive_window_count: int
    negative_window_count: int
    selected_positive_count: int
    selected_negative_count: int
    sessions_with_positive_windows: tuple[str, ...]
    sampling_strategy: str = "sequential"
    scan_max_batches: int | None = None
    selected_window_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sessions_with_positive_windows"] = list(self.sessions_with_positive_windows)
        return payload


@dataclass(frozen=True)
class CandidateTokenRecord:
    candidate_id: str
    cluster_id: int
    recording_id: str
    session_id: str
    subject_id: str
    unit_id: str
    unit_region: str
    unit_depth_um: float
    patch_index: int
    patch_start_s: float
    patch_end_s: float
    window_start_s: float
    window_end_s: float
    label: int
    score: float
    embedding: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiscoveryArtifact:
    dataset_id: str
    split_name: str
    checkpoint_path: str
    config_snapshot: dict[str, Any]
    decoder_summary: DecoderSummary
    candidates: tuple[CandidateTokenRecord, ...]
    cluster_stats: dict[str, Any]
    cluster_quality_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "split_name": self.split_name,
            "checkpoint_path": self.checkpoint_path,
            "config_snapshot": self.config_snapshot,
            "decoder_summary": self.decoder_summary.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "cluster_stats": self.cluster_stats,
            "cluster_quality_summary": self.cluster_quality_summary,
        }


@dataclass(frozen=True)
class ValidationSummary:
    dataset_id: str
    checkpoint_path: str
    discovery_artifact_path: str
    real_label_metrics: dict[str, float]
    shuffled_label_metrics: dict[str, float]
    held_out_test_metrics: dict[str, float]
    held_out_similarity_summary: dict[str, Any]
    baseline_sensitivity_summary: dict[str, Any]
    candidate_count: int
    cluster_count: int
    cluster_quality_summary: dict[str, Any]
    provenance_issues: tuple[str, ...]
    sampling_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "checkpoint_path": self.checkpoint_path,
            "discovery_artifact_path": self.discovery_artifact_path,
            "real_label_metrics": self.real_label_metrics,
            "shuffled_label_metrics": self.shuffled_label_metrics,
            "held_out_test_metrics": self.held_out_test_metrics,
            "held_out_similarity_summary": self.held_out_similarity_summary,
            "baseline_sensitivity_summary": self.baseline_sensitivity_summary,
            "candidate_count": self.candidate_count,
            "cluster_count": self.cluster_count,
            "cluster_quality_summary": self.cluster_quality_summary,
            "provenance_issues": list(self.provenance_issues),
            "sampling_summary": self.sampling_summary,
        }


def write_json_payload(payload: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return target
