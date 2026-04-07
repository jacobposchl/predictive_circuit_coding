from __future__ import annotations

import csv
from pathlib import Path

import torch

from predictive_circuit_coding.training.contracts import (
    CheckpointMetadata,
    EvaluationSummary,
    TrainingCheckpoint,
    TrainingSummary,
    ValidationSummary,
    write_json_payload,
)


def write_checkpoint_metadata(metadata: CheckpointMetadata, path: str | Path) -> Path:
    return write_json_payload(metadata.to_dict(), path)


def write_training_summary(summary: TrainingSummary, path: str | Path) -> Path:
    return write_json_payload(summary.to_dict(), path)


def write_evaluation_summary(summary: EvaluationSummary, path: str | Path) -> Path:
    return write_json_payload(summary.to_dict(), path)


def write_validation_summary(summary: ValidationSummary, path: str | Path) -> Path:
    return write_json_payload(summary.to_dict(), path)


def write_run_manifest(payload: dict, path: str | Path) -> Path:
    return write_json_payload(payload, path)


def write_validation_summary_csv(summary: ValidationSummary, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset_id",
                "checkpoint_path",
                "discovery_artifact_path",
                "candidate_count",
                "cluster_count",
                "real_probe_accuracy",
                "shuffled_probe_accuracy",
                "held_out_test_probe_accuracy",
                "held_out_similarity_roc_auc",
                "held_out_similarity_pr_auc",
                "real_probe_bce",
                "shuffled_probe_bce",
                "held_out_test_probe_bce",
                "provenance_issue_count",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset_id": summary.dataset_id,
                "checkpoint_path": summary.checkpoint_path,
                "discovery_artifact_path": summary.discovery_artifact_path,
                "candidate_count": summary.candidate_count,
                "cluster_count": summary.cluster_count,
                "real_probe_accuracy": summary.real_label_metrics.get("probe_accuracy", 0.0),
                "shuffled_probe_accuracy": summary.shuffled_label_metrics.get("probe_accuracy", 0.0),
                "held_out_test_probe_accuracy": summary.held_out_test_metrics.get("probe_accuracy", 0.0),
                "held_out_similarity_roc_auc": summary.held_out_similarity_summary.get("window_roc_auc", 0.0),
                "held_out_similarity_pr_auc": summary.held_out_similarity_summary.get("window_pr_auc", 0.0),
                "real_probe_bce": summary.real_label_metrics.get("probe_bce", 0.0),
                "shuffled_probe_bce": summary.shuffled_label_metrics.get("probe_bce", 0.0),
                "held_out_test_probe_bce": summary.held_out_test_metrics.get("probe_bce", 0.0),
                "provenance_issue_count": len(summary.provenance_issues),
            }
        )
    return target


def save_training_checkpoint(checkpoint: TrainingCheckpoint, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint.to_dict(), target)
    return target


def load_training_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict:
    return torch.load(Path(path), map_location=map_location)
