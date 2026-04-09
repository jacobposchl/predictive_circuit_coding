from __future__ import annotations

import torch

from predictive_circuit_coding.decoding.geometry import (
    summarize_candidate_neighbor_geometry,
    summarize_neighbor_geometry,
    summarize_session_alignment_geometry,
)
from predictive_circuit_coding.training.contracts import CandidateTokenRecord


def test_summarize_neighbor_geometry_detects_label_enrichment() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=torch.float32,
    )
    summary = summarize_neighbor_geometry(
        features=features,
        attributes={
            "label": ("positive", "positive", "negative", "negative"),
            "session_id": ("session_a", "session_a", "session_b", "session_b"),
            "subject_id": ("subject_a", "subject_a", "subject_b", "subject_b"),
        },
        neighbor_k=1,
    )

    assert summary["sample_count"] == 4
    assert summary["neighbor_k"] == 1
    assert summary["metrics"]["label"]["mean_neighbor_match_fraction"] == 1.0
    assert summary["metrics"]["label"]["enrichment_over_base"] == 3.0


def test_summarize_candidate_neighbor_geometry_detects_region_enrichment() -> None:
    candidates = (
        CandidateTokenRecord(
            candidate_id="candidate_0000",
            cluster_id=0,
            recording_id="dataset/session_a",
            session_id="session_a",
            subject_id="subject_a",
            unit_id="unit_a0",
            unit_region="VISp",
            unit_depth_um=100.0,
            patch_index=0,
            patch_start_s=0.0,
            patch_end_s=0.5,
            window_start_s=0.0,
            window_end_s=1.0,
            label=1,
            score=0.5,
            embedding=(1.0, 0.0),
        ),
        CandidateTokenRecord(
            candidate_id="candidate_0001",
            cluster_id=0,
            recording_id="dataset/session_b",
            session_id="session_b",
            subject_id="subject_b",
            unit_id="unit_b0",
            unit_region="VISp",
            unit_depth_um=120.0,
            patch_index=0,
            patch_start_s=0.0,
            patch_end_s=0.5,
            window_start_s=0.0,
            window_end_s=1.0,
            label=1,
            score=0.4,
            embedding=(0.95, 0.05),
        ),
        CandidateTokenRecord(
            candidate_id="candidate_0002",
            cluster_id=1,
            recording_id="dataset/session_c",
            session_id="session_c",
            subject_id="subject_c",
            unit_id="unit_c0",
            unit_region="APN",
            unit_depth_um=200.0,
            patch_index=1,
            patch_start_s=0.5,
            patch_end_s=1.0,
            window_start_s=0.0,
            window_end_s=1.0,
            label=1,
            score=0.6,
            embedding=(0.0, 1.0),
        ),
        CandidateTokenRecord(
            candidate_id="candidate_0003",
            cluster_id=1,
            recording_id="dataset/session_d",
            session_id="session_d",
            subject_id="subject_d",
            unit_id="unit_d0",
            unit_region="APN",
            unit_depth_um=220.0,
            patch_index=1,
            patch_start_s=0.5,
            patch_end_s=1.0,
            window_start_s=0.0,
            window_end_s=1.0,
            label=1,
            score=0.7,
            embedding=(0.05, 0.95),
        ),
    )

    summary = summarize_candidate_neighbor_geometry(candidates=candidates, neighbor_k=1)

    assert summary["sample_count"] == 4
    assert summary["neighbor_k"] == 1
    assert summary["metrics"]["unit_region"]["mean_neighbor_match_fraction"] == 1.0
    assert summary["metrics"]["unit_region"]["enrichment_over_base"] == 3.0


def test_summarize_session_alignment_geometry_improves_rotated_label_axes() -> None:
    features = torch.tensor(
        [
            [2.0, 0.0],
            [1.5, 0.1],
            [-2.0, 0.0],
            [-1.5, -0.1],
            [0.0, 2.0],
            [-0.1, 1.5],
            [0.0, -2.0],
            [0.1, -1.5],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([1, 1, 0, 0, 1, 1, 0, 0], dtype=torch.float32)
    session_ids = (
        "session_a",
        "session_a",
        "session_a",
        "session_a",
        "session_b",
        "session_b",
        "session_b",
        "session_b",
    )
    subject_ids = (
        "subject_a",
        "subject_a",
        "subject_a",
        "subject_a",
        "subject_b",
        "subject_b",
        "subject_b",
        "subject_b",
    )

    summary = summarize_session_alignment_geometry(
        features=features,
        labels=labels,
        session_ids=session_ids,
        subject_ids=subject_ids,
        neighbor_k=1,
        reference_session_id="session_a",
    )

    aggregate = summary["aggregate_metrics"]
    assert summary["reference_session_id"] == "session_a"
    assert summary["session_count"] == 2
    assert aggregate["mean_label_axis_cosine_before"] is not None
    assert aggregate["mean_label_axis_cosine_after"] is not None
    assert aggregate["mean_label_axis_cosine_after"] > aggregate["mean_label_axis_cosine_before"]
    assert summary["geometry_aligned"]["metrics"]["session_id"]["mean_neighbor_match_fraction"] < 1.0
