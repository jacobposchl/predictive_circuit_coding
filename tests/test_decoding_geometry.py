from __future__ import annotations

import torch

from predictive_circuit_coding.decoding.geometry import (
    apply_session_linear_transforms_to_features,
    apply_session_linear_transforms_to_tokens,
    build_session_stratified_holdout_split,
    fit_session_alignment_transforms,
    fit_session_whitening_transforms,
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


def test_build_session_stratified_holdout_split_is_deterministic_and_balanced() -> None:
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

    split_a = build_session_stratified_holdout_split(
        labels=labels,
        session_ids=session_ids,
        subject_ids=subject_ids,
        holdout_fraction=0.5,
        seed=17,
    )
    split_b = build_session_stratified_holdout_split(
        labels=labels,
        session_ids=session_ids,
        subject_ids=subject_ids,
        holdout_fraction=0.5,
        seed=17,
    )

    assert torch.equal(split_a.fit_indices, split_b.fit_indices)
    assert torch.equal(split_a.heldout_indices, split_b.heldout_indices)
    assert split_a.valid_session_ids == ("session_a", "session_b")
    for split_indices in (split_a.fit_indices, split_a.heldout_indices):
        split_labels = labels.index_select(0, split_indices)
        split_sessions = tuple(session_ids[index] for index in split_indices.tolist())
        for session_id in ("session_a", "session_b"):
            session_mask = [index for index, value in enumerate(split_sessions) if value == session_id]
            session_labels = split_labels[torch.tensor(session_mask, dtype=torch.long)]
            assert int((session_labels > 0.5).sum().item()) == 1
            assert int((session_labels <= 0.5).sum().item()) == 1


def test_fit_session_whitening_transforms_centers_each_session() -> None:
    features = torch.tensor(
        [
            [4.0, 0.0],
            [0.0, 2.0],
            [-4.0, 0.0],
            [0.0, -2.0],
            [13.0, 10.0],
            [10.0, 11.5],
            [7.0, 10.0],
            [10.0, 8.5],
        ],
        dtype=torch.float32,
    )
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
    fit_indices = torch.arange(features.shape[0], dtype=torch.long)

    transforms, _ = fit_session_whitening_transforms(
        features=features,
        session_ids=session_ids,
        fit_indices=fit_indices,
    )
    whitened = apply_session_linear_transforms_to_features(
        features=features,
        session_ids=session_ids,
        transforms=transforms,
    )

    for session_id in ("session_a", "session_b"):
        indices = [index for index, value in enumerate(session_ids) if value == session_id]
        session_tensor = whitened[torch.tensor(indices, dtype=torch.long)]
        session_mean = session_tensor.mean(dim=0)
        covariance = (session_tensor.T @ session_tensor) / float(session_tensor.shape[0] - 1)
        assert torch.allclose(session_mean, torch.zeros_like(session_mean), atol=1.0e-4)
        assert torch.allclose(covariance, torch.eye(2), atol=2.0e-2)


def test_fit_session_alignment_transforms_improve_anchor_alignment_and_preserve_token_shape() -> None:
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

    transforms, summary = fit_session_alignment_transforms(
        features=features,
        labels=labels,
        session_ids=session_ids,
        fit_indices=torch.arange(features.shape[0], dtype=torch.long),
        reference_session_id="session_a",
    )
    aligned = apply_session_linear_transforms_to_features(
        features=features,
        session_ids=session_ids,
        transforms=transforms,
    )
    tokens = features.unsqueeze(1).repeat(1, 3, 1)
    aligned_tokens = apply_session_linear_transforms_to_tokens(
        tokens=tokens,
        session_ids=session_ids,
        transforms=transforms,
    )

    session_a_positive = aligned[torch.tensor([0, 1], dtype=torch.long)].mean(dim=0)
    session_b_positive = aligned[torch.tensor([4, 5], dtype=torch.long)].mean(dim=0)
    cosine_after = torch.nn.functional.cosine_similarity(
        session_a_positive.unsqueeze(0),
        session_b_positive.unsqueeze(0),
    ).item()

    assert summary["reference_session_id"] == "session_a"
    assert summary["aggregate_metrics"]["mean_label_axis_cosine_after"] > summary["aggregate_metrics"]["mean_label_axis_cosine_before"]
    assert cosine_after > 0.95
    assert aligned_tokens.shape == tokens.shape
