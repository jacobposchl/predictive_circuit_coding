from __future__ import annotations

import pytest
import torch

from predictive_circuit_coding.decoding import fit_additive_probe
from predictive_circuit_coding.validation.artifact_checks import validate_discovery_artifact_identity
from predictive_circuit_coding.validation.run import (
    _validate_candidate_schema,
    _window_similarity_scores,
)


def _base_artifact(checkpoint_path):
    return {
        "dataset_id": "allen_visual_behavior_neuropixels",
        "split_name": "discovery",
        "checkpoint_path": str(checkpoint_path),
        "decoder_summary": {"target_label": "stimulus_change"},
    }


def test_validate_artifact_provenance_rejects_dataset_mismatch(tmp_path):
    checkpoint_path = tmp_path / "pcc_best.pt"
    artifact = _base_artifact(checkpoint_path)
    artifact["dataset_id"] = "other_dataset"

    with pytest.raises(ValueError, match="dataset_id does not match"):
        validate_discovery_artifact_identity(
            artifact=artifact,
            checkpoint_path=checkpoint_path,
            dataset_id="allen_visual_behavior_neuropixels",
            split_name="discovery",
            target_label="stimulus_change",
            require_fields=True,
        )


def test_validate_artifact_provenance_rejects_split_mismatch(tmp_path):
    checkpoint_path = tmp_path / "pcc_best.pt"
    artifact = _base_artifact(checkpoint_path)
    artifact["split_name"] = "test"

    with pytest.raises(ValueError, match="split_name does not match"):
        validate_discovery_artifact_identity(
            artifact=artifact,
            checkpoint_path=checkpoint_path,
            dataset_id="allen_visual_behavior_neuropixels",
            split_name="discovery",
            target_label="stimulus_change",
            require_fields=True,
        )


def test_validate_candidate_schema_rejects_missing_non_noise_clusters():
    candidate = {
        "candidate_id": "candidate_0",
        "cluster_id": -1,
        "recording_id": "recording",
        "session_id": "session",
        "subject_id": "subject",
        "unit_id": "unit",
        "unit_region": "VISp",
        "unit_depth_um": 100.0,
        "patch_index": 0,
        "patch_start_s": 0.0,
        "patch_end_s": 1.0,
        "window_start_s": 0.0,
        "window_end_s": 2.0,
        "label": 1,
        "score": 0.5,
        "embedding": [0.1, 0.2],
    }

    with pytest.raises(ValueError, match="non-noise candidate clusters"):
        _validate_candidate_schema(candidates=[candidate], expected_embedding_dim=2)


def test_validate_candidate_schema_rejects_bad_embedding_dimension():
    candidate = {
        "candidate_id": "candidate_0",
        "cluster_id": 0,
        "recording_id": "recording",
        "session_id": "session",
        "subject_id": "subject",
        "unit_id": "unit",
        "unit_region": "VISp",
        "unit_depth_um": 100.0,
        "patch_index": 0,
        "patch_start_s": 0.0,
        "patch_end_s": 1.0,
        "window_start_s": 0.0,
        "window_end_s": 2.0,
        "label": 1,
        "score": 0.5,
        "embedding": [0.1],
    }

    with pytest.raises(ValueError, match="embedding dimension"):
        _validate_candidate_schema(candidates=[candidate], expected_embedding_dim=2)


def test_window_similarity_scores_are_finite_for_all_masked_windows():
    tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]])
    token_mask = torch.tensor([[True, False], [False, False]])
    scores = _window_similarity_scores(
        tokens=tokens,
        token_mask=token_mask,
        centroids=[torch.tensor([1.0, 0.0])],
    )

    assert torch.isfinite(scores).all()
    assert scores.tolist() == [1.0, 0.0]


def test_fit_additive_probe_is_deterministic_with_seed():
    tokens = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.9, 0.1]],
            [[0.0, 1.0]],
            [[0.1, 0.9]],
        ],
        dtype=torch.float32,
    )
    token_mask = torch.ones((4, 1), dtype=torch.bool)
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32)

    first = fit_additive_probe(
        tokens=tokens,
        token_mask=token_mask,
        labels=labels,
        epochs=3,
        learning_rate=0.1,
        mini_batch_size=2,
        seed=123,
    )
    second = fit_additive_probe(
        tokens=tokens,
        token_mask=token_mask,
        labels=labels,
        epochs=3,
        learning_rate=0.1,
        mini_batch_size=2,
        seed=123,
    )

    assert first.metrics == second.metrics
    for key, value in first.state_dict.items():
        assert torch.equal(value, second.state_dict[key])
