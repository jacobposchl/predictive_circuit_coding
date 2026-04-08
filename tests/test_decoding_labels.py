from __future__ import annotations

import torch

from predictive_circuit_coding.decoding.labels import (
    extract_binary_label_from_annotations,
    extract_binary_labels,
    extract_matching_values_from_annotations,
)
from predictive_circuit_coding.decoding.probes import evaluate_additive_probe, fit_additive_probe
from predictive_circuit_coding.training.contracts import PopulationWindowBatch, TokenProvenanceBatch


def _build_batch(*, event_annotations: tuple[dict, ...]) -> PopulationWindowBatch:
    batch_size = len(event_annotations)
    counts = torch.zeros((batch_size, 1, 4), dtype=torch.float32)
    patch_counts = torch.zeros((batch_size, 1, 2, 2), dtype=torch.float32)
    unit_mask = torch.ones((batch_size, 1), dtype=torch.bool)
    patch_mask = torch.ones((batch_size, 1, 2), dtype=torch.bool)
    provenance = TokenProvenanceBatch(
        recording_ids=tuple(f"recording_{index}" for index in range(batch_size)),
        session_ids=tuple(f"session_{index}" for index in range(batch_size)),
        subject_ids=tuple(f"subject_{index}" for index in range(batch_size)),
        unit_ids=tuple((f"unit_{index}",) for index in range(batch_size)),
        unit_regions=tuple((f"region_{index}",) for index in range(batch_size)),
        unit_depth_um=torch.ones((batch_size, 1), dtype=torch.float32),
        patch_start_s=torch.zeros((batch_size, 2), dtype=torch.float32),
        patch_end_s=torch.ones((batch_size, 2), dtype=torch.float32),
        window_start_s=torch.zeros((batch_size,), dtype=torch.float32),
        window_end_s=torch.ones((batch_size,), dtype=torch.float32),
        event_annotations=event_annotations,
    )
    return PopulationWindowBatch(
        counts=counts,
        patch_counts=patch_counts,
        unit_mask=unit_mask,
        patch_mask=patch_mask,
        bin_width_s=0.02,
        provenance=provenance,
    )


def test_extract_binary_labels_supports_alias_and_nested_paths() -> None:
    batch = _build_batch(
        event_annotations=(
            {
                "stimulus_presentations": {"is_change": [False, True]},
                "behavior": {"outcome": {"hit": [False, False]}},
            },
            {
                "stimulus_presentations": {"is_change": [False, False]},
                "behavior": {"outcome": {"hit": [0, 1]}},
            },
            {
                "trials": {"is_change": [False, True]},
                "behavior": {"outcome": {"hit": [False, False]}},
            },
        )
    )

    stimulus_change = extract_binary_labels(batch, target_label="stimulus_change")
    behavioral_hit = extract_binary_labels(batch, target_label="behavior.outcome.hit")

    assert torch.equal(stimulus_change, torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32))
    assert torch.equal(behavioral_hit, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32))


def test_event_local_labels_default_to_onset_within_window() -> None:
    annotation = {
        "trials": {
            "start_s": (-0.4, 0.2),
            "end_s": (0.6, 0.8),
            "go": (True, False),
        }
    }

    overlap_label = extract_binary_label_from_annotations(
        annotation,
        target_label="trials.go",
        target_label_mode="overlap",
        window_duration_s=1.0,
    )
    onset_label = extract_binary_label_from_annotations(
        annotation,
        target_label="trials.go",
        target_label_mode="onset_within_window",
        window_duration_s=1.0,
    )
    auto_label = extract_binary_label_from_annotations(
        annotation,
        target_label="trials.go",
        window_duration_s=1.0,
    )

    assert overlap_label == 1.0
    assert onset_label == 0.0
    assert auto_label == 0.0


def test_extract_binary_labels_supports_string_match_value_for_image_identity() -> None:
    batch = _build_batch(
        event_annotations=(
            {
                "stimulus_presentations": {
                    "start_s": (-0.4, 0.2),
                    "end_s": (0.0, 0.5),
                    "image_name": ("im0", "im1"),
                }
            },
            {
                "stimulus_presentations": {
                    "start_s": (0.1,),
                    "end_s": (0.4,),
                    "image_name": ("im2",),
                }
            },
        )
    )

    labels = extract_binary_labels(
        batch,
        target_label="stimulus_presentations.image_name",
        target_label_mode="onset_within_window",
        target_label_match_value="im1",
    )

    assert torch.equal(labels, torch.tensor([1.0, 0.0], dtype=torch.float32))


def test_extract_matching_values_from_annotations_filters_by_window_mode() -> None:
    values = extract_matching_values_from_annotations(
        {
            "stimulus_presentations": {
                "start_s": (-0.3, 0.15, 0.45),
                "end_s": (0.0, 0.25, 0.6),
                "image_name": ("im0", "im1", "im1"),
            }
        },
        target_label="stimulus_presentations.image_name",
        target_label_mode="onset_within_window",
        window_duration_s=1.0,
    )

    assert values == ("im1",)


def test_evaluate_additive_probe_scores_held_out_tokens() -> None:
    tokens = torch.tensor(
        [
            [[2.0, 0.0], [2.0, 0.0]],
            [[1.5, 0.0], [1.5, 0.0]],
            [[0.0, 2.0], [0.0, 2.0]],
            [[0.0, 1.5], [0.0, 1.5]],
        ],
        dtype=torch.float32,
    )
    token_mask = torch.ones((4, 2), dtype=torch.bool)
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32)

    torch.manual_seed(0)
    fit = fit_additive_probe(
        tokens=tokens,
        token_mask=token_mask,
        labels=labels,
        epochs=200,
        learning_rate=1.0e-1,
        label_name="stimulus_change",
    )

    held_out_metrics = evaluate_additive_probe(
        state_dict=fit.state_dict,
        tokens=tokens,
        token_mask=token_mask,
        labels=labels,
    )

    assert held_out_metrics["probe_accuracy"] >= 0.99
    assert held_out_metrics["probe_bce"] < 0.2
