from __future__ import annotations

import torch

from predictive_circuit_coding.decoding.labels import extract_binary_labels
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
        )
    )

    stimulus_change = extract_binary_labels(batch, target_label="stimulus_change")
    behavioral_hit = extract_binary_labels(batch, target_label="behavior.outcome.hit")

    assert torch.equal(stimulus_change, torch.tensor([1.0, 0.0], dtype=torch.float32))
    assert torch.equal(behavioral_hit, torch.tensor([0.0, 1.0], dtype=torch.float32))
