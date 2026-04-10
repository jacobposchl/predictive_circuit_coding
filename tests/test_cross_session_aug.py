from __future__ import annotations

import torch

from predictive_circuit_coding.models import RegionRatePredictiveHead
from predictive_circuit_coding.objectives import CrossSessionRegionLoss, RegionRateDonorCache, RegionRateTargetBuilder
from predictive_circuit_coding.training.contracts import PopulationWindowBatch, TokenProvenanceBatch


def _build_batch(
    *,
    session_ids: tuple[str, ...],
    subject_ids: tuple[str, ...],
    labels: tuple[bool, ...],
    patch_counts: torch.Tensor,
    unit_regions: tuple[tuple[str, ...], ...],
) -> PopulationWindowBatch:
    batch_size, unit_count, patch_count, patch_bins = patch_counts.shape
    counts = patch_counts.reshape(batch_size, unit_count, patch_count * patch_bins)
    unit_mask = torch.ones((batch_size, unit_count), dtype=torch.bool)
    patch_mask = unit_mask.unsqueeze(-1).expand(-1, -1, patch_count).clone()
    provenance = TokenProvenanceBatch(
        recording_ids=tuple(f"allen_visual_behavior_neuropixels/{session_id}" for session_id in session_ids),
        session_ids=session_ids,
        subject_ids=subject_ids,
        unit_ids=tuple(tuple(f"{session_id}_u{index}" for index in range(unit_count)) for session_id in session_ids),
        unit_regions=unit_regions,
        unit_depth_um=torch.zeros((batch_size, unit_count), dtype=torch.float32),
        patch_start_s=torch.zeros((batch_size, patch_count), dtype=torch.float32),
        patch_end_s=torch.ones((batch_size, patch_count), dtype=torch.float32),
        window_start_s=torch.zeros((batch_size,), dtype=torch.float32),
        window_end_s=torch.ones((batch_size,), dtype=torch.float32),
        event_annotations=tuple(
            {
                "stimulus_presentations": {
                    "start_s": (0.1,),
                    "end_s": (0.2,),
                    "is_change": (bool(label),),
                }
            }
            for label in labels
        ),
    )
    return PopulationWindowBatch(
        counts=counts,
        patch_counts=patch_counts,
        unit_mask=unit_mask,
        patch_mask=patch_mask,
        bin_width_s=0.1,
        provenance=provenance,
    )


def test_region_rate_target_builder_without_augmentation_matches_expected_future() -> None:
    patch_counts = torch.tensor(
        [[[[1.0, 1.0], [3.0, 3.0]], [[2.0, 2.0], [4.0, 4.0]]]],
        dtype=torch.float32,
    )
    batch = _build_batch(
        session_ids=("s1",),
        subject_ids=("m1",),
        labels=(True,),
        patch_counts=patch_counts,
        unit_regions=(("VISp", "LP"),),
    )
    builder = RegionRateTargetBuilder(
        canonical_regions=("LP", "VISp"),
        donor_cache=RegionRateDonorCache(max_size_per_label_session=4),
        min_shared_regions=1,
        exclude_final_prediction_patch=True,
    )
    targets = builder(batch, labels=torch.tensor([1.0]), aug_prob=0.0)

    assert not bool(targets.is_augmented.any().item())
    assert torch.allclose(
        targets.region_rate_future[0, 0],
        torch.tensor([40.0, 0.0], dtype=torch.float32),
    )
    assert torch.allclose(
        targets.region_rate_future[0, 1],
        torch.tensor([30.0, 0.0], dtype=torch.float32),
    )
    assert bool(targets.valid_patch_mask[0, 0, 0].item()) is True
    assert bool(targets.valid_patch_mask[0, 0, 1].item()) is False


def test_region_rate_target_builder_uses_cached_cross_session_donor() -> None:
    donor_cache = RegionRateDonorCache(max_size_per_label_session=4)
    builder = RegionRateTargetBuilder(
        canonical_regions=("VISp", "LP"),
        donor_cache=donor_cache,
        min_shared_regions=1,
        exclude_final_prediction_patch=True,
    )
    first_batch = _build_batch(
        session_ids=("s1",),
        subject_ids=("m1",),
        labels=(True,),
        patch_counts=torch.tensor([[[[1.0, 1.0], [3.0, 3.0]], [[5.0, 5.0], [7.0, 7.0]]]], dtype=torch.float32),
        unit_regions=(("VISp", "LP"),),
    )
    second_batch = _build_batch(
        session_ids=("s2",),
        subject_ids=("m2",),
        labels=(True,),
        patch_counts=torch.tensor([[[[9.0, 9.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]]]], dtype=torch.float32),
        unit_regions=(("VISp", "LP"),),
    )

    first_targets = builder(first_batch, labels=torch.tensor([1.0]), aug_prob=0.0)
    second_targets = builder(second_batch, labels=torch.tensor([1.0]), aug_prob=1.0)

    assert not bool(first_targets.is_augmented.any().item())
    assert bool(second_targets.is_augmented[0].item()) is True
    assert torch.allclose(second_targets.region_rate_future[0], first_targets.region_rate_future[0])
    assert float(second_targets.diagnostics["cross_session_aug_fraction"]) == 1.0


def test_region_rate_target_builder_excludes_same_session_donors() -> None:
    donor_cache = RegionRateDonorCache(max_size_per_label_session=4)
    builder = RegionRateTargetBuilder(
        canonical_regions=("VISp",),
        donor_cache=donor_cache,
        min_shared_regions=1,
    )
    batch = _build_batch(
        session_ids=("same",),
        subject_ids=("m1",),
        labels=(True,),
        patch_counts=torch.tensor([[[[1.0, 1.0], [2.0, 2.0]]]], dtype=torch.float32),
        unit_regions=(("VISp",),),
    )
    builder(batch, labels=torch.tensor([1.0]), aug_prob=0.0)
    targets = builder(batch, labels=torch.tensor([1.0]), aug_prob=1.0)

    assert bool(targets.is_augmented.any().item()) is False
    assert float(targets.diagnostics["cross_session_donor_available_fraction"]) == 0.0


def test_region_rate_predictive_head_masks_invalid_units() -> None:
    head = RegionRatePredictiveHead(d_model=2, num_regions=2)
    with torch.no_grad():
        head.proj.weight.copy_(torch.eye(2, dtype=torch.float32))
        head.proj.bias.zero_()
    tokens = torch.tensor(
        [[[[1.0, 2.0], [1.0, 2.0]], [[100.0, 100.0], [100.0, 100.0]]]],
        dtype=torch.float32,
    )
    unit_mask = torch.tensor([[True, False]])
    outputs = head(tokens, unit_mask)

    assert outputs.shape == (1, 2, 2)
    assert torch.allclose(outputs[0, :, 0], torch.tensor([1.0, 2.0], dtype=torch.float32))


def test_cross_session_region_loss_zero_without_augmented_items() -> None:
    loss = CrossSessionRegionLoss()
    targets = builder_targets = _build_batch(
        session_ids=("s1",),
        subject_ids=("m1",),
        labels=(True,),
        patch_counts=torch.tensor([[[[1.0, 1.0], [2.0, 2.0]]]], dtype=torch.float32),
        unit_regions=(("VISp",),),
    )
    builder = RegionRateTargetBuilder(
        canonical_regions=("VISp",),
        donor_cache=RegionRateDonorCache(max_size_per_label_session=2),
        min_shared_regions=1,
    )
    region_targets = builder(builder_targets, labels=torch.tensor([1.0]), aug_prob=0.0)
    predicted = torch.zeros_like(region_targets.region_rate_future)
    weighted_loss, metrics = loss.evaluate(
        predicted_region_rates=predicted,
        region_targets=region_targets,
        region_loss_weight=0.5,
    )

    assert float(weighted_loss.detach().item()) == 0.0
    assert metrics["cross_session_region_loss"] == 0.0
    assert metrics["cross_session_region_loss_weighted"] == 0.0


def test_cross_session_region_loss_positive_for_augmented_items() -> None:
    donor_cache = RegionRateDonorCache(max_size_per_label_session=4)
    builder = RegionRateTargetBuilder(
        canonical_regions=("VISp",),
        donor_cache=donor_cache,
        min_shared_regions=1,
    )
    donor_batch = _build_batch(
        session_ids=("s1",),
        subject_ids=("m1",),
        labels=(True,),
        patch_counts=torch.tensor([[[[1.0, 1.0], [5.0, 5.0]]]], dtype=torch.float32),
        unit_regions=(("VISp",),),
    )
    target_batch = _build_batch(
        session_ids=("s2",),
        subject_ids=("m2",),
        labels=(True,),
        patch_counts=torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]], dtype=torch.float32),
        unit_regions=(("VISp",),),
    )
    builder(donor_batch, labels=torch.tensor([1.0]), aug_prob=0.0)
    region_targets = builder(target_batch, labels=torch.tensor([1.0]), aug_prob=1.0)
    predicted = torch.zeros_like(region_targets.region_rate_future)

    weighted_loss, metrics = CrossSessionRegionLoss().evaluate(
        predicted_region_rates=predicted,
        region_targets=region_targets,
        region_loss_weight=0.5,
    )

    assert float(weighted_loss.detach().item()) > 0.0
    assert metrics["cross_session_region_loss"] > 0.0
    assert metrics["cross_session_aug_fraction"] == 1.0
