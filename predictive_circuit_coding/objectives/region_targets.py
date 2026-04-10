from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import torch

from predictive_circuit_coding.training.contracts import PopulationWindowBatch


@dataclass(frozen=True)
class RegionRateTargets:
    region_rate_future: torch.Tensor
    region_mask: torch.Tensor
    valid_patch_mask: torch.Tensor
    is_augmented: torch.Tensor
    donor_available: torch.Tensor
    shared_region_counts: torch.Tensor
    diagnostics: dict[str, float]


@dataclass(frozen=True)
class CachedRegionDonor:
    label_value: int
    session_id: str
    region_rate_future: torch.Tensor
    region_mask: torch.Tensor


class RegionRateDonorCache:
    def __init__(self, *, max_size_per_label_session: int) -> None:
        self.max_size_per_label_session = int(max_size_per_label_session)
        self._items: dict[tuple[int, str], deque[CachedRegionDonor]] = {}

    def _bucket(self, *, label_value: int, session_id: str) -> deque[CachedRegionDonor]:
        key = (int(label_value), str(session_id))
        if key not in self._items:
            self._items[key] = deque(maxlen=self.max_size_per_label_session)
        return self._items[key]

    def add(
        self,
        *,
        label_value: int,
        session_id: str,
        region_rate_future: torch.Tensor,
        region_mask: torch.Tensor,
    ) -> None:
        bucket = self._bucket(label_value=int(label_value), session_id=str(session_id))
        bucket.append(
            CachedRegionDonor(
                label_value=int(label_value),
                session_id=str(session_id),
                region_rate_future=region_rate_future.detach().cpu().to(dtype=torch.float32).clone(),
                region_mask=region_mask.detach().cpu().to(dtype=torch.bool).clone(),
            )
        )

    def state_dict(self) -> dict[str, object]:
        payload_items: list[dict[str, object]] = []
        for (label_value, session_id), bucket in self._items.items():
            payload_items.append(
                {
                    "label_value": int(label_value),
                    "session_id": str(session_id),
                    "donors": [
                        {
                            "region_rate_future": donor.region_rate_future.clone(),
                            "region_mask": donor.region_mask.clone(),
                        }
                        for donor in bucket
                    ],
                }
            )
        return {
            "max_size_per_label_session": int(self.max_size_per_label_session),
            "items": payload_items,
        }

    def load_state_dict(self, state: dict[str, object] | None) -> None:
        self._items = {}
        if not state:
            return
        self.max_size_per_label_session = int(state.get("max_size_per_label_session", self.max_size_per_label_session))
        for item in list(state.get("items", []) or []):
            if not isinstance(item, dict):
                continue
            bucket = self._bucket(
                label_value=int(item.get("label_value", 0)),
                session_id=str(item.get("session_id", "")),
            )
            for donor_payload in list(item.get("donors", []) or []):
                if not isinstance(donor_payload, dict):
                    continue
                bucket.append(
                    CachedRegionDonor(
                        label_value=int(item.get("label_value", 0)),
                        session_id=str(item.get("session_id", "")),
                        region_rate_future=torch.as_tensor(
                            donor_payload.get("region_rate_future"),
                            dtype=torch.float32,
                        ).detach().cpu(),
                        region_mask=torch.as_tensor(
                            donor_payload.get("region_mask"),
                            dtype=torch.bool,
                        ).detach().cpu(),
                    )
                )

    def available_donor_count(self, *, label_value: int, exclude_session_id: str) -> int:
        count = 0
        for (candidate_label, session_id), bucket in self._items.items():
            if int(candidate_label) != int(label_value) or str(session_id) == str(exclude_session_id):
                continue
            count += len(bucket)
        return int(count)

    def sample(
        self,
        *,
        label_value: int,
        exclude_session_id: str,
        rng: random.Random,
    ) -> CachedRegionDonor | None:
        candidates: list[CachedRegionDonor] = []
        for (candidate_label, session_id), bucket in self._items.items():
            if int(candidate_label) != int(label_value) or str(session_id) == str(exclude_session_id):
                continue
            candidates.extend(list(bucket))
        if not candidates:
            return None
        return candidates[int(rng.randrange(len(candidates)))]


class RegionRateTargetBuilder:
    def __init__(
        self,
        *,
        canonical_regions: tuple[str, ...],
        donor_cache: RegionRateDonorCache,
        min_shared_regions: int = 1,
        exclude_final_prediction_patch: bool = True,
        rng: random.Random | None = None,
    ) -> None:
        self.canonical_regions = tuple(str(region) for region in canonical_regions)
        self.region_to_index = {region: index for index, region in enumerate(self.canonical_regions)}
        self.donor_cache = donor_cache
        self.min_shared_regions = int(min_shared_regions)
        self.exclude_final_prediction_patch = bool(exclude_final_prediction_patch)
        self.rng = rng or random.Random(0)

    def _compute_region_rates(self, batch: PopulationWindowBatch) -> tuple[torch.Tensor, torch.Tensor]:
        patch_counts = batch.patch_counts.detach().cpu().to(dtype=torch.float32)
        unit_mask = batch.unit_mask.detach().cpu().to(dtype=torch.bool)
        batch_size = int(batch.batch_size)
        num_regions = len(self.canonical_regions)
        num_patches = int(batch.num_patches)
        region_rates = torch.zeros((batch_size, num_regions, num_patches), dtype=torch.float32)
        region_mask = torch.zeros((batch_size, num_regions), dtype=torch.bool)
        region_counts = torch.zeros((batch_size, num_regions), dtype=torch.float32)
        bin_width_s = float(batch.bin_width_s)

        for batch_index in range(batch_size):
            unit_regions = batch.provenance.unit_regions[batch_index]
            for unit_index, region in enumerate(unit_regions):
                if not bool(unit_mask[batch_index, unit_index].item()):
                    continue
                region_index = self.region_to_index.get(str(region))
                if region_index is None:
                    continue
                unit_patch_rate = patch_counts[batch_index, unit_index].mean(dim=-1) / max(bin_width_s, 1.0e-8)
                region_rates[batch_index, region_index] += unit_patch_rate
                region_counts[batch_index, region_index] += 1.0
                region_mask[batch_index, region_index] = True

        denom = region_counts.unsqueeze(-1).clamp_min(1.0)
        region_rates = region_rates / denom
        return region_rates, region_mask

    def __call__(self, batch: PopulationWindowBatch, *, labels: torch.Tensor, aug_prob: float) -> RegionRateTargets:
        region_rates, region_mask = self._compute_region_rates(batch)
        future_region_rates = torch.zeros_like(region_rates)
        future_region_rates[:, :, :-1] = region_rates[:, :, 1:]

        valid_patch_mask = region_mask.unsqueeze(-1).expand(-1, -1, region_rates.shape[-1]).clone()
        if self.exclude_final_prediction_patch and valid_patch_mask.shape[-1] > 0:
            valid_patch_mask[:, :, -1] = False

        labels = labels.detach().cpu()

        augmented_future = future_region_rates.clone()
        effective_region_mask = region_mask.clone()
        is_augmented = torch.zeros((batch.batch_size,), dtype=torch.bool)
        donor_available = torch.zeros((batch.batch_size,), dtype=torch.bool)
        shared_region_counts = torch.zeros((batch.batch_size,), dtype=torch.long)

        for batch_index in range(batch.batch_size):
            session_id = str(batch.provenance.session_ids[batch_index])
            label_value = int(float(labels[batch_index].item()) > 0.5)
            donor_count = self.donor_cache.available_donor_count(
                label_value=label_value,
                exclude_session_id=session_id,
            )
            donor_available[batch_index] = donor_count > 0
            if donor_count <= 0:
                continue
            if float(aug_prob) <= 0.0 or self.rng.random() > float(aug_prob):
                continue
            donor = self.donor_cache.sample(
                label_value=label_value,
                exclude_session_id=session_id,
                rng=self.rng,
            )
            if donor is None:
                continue
            shared_mask = region_mask[batch_index] & donor.region_mask
            shared_count = int(shared_mask.sum().item())
            if shared_count < self.min_shared_regions:
                continue
            augmented_future[batch_index, shared_mask] = donor.region_rate_future[shared_mask]
            effective_region_mask[batch_index] = shared_mask
            valid_patch_mask[batch_index] = shared_mask.unsqueeze(-1).expand_as(valid_patch_mask[batch_index])
            if self.exclude_final_prediction_patch and valid_patch_mask.shape[-1] > 0:
                valid_patch_mask[batch_index, :, -1] = False
            is_augmented[batch_index] = True
            shared_region_counts[batch_index] = shared_count

        for batch_index in range(batch.batch_size):
            self.donor_cache.add(
                label_value=int(float(labels[batch_index].item()) > 0.5),
                session_id=str(batch.provenance.session_ids[batch_index]),
                region_rate_future=future_region_rates[batch_index],
                region_mask=region_mask[batch_index],
            )

        augmented_count = int(is_augmented.sum().item())
        shared_region_mean = (
            float(shared_region_counts[is_augmented].to(dtype=torch.float32).mean().item())
            if augmented_count > 0
            else 0.0
        )
        diagnostics = {
            "cross_session_aug_fraction": float(is_augmented.to(dtype=torch.float32).mean().item()),
            "cross_session_donor_available_fraction": float(
                donor_available.to(dtype=torch.float32).mean().item()
            ),
            "cross_session_shared_region_mean": shared_region_mean,
            "cross_session_requested_aug_prob": float(aug_prob),
        }
        return RegionRateTargets(
            region_rate_future=augmented_future,
            region_mask=effective_region_mask,
            valid_patch_mask=valid_patch_mask,
            is_augmented=is_augmented,
            donor_available=donor_available,
            shared_region_counts=shared_region_counts,
            diagnostics=diagnostics,
        )
