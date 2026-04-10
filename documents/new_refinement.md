This is a substantive implementation task. Let me synthesize the full picture precisely, grounding it in the existing codebase structure.

---

## Scientific Motivation

The core problem diagnosed across all experimental runs is that the predictive objective encodes session-specific covariance as the dominant geometric signal in the latent space. It does this because session-level statistics — baseline firing rates, oscillatory regime, arousal state — are the highest-variance and most temporally predictable component of population spiking data. The encoder learns to represent them because doing so minimizes the predictive loss, even though they're nuisance variables for cross-session motif discovery.

The standard predictive objective gives the encoder no pressure to disentangle session-identity variance from task-relevant variance. Whitening partially corrects for this post-hoc, but it's a second-order statistical correction applied after the representations are already learned. What you want instead is a training signal that explicitly penalizes the encoder for relying on session-specific statistics to minimize the loss.

Task-aligned cross-session augmentation does this by replacing the future patch target for some fraction of training windows with a target from a different session, matched on task label class. When the future target comes from a different session, any prediction strategy that relies on session-identity features — baseline firing rate, mean population activation level — will produce a high loss, because those features are wrong for the target session. The encoder is therefore pressured to predict features that generalize across sessions, which are precisely the task-relevant dynamic patterns you want it to encode.

---

## The Unit Mismatch Problem and Its Solution

The immediate obstacle is that `patch_counts` has shape `(batch, units, patches, bins)` where the `units` dimension is session-specific. Session A may have 87 units, session B may have 143 units. You cannot directly substitute session B's `future_counts` tensor into session A's target because the unit dimension is incompatible.

There are three approaches to this. The first is unit identity matching — only swap units that are anatomically matched across sessions by region and approximate depth. This is complex to implement and fragile because unit identity across sessions is not well-defined even for the same animal. The second is operating on raw population counts through a shared region-indexed space. The third, and most scientifically principled, is to augment at the **region-pooled population rate level** rather than the unit level.

Region-pooled population rates are session-agnostic by construction: you compute the mean firing rate across all units in each brain region for each temporal patch, producing a fixed-size vector indexed by region rather than by unit. This vector can be directly compared and swapped across sessions regardless of unit count. It also happens to be the most interpretable cross-session signal — the question "does VISp fire more or less in the next patch during a change trial" is meaningful across animals, while "does unit 47 fire more in the next patch" is not.

The implementation adds a **region-rate predictive head** as an auxiliary objective alongside the existing unit-level predictive head. This auxiliary head takes the mean-pooled context tokens and predicts a region-rate vector for the next patch. During augmented training steps, the target for this head comes from a different session's future window, matched on label class. The existing unit-level predictive objective is unchanged.

---

## Architecture Changes

### New module: `predictive_circuit_coding/objectives/region_targets.py`

This module is responsible for computing the region-pooled future rate vector from a batch, and building the cross-session augmented version.

```python
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import torch

from predictive_circuit_coding.training.contracts import PopulationWindowBatch


@dataclass(frozen=True)
class RegionRateTargets:
    # shape: (batch, num_canonical_regions, patches)
    # target is the mean firing rate per region per future patch
    region_rate_future: torch.Tensor
    # which batch items have cross-session augmented targets
    is_augmented: torch.Tensor  # (batch,) bool
    # valid mask: (batch, num_canonical_regions) — False where region absent
    region_mask: torch.Tensor


class RegionRateTargetBuilder:
    """
    Builds region-pooled future patch firing rate targets.

    For each batch item, computes mean firing rate across units per
    canonical region per future patch. When cross-session augmentation
    is enabled, swaps these targets across same-label batch items from
    different sessions with probability aug_prob.
    """

    def __init__(
        self,
        *,
        canonical_regions: tuple[str, ...],
        aug_prob: float = 0.0,
        aug_label_key: str = "stimulus_change",
        rng: random.Random | None = None,
    ):
        self.canonical_regions = canonical_regions
        self.region_to_index: dict[str, int] = {
            r: i for i, r in enumerate(canonical_regions)
        }
        self.aug_prob = aug_prob
        self.aug_label_key = aug_label_key
        self.rng = rng or random.Random(0)

    def _compute_region_rates(
        self, batch: PopulationWindowBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          region_rates: (batch, num_regions, num_patches)
          region_mask:  (batch, num_regions) bool
        """
        B = batch.batch_size
        R = len(self.canonical_regions)
        P = batch.num_patches
        region_rates = torch.zeros(B, R, P, dtype=torch.float32)
        region_mask = torch.zeros(B, R, dtype=torch.bool)
        region_counts = torch.zeros(B, R, dtype=torch.float32)

        for b in range(B):
            unit_regions = batch.provenance.unit_regions[b]
            unit_mask = batch.unit_mask[b]  # (units,)
            # patch_counts shape: (units, patches, bins)
            patch_counts = batch.patch_counts[b]
            for u_idx, region in enumerate(unit_regions):
                if not unit_mask[u_idx]:
                    continue
                r_idx = self.region_to_index.get(region)
                if r_idx is None:
                    continue
                # mean firing rate per patch: mean over bins, multiply by bin rate
                unit_rate = patch_counts[u_idx].mean(dim=-1)  # (patches,)
                region_rates[b, r_idx] += unit_rate
                region_counts[b, r_idx] += 1.0
                region_mask[b, r_idx] = True

        denom = region_counts.unsqueeze(-1).clamp_min(1.0)
        region_rates = region_rates / denom
        return region_rates, region_mask

    def _extract_label(self, batch: PopulationWindowBatch, b: int) -> int | None:
        """Extract binary label for item b from event_annotations."""
        ann = batch.provenance.event_annotations[b]
        # support dotted key paths e.g. "stimulus_presentations.is_change"
        parts = self.aug_label_key.replace(".", "/").split("/")
        node = ann
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        if isinstance(node, (list, tuple)) and len(node) > 0:
            return int(bool(node[-1]))
        if isinstance(node, (bool, int, float)):
            return int(bool(node))
        return None

    def __call__(
        self, batch: PopulationWindowBatch
    ) -> RegionRateTargets:
        # compute future region rates: shift by one patch
        region_rates, region_mask = self._compute_region_rates(batch)
        # future = rates shifted forward by one patch (same logic as CountTargetBuilder)
        future_region_rates = torch.zeros_like(region_rates)
        future_region_rates[:, :, :-1] = region_rates[:, :, 1:]

        B = batch.batch_size
        is_augmented = torch.zeros(B, dtype=torch.bool)

        if self.aug_prob <= 0.0:
            return RegionRateTargets(
                region_rate_future=future_region_rates,
                is_augmented=is_augmented,
                region_mask=region_mask,
            )

        # group batch items by label class and session
        label_map: dict[int, list[int]] = {0: [], 1: []}
        labels: list[int | None] = []
        for b in range(B):
            label = self._extract_label(batch, b)
            labels.append(label)
            if label is not None:
                label_map[label].append(b)

        augmented_rates = future_region_rates.clone()

        for b in range(B):
            if self.rng.random() > self.aug_prob:
                continue
            label = labels[b]
            if label is None:
                continue
            session = batch.provenance.session_ids[b]
            # find candidates from same label, different session
            candidates = [
                c for c in label_map[label]
                if c != b and batch.provenance.session_ids[c] != session
            ]
            if not candidates:
                continue
            donor = self.rng.choice(candidates)
            # swap only regions present in BOTH items
            shared_mask = region_mask[b] & region_mask[donor]
            augmented_rates[b, shared_mask] = future_region_rates[donor, shared_mask]
            is_augmented[b] = True

        return RegionRateTargets(
            region_rate_future=augmented_rates,
            is_augmented=is_augmented,
            region_mask=region_mask,
        )
```

---

### New module: `predictive_circuit_coding/models/heads.py` — add `RegionRatePredictiveHead`

This adds a new head alongside the existing `PredictiveHead` and `ReconstructionHead`. It takes the mean-pooled token tensor across the unit dimension and projects to the canonical region space.

```python
class RegionRatePredictiveHead(nn.Module):
    """
    Predicts region-pooled future firing rates from mean-pooled context tokens.

    Input:  mean-pooled tokens, shape (batch, patches, d_model)
    Output: predicted region rates, shape (batch, num_regions, patches)
    """
    def __init__(self, *, d_model: int, num_regions: int):
        super().__init__()
        self.proj = nn.Linear(d_model, num_regions)

    def forward(self, tokens: torch.Tensor, unit_mask: torch.Tensor) -> torch.Tensor:
        # tokens: (batch, units, patches, d_model)
        # unit_mask: (batch, units)
        mask_f = unit_mask.float().unsqueeze(-1).unsqueeze(-1)  # (B, U, 1, 1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)               # (B, 1, 1, 1) → broadcast
        mean_tokens = (tokens * mask_f).sum(dim=1) / denom      # (B, patches, d_model)
        predicted = self.proj(mean_tokens)                       # (B, patches, num_regions)
        return predicted.permute(0, 2, 1)                        # (B, num_regions, patches)
```

---

### Changes to `predictive_circuit_coding/objectives/losses.py`

Add `CrossSessionRegionLoss` and extend `CombinedObjective` to include it.

```python
class CrossSessionRegionLoss:
    """
    MSE between predicted region rates and cross-session augmented future region rates.
    Loss is computed only over augmented items and shared region masks.
    """
    def __init__(self, *, weight: float = 1.0):
        self.weight = weight

    def evaluate(
        self,
        predicted_region_rates: torch.Tensor,   # (B, R, P)
        region_targets: RegionRateTargets,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        target = region_targets.region_rate_future   # (B, R, P)
        mask = region_targets.region_mask            # (B, R)
        aug_mask = region_targets.is_augmented       # (B,)

        if not aug_mask.any():
            zero = predicted_region_rates.sum() * 0.0
            return zero, {
                "cross_session_region_loss": 0.0,
                "cross_session_aug_fraction": 0.0,
            }

        # only compute loss on augmented items
        pred_aug = predicted_region_rates[aug_mask]    # (n_aug, R, P)
        target_aug = target[aug_mask]                  # (n_aug, R, P)
        mask_aug = mask[aug_mask]                      # (n_aug, R)

        expanded_mask = mask_aug.unsqueeze(-1).float() # (n_aug, R, 1)
        sq_err = ((pred_aug - target_aug) ** 2) * expanded_mask
        denom = expanded_mask.sum().clamp_min(1.0) * pred_aug.shape[-1]
        loss = sq_err.sum() / denom

        aug_fraction = float(aug_mask.float().mean().item())
        return loss * self.weight, {
            "cross_session_region_loss": float(loss.detach().item()),
            "cross_session_aug_fraction": aug_fraction,
        }
```

`CombinedObjective` then needs to conditionally invoke this loss if the region head and region target builder are present. The cleanest way is to make `CombinedObjective` accept optional components so the existing interface doesn't break for runs that don't use augmentation.

---

### Changes to `predictive_circuit_coding/training/config.py`

Add a new `CrossSessionAugConfig` dataclass and wire it into `ObjectiveConfig`:

```python
@dataclass(frozen=True)
class CrossSessionAugConfig:
    enabled: bool = False
    aug_prob: float = 0.5
    aug_label_key: str = "stimulus_presentations.is_change"
    region_loss_weight: float = 1.0
    # canonical regions: must match prepared session catalog
    # if empty, derived at runtime from training split regions
    canonical_regions: tuple[str, ...] = ()
```

Add to `ExperimentConfig` and expose via YAML under `objective.cross_session_aug`. Default off so no existing configs break.

---

### Changes to `predictive_circuit_coding/training/loop.py`

The training loop needs three additions when augmentation is enabled:

**Before the epoch loop:** build the `RegionRateTargetBuilder` from the canonical regions list (which can be derived from the training split session catalog if not provided in config), and build the `RegionRatePredictiveHead` and `CrossSessionRegionLoss`.

**Inside the step loop:** after collation but before the forward pass, call `region_target_builder(batch)` to produce `RegionRateTargets`. After the forward pass, call the region head on `output.tokens` and `batch.unit_mask`, then compute the cross-session loss and add it to the total loss.

**Metric logging:** emit `cross_session_region_loss` and `cross_session_aug_fraction` in the step metrics so you can monitor augmentation coverage during training.

The key structural point is that `run_training_step` in `step.py` should not be modified — the cross-session logic is higher-level and belongs in the loop, not in the step function, because it requires inter-batch state that the step function doesn't have access to.

---

### Changes to `predictive_circuit_coding/models/encoder.py`

`PredictiveCircuitModel.forward` currently returns `ModelForwardOutput` containing `tokens`, `predictive_outputs`, and `reconstruction_outputs`. The region head output should be added as an optional field:

```python
@dataclass(frozen=True)
class ModelForwardOutput:
    tokens: torch.Tensor
    predictive_outputs: torch.Tensor
    reconstruction_outputs: torch.Tensor
    region_rate_outputs: torch.Tensor | None = None  # (B, R, P) when head present
```

The region head is not part of the core encoder — it's an auxiliary training component. It should be instantiated in the training loop and called separately from the encoder forward pass, using the returned `tokens` tensor. This keeps the frozen encoder clean and ensures that when you freeze the encoder for discovery, you're not carrying the region head with you.

---

## Config Surface

New fields in `predictive_circuit_coding_base.yaml` under the `objective` block:

```yaml
objective:
  # existing fields unchanged
  predictive_target_type: delta
  continuation_baseline_type: previous_patch
  predictive_loss: mse
  reconstruction_loss: mse
  reconstruction_weight: 0.2
  exclude_final_prediction_patch: true

  # new: cross-session augmentation
  cross_session_aug:
    enabled: false
    aug_prob: 0.5
    aug_label_key: "stimulus_presentations.is_change"
    region_loss_weight: 1.0
    canonical_regions: []   # empty = derive from training split at runtime
```

You'd run the augmented training experiment with a separate config file (e.g. `predictive_circuit_coding_cross_session_aug.yaml`) that inherits from the base config and sets `enabled: true`, rather than modifying the base config.

---

## Testing Strategy

Testing needs to cover four distinct concerns: correctness of the target builder, correctness of the loss, that augmentation doesn't break the training step when disabled, and that it produces the intended geometric effect when enabled.

### Unit tests: `tests/test_cross_session_aug.py`

**Test 1: RegionRateTargetBuilder with aug_prob=0 returns same-session targets**
Construct a batch with two items from different sessions and different labels. With `aug_prob=0`, verify that `is_augmented` is all False and `region_rate_future` matches what you'd compute manually from `batch.patch_counts`.

**Test 2: RegionRateTargetBuilder with aug_prob=1.0 swaps across sessions**
Construct a batch where items 0 and 1 are same label, different sessions, and items 2 and 3 are same label, different sessions. With `aug_prob=1.0` and a seeded RNG, verify that `is_augmented` is all True and that the region rates for item 0 match item 1's future rates (for shared regions) and vice versa.

**Test 3: Augmentation does not swap same-session items**
Construct a batch where all items are from the same session. With `aug_prob=1.0`, verify `is_augmented` is all False — no donors available from different sessions.

**Test 4: Augmentation does not swap across different label classes**
Construct a batch with two positive items from different sessions and two negative items from different sessions. Verify that positive items only receive targets from positive donors and negative items from negative donors.

**Test 5: CrossSessionRegionLoss is zero when no augmented items**
Pass a `RegionRateTargets` where `is_augmented` is all False. Verify the loss is exactly zero.

**Test 6: CrossSessionRegionLoss is finite and positive when augmented items present**
Pass a `RegionRateTargets` with some augmented items and mismatched predictions. Verify loss is finite, positive, and proportional to prediction error.

**Test 7: Region head output shape**
Construct a batch with known `(B, U, P, D)` token shape and `unit_mask`. Pass through `RegionRatePredictiveHead`. Verify output shape is `(B, R, P)`.

**Test 8: Region head zeros out masked units in mean pooling**
Construct tokens where masked-out units have large values. Verify that the mean-pooled result is not affected by those units.

### Integration test: `tests/test_cross_session_aug_workflow.py`

**Test: training step with augmentation enabled completes without error and emits expected metrics**
Use the existing `_create_prepared_workspace` fixture from `test_stage5_stage6_workflow.py`. Write an experiment config with `cross_session_aug.enabled: true`, `aug_prob: 0.5`. Run two epochs. Verify:
- Training completes without raising
- `cross_session_aug_fraction` appears in step metrics
- Final checkpoint contains model state normally
- Evaluation on the validation split produces finite metrics

**Test: augmented training does not degrade frozen evaluation compared to standard**
This is a statistical rather than exact test. Run both configs on the same synthetic workspace. Verify that `predictive_improvement` is positive in both cases. You don't expect the augmented run to be worse on unit-level prediction — the auxiliary loss is additive, not a replacement.

### Scientific validation test (run once, not part of CI)

**Geometry check:** train with augmentation enabled, freeze the encoder, compute session neighbor enrichment and label neighbor enrichment on the discovery split (the same geometry diagnostic you've already implemented). Compare against the non-augmented checkpoint. The expected result is that session enrichment decreases and label enrichment increases. If session enrichment does not decrease, augmentation is not having its intended effect and the aug_prob or region_loss_weight need tuning.

**Ablation check:** run the full benchmark (encoder_raw, encoder_whitened, encoder_aug_raw, encoder_aug_whitened) and compare held-out motif ROC-AUC across all four arms. The scientific hypothesis is that `encoder_aug_raw` should outperform `encoder_raw` on motif similarity, possibly without needing whitening.

---

## Implementation Order

Do these in sequence and verify each step before proceeding:

First, implement and unit-test `RegionRateTargetBuilder` with `aug_prob=0`. This establishes the region-rate computation is correct before any augmentation logic is involved.

Second, implement and unit-test the augmentation swapping logic in `RegionRateTargetBuilder`. Specifically test the case where no cross-session donors exist before testing the happy path.

Third, implement `RegionRatePredictiveHead` and unit-test its shape and masking behavior.

Fourth, implement `CrossSessionRegionLoss` and unit-test zero-aug and nonzero-aug cases.

Fifth, wire everything into the training loop behind the config flag and run the integration test.

Sixth, run the geometry check on a short training run to verify the augmentation is actually reducing session enrichment before investing in a full retraining run.

The geometry check in step six is the most important gate. If session enrichment doesn't decrease after a short augmented training run, something is wrong with the mechanism and you should debug before committing to a full 45-epoch training run.