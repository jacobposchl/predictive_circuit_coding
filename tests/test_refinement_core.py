from __future__ import annotations

from pathlib import Path

import torch

from predictive_circuit_coding.benchmarks.run import default_motif_arm_specs
from predictive_circuit_coding.models import PredictiveCircuitModel
from predictive_circuit_coding.objectives import CombinedObjective
from predictive_circuit_coding.tokenization import CountNormalizationStats, apply_count_normalization
from predictive_circuit_coding.training import ModelConfig, ObjectiveConfig, load_experiment_config

from test_model_core import _build_batch


def test_refined_config_family_loads() -> None:
    config_dir = Path("configs/pcc")
    names = [
        "predictive_circuit_coding_refined_debug.yaml",
        "predictive_circuit_coding_refined_full.yaml",
        "predictive_circuit_coding_refined_recon000_debug.yaml",
        "predictive_circuit_coding_refined_l2off_debug.yaml",
        "predictive_circuit_coding_refined_countnorm_debug.yaml",
        "predictive_circuit_coding_refined_cls_debug.yaml",
    ]
    variants = [load_experiment_config(config_dir / name).experiment.variant_name for name in names]

    assert variants == [
        "refined_core",
        "refined_core",
        "refined_recon000",
        "refined_l2off",
        "refined_countnorm",
        "refined_cls",
    ]


def test_refinement_arms_are_trained_encoder_only() -> None:
    arms = default_motif_arm_specs()

    assert [arm.name for arm in arms] == [
        "encoder_raw",
        "encoder_token_normalized",
        "encoder_probe_weighted",
        "encoder_aligned_oracle",
    ]
    assert [arm.claim_safe for arm in arms] == [True, True, True, False]


def test_window_zscore_reconstruction_loss_is_finite() -> None:
    data_config, batch = _build_batch()
    model = PredictiveCircuitModel(
        model_config=ModelConfig(
            d_model=64,
            num_heads=8,
            temporal_layers=1,
            spatial_layers=1,
            dropout=0.1,
            mlp_ratio=2.0,
            l2_normalize_tokens=True,
            norm_eps=1.0e-5,
        ),
        patch_bins=data_config.patch_bins,
        num_patches=data_config.patches_per_window,
    )
    objective = CombinedObjective(
        ObjectiveConfig(
            predictive_target_type="delta",
            continuation_baseline_type="previous_patch",
            predictive_loss="mse",
            reconstruction_loss="mse",
            reconstruction_weight=0.05,
            reconstruction_target_mode="window_zscore",
            exclude_final_prediction_patch=True,
        )
    )

    output = objective(batch, model(batch))

    assert torch.isfinite(output.loss)
    assert output.losses["reconstruction_loss"] >= 0.0


def test_count_normalization_applies_log_train_zscore() -> None:
    counts = torch.tensor([[0.0, 1.0, 3.0]], dtype=torch.float32)
    stats = CountNormalizationStats(mode="log1p_train_zscore", mean=0.5, std=2.0, count=3)

    normalized = apply_count_normalization(counts, stats)

    expected = (torch.log1p(counts) - 0.5) / 2.0
    assert torch.allclose(normalized, expected)


def test_cls_population_tokens_preserve_unit_outputs() -> None:
    data_config, batch = _build_batch()
    model = PredictiveCircuitModel(
        model_config=ModelConfig(
            d_model=64,
            num_heads=8,
            temporal_layers=1,
            spatial_layers=1,
            dropout=0.1,
            mlp_ratio=2.0,
            l2_normalize_tokens=True,
            norm_eps=1.0e-5,
            population_token_mode="per_patch_cls",
        ),
        patch_bins=data_config.patch_bins,
        num_patches=data_config.patches_per_window,
    )

    output = model(batch)

    assert output.tokens.shape == (2, 3, 10, 64)
    assert output.predictive_outputs.shape == (2, 3, 10, 50)
    assert output.summary_tokens is not None
    assert output.summary_tokens.shape == (2, 10, 64)
