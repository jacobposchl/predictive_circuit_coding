from __future__ import annotations

import numpy as np
import torch

from predictive_circuit_coding.models import PredictiveCircuitModel
from predictive_circuit_coding.objectives import CombinedObjective, CountTargetBuilder
from predictive_circuit_coding.tokenization import build_population_window_collator
from predictive_circuit_coding.training import (
    DataRuntimeConfig,
    ExperimentConfig,
    ModelConfig,
    ObjectiveConfig,
    OptimizationConfig,
    ArtifactConfig,
    build_model_from_config,
    build_objective_from_config,
    build_tokenizer_from_config,
    run_training_step,
)


def _build_window_sample(*, session_id: str, subject_id: str, window_start_s: float, unit_ids: list[str], regions: list[str], depths: list[float], spikes: dict[int, list[float]]):
    from temporaldata import ArrayDict, Data, Interval, IrregularTimeSeries

    domain = Interval(
        start=np.asarray([window_start_s], dtype=np.float64),
        end=np.asarray([window_start_s + 10.0], dtype=np.float64),
    )
    timestamps: list[float] = []
    unit_index: list[int] = []
    for index, values in spikes.items():
        timestamps.extend(values)
        unit_index.extend([index] * len(values))
    order = np.argsort(np.asarray(timestamps, dtype=np.float64))
    timestamps_np = np.asarray(timestamps, dtype=np.float64)[order]
    unit_index_np = np.asarray(unit_index, dtype=np.int64)[order]
    return Data(
        brainset=Data(id="allen_visual_behavior_neuropixels"),
        session=Data(id=session_id),
        subject=Data(id=subject_id),
        units=ArrayDict(
            id=np.asarray(unit_ids, dtype=object),
            brain_region=np.asarray(regions, dtype=object),
            probe_depth_um=np.asarray(depths, dtype=np.float32),
        ),
        spikes=IrregularTimeSeries(
            timestamps=timestamps_np,
            unit_index=unit_index_np,
            domain=domain,
        ),
        domain=domain,
    )


def _build_batch():
    config = DataRuntimeConfig(
        bin_width_ms=20.0,
        context_bins=500,
        patch_bins=50,
        min_unit_spikes=0,
        max_units=None,
        padding_strategy="mask",
        include_trials=True,
        include_stimulus_presentations=True,
        include_optotagging=True,
    )
    collator = build_population_window_collator(config)
    sample_a = _build_window_sample(
        session_id="session_a",
        subject_id="mouse_a",
        window_start_s=0.0,
        unit_ids=["u0", "u1"],
        regions=["VISp", "LP"],
        depths=[100.0, 200.0],
        spikes={
            0: [0.005, 0.025, 1.005, 2.005],
            1: [0.505, 1.505],
        },
    )
    sample_b = _build_window_sample(
        session_id="session_b",
        subject_id="mouse_b",
        window_start_s=10.0,
        unit_ids=["u2", "u3", "u4"],
        regions=["VISl", "VISpm", "LP"],
        depths=[150.0, 250.0, 350.0],
        spikes={
            0: [10.005, 10.405],
            1: [11.005, 11.405],
            2: [12.005, 12.405, 12.805],
        },
    )
    return config, collator([sample_a, sample_b])


def test_model_forward_shapes_and_masks():
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

    output = model(batch)

    assert output.tokens.shape == (2, 3, 10, 64)
    assert output.predictive_outputs.shape == (2, 3, 10, 50)
    assert output.reconstruction_outputs.shape == (2, 3, 10, 50)
    assert torch.allclose(output.tokens[0, 2], torch.zeros_like(output.tokens[0, 2]))


def test_target_builder_and_combined_objective_are_finite():
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
    objective_config = ObjectiveConfig(
        predictive_target_type="delta",
        continuation_baseline_type="previous_patch",
        predictive_loss="mse",
        reconstruction_loss="mse",
        reconstruction_weight=0.2,
        exclude_final_prediction_patch=True,
    )
    target_builder = CountTargetBuilder(objective_config)
    targets = target_builder(batch)
    forward_output = model(batch)
    objective = CombinedObjective(objective_config)
    result = objective(batch, forward_output)

    assert targets.target.shape == (2, 3, 10, 50)
    assert targets.valid_mask.shape == (2, 3, 10)
    assert torch.isfinite(result.loss)
    assert "predictive_improvement" in result.metrics
    assert "reconstruction_loss" in result.metrics


def test_training_step_runs_end_to_end_from_factories(tmp_path):
    data_config, batch = _build_batch()
    experiment = ExperimentConfig(
        dataset_id="allen_visual_behavior_neuropixels",
        split_name="train",
        seed=7,
        data_runtime=data_config,
        model=ModelConfig(
            d_model=64,
            num_heads=8,
            temporal_layers=1,
            spatial_layers=1,
            dropout=0.1,
            mlp_ratio=2.0,
            l2_normalize_tokens=True,
            norm_eps=1.0e-5,
        ),
        objective=ObjectiveConfig(
            predictive_target_type="delta",
            continuation_baseline_type="previous_patch",
            predictive_loss="mse",
            reconstruction_loss="mse",
            reconstruction_weight=0.2,
            exclude_final_prediction_patch=True,
        ),
        optimization=OptimizationConfig(
            learning_rate=1.0e-4,
            weight_decay=1.0e-4,
            grad_clip_norm=1.0,
            batch_size=4,
        ),
        artifacts=ArtifactConfig(
            checkpoint_dir=tmp_path / "checkpoints",
            summary_path=tmp_path / "summary.json",
            checkpoint_prefix="pcc",
            save_config_snapshot=True,
        ),
        config_path=tmp_path / "experiment.yaml",
    )

    tokenizer = build_tokenizer_from_config(experiment)
    model = build_model_from_config(experiment)
    objective = build_objective_from_config(experiment)
    step_output = run_training_step(model, objective, batch)

    assert tokenizer.config.context_bins == 500
    assert step_output.batch_size == 2
    assert step_output.token_count == int(batch.patch_mask.sum().item())
    assert torch.isfinite(step_output.loss)
    assert "predictive_raw_mse" in step_output.metrics
