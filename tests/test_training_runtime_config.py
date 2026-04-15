from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from predictive_circuit_coding.training import (
    CheckpointMetadata,
    TrainingSummary,
    load_experiment_config,
    write_checkpoint_metadata,
    write_training_summary,
)
from predictive_circuit_coding.evaluation.metrics import aggregate_metric_dicts
from predictive_circuit_coding.training.runtime import resolve_device


def _deep_merge(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _base_experiment_payload() -> dict[str, object]:
    return {
        "dataset_id": "allen_visual_behavior_neuropixels",
        "split_name": "train",
        "seed": 11,
        "data_runtime": {
            "bin_width_ms": 20.0,
            "context_bins": 500,
            "patch_bins": 50,
            "min_unit_spikes": 1,
            "max_units": 32,
            "padding_strategy": "mask",
            "include_trials": True,
            "include_stimulus_presentations": True,
            "include_optotagging": False,
        },
        "model": {
            "d_model": 128,
            "num_heads": 8,
            "temporal_layers": 2,
            "spatial_layers": 2,
            "dropout": 0.1,
            "mlp_ratio": 4.0,
            "l2_normalize_tokens": True,
            "norm_eps": 1.0e-5,
        },
        "objective": {
            "predictive_target_type": "delta",
            "continuation_baseline_type": "previous_patch",
            "predictive_loss": "mse",
            "reconstruction_loss": "mse",
            "reconstruction_weight": 0.25,
            "exclude_final_prediction_patch": True,
        },
        "optimization": {
            "learning_rate": 1.0e-4,
            "weight_decay": 1.0e-4,
            "grad_clip_norm": 1.0,
            "batch_size": 4,
        },
        "artifacts": {
            "checkpoint_dir": "../../artifacts/checkpoints",
            "summary_path": "../../artifacts/summary.json",
            "checkpoint_prefix": "pcc",
            "save_config_snapshot": True,
        },
    }


def _write_experiment_payload(tmp_path: Path, override: dict[str, object] | None = None) -> Path:
    config_path = tmp_path / "configs" / "pcc" / "experiment.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _base_experiment_payload()
    if override is not None:
        payload = _deep_merge(payload, override)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def test_load_experiment_config_parses_stage_3_and_4_defaults(tmp_path: Path):
    config_path = tmp_path / "configs" / "pcc" / "experiment.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "\n".join(
            [
                "dataset_id: allen_visual_behavior_neuropixels",
                "split_name: train",
                "seed: 11",
                "data_runtime:",
                "  bin_width_ms: 20.0",
                "  context_bins: 500",
                "  patch_bins: 50",
                "  min_unit_spikes: 1",
                "  max_units: 32",
                "  padding_strategy: mask",
                "  include_trials: true",
                "  include_stimulus_presentations: true",
                "  include_optotagging: false",
                "model:",
                "  d_model: 128",
                "  num_heads: 8",
                "  temporal_layers: 2",
                "  spatial_layers: 2",
                "  dropout: 0.1",
                "  mlp_ratio: 4.0",
                "  l2_normalize_tokens: true",
                "  norm_eps: 1.0e-5",
                "objective:",
                "  predictive_target_type: delta",
                "  continuation_baseline_type: previous_patch",
                "  predictive_loss: mse",
                "  reconstruction_loss: mse",
                "  reconstruction_weight: 0.25",
                "  exclude_final_prediction_patch: true",
                "optimization:",
                "  learning_rate: 1.0e-4",
                "  weight_decay: 1.0e-4",
                "  grad_clip_norm: 1.0",
                "  batch_size: 4",
                "artifacts:",
                "  checkpoint_dir: ../../artifacts/checkpoints",
                "  summary_path: ../../artifacts/summary.json",
                "  checkpoint_prefix: pcc",
                "  save_config_snapshot: true",
            ]
        ),
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert config.dataset_id == "allen_visual_behavior_neuropixels"
    assert config.data_runtime.bin_width_s == 0.02
    assert config.data_runtime.patches_per_window == 10
    assert config.model.d_model == 128
    assert config.objective.reconstruction_weight == 0.25
    assert config.optimization.batch_size == 4
    assert config.artifacts.checkpoint_dir == tmp_path / "artifacts" / "checkpoints"


def test_checkpoint_and_training_summary_serialize_to_json(tmp_path: Path):
    metadata = CheckpointMetadata(
        dataset_id="allen_visual_behavior_neuropixels",
        split_name="train",
        seed=7,
        config_snapshot={"dataset_id": "allen_visual_behavior_neuropixels", "seed": 7},
        model_hparams={"d_model": 256, "temporal_layers": 2, "spatial_layers": 2},
        continuation_baseline_type="previous_patch",
    )
    summary = TrainingSummary(
        dataset_id="allen_visual_behavior_neuropixels",
        split_name="valid",
        epoch=3,
        best_epoch=2,
        metrics={"predictive_raw_mse": 0.12},
        losses={"total_loss": 0.2},
        checkpoint_path="artifacts/checkpoints/pcc_epoch_003.pt",
    )

    metadata_path = write_checkpoint_metadata(metadata, tmp_path / "metadata.json")
    summary_path = write_training_summary(summary, tmp_path / "summary.json")

    loaded_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert loaded_metadata["continuation_baseline_type"] == "previous_patch"
    assert loaded_metadata["model_hparams"]["d_model"] == 256
    assert loaded_summary["best_epoch"] == 2
    assert loaded_summary["metrics"]["predictive_raw_mse"] == 0.12


def test_load_experiment_config_accepts_generic_discovery_targets_and_sampling_fields(tmp_path: Path):
    config_path = tmp_path / "configs" / "pcc" / "experiment.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "\n".join(
            [
                "dataset_id: generic_dataset",
                "split_name: train",
                "seed: 3",
                "data_runtime:",
                "  bin_width_ms: 20.0",
                "  context_bins: 500",
                "  patch_bins: 50",
                "  min_unit_spikes: 1",
                "  max_units: 32",
                "  padding_strategy: mask",
                "  include_trials: true",
                "  include_stimulus_presentations: true",
                "  include_optotagging: false",
                "model:",
                "  d_model: 128",
                "  num_heads: 8",
                "  temporal_layers: 2",
                "  spatial_layers: 2",
                "  dropout: 0.1",
                "  mlp_ratio: 4.0",
                "  l2_normalize_tokens: true",
                "  norm_eps: 1.0e-5",
                "objective:",
                "  predictive_target_type: delta",
                "  continuation_baseline_type: previous_patch",
                "  predictive_loss: mse",
                "  reconstruction_loss: mse",
                "  reconstruction_weight: 0.25",
                "  exclude_final_prediction_patch: true",
                "optimization:",
                "  learning_rate: 1.0e-4",
                "  weight_decay: 1.0e-4",
                "  grad_clip_norm: 1.0",
                "  batch_size: 4",
                "artifacts:",
                "  checkpoint_dir: ../../artifacts/checkpoints",
                "  summary_path: ../../artifacts/summary.json",
                "  checkpoint_prefix: pcc",
                "  save_config_snapshot: true",
                "discovery:",
                "  target_label: behavior.outcome.hit",
                "  target_label_match_value: hit",
                "  sampling_strategy: label_balanced",
                "  min_positive_windows: 4",
                "  negative_to_positive_ratio: 2.0",
                "  search_max_batches: 32",
                "  candidate_session_balance_fraction: 0.3",
            ]
        ),
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert config.discovery.target_label == "behavior.outcome.hit"
    assert config.discovery.target_label_match_value == "hit"
    assert config.discovery.sampling_strategy == "label_balanced"
    assert config.discovery.min_positive_windows == 4
    assert config.discovery.negative_to_positive_ratio == 2.0
    assert config.discovery.search_max_batches == 32
    assert config.discovery.candidate_session_balance_fraction == 0.3


def test_load_experiment_config_rejects_nonzero_dataloader_workers(tmp_path: Path):
    config_path = tmp_path / "configs" / "pcc" / "experiment.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "\n".join(
            [
                "dataset_id: allen_visual_behavior_neuropixels",
                "split_name: train",
                "seed: 11",
                "data_runtime:",
                "  bin_width_ms: 20.0",
                "  context_bins: 500",
                "  patch_bins: 50",
                "  min_unit_spikes: 1",
                "  max_units: 32",
                "  padding_strategy: mask",
                "  include_trials: true",
                "  include_stimulus_presentations: true",
                "  include_optotagging: false",
                "model:",
                "  d_model: 128",
                "  num_heads: 8",
                "  temporal_layers: 2",
                "  spatial_layers: 2",
                "  dropout: 0.1",
                "  mlp_ratio: 4.0",
                "  l2_normalize_tokens: true",
                "  norm_eps: 1.0e-5",
                "objective:",
                "  predictive_target_type: delta",
                "  continuation_baseline_type: previous_patch",
                "  predictive_loss: mse",
                "  reconstruction_loss: mse",
                "  reconstruction_weight: 0.25",
                "  exclude_final_prediction_patch: true",
                "optimization:",
                "  learning_rate: 1.0e-4",
                "  weight_decay: 1.0e-4",
                "  grad_clip_norm: 1.0",
                "  batch_size: 4",
                "training:",
                "  dataloader_workers: 2",
                "artifacts:",
                "  checkpoint_dir: ../../artifacts/checkpoints",
                "  summary_path: ../../artifacts/summary.json",
                "  checkpoint_prefix: pcc",
                "  save_config_snapshot: true",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dataloader_workers"):
        load_experiment_config(config_path)


def test_load_experiment_config_rejects_singleton_clusters(tmp_path: Path):
    config_path = tmp_path / "configs" / "pcc" / "experiment.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "\n".join(
            [
                "dataset_id: allen_visual_behavior_neuropixels",
                "split_name: train",
                "seed: 11",
                "data_runtime:",
                "  bin_width_ms: 20.0",
                "  context_bins: 500",
                "  patch_bins: 50",
                "  min_unit_spikes: 1",
                "  max_units: 32",
                "  padding_strategy: mask",
                "  include_trials: true",
                "  include_stimulus_presentations: true",
                "  include_optotagging: false",
                "model:",
                "  d_model: 128",
                "  num_heads: 8",
                "  temporal_layers: 2",
                "  spatial_layers: 2",
                "  dropout: 0.1",
                "  mlp_ratio: 4.0",
                "  l2_normalize_tokens: true",
                "  norm_eps: 1.0e-5",
                "objective:",
                "  predictive_target_type: delta",
                "  continuation_baseline_type: previous_patch",
                "  predictive_loss: mse",
                "  reconstruction_loss: mse",
                "  reconstruction_weight: 0.25",
                "  exclude_final_prediction_patch: true",
                "optimization:",
                "  learning_rate: 1.0e-4",
                "  weight_decay: 1.0e-4",
                "  grad_clip_norm: 1.0",
                "  batch_size: 4",
                "artifacts:",
                "  checkpoint_dir: ../../artifacts/checkpoints",
                "  summary_path: ../../artifacts/summary.json",
                "  checkpoint_prefix: pcc",
                "  save_config_snapshot: true",
                "discovery:",
                "  min_cluster_size: 1",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="min_cluster_size"):
        load_experiment_config(config_path)


def test_load_experiment_config_rejects_invalid_candidate_session_balance_fraction(tmp_path: Path):
    config_path = tmp_path / "configs" / "pcc" / "experiment.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "\n".join(
            [
                "dataset_id: allen_visual_behavior_neuropixels",
                "split_name: train",
                "seed: 11",
                "data_runtime:",
                "  bin_width_ms: 20.0",
                "  context_bins: 500",
                "  patch_bins: 50",
                "  min_unit_spikes: 1",
                "  max_units: 32",
                "  padding_strategy: mask",
                "  include_trials: true",
                "  include_stimulus_presentations: true",
                "  include_optotagging: false",
                "model:",
                "  d_model: 128",
                "  num_heads: 8",
                "  temporal_layers: 2",
                "  spatial_layers: 2",
                "  dropout: 0.1",
                "  mlp_ratio: 4.0",
                "  l2_normalize_tokens: true",
                "  norm_eps: 1.0e-5",
                "objective:",
                "  predictive_target_type: delta",
                "  continuation_baseline_type: previous_patch",
                "  predictive_loss: mse",
                "  reconstruction_loss: mse",
                "  reconstruction_weight: 0.25",
                "  exclude_final_prediction_patch: true",
                "optimization:",
                "  learning_rate: 1.0e-4",
                "  weight_decay: 1.0e-4",
                "  grad_clip_norm: 1.0",
                "  batch_size: 4",
                "artifacts:",
                "  checkpoint_dir: ../../artifacts/checkpoints",
                "  summary_path: ../../artifacts/summary.json",
                "  checkpoint_prefix: pcc",
                "  save_config_snapshot: true",
                "discovery:",
                "  candidate_session_balance_fraction: 0.0",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="candidate_session_balance_fraction"):
        load_experiment_config(config_path)


def test_load_experiment_config_rejects_blank_target_label_match_value(tmp_path: Path):
    config_path = tmp_path / "configs" / "pcc" / "experiment.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "\n".join(
            [
                "dataset_id: allen_visual_behavior_neuropixels",
                "split_name: train",
                "seed: 11",
                "data_runtime:",
                "  bin_width_ms: 20.0",
                "  context_bins: 500",
                "  patch_bins: 50",
                "  min_unit_spikes: 1",
                "  max_units: 32",
                "  padding_strategy: mask",
                "  include_trials: true",
                "  include_stimulus_presentations: true",
                "  include_optotagging: false",
                "model:",
                "  d_model: 128",
                "  num_heads: 8",
                "  temporal_layers: 2",
                "  spatial_layers: 2",
                "  dropout: 0.1",
                "  mlp_ratio: 4.0",
                "  l2_normalize_tokens: true",
                "  norm_eps: 1.0e-5",
                "objective:",
                "  predictive_target_type: delta",
                "  continuation_baseline_type: previous_patch",
                "  predictive_loss: mse",
                "  reconstruction_loss: mse",
                "  reconstruction_weight: 0.25",
                "  exclude_final_prediction_patch: true",
                "optimization:",
                "  learning_rate: 1.0e-4",
                "  weight_decay: 1.0e-4",
                "  grad_clip_norm: 1.0",
                "  batch_size: 4",
                "artifacts:",
                "  checkpoint_dir: ../../artifacts/checkpoints",
                "  summary_path: ../../artifacts/summary.json",
                "  checkpoint_prefix: pcc",
                "  save_config_snapshot: true",
                "discovery:",
                "  target_label: stimulus_presentations.image_name",
                "  target_label_match_value: '   '",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="target_label_match_value"):
        load_experiment_config(config_path)


def test_load_experiment_config_rejects_unknown_top_level_keys(tmp_path: Path):
    config_path = _write_experiment_payload(tmp_path, {"variant_name": "legacy_field"})

    with pytest.raises(ValueError, match="Unsupported experiment config keys: variant_name"):
        load_experiment_config(config_path)


def test_load_experiment_config_rejects_unknown_nested_keys(tmp_path: Path):
    config_path = _write_experiment_payload(tmp_path, {"model": {"unused_flag": 3}})

    with pytest.raises(ValueError, match="Unsupported model section keys: unused_flag"):
        load_experiment_config(config_path)


def test_load_experiment_config_rejects_string_booleans(tmp_path: Path):
    config_path = _write_experiment_payload(tmp_path, {"execution": {"mixed_precision": "false"}})

    with pytest.raises(ValueError, match="execution.mixed_precision must be a boolean"):
        load_experiment_config(config_path)


def test_load_experiment_config_rejects_non_mapping_sections(tmp_path: Path):
    config_path = _write_experiment_payload(tmp_path, {"model": 3})

    with pytest.raises(ValueError, match="Section 'model' must be a mapping"):
        load_experiment_config(config_path)


def test_load_experiment_config_accepts_cuda_device_strings(tmp_path: Path):
    config_path = _write_experiment_payload(tmp_path, {"execution": {"device": "cuda:0"}})

    config = load_experiment_config(config_path)

    assert config.execution.device == "cuda:0"


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"model": {"mlp_ratio": 0.0}}, "model.mlp_ratio"),
        ({"model": {"norm_eps": 0.0}}, "model.norm_eps"),
        ({"optimization": {"grad_clip_norm": 0.0}}, "optimization.grad_clip_norm"),
        ({"evaluation": {"sequential_step_s": 0.0}}, "evaluation.sequential_step_s"),
        ({"discovery": {"probe_learning_rate": 0.0}}, "discovery.probe_learning_rate"),
        ({"data_runtime": {"min_unit_spikes": -1}}, "data_runtime.min_unit_spikes"),
        ({"data_runtime": {"max_units": 0}}, "data_runtime.max_units"),
    ],
)
def test_load_experiment_config_rejects_invalid_numeric_ranges(
    tmp_path: Path,
    override: dict[str, object],
    match: str,
):
    config_path = _write_experiment_payload(tmp_path, override)

    with pytest.raises(ValueError, match=match):
        load_experiment_config(config_path)


def test_dataset_selection_split_overrides_mark_selection_active(tmp_path: Path):
    config_path = _write_experiment_payload(tmp_path, {"dataset_selection": {"split_seed": 11}})

    config = load_experiment_config(config_path)

    assert config.dataset_selection.has_split_overrides is True
    assert config.dataset_selection.is_active is True


def test_resolve_device_falls_back_to_cpu_for_unavailable_cuda_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    device = resolve_device("cuda:0")

    assert device.type == "cpu"


def test_aggregate_metric_dicts_supports_window_weighting() -> None:
    metrics = aggregate_metric_dicts(
        [
            {"predictive_improvement": 1.0},
            {"predictive_improvement": 0.0},
        ],
        weights=[3.0, 1.0],
    )

    assert metrics["predictive_improvement"] == 0.75
