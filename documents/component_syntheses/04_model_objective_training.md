# Model, Objective, And Training Synthesis

## What This Subsystem Owns

The model/objective/training subsystem learns the predictive population-dynamics encoder and writes checkpointed training artifacts. It owns typed experiment config, model construction, objective construction, training/evaluation loops, checkpoint metadata, training summaries, training histories, and auxiliary cross-session diagnostics.

The baseline encoder objective is task-agnostic. It predicts future neural dynamics relative to a continuation baseline and includes reconstruction anchoring. The cross-session auxiliary variant keeps the baseline objective intact while adding a region-rate prediction signal designed to reduce session-specific latent geometry.

## Inputs And Outputs

Inputs:

- `PopulationWindowBatch` objects from the runtime data-flow layer.
- Experiment configs under `configs/pcc/predictive_circuit_coding_*.yaml`.
- Prepared data view and split manifests.
- Optional resume checkpoints.

Outputs:

- Best and latest training checkpoints.
- `training_summary.json`.
- `training_history.json` and `training_history.csv`.
- Optional `cross_session_geometry_monitor.json` and `.csv`.
- Training run-manifest sidecars.

## Key Code/Config Anchors

- `predictive_circuit_coding/training/config.py`: experiment config dataclasses, including `CrossSessionAugConfig`.
- `predictive_circuit_coding/models/encoder.py`: patch embedder, encoder, and predictive circuit model.
- `predictive_circuit_coding/models/blocks.py`: temporal/spatial attention blocks.
- `predictive_circuit_coding/models/heads.py`: predictive, reconstruction, and region-rate heads.
- `predictive_circuit_coding/objectives/targets.py`: predictive target and continuation baseline builders.
- `predictive_circuit_coding/objectives/losses.py`: combined objective and cross-session region loss.
- `predictive_circuit_coding/objectives/region_targets.py`: donor cache and region-rate target builder.
- `predictive_circuit_coding/training/loop.py`: training loop, checkpointing, history writing, geometry monitor.
- `predictive_circuit_coding/training/artifacts.py`: checkpoint and summary writers.
- `predictive_circuit_coding/cli/train.py`: public `pcc-train` entry point.
- `tests/test_model_core.py`, `tests/test_training_runtime_config.py`, `tests/test_cross_session_aug.py`, and benchmark workflow tests.

## Workflow Dependencies

The training subsystem depends on stable batch shapes and provenance from tokenization. Downstream stages depend on checkpoint metadata matching the dataset/config family, especially for validation and artifact provenance checks.

The cross-session auxiliary variant has additional dependencies:

- canonical regions are derived from the training split unless explicitly configured
- donor targets are cached by label and session
- auxiliary state is stored in checkpoints when enabled
- geometry monitoring uses frozen features from a configured split

These mechanics make checkpoint/config compatibility important. A baseline config should not resume an augmented checkpoint, and an augmented config should require auxiliary state when resuming.

## Refinement Opportunities

- Keep the implemented cross-session auxiliary design documented separately from the earlier design sketch in `documents/new_refinement.md`; the current implementation uses a donor cache and scheduled probabilities/weights.
- Consider extracting some auxiliary-training wiring from `training/loop.py` if the loop grows further.
- Make checkpoint compatibility rules easy to find in troubleshooting docs.
- Keep `training_summary.json` semantics explicit: it describes the selected best checkpoint, while `training_history.*` describes the full training trajectory.
- Add or maintain tests that assert auxiliary state, training variant name, and geometry monitor fields remain present when the augmented variant is enabled.
