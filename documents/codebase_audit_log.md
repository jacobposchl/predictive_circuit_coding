# Codebase Audit Log

This file records focused findings from the codebase audit as we work through the major surfaces of `predictive_circuit_coding`.

## 2026-04-14

### `predictive_circuit_coding/training/config.py`

Role in pipeline:
This file defines the experiment schema for training, evaluation, discovery, validation, and runtime dataset selection. It is the main boundary between handwritten experiment YAMLs and the executable config objects used throughout the rest of the project.

Findings:

1. High: unknown config keys are silently ignored.
The loader accepts arbitrary extra keys at the top level and inside sections such as `data_runtime`, `model`, `objective`, `training`, and `discovery`, then quietly falls back to defaults for any misspelled fields. That makes config typos easy to miss in a research workflow.
Evidence:
- `predictive_circuit_coding/training/config.py:282-487`

2. High: boolean parsing is unsafe because it relies on `bool(...)` coercion.
Fields such as `include_trials`, `include_stimulus_presentations`, `include_optotagging`, `exclude_final_prediction_patch`, `save_config_snapshot`, and `mixed_precision` are parsed with `bool(raw_value)`. If a config contains quoted strings like `"false"` or `"0"`, they will parse as `True`, which is the opposite of user intent.
Evidence:
- `predictive_circuit_coding/training/config.py:323-326`
- `predictive_circuit_coding/training/config.py:350`
- `predictive_circuit_coding/training/config.py:368`
- `predictive_circuit_coding/training/config.py:446`

3. High: dataset-selection split overrides can be silently ignored.
`DatasetSelectionConfig.is_active` does not consider `split_seed`, `split_primary_axis`, `train_fraction`, `valid_fraction`, `discovery_fraction`, or `test_fraction`. That means a config that only wants to override split planning is treated as inactive, and the runtime selection path is skipped entirely.
Evidence:
- `predictive_circuit_coding/training/config.py:135-159`
- `predictive_circuit_coding/data/selection.py:196-205`
- `predictive_circuit_coding/data/selection.py:310-314`

4. Medium: malformed YAML section shapes fail late with opaque errors.
The loader assumes the top-level payload is a mapping and that sections like `data_runtime`, `model`, and `optimization` are mappings too. If any of those sections are missing or accidentally written as a scalar/list, the failure will be a `TypeError`, `KeyError`, or attribute error instead of a clear config-schema message.
Evidence:
- `predictive_circuit_coding/training/config.py:282-308`

5. Medium: several numerically important fields are still under-validated.
The validator covers a good amount, but some values used directly in downstream code still have no range checks. Notable examples are `model.mlp_ratio`, `model.norm_eps`, `optimization.grad_clip_norm`, `evaluation.sequential_step_s`, `discovery.probe_learning_rate`, `data_runtime.min_unit_spikes`, and `data_runtime.max_units`.
Why this matters:
- negative or zero `mlp_ratio` feeds invalid hidden dimensions into transformer blocks
- non-positive `norm_eps` is invalid for `LayerNorm`
- negative `grad_clip_norm` reaches gradient clipping directly
- non-positive `sequential_step_s` reaches sequential window sampling directly
- non-positive `probe_learning_rate` reaches probe fitting directly
Evidence:
- `predictive_circuit_coding/training/config.py:338`
- `predictive_circuit_coding/training/config.py:340`
- `predictive_circuit_coding/training/config.py:355-358`
- `predictive_circuit_coding/training/config.py:450-453`
- `predictive_circuit_coding/training/config.py:474`
- `predictive_circuit_coding/training/config.py:321-322`
- `predictive_circuit_coding/models/blocks.py:12-20`

6. Medium: device validation is stricter than the runtime actually is.
`validate_experiment_config()` only accepts `cpu`, `cuda`, or `auto`, but `resolve_device()` can handle general torch device strings like `cuda:0`. That means the schema currently blocks a runtime capability that the lower-level device resolver already supports.
Evidence:
- `predictive_circuit_coding/training/config.py:594-595`
- `predictive_circuit_coding/training/runtime.py:8-13`

Removal or deprecation candidates:

- `ExperimentConfig.split_name` appears unused in the executable code path. The train/evaluate/discovery/validation entry points use explicit split arguments or `config.splits.*`, not `config.split_name`.
- the legacy fallback to top-level `variant_name` in `load_experiment_config()` looks stale now that the canonical config shape uses `experiment.variant_name`
Evidence:
- `predictive_circuit_coding/training/config.py:211`
- `predictive_circuit_coding/training/config.py:315`

Recommended fix path:

1. Harden `training/config.py` the same way `workflows/config.py` was hardened:
   add strict section/key validation, typed readers, and explicit schema errors.
2. Split dataset-selection activation into clearer concepts:
   one check for selection filters and one check for split overrides, or broaden `is_active` so split-only overrides are honored.
3. Add missing numeric validation for downstream-sensitive fields.
4. Decide whether to officially support `cuda:N` device strings.
   If yes, widen validation. If not, narrow the runtime resolver for consistency.
5. Remove or deprecate truly unused compatibility fields once docs and tests are aligned.
