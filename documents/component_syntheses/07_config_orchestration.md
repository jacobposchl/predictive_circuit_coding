# Config And Orchestration Synthesis

## What This Subsystem Owns

The config and orchestration subsystem defines how users choose datasets, subsets, model settings, workflow stages, tasks, arms, and output locations. It also owns the supported Colab pipeline surface and its resumable stage state.

The repo now uses repo-native configs as the normal entry point. Pipeline configs point to experiment configs and data configs. The unified notebook reads those configs and calls package APIs rather than carrying duplicated core logic.

## Inputs And Outputs

Inputs:

- Preparation config: `configs/pcc/allen_visual_behavior_neuropixels_local.yaml`.
- Experiment configs: baseline/debug/full and cross-session augmented variants.
- Pipeline configs: debug/full and cross-session augmented debug/full variants.
- Optional `PIPELINE_RUN_ID`.
- Drive/local artifact roots and source dataset roots.

Outputs:

- Resolved runtime experiment config.
- Runtime subset bundle.
- Colab run root under `pcc_colab_outputs/<run_id>/run_1/`.
- `pipeline/pipeline_manifest.json`.
- `pipeline/pipeline_state.json`.
- Stage summaries and report artifacts.

## Key Code/Config Anchors

- `configs/pcc/allen_visual_behavior_neuropixels_local.yaml`: local preparation surface.
- `configs/pcc/predictive_circuit_coding_base.yaml`: baseline experiment defaults.
- `configs/pcc/predictive_circuit_coding_debug.yaml` and `predictive_circuit_coding_full.yaml`.
- `configs/pcc/predictive_circuit_coding_cross_session_aug_debug.yaml` and `predictive_circuit_coding_cross_session_aug_full.yaml`.
- `configs/pcc/pipeline_debug.yaml`, `pipeline_full.yaml`, `pipeline_cross_session_aug_debug.yaml`, and `pipeline_cross_session_aug_full.yaml`.
- `predictive_circuit_coding/benchmarks/config.py`: notebook pipeline config loader.
- `predictive_circuit_coding/benchmarks/pipeline.py`: stage runner, state reuse, Drive/local sync, final reports.
- `predictive_circuit_coding/utils/notebook.py`: notebook-facing runtime helpers.
- `notebooks/run_predictive_circuit_coding_pipeline_colab.ipynb`: supported Colab entry point.
- `tests/test_benchmarks_pipeline.py` and `tests/test_notebook_runtime_helpers.py`.

## Workflow Dependencies

Config orchestration sits above all major stages. It must keep paths, stage toggles, task panels, arms, runtime subsets, and checkpoint outputs aligned. A completed stage can only be reused when config hash, upstream inputs, and declared outputs still match.

The unified notebook is intentionally thin: it sets a pipeline config path and optional run ID, loads the pipeline config, and delegates execution to `run_notebook_pipeline_from_config`. This keeps implementation logic in the package where tests can cover it.

## Refinement Opportunities

- Reduce YAML duplication between debug/full and baseline/augmented config families where a safe include/overlay pattern becomes available.
- Keep the pipeline state reuse rules visible in docs, because resume behavior can otherwise feel magical.
- Add a compact config map showing which files are prep configs, experiment configs, and pipeline configs.
- Continue testing notebook JSON parsing and stage-runner wiring so the notebook remains an orchestration surface rather than an implementation surface.
- Keep task panels and benchmark arms config-first so final-run changes are visible in version control.
