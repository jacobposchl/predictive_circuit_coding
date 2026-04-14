# Evaluation And Benchmarking Synthesis

## What This Subsystem Owns

The evaluation and benchmarking subsystem measures trained checkpoints on held-out data and runs the claim-facing comparison matrix. It owns predictive held-out evaluation, representation benchmark arms, motif benchmark arms, compact report writing, and full-run readiness verification.

This synthesis treats benchmarking as workflow infrastructure rather than as a results review. The important refinement question is whether the system reliably makes comparisons reproducible, traceable, and skippable when a task is unsupported.

## Inputs And Outputs

Inputs:

- Experiment config and data config.
- A checkpoint matching the dataset/config family.
- Runtime subset assets when notebook-driven selection is active.
- Pipeline configs specifying task panels and benchmark arms.

Outputs:

- Evaluation summary JSON.
- Representation benchmark summary JSON and CSV.
- Motif benchmark summary JSON and CSV.
- Final project summary JSON and CSV.
- Full-run verification summary JSON and task coverage CSV.
- Per-stage run manifests and pipeline state entries.

## Key Code/Config Anchors

- `predictive_circuit_coding/evaluation/run.py`: held-out checkpoint evaluation.
- `predictive_circuit_coding/evaluation/metrics.py`: metric aggregation helpers.
- `predictive_circuit_coding/benchmarks/contracts.py`: benchmark task, arm, result, and pipeline-state dataclasses.
- `predictive_circuit_coding/benchmarks/features.py`: benchmark feature extraction and transforms.
- `predictive_circuit_coding/benchmarks/run.py`: representation and motif benchmark matrices.
- `predictive_circuit_coding/benchmarks/reports.py`: compact report writers.
- `predictive_circuit_coding/benchmarks/verification.py`: full-run readiness gate.
- `predictive_circuit_coding/cli/evaluate.py`, `cli/benchmark.py`, and `cli/verify_full_run.py`.
- `configs/pcc/pipeline_*.yaml`: stage toggles, task panels, arms, Drive/local paths.
- `tests/test_benchmarks_pipeline.py` and `tests/test_stage7_hardening.py`.

## Workflow Dependencies

Evaluation depends on training checkpoints and the same dataset view used by the training run. Benchmarks depend on frozen feature extraction, label extraction, split-aware sampling, and artifact output paths. The full-run verifier depends on real local prepared metadata and label coverage scans; it is a no-training gate meant to prevent expensive runs that would degrade into missing-field or no-positive-window summaries.

Representation benchmarks and motif benchmarks answer different workflow questions. Representation rows check feature readout quality across arms. Motif rows check whether candidate selection and clustering produce transferable motif summaries. The synthesis should preserve that distinction without ranking arms or repeating result tables.

## Refinement Opportunities

- Keep benchmark skip/degraded statuses explicit so unsupported tasks do not look like silent failures.
- Continue treating `pcc-verify-full-run` as a required gate for claim-facing full configs.
- Centralize task naming and aliases where practical, since task labels appear in configs, notebook UI, label extraction, benchmark rows, and reports.
- Strengthen schema tests for representation and motif summary rows as these reports become public-facing artifacts.
- Maintain the difference between benchmark infrastructure and empirical conclusions in docs.
