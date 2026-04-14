# Artifact Contracts Synthesis

## What This Subsystem Owns

Artifact contracts define the stable files that let the project remain reproducible, inspectable, and resumable. They are the bridge between code stages, notebook execution, future contributors, and downstream scientific interpretation.

The artifact layer is effectively a public interface. Changing checkpoint payloads, manifest shapes, discovery artifacts, validation summaries, benchmark reports, or run-manifest sidecars should be treated like changing an API.

## Inputs And Outputs

Inputs:

- Training, evaluation, benchmark, discovery, validation, and pipeline-stage outputs.
- Config snapshots and runtime subset metadata.
- Checkpoint metadata and provenance records.

Outputs:

- Checkpoint `.pt` files.
- JSON summaries.
- CSV summaries.
- Discovery artifacts and cluster reports.
- Pipeline manifests and state.
- Stage run-manifest sidecars.

## Key Code/Config Anchors

- `documents/artifact_contracts.md`: canonical artifact field reference.
- `predictive_circuit_coding/training/contracts.py`: dataclass-backed checkpoint, summary, discovery, and validation payloads.
- `predictive_circuit_coding/training/artifacts.py`: JSON/CSV/checkpoint writers.
- `predictive_circuit_coding/benchmarks/contracts.py`: benchmark task, arm, pipeline state, and run-manifest dataclasses.
- `predictive_circuit_coding/benchmarks/reports.py`: benchmark and final summary report writers.
- `predictive_circuit_coding/discovery/reporting.py`: cluster summary report writers.
- `predictive_circuit_coding/cli/common.py`: run-manifest sidecar emission and provenance guard helpers.
- `tests/test_training_runtime_config.py`, `tests/test_benchmarks_pipeline.py`, `tests/test_stage7_hardening.py`, and workflow tests.

## Workflow Dependencies

Every stage depends on upstream artifacts remaining interpretable. Evaluation, discovery, validation, and benchmark commands check dataset and checkpoint compatibility through sidecar or checkpoint metadata. Notebook stages use pipeline state and output existence to decide whether a completed stage is reusable.

Artifacts also carry scientific meaning. Discovery candidate records preserve token provenance. Validation summaries distinguish real labels from shuffled controls and held-out similarity. Benchmark rows carry task, arm, training variant, and geometry metadata.

## Refinement Opportunities

- Add schema-oriented tests for high-value JSON/CSV artifacts, especially benchmark summaries and pipeline state.
- Consider explicit artifact schema versions if multiple historical runs need to coexist.
- Keep run-manifest sidecars complete for notebook-subset runs, including runtime split manifest and session catalog inputs.
- Separate "required public fields" from "diagnostic extra fields" in docs if artifacts continue growing.
- Ensure docs and dataclasses stay aligned when adding fields such as auxiliary-state metadata or new benchmark metrics.
