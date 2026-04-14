# Project Frame Synthesis

## What This Subsystem Owns

The project frame defines why the repository exists, what workflow it supports, and what claims it is designed to make possible. `predictive_circuit_coding` is an Allen-first Neuropixels research codebase for learning task-agnostic predictive population-dynamics representations, then using frozen tokens for downstream decoding, motif discovery, and conservative validation.

The central design stance is dynamics first. The encoder is trained without task labels, using predictive and reconstruction objectives. Task variables enter after training through frozen-token analysis, not through supervised representation learning. This keeps the interpretation workflow separate from the encoder objective and supports conservative claims about recurring candidate motifs rather than automatic causal discovery.

The supported workflow is:

1. `pcc-prepare-data`
2. `pcc-train`
3. `pcc-evaluate`
4. `pcc-benchmark`
5. `pcc-discover`
6. `pcc-validate`
7. `pcc-verify-full-run` before claim-facing full runs

The compute split is part of the architecture, not just an operational preference. Local CPU handles Allen preparation, manifest/split planning, and inspection. Google Colab A100 handles training, evaluation, benchmarks, discovery, and validation.

## Inputs And Outputs

Inputs:

- Allen Visual Behavior Neuropixels data or prepared `temporaldata` sessions.
- Repo-native preparation, experiment, and pipeline configs under `configs/pcc/`.
- The canonical docs: `AGENTS.md`, `README.md`, `documents/project_guide.md`, and `documents/artifact_contracts.md`.

Outputs:

- Prepared sessions and split manifests.
- Model checkpoints and training summaries.
- Evaluation, benchmark, discovery, and validation artifacts.
- Run-manifest sidecars and notebook pipeline state.

## Key Code/Config Anchors

- `AGENTS.md`: top-level agent contract, package boundaries, edit policy, validation policy.
- `README.md`: quick-start commands, current supported workflow, troubleshooting.
- `documents/project_guide.md`: main human-facing project guide.
- `documents/artifact_contracts.md`: stable artifact contract reference.
- `pyproject.toml`: package metadata and public CLI entry points.
- `predictive_circuit_coding/`: core implementation packages.
- `configs/pcc/`: preparation, experiment, pipeline, debug/full, and cross-session configs.
- `notebooks/run_predictive_circuit_coding_pipeline_colab.ipynb`: supported Colab orchestration surface.

## Workflow Dependencies

The project frame depends on the data and artifact layers staying explicit. Training and downstream interpretation are only meaningful when checkpoint metadata, runtime subset assets, split manifests, and token provenance agree. The docs also rely on the package boundaries staying stable enough that future contributors can find responsibilities without reading every module.

The full pipeline has a natural dependency chain:

- local data prep creates prepared sessions and manifests
- runtime selection creates artifact-local selected catalogs and split manifests
- training creates checkpoints and summaries
- evaluation and benchmarks consume checkpoints
- discovery consumes checkpoints and split-aware frozen features
- validation consumes discovery artifacts and checks provenance against checkpoint/config inputs

## Refinement Opportunities

- Keep `AGENTS.md`, `README.md`, and `project_guide.md` synchronized around the same supported command surface.
- Preserve a clear distinction between architecture/workflow docs and result-review docs so refinement notes do not become benchmark summaries.
- Add a small "project invariants" section to the main guide for non-negotiables such as provenance preservation, thin notebooks, and artifact schema discipline.
- Consider pinning final-run session membership through committed manifests when the project reaches a paper-ready run.
- Keep result-oriented synthesis documents as historical context, while letting refinement documents emphasize system shape, interfaces, and next implementation priorities.
