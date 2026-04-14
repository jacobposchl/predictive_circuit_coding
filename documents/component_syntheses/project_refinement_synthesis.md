# Project Refinement Synthesis

## Project Purpose

`predictive_circuit_coding` is an Allen-first Neuropixels research codebase for building predictive population-dynamics representations and using them to support downstream, provenance-preserving interpretation. The repository is designed to prepare Allen Visual Behavior Neuropixels sessions, train a task-agnostic predictive encoder, evaluate held-out predictive quality, discover candidate task-relevant motifs from frozen tokens, and validate those discoveries with conservative observational checks.

This synthesis is a refinement map, not a results review. It describes how the project is organized, where the key contracts live, and which areas are natural candidates for cleanup or hardening. Result-oriented documents remain useful methodological context, but they should not drive this document's structure.

The guiding scientific separation is:

- encoder learning is dynamics first and task-agnostic in the baseline objective
- task-specific interpretation happens after training on frozen tokens
- discovery artifacts must preserve provenance back to sessions, units, windows, patches, and labels
- validation checks support cautious observational claims, not causal discovery

## System Map

The core package is organized around workflow responsibilities:

- `predictive_circuit_coding/data/`: preparation config, workspace layout, Allen/brainsets integration, prepared-session scanning, catalogs, manifests, splits, and runtime subset selection.
- `predictive_circuit_coding/windowing/`: split-aware `torch_brain` dataset bundles and fixed-window samplers.
- `predictive_circuit_coding/tokenization/`: spike binning, temporal patching, masks, event annotations, and `PopulationWindowBatch` collation.
- `predictive_circuit_coding/models/`: spatiotemporal encoder blocks and prediction/reconstruction/region-rate heads.
- `predictive_circuit_coding/objectives/`: predictive targets, continuation baselines, reconstruction/predictive losses, and cross-session region-rate auxiliary targets.
- `predictive_circuit_coding/training/`: typed experiment config, training loop, checkpointing, summaries, training history, and geometry monitor output.
- `predictive_circuit_coding/evaluation/`: held-out predictive evaluation.
- `predictive_circuit_coding/benchmarks/`: representation and motif benchmark matrices, reports, full-run verifier, and notebook pipeline runner.
- `predictive_circuit_coding/decoding/`: frozen-token extraction, labels, probes, scoring, and geometry transforms.
- `predictive_circuit_coding/discovery/`: candidate selection orchestration, clustering, stability, and cluster reports.
- `predictive_circuit_coding/validation/`: provenance checks, shuffled-label controls, held-out probe transfer, motif similarity, and baseline sensitivity.
- `predictive_circuit_coding/utils/`: dependency checks, console helpers, and notebook-facing runtime helpers.

Supporting surfaces:

- `brainsets_local_pipelines/`: repo-local Allen Visual Behavior Neuropixels `BrainsetPipeline`.
- `configs/pcc/`: preparation, experiment, and pipeline configs.
- `notebooks/`: thin Colab orchestration, currently centered on `run_predictive_circuit_coding_pipeline_colab.ipynb`.
- `documents/`: project guide, artifact contracts, synthesis, and refinement docs.
- `tests/`: unit, workflow, hardening, and notebook orchestration coverage.

## End-To-End Workflow

Local preparation starts with `pcc-prepare-data` and `configs/pcc/allen_visual_behavior_neuropixels_local.yaml`. The real Allen path invokes `brainsets.runner` with the repo-local Allen Visual Behavior Neuropixels pipeline, writes processed `.h5` sessions, scans them into rich catalogs, plans splits, and writes split-specific dataset configs. The local machine is also the right place to inspect windows before using Colab compute.

Training is driven by `pcc-train` or the unified Colab pipeline. It resolves a dataset view, builds fixed-window batches, constructs the encoder and objective from the experiment config, writes best/latest checkpoints, and emits training summaries. The cross-session augmented config family adds a region-rate auxiliary objective with donor-cache targets and optional geometry monitoring, while keeping the baseline predictive/reconstruction objective present.

Evaluation and benchmarking consume checkpoints. `pcc-evaluate` records held-out predictive metrics for a chosen split. `pcc-benchmark` runs representation and motif benchmark matrices over configured tasks and arms. `pcc-verify-full-run` is a no-training readiness gate for full pipeline configs; it checks configuration and task coverage before expensive Colab runs.

Discovery and validation are downstream interpretation stages. `pcc-discover` scans discovery windows, labels them, extracts frozen features, fits probes, scores tokens, selects candidates, clusters them, estimates stability, and writes both machine-readable and human-readable artifacts. `pcc-validate` checks artifact/checkpoint/target provenance, recomputes real-label metrics, runs shuffled-label controls, evaluates held-out probe transfer, and computes held-out motif-similarity summaries.

The unified notebook orchestrates these stages through config-first package calls. It should remain a thin interface for setup, stage selection, progress display, and artifact inspection.

## Contract Surfaces

The main public interfaces are:

- CLI commands declared in `pyproject.toml`.
- Preparation configs, experiment configs, and pipeline configs under `configs/pcc/`.
- Checkpoint payloads and metadata.
- JSON/CSV artifact schemas listed in `documents/artifact_contracts.md`.
- Runtime subset bundles created for notebook-driven subset runs.
- Pipeline state and manifest files used for stage resume.
- Token and candidate provenance fields used by discovery and validation.

The artifact layer is especially important. Checkpoints, summaries, discovery artifacts, validation outputs, benchmark reports, and run-manifest sidecars are not temporary logs; they are how the project records what happened and whether downstream interpretation is aligned with the original dataset/config/checkpoint.

Schema-affecting changes should be treated like API changes. They should update docs and tests together.

## Current Design Strengths

The package boundaries are clear and mostly workflow-aligned. Data preparation, runtime batching, training, evaluation, discovery, validation, and benchmarking are separate enough that future contributors can usually find the right layer quickly.

Provenance is treated as a first-class requirement. The batch contract preserves session, subject, unit, region, depth, timing, and annotation context, and downstream artifacts keep those fields visible.

The local/Colab compute split is explicit. Fragile Allen preparation and CPU-heavy processing stay local, while GPU-heavy training and downstream analyses run in Colab.

The notebook policy is healthy. The supported notebook calls package APIs and config loaders rather than embedding core implementation logic.

The artifact discipline is strong. Major stages write compact summaries, sidecars, and CSV versions where inspection matters.

The test suite includes both unit-style contract coverage and synthetic workflow coverage, which is a good match for a pipeline where many risks are cross-stage interface risks.

## Refinement Opportunities

Keep docs aligned around one supported workflow. `AGENTS.md`, `README.md`, `project_guide.md`, and `artifact_contracts.md` should continue to agree on command names, artifact names, config families, and notebook role.

Clarify the data-prep pathways. The real Allen/brainsets route and the CSV/synthetic preparation helpers both matter, but they serve different audiences. Naming that distinction in docs will reduce confusion for future agents.

Add a compact batch-contract reference. Many downstream systems depend on `PopulationWindowBatch` shapes and provenance fields, so a small stable description would help contributors reason about tokenization, objectives, discovery, and validation.

Harden artifact schemas with tests. The dataclasses cover many core JSON payloads, but summary CSV/JSON report rows and pipeline state are also public-facing contracts.

Make config families easier to navigate. The baseline, debug, full, cross-session, and pipeline configs are explicit but somewhat duplicated. A config map in docs, and eventually a safe overlay/include pattern, would make refinements less error-prone.

Keep pipeline resume semantics visible. Stage reuse depends on config hashes, upstream inputs, and output existence. This is useful behavior, but it should be easy to audit when a run reuses a stage.

Separate implemented auxiliary behavior from earlier design notes. `documents/new_refinement.md` is useful background, while the current implementation uses scheduled augmentation parameters, donor-cache state, canonical region derivation, auxiliary checkpoint state, and geometry monitoring.

Treat final-run reproducibility as a distinct refinement track. Runtime Familiar/G selection is convenient for iteration, but claim-facing runs are easier to explain when final session membership and task coverage are pinned in committed artifacts.

## Suggested Refinement Roadmap

Near term:

- Add a config map section to the project guide.
- Add a batch-contract note covering shapes, masks, and provenance.
- Add doc/code drift checks for CLI command names and key config filenames.
- Keep this refinement synthesis updated when workflow behavior changes.

Medium term:

- Add schema tests for benchmark summaries, pipeline state, run manifests, and discovery/validation CSVs.
- Consolidate repeated task-label naming and alias handling where practical.
- Clarify preparation helper roles in data docs and tests.
- Review training-loop complexity after auxiliary-training additions and extract helpers if needed.

Later:

- Pin final-run session manifests for paper-facing experiments.
- Introduce explicit artifact schema versions if historical run compatibility becomes important.
- Expand scientific-method docs around validation controls and accepted claim language without turning architecture docs into result summaries.
- Revisit config composition to reduce duplication between debug/full and baseline/augmented families.

## Navigation Guide

For data preparation questions, start with `configs/pcc/allen_visual_behavior_neuropixels_local.yaml`, `predictive_circuit_coding/data/brainsets_runner.py`, `brainsets_local_pipelines/allen_visual_behavior_neuropixels/pipeline.py`, `predictive_circuit_coding/data/catalog.py`, and `predictive_circuit_coding/data/selection.py`.

For windowing and batching questions, start with `predictive_circuit_coding/windowing/dataset.py`, `predictive_circuit_coding/tokenization/batching.py`, and `predictive_circuit_coding/training/contracts.py`.

For model and objective questions, start with `predictive_circuit_coding/models/encoder.py`, `predictive_circuit_coding/models/heads.py`, `predictive_circuit_coding/objectives/targets.py`, `predictive_circuit_coding/objectives/losses.py`, and `predictive_circuit_coding/objectives/region_targets.py`.

For training runtime questions, start with `predictive_circuit_coding/training/config.py`, `predictive_circuit_coding/training/loop.py`, and `predictive_circuit_coding/training/artifacts.py`.

For benchmarks and notebook orchestration, start with `configs/pcc/pipeline_*.yaml`, `predictive_circuit_coding/benchmarks/run.py`, `predictive_circuit_coding/benchmarks/pipeline.py`, `predictive_circuit_coding/benchmarks/verification.py`, and the unified notebook.

For discovery and validation, start with `predictive_circuit_coding/decoding/extract.py`, `predictive_circuit_coding/decoding/labels.py`, `predictive_circuit_coding/decoding/scoring.py`, `predictive_circuit_coding/discovery/run.py`, and `predictive_circuit_coding/validation/run.py`.

For artifact expectations, start with `documents/artifact_contracts.md`, `predictive_circuit_coding/training/contracts.py`, and `predictive_circuit_coding/benchmarks/contracts.py`.

For test coverage, start with `tests/test_data_foundation.py`, `tests/test_allen_brainsets_prep.py`, `tests/test_tokenization_core.py`, `tests/test_model_core.py`, `tests/test_cross_session_aug.py`, `tests/test_benchmarks_pipeline.py`, `tests/test_stage5_stage6_workflow.py`, `tests/test_stage7_hardening.py`, and `tests/test_notebook_runtime_helpers.py`.
