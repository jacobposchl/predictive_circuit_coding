# Predictive Circuit Coding Rewrite Plan

## Summary

This repository has been intentionally reset to prepare for a full rewrite around Allen Institute Neuropixels data and a dynamics-first representation-learning workflow. The next implementation should build a new codebase from scratch rather than adapting the removed image-era `flow_circuits` project.

Public surface for the rebuild:

- Python package: `predictive_circuit_coding`
- CLI family: `pcc-*`
- Initial scope: Allen-first Neuropixels workflow

## Operating Rules

The rebuild should follow these execution rules from the beginning:

1. Google Colab notebooks with A100 GPUs are the primary training compute surface.
2. Any preprocessing, dataset inspection, split creation, metadata validation, and other CPU-only work should be done locally first.
3. Colab training notebooks must save resumable checkpoints to Google Drive often enough to protect expensive compute, while still keeping artifact size disciplined.
4. The codebase should be professionally structured with clear modularization, dependency inversion at package boundaries, high-signal documentation, concise comments, and clear tests.
5. Notebooks should provide clear runtime status updates, stage boundaries, and rough timing expectations so the user always knows what is happening.
6. Existing packages should be preferred whenever they cover the need cleanly; avoid rewriting infrastructure code without a clear project-specific reason.

## Target Repo Structure

```text
predictive_circuit_coding/
  data/
  windowing/
  models/
  objectives/
  decoding/
  discovery/
  analysis/
  evaluation/
  cli/
  utils/
configs/
  pcc/
documents/
notebooks/
tests/
```

## Dependency Direction

The rebuild should be designed around these packages:

- `brainsets`
- `temporaldata`
- `torch_brain`

Use them as first-class dependencies rather than re-implementing their core capabilities locally.

Planned package roles:

- `temporaldata`: canonical in-memory and serialized data model for sessions, spikes, intervals, trials, and aligned covariates
- `brainsets`: preferred source for processed dataset pipelines and dataset preparation templates when the target dataset already exists there, or as the pattern to follow when a custom prepare step is needed
- `torch_brain`: primary package for lazy recording access, split-aware sampling intervals, fixed-window sampling, and reusable neural-data training input logic

Implementation stance:

- Start with `temporaldata` and `brainsets` first so the project has a solid data contract and a reproducible local preparation path.
- Then build the project runtime around `torch_brain` for dataset access and window sampling.
- Do not make `torch_brain` the owner of project-specific scientific logic; keep the encoder, objectives, motif discovery, and evaluation logic inside `predictive_circuit_coding`.

Allen dataset note:

- Treat Allen Visual Behavior Neuropixels as the first target dataset.
- First verify whether the exact Allen dataset is already supported by `brainsets`.
- If it is not, implement a local preparation pipeline that converts the Allen source data into `temporaldata.Data` objects using a `brainsets`-style preparation flow, then consume the processed output through `torch_brain`.

## First-Phase Workflow

The first working version of the new repo should support this four-stage flow:

1. Train a predictive encoder on neural population dynamics.
2. Evaluate held-out predictive-dynamics performance against explicit baselines.
3. Discover task-relevant motifs from frozen token representations.
4. Validate motif robustness and specificity on held-out data.

## Implementation Stages

The implementation should be done in this order:

### Stage 1: Data Foundation

- Set up dependency installation and version pinning around `temporaldata`, `brainsets`, and `torch_brain`.
- Decide the exact Allen Visual Behavior Neuropixels source path and local storage layout.
- Build a local CPU-first dataset preparation workflow that standardizes session metadata, trials/events, units, spike times or counts, and split metadata.
- Store prepared sessions in a form that can be loaded as `temporaldata.Data` objects with stable recording IDs and provenance fields.
- Define the canonical provenance contract: session ID, subject ID, unit ID, region, depth, trial ID, window start/end, patch index, and any aligned task/event fields.

### Stage 2: Dataset Access And Windowing

- Wrap the processed dataset with `torch_brain` dataset access.
- Use `torch_brain` sampling intervals and fixed-window samplers for train, validation, test, and discovery workflows.
- Add project-local transforms only where needed to turn sampled windows into the exact tensors required by the model.
- Keep this stage CPU-friendly and runnable locally for inspection and debugging before any GPU training starts.

### Stage 3: Core Package Skeleton

- Create the new `predictive_circuit_coding` package tree with clean interfaces between data, windowing, models, objectives, decoding, discovery, analysis, evaluation, cli, and utils.
- Define typed config schemas and artifact schemas before implementing training logic deeply.
- Keep external-package integrations behind narrow adapters so the model code does not depend directly on vendor-specific details everywhere.

### Stage 4: Model And Objective Implementation

- Implement patch construction, tokenization, encoder blocks, continuation baselines, delta targets, predictive loss, and reconstruction anchoring.
- Keep training logic package-based and notebook-thin.
- Ensure all model outputs preserve the reversible provenance path back to raw data coordinates.

### Stage 5: Training And Evaluation Workflow

- Implement `pcc-train` and `pcc-evaluate`.
- Build Colab-first training notebooks that mount Drive, install the package, resume from checkpoints, and report progress clearly.
- Keep local CPU utilities for config validation, dataset sanity checks, and small dry runs.

### Stage 6: Decoding, Discovery, And Validation

- Implement frozen-token extraction, additive decoders, token contribution scoring, candidate selection, clustering, and stability analysis.
- Implement `pcc-discover` and `pcc-validate`.
- Keep early validation focused on predictive baselines, label shuffles, robustness, target sensitivity, and cross-session checks before any stronger intervention story is introduced.

### Stage 7: Notebook UX, Documentation, And Hardening

- Build notebook progress UI with existing packages such as `rich`, `tqdm.auto`, and `ipywidgets` where appropriate.
- Add stage banners, elapsed-time reporting, checkpoint-save messages, and expected-duration notes for long Colab cells.
- Finish repo docs, artifact contracts, and tests only after the core pipeline shape is stable.

## Planned First Rebuild Artifacts

The new implementation should define and document first-class artifacts for:

- training checkpoint
- evaluation summary JSON
- motif discovery artifact JSON
- validation summary JSON/CSV

## Explicit Non-Goals

The rewrite should not carry forward any of the following:

- CIFAR-based data handling
- ResNet observer logic
- `torchvision` dependencies
- residual-patch intervention logic from the image project
- backward compatibility with `flow_circuits`

## Rebuild Notes

- The preserved design doc at `documents/predictive_circuit_coding_design_doc.docx` is the scientific source of truth during the rebuild.
- `AGENTS.md` and `CODEX.md` remain only as collaboration scaffolding.
- Git history is the archive for the removed image-era project; no in-repo compatibility layer or archive tree should be reintroduced.
- Local work should own dataset preparation, schema checks, and CPU-only preprocessing; Colab should be reserved primarily for GPU-dependent training and evaluation runs.
