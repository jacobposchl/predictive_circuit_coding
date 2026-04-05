# AGENTS

This file is the shared top-level contract for coding agents working in this repository. It is the canonical agent-facing guide for the current `predictive_circuit_coding` project.

## Purpose

`predictive_circuit_coding` is an Allen-first Neuropixels research codebase for:

- preparing Allen Visual Behavior Neuropixels sessions into `temporaldata`/`torch_brain`-compatible processed sessions
- training a predictive population-dynamics encoder
- evaluating held-out predictive performance against explicit baselines
- discovering candidate task-relevant motifs from frozen tokens
- validating those discoveries with conservative observational checks

The supported end-to-end workflow is:

1. `pcc-prepare-data`
2. `pcc-train`
3. `pcc-evaluate`
4. `pcc-discover`
5. `pcc-validate`


## Canonical Sources Of Truth

Read these before making substantial changes:

1. [`AGENTS.md`](C:/Users/Jacob%20Poschl/Desktop/population-dynamics/AGENTS.md)
2. [`README.md`](C:/Users/Jacob%20Poschl/Desktop/population-dynamics/README.md)
3. [`documents/project_guide.md`](C:/Users/Jacob%20Poschl/Desktop/population-dynamics/documents/project_guide.md)
4. [`documents/artifact_contracts.md`](C:/Users/Jacob%20Poschl/Desktop/population-dynamics/documents/artifact_contracts.md)

`AGENTS.md` is the canonical shared agent guide. Tool-specific files such as `CODEX.md` should point back here instead of redefining project truth.

## Compute Split

The repo is built around a strict compute split:

- local CPU:
  - Allen data preparation
  - manifest and split planning
  - local inspection and debugging
- Google Colab A100:
  - training
  - evaluation
  - discovery
  - validation

When `allen_sdk.cache_root` is configured, the Allen raw cache is external to this repo. Keep processed outputs, manifests, splits, and runtime artifacts inside this repo; do not copy raw Allen session data into the project workspace unless explicitly required.

## Package Boundaries

- `predictive_circuit_coding/data/`: preparation configs, workspace layout, manifests, split logic, processed-session scanning, `brainsets` integration
- `predictive_circuit_coding/windowing/`: `torch_brain` dataset config generation, split-aware dataset bundles, fixed-window sampling, inspection helpers
- `predictive_circuit_coding/tokenization/`: spike binning, temporal patch construction, canonical training batches, provenance-preserving collation
- `predictive_circuit_coding/models/`: encoder blocks, spatiotemporal mixing, predictive head, reconstruction head
- `predictive_circuit_coding/objectives/`: target builders, continuation baselines, predictive and reconstruction losses
- `predictive_circuit_coding/training/`: typed experiment config, runtime contracts, training loop, checkpointing, summaries, logging
- `predictive_circuit_coding/decoding/`: frozen-token extraction, label extraction, additive probes, token scoring
- `predictive_circuit_coding/discovery/`: candidate selection, clustering, stability estimates, cluster reporting
- `predictive_circuit_coding/evaluation/`: held-out evaluation and metric aggregation
- `predictive_circuit_coding/validation/`: label-shuffle, recurrence, baseline-sensitivity, and provenance-integrity checks
- `predictive_circuit_coding/utils/`: dependency checks, console helpers, notebook-facing helpers
- `brainsets_local_pipelines/`: repo-local Allen Visual Behavior Neuropixels `BrainsetPipeline`
- `configs/pcc/`: preparation and experiment configs
- `notebooks/`: thin Colab orchestration only
- `tests/`: synthetic and workflow-level validation

## Edit Policy

- Put core logic in `predictive_circuit_coding/`, not in notebooks.
- Treat notebooks as thin orchestration and inspection surfaces.
- Prefer reusing `brainsets`, `temporaldata`, and `torch_brain` when they fit the problem instead of rebuilding equivalent infrastructure locally.
- Do not silently change checkpoint schemas, manifest JSON, discovery artifacts, validation CSVs, or run-manifest sidecars without updating docs and tests.
- Preserve provenance. Tokens, candidates, clusters, and summaries should remain traceable back to session/unit/window/patch context.
- Keep the local prep workflow clean: raw Allen cache may live outside the repo, while processed sessions and downstream artifacts should live inside the repo.

## Validation Policy

After non-trivial changes, run:

```bash
python -m pytest -q
```

If you change docs, configs, notebooks, or artifact formats, also sanity-check:

- links and filenames in `README.md` and `documents/project_guide.md`
- command examples for the `pcc-*` surface
- artifact references in [`documents/artifact_contracts.md`](C:/Users/Jacob%20Poschl/Desktop/population-dynamics/documents/artifact_contracts.md)

## Notebook Policy

- Notebooks are Google Colab-first.
- Setup cells may clone the repo, install the package, mount Drive, and reuse saved checkpoints and processed outputs.
- Notebook code should call library APIs or `pcc-*` CLIs rather than duplicating implementation logic.
- Install first, import repo helpers second.
- Reusable notebook helpers belong in the package, not copy-pasted across notebooks.

## Artifact Policy

Current first-class artifacts are:

- training checkpoints
- training summary JSON
- evaluation summary JSON
- discovery artifact JSON
- cluster summary JSON and CSV
- validation summary JSON and CSV
- run-manifest sidecars

See [`documents/artifact_contracts.md`](C:/Users/Jacob%20Poschl/Desktop/population-dynamics/documents/artifact_contracts.md) for required keys and shapes.

## High-Signal Working Pattern

When making a change:

1. Read the relevant workflow and artifact docs.
2. Inspect the target module boundary in `predictive_circuit_coding/`.
3. Implement the smallest coherent change in code, not notebooks.
4. Update docs if repo behavior, paths, or artifact expectations changed.
5. Add or update tests when behavior changed.
6. Run tests.

## Supporting Docs

- [`README.md`](C:/Users/Jacob%20Poschl/Desktop/population-dynamics/README.md): quick-start, install, commands, troubleshooting
- [`documents/project_guide.md`](C:/Users/Jacob%20Poschl/Desktop/population-dynamics/documents/project_guide.md): full repo organization and experiment workflow
- [`documents/artifact_contracts.md`](C:/Users/Jacob%20Poschl/Desktop/population-dynamics/documents/artifact_contracts.md): stable artifact schemas
- [`environments/allen_visual_behavior_prep/README.md`](C:/Users/Jacob%20Poschl/Desktop/population-dynamics/environments/allen_visual_behavior_prep/README.md): dedicated Allen prep environment
