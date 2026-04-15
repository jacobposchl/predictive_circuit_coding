# Project Guide

This repository is now organized around refinement of the trained predictive encoder, not broad comparison runs.

## Canonical Workflow

1. Prepare Allen sessions with `pcc-prepare-data` subcommands.
2. Train a refined predictive encoder with `pcc-train`.
3. Run trained-encoder refinement discovery with `pcc-refine`.
4. Validate selected discoveries with `pcc-validate`.
5. Use `pcc-evaluate` only as a predictive sanity check.

`pcc-discover` still exists for lower-level single-target discovery runs, but the default project workflow should go through `pcc-refine`.

Common preparation commands:

```bash
pcc-prepare-data init-workspace --config configs/pcc/allen_visual_behavior_neuropixels_local.yaml
pcc-prepare-data prepare-allen-visual-behavior-neuropixels --config configs/pcc/allen_visual_behavior_neuropixels_local.yaml
```

If processed sessions already exist and you only need canonical manifests, splits, and upload metadata:

```bash
pcc-prepare-data build-session-catalog --config configs/pcc/allen_visual_behavior_neuropixels_local.yaml
```

Before expensive runs, use:

```bash
pcc-verify-refinement --pipeline-config configs/pcc/pipeline_refined_full.yaml --output-root artifacts/refinement_verification
```

For the Colab/notebook end-to-end path, use:

```bash
pcc-run-pipeline --pipeline-config configs/pcc/pipeline_refined_debug.yaml
```

`pcc-run-pipeline` writes run-scoped outputs under `artifacts/pipeline_runs/<run_id>/run_1/`, including pipeline manifest/state files, train/evaluate/refinement stage artifacts, and final report summaries.

## CLI Surface

- `pcc-prepare-data`: local CPU preparation subcommands, manifest planning, runtime selection materialization
- `pcc-train`: refined predictive encoder training
- `pcc-evaluate`: split-level predictive sanity checks
- `pcc-discover`: direct single-target frozen-token discovery
- `pcc-refine`: trained-encoder refinement benchmark matrix and summary reports
- `pcc-validate`: conservative validation of a selected discovery artifact
- `pcc-run-pipeline`: Colab-friendly stage-resume orchestration
- `pcc-verify-refinement`: readiness and coverage gating for a pipeline config
- `pcc-preview-notebook-ui`: notebook/report preview utility

## Package Map

- `predictive_circuit_coding/data/`: Allen preparation, manifests, catalogs, split planning, local pipeline integration
- `predictive_circuit_coding/windowing/`: dataset views and fixed-window sampling
- `predictive_circuit_coding/tokenization/`: spike binning, patch construction, provenance-preserving batches, optional count normalization
- `predictive_circuit_coding/models/`: predictive encoder, predictive/reconstruction heads, optional per-patch population tokens
- `predictive_circuit_coding/objectives/`: predictive continuation objective and reconstruction objective
- `predictive_circuit_coding/training/`: typed configs, training loop, checkpoint and summary writing
- `predictive_circuit_coding/decoding/`: frozen-token extraction, labels, probes, geometry transforms
- `predictive_circuit_coding/discovery/`: candidate selection, clustering, stability, reports
- `predictive_circuit_coding/benchmarks/`: refinement discovery matrices and reporting
- `predictive_circuit_coding/validation/`: conservative validation checks for discovery artifacts
- `predictive_circuit_coding/workflows/`: end-to-end pipeline orchestration, preflight, stage state, and Colab runner integration
- `predictive_circuit_coding/cli/`: `pcc-*` entry points and command-sidecar manifests

## Config Families

Core:

- `predictive_circuit_coding_refined_debug.yaml`
- `predictive_circuit_coding_refined_full.yaml`

Ablations:

- `predictive_circuit_coding_refined_recon000_*`
- `predictive_circuit_coding_refined_l2off_*`
- `predictive_circuit_coding_refined_countnorm_*`
- `predictive_circuit_coding_refined_cls_*`

Pipeline:

- `pipeline_refined_debug.yaml`
- `pipeline_refined_full.yaml`

Short ablation configs use `extends` to inherit from the refined core config and override one axis. Some compatibility variants preserve earlier comparison names even when the current full-core defaults already include that setting.

Pipeline task coverage:

- `pipeline_refined_debug.yaml`: `stimulus_change`
- `pipeline_refined_full.yaml`: `stimulus_change`, `trials_go`

## Refinement Axes

- Reconstruction: full-core training uses `reconstruction_weight: 0.0`; the debug config still keeps a lightweight reconstruction term for quick smoke runs
- Reconstruction target mode: `window_zscore` in the core config
- Token geometry: model token L2 normalization plus post-hoc token normalization arm
- Count normalization: full-core training uses train-split `log1p_train_zscore`
- Pooling: mean tokens by default, optional per-patch population CLS tokens
- Discovery geometry: raw, token-normalized, probe-weighted, and oracle-aligned diagnostic arms

## Artifact Surfaces

The refinement-centered workflow now writes:

- training checkpoints, training summary JSON, training history JSON/CSV
- evaluation summary JSON
- discovery decode-coverage JSON, discovery artifact JSON, cluster summary JSON/CSV
- refinement summary JSON/CSV and final project summary JSON/CSV
- per-arm transform summary JSON/CSV
- validation summary JSON/CSV
- command-sidecar run manifests
- pipeline manifest/state JSON for notebook pipeline runs
- refinement verification summary JSON and task-coverage CSV

## Notes For Contributors

Keep notebooks thin. Core logic belongs in package modules.

Do not add broad comparison arms back into the default path until the refined idea is stable.

Any artifact schema change should update `documents/artifact_contracts.md` and tests in the same change.
