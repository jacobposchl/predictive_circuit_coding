# Project Guide

This repository is now organized around refinement of the trained predictive encoder, not broad comparison runs.

## Canonical Workflow

1. Prepare Allen sessions with `pcc-prepare-data`.
2. Train a refined predictive encoder with `pcc-train`.
3. Run trained-encoder refinement discovery with `pcc-refine`.
4. Validate selected discoveries with `pcc-validate`.
5. Use `pcc-evaluate` only as a predictive sanity check.

Before expensive runs, use:

```bash
pcc-verify-refinement --pipeline-config configs/pcc/pipeline_refined_full.yaml --output-root artifacts/refinement_verification
```

## Package Map

- `predictive_circuit_coding/data/`: Allen preparation, manifests, catalogs, split planning, local pipeline integration
- `predictive_circuit_coding/windowing/`: dataset views and fixed-window sampling
- `predictive_circuit_coding/tokenization/`: spike binning, patch construction, provenance-preserving batches, optional count normalization
- `predictive_circuit_coding/models/`: predictive encoder, predictive/reconstruction heads, optional per-patch population tokens
- `predictive_circuit_coding/objectives/`: predictive continuation objective and reconstruction objective
- `predictive_circuit_coding/training/`: typed configs, training loop, checkpoint and summary writing
- `predictive_circuit_coding/decoding/`: frozen-token extraction, labels, probes, geometry transforms
- `predictive_circuit_coding/discovery/`: candidate selection, clustering, stability, reports
- `predictive_circuit_coding/benchmarks/`: refinement discovery matrices and notebook pipeline orchestration
- `predictive_circuit_coding/validation/`: conservative validation checks for discovery artifacts

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

Short ablation configs use `extends` to inherit from the refined core config and override one axis.

## Refinement Axes

- Reconstruction: `reconstruction_weight: 0.05` in core, `0.0` in the reconstruction-free ablation
- Reconstruction target mode: `window_zscore` in the core config
- Token geometry: model token L2 normalization plus post-hoc token normalization arm
- Count normalization: optional train-split `log1p_train_zscore`
- Pooling: mean tokens by default, optional per-patch population CLS tokens
- Discovery geometry: raw, token-normalized, probe-weighted, and oracle-aligned diagnostic arms

## Notes For Contributors

Keep notebooks thin. Core logic belongs in package modules.

Do not add broad comparison arms back into the default path until the refined idea is stable.

Any artifact schema change should update `documents/artifact_contracts.md` and tests in the same change.
