# predictive_circuit_coding

Allen-first Neuropixels research code for testing predictive circuit coding refinements on neural population recordings.

The current workflow is refinement-centered:

1. `pcc-prepare-data`
2. `pcc-train`
3. `pcc-refine`
4. `pcc-validate`
5. optional `pcc-evaluate` for predictive sanity checks

The project now focuses on making the main trained-encoder idea work before spending compute on broader comparison suites.

## Install

```bash
python -m pip install -e ".[dev]"
```

## Quick Commands

```bash
pcc-prepare-data --config configs/pcc/allen_visual_behavior_neuropixels_local.yaml

pcc-train --config configs/pcc/predictive_circuit_coding_refined_debug.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml

pcc-refine --config configs/pcc/predictive_circuit_coding_refined_debug.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml --checkpoint artifacts/checkpoints/pcc_refined_debug_best.pt --output-root artifacts/refinement

pcc-validate --config configs/pcc/predictive_circuit_coding_refined_debug.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml --checkpoint artifacts/checkpoints/pcc_refined_debug_best.pt --discovery-artifact artifacts/refinement/refinement/stimulus_change/encoder_raw/discovery_artifact.json --output artifacts/validation_summary.json

pcc-run-pipeline --pipeline-config configs/pcc/pipeline_refined_debug.yaml

pcc-verify-refinement --pipeline-config configs/pcc/pipeline_refined_debug.yaml --output-root artifacts/refinement_verification
```

## Refinement Configs

Core configs:

- `configs/pcc/predictive_circuit_coding_refined_debug.yaml`
- `configs/pcc/predictive_circuit_coding_refined_full.yaml`

One-axis ablations:

- `refined_recon000`: reconstruction weight set to `0.0`
- `refined_l2off`: token L2 normalization disabled
- `refined_countnorm`: train-split `log1p` count normalization
- `refined_cls`: per-patch population CLS tokens for pooled discovery features

Pipeline configs:

- `configs/pcc/pipeline_refined_debug.yaml`
- `configs/pcc/pipeline_refined_full.yaml`

## Refinement Arms

`pcc-refine` runs trained-encoder discovery arms only:

- `encoder_raw`
- `encoder_token_normalized`
- `encoder_probe_weighted`
- `encoder_aligned_oracle`

`encoder_aligned_oracle` is diagnostic and writes `claim_safe: false`.

## Compute Split

Local CPU:

- Allen data preparation
- manifest and split planning
- small debug runs and inspection

Google Colab A100:

- full training
- refinement discovery
- validation

Raw Allen cache should stay outside the repo when `allen_sdk.cache_root` is configured. Processed sessions, manifests, splits, checkpoints, and summaries are expected inside the configured project artifact roots.

## Tests

```bash
python -m pytest -q
```
