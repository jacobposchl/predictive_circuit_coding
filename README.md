# predictive_circuit_coding

Allen-first Neuropixels research code for testing predictive circuit coding refinements on neural population recordings.

The default project path is refinement-centered:

1. local data preparation with `pcc-prepare-data` subcommands
2. `pcc-train`
3. `pcc-refine`
4. `pcc-validate`
5. optional `pcc-evaluate` for predictive sanity checks

`pcc-discover` remains supported as a lower-level single-target discovery CLI, but `pcc-refine` is the canonical discovery surface for this repo.

## Install

```bash
python -m pip install -e ".[dev]"
```

## Canonical Commands

Data preparation is a subcommand group, not a single flat command:

```bash
pcc-prepare-data init-workspace --config configs/pcc/allen_visual_behavior_neuropixels_local.yaml

pcc-prepare-data prepare-allen-visual-behavior-neuropixels --config configs/pcc/allen_visual_behavior_neuropixels_local.yaml
```

If processed `.h5` sessions already exist and you only need manifests, splits, and upload metadata:

```bash
pcc-prepare-data build-session-catalog --config configs/pcc/allen_visual_behavior_neuropixels_local.yaml
```

Core refinement workflow:

```bash
pcc-train --config configs/pcc/predictive_circuit_coding_refined_debug.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml

pcc-refine --config configs/pcc/predictive_circuit_coding_refined_debug.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml --checkpoint artifacts/checkpoints/pcc_refined_debug_best.pt --output-root artifacts/refinement_run

pcc-validate --config configs/pcc/predictive_circuit_coding_refined_debug.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml --checkpoint artifacts/checkpoints/pcc_refined_debug_best.pt --discovery-artifact artifacts/refinement_run/refinement/stimulus_change/encoder_raw/discovery_artifact.json --output-json artifacts/validation_summary.json --output-csv artifacts/validation_summary.csv

pcc-evaluate --config configs/pcc/predictive_circuit_coding_refined_debug.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml --checkpoint artifacts/checkpoints/pcc_refined_debug_best.pt --split test
```

Pipeline and preflight helpers:

```bash
pcc-run-pipeline --pipeline-config configs/pcc/pipeline_refined_debug.yaml

pcc-verify-refinement --pipeline-config configs/pcc/pipeline_refined_debug.yaml --output-root artifacts/refinement_verification
```

## CLI Notes

- `pcc-prepare-data materialize-runtime-selection --config <experiment.yaml> --data-config <prep.yaml>` writes a filtered runtime dataset view when `dataset_selection` is active in the experiment config.
- `pcc-run-pipeline` writes per-run outputs under `artifacts/pipeline_runs/<run_id>/run_1/`.
- `pcc-preview-notebook-ui` is a notebook/debugging utility, not part of the main workflow.

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

- `configs/pcc/pipeline_refined_debug.yaml`: benchmarks `stimulus_change`
- `configs/pcc/pipeline_refined_full.yaml`: benchmarks `stimulus_change` and `trials_go`

## Refinement Arms

`pcc-refine` runs trained-encoder discovery arms only:

- `encoder_raw`
- `encoder_token_normalized`
- `encoder_probe_weighted`
- `encoder_aligned_oracle`

`encoder_aligned_oracle` is diagnostic and writes `claim_safe: false`.

## Key Outputs

- `pcc-train`: best checkpoint, training summary JSON, training history JSON/CSV, command sidecar manifest
- `pcc-refine`: per-task/per-arm discovery artifacts, cluster summaries, transform summaries, refinement summary JSON/CSV, final project summary JSON/CSV, command sidecar manifest
- `pcc-validate`: validation summary JSON/CSV and command sidecar manifest
- `pcc-run-pipeline`: pipeline manifest/state plus staged train, evaluation, refinement, and report outputs

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
