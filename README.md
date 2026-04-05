# Predictive Circuit Coding

This repository is an Allen-first Neuropixels pipeline for predictive population-dynamics modeling. The workflow is intentionally split across two compute surfaces:

- local CPU for Allen data preparation, split planning, and inspection
- Google Colab A100 for training, evaluation, discovery, and validation

The current repo is Stage 7 complete: it supports `pcc-prepare-data`, `pcc-train`, `pcc-evaluate`, `pcc-discover`, and `pcc-validate`, plus two thin Colab notebooks that sit on top of the CLI/library surface.

The main human-facing guide for how the repo is organized and how to run the experiment is:

- [project_guide.md](/C:/Users/Jacob%20Poschl/Desktop/population-dynamics/documents/project_guide.md)

## Main Commands

```bash
pcc-prepare-data prepare-allen-visual-behavior-neuropixels --config configs/pcc/allen_visual_behavior_neuropixels_local.yaml
pcc-train --config configs/pcc/predictive_circuit_coding_base.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml
pcc-evaluate --config configs/pcc/predictive_circuit_coding_base.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml --checkpoint artifacts/checkpoints/pcc_best.pt --split valid
pcc-discover --config configs/pcc/predictive_circuit_coding_base.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml --checkpoint artifacts/checkpoints/pcc_best.pt --split discovery
pcc-validate --config configs/pcc/predictive_circuit_coding_base.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml --checkpoint artifacts/checkpoints/pcc_best.pt --discovery-artifact artifacts/checkpoints/pcc_best_discovery_discovery.json
```

Each Stage 5-7 command writes:

- its main artifact
- a JSON run-manifest sidecar that records the config paths, split names, input artifacts, and output artifacts used for that run
- `pcc-discover` also writes companion cluster summary JSON and CSV outputs for easier inspection

## Notebooks

The repo ends with two Colab notebooks:

- [train_predictive_circuit_coding_colab.ipynb](/C:/Users/Jacob%20Poschl/Desktop/population-dynamics/notebooks/train_predictive_circuit_coding_colab.ipynb)
- [discover_validate_inspect_colab.ipynb](/C:/Users/Jacob%20Poschl/Desktop/population-dynamics/notebooks/discover_validate_inspect_colab.ipynb)

They are intentionally thin and call the same CLI surface listed above.

## Documentation

Stage 7 docs live under `documents/`:

- [project_guide.md](/C:/Users/Jacob%20Poschl/Desktop/population-dynamics/documents/project_guide.md)
- [artifact_contracts.md](/C:/Users/Jacob%20Poschl/Desktop/population-dynamics/documents/artifact_contracts.md)

## Install

Main environment:

```bash
pip install -e .
pip install -e ".[dev]"
pip install -e ".[notebook]"
```

Allen prep environment:

- use [environments/allen_visual_behavior_prep/README.md](/C:/Users/Jacob%20Poschl/Desktop/population-dynamics/environments/allen_visual_behavior_prep/README.md)
- use [scripts/setup_allen_visual_behavior_prep_env.ps1](/C:/Users/Jacob%20Poschl/Desktop/population-dynamics/scripts/setup_allen_visual_behavior_prep_env.ps1)

## Troubleshooting

Allen prep problems:

- if `brainsets` or AllenSDK dependencies conflict, use the dedicated Allen prep environment rather than the main training environment
- if no processed sessions are produced, verify the prep command completed and rerun with `--max-sessions 1` first

Training problems:

- if `split_manifest.json` is missing, rerun local prep first
- if a resume checkpoint is missing, clear `training.resume_checkpoint` or point it at a real checkpoint
- if a checkpoint dataset mismatch is reported, use a checkpoint produced from the same dataset/config family

Discovery problems:

- if no positive `stimulus_change` labels are found, increase discovery coverage or adjust the sampled window coverage
- if no candidate tokens are selected, lower `discovery.min_candidate_score` or increase `discovery.max_batches`
- if clustering produces no motif clusters, lower `discovery.cluster_similarity_threshold` or reduce `discovery.min_cluster_size`

Validation problems:

- if no recurrence hits are found, validation still succeeds, but the held-out recurrence story is weak for that run
- if the discovery artifact dataset does not match the checkpoint/config dataset, rerun with aligned artifacts

## Verification

For repo-level validation:

```bash
.venv\Scripts\python.exe -m pytest -q
```
