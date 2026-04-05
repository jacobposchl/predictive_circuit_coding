# Predictive Circuit Coding

This repository is an Allen-first Neuropixels pipeline for predictive population-dynamics modeling. The workflow is intentionally split across two compute surfaces:

- local CPU for Allen data preparation, split planning, and inspection
- Google Colab A100 for training, evaluation, discovery, and validation

The current repo is Stage 7 complete: it supports `pcc-prepare-data`, `pcc-train`, `pcc-evaluate`, `pcc-discover`, and `pcc-validate`, plus two thin Colab notebooks that sit on top of the CLI/library surface.

If you want to preview the notebook-style stage flow and artifact summaries without touching Allen data or Colab, run:

```bash
pcc-preview-notebook-ui --output-root artifacts/notebook_ui_preview --force
```

The main human-facing guide for how the repo is organized and how to run the experiment is:

- [project_guide.md](/C:/Users/Jacob%20Poschl/Desktop/population-dynamics/documents/project_guide.md)

## Main Commands

```bash
pcc-prepare-data prepare-allen-visual-behavior-neuropixels --config configs/pcc/allen_visual_behavior_neuropixels_local.yaml
pcc-prepare-data process-allen-visual-behavior-neuropixels --config configs/pcc/allen_visual_behavior_neuropixels_local.yaml
pcc-prepare-data build-session-catalog --config configs/pcc/allen_visual_behavior_neuropixels_local.yaml
pcc-prepare-data materialize-runtime-selection --config configs/pcc/predictive_circuit_coding_base.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml
pcc-train --config configs/pcc/predictive_circuit_coding_base.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml
pcc-evaluate --config configs/pcc/predictive_circuit_coding_base.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml --checkpoint artifacts/checkpoints/pcc_best.pt --split valid
pcc-discover --config configs/pcc/predictive_circuit_coding_base.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml --checkpoint artifacts/checkpoints/pcc_best.pt --split discovery
pcc-validate --config configs/pcc/predictive_circuit_coding_base.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml --checkpoint artifacts/checkpoints/pcc_best.pt --discovery-artifact artifacts/checkpoints/pcc_best_discovery_discovery.json
pcc-preview-notebook-ui --output-root artifacts/notebook_ui_preview --force
```

Each Stage 5-7 command writes:

- its main artifact
- a JSON run-manifest sidecar that records the config paths, split names, input artifacts, and output artifacts used for that run
- `pcc-discover` also writes companion cluster summary JSON and CSV outputs for easier inspection

## Dataset Selection

The canonical local dataset can contain the full processed Allen session store. Subset experiments are now handled at runtime through `dataset_selection` in [predictive_circuit_coding_base.yaml](/C:/Users/Jacob%20Poschl/Desktop/population-dynamics/configs/pcc/predictive_circuit_coding_base.yaml), not by reprocessing raw data.

You can:

- process the full dataset once
- rebuild the rich session catalog cheaply
- select subsets by metadata like `experience_level`, `session_type`, `image_set`, `subject_id`, `brain_regions_any`, or explicit `session_ids_file`
- recompute split manifests for the selected subset without recreating `.h5` files

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
- if metadata or subset behavior changes, rerun `pcc-prepare-data build-session-catalog` instead of reprocessing `.h5` files

Training problems:

- if `split_manifest.json` is missing, rerun local prep first
- if `session_catalog.json` is missing, run `pcc-prepare-data build-session-catalog`
- if a runtime subset should be used, edit `dataset_selection` in the experiment config and optionally run `pcc-prepare-data materialize-runtime-selection` before launching Colab
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
