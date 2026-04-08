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
pcc-train --config configs/pcc/predictive_circuit_coding_base.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml
pcc-evaluate --config configs/pcc/predictive_circuit_coding_base.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml --checkpoint artifacts/checkpoints/pcc_best.pt --split valid
pcc-discover --config configs/pcc/predictive_circuit_coding_base.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml --checkpoint artifacts/checkpoints/pcc_best.pt --split discovery
pcc-validate --config configs/pcc/predictive_circuit_coding_base.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml --checkpoint artifacts/checkpoints/pcc_best.pt --discovery-artifact artifacts/checkpoints/pcc_best_discovery_discovery.json
pcc-preview-notebook-ui --output-root artifacts/notebook_ui_preview --force
```

Each Stage 5-7 command writes:

- its main artifact
- a JSON run-manifest sidecar that records the config paths, split names, input artifacts, and output artifacts used for that run
- `pcc-discover` also writes a decode-coverage summary JSON plus companion cluster summary JSON and CSV outputs for easier inspection

## Dataset Selection

The canonical local dataset can contain the full processed Allen session store. Normal Colab runs are now notebook-first:

- the training notebook defines the subset with simple scalars like `EXPERIENCE_LEVEL`, `MAX_SESSIONS`, and split fractions
- the notebook writes an artifact-local runtime subset bundle under `artifacts/runtime_subset/`
- the discovery notebook restores a selected training `run_id`, reuses that exact saved subset, and exposes `DECODE_TYPE` plus focused discovery/validation overrides in one config cell

You can:

- process the full dataset once
- rebuild the rich session catalog cheaply
- select notebook subsets by metadata like `experience_level` and `max_sessions`
- recompute split manifests for the selected subset without recreating `.h5` files

## Notebooks

The repo ends with two Colab notebooks:

- [train_predictive_circuit_coding_colab.ipynb](/C:/Users/Jacob%20Poschl/Desktop/population-dynamics/notebooks/train_predictive_circuit_coding_colab.ipynb)
- [discover_validate_inspect_colab.ipynb](/C:/Users/Jacob%20Poschl/Desktop/population-dynamics/notebooks/discover_validate_inspect_colab.ipynb)
- [diagnose_representation_motifs_colab.ipynb](/C:/Users/Jacob%20Poschl/Desktop/population-dynamics/notebooks/diagnose_representation_motifs_colab.ipynb)

They are intentionally thin and call the same CLI surface listed above.

- the training notebook owns subset choice and split fractions
- the training notebook creates a fresh `run_id` and exports to `pcc_colab_outputs/<run_id>/run_1/train/`
- the discovery notebook restores `TRAINING_RUN_ID` or the latest exported training run, then writes task-specific attempts under `pcc_colab_outputs/<run_id>/run_1/discovery/<decode_type>__<timestamp>/`
- the diagnostics notebook restores the same training run and writes grouped multi-experiment outputs under `pcc_colab_outputs/<run_id>/run_1/diagnostics/<timestamp>/`
- discovery supports both capped `sequential` planning and `label_balanced` planning with explicit `max_batches`, `search_max_batches`, `min_positive_windows`, and `negative_to_positive_ratio` controls
- major Allen decode targets default to event-local onset labeling rather than broad overlap labeling
- discovery candidate selection is session-balanced by default via `discovery.candidate_session_balance_fraction` so one session does not dominate the top-k motif pool; set it to `1.0` to restore pure global top-k scoring for comparisons
- discovery artifacts and cluster summaries now include raw probe-score diagnostics so you can inspect sign direction separately from the final contrastive ranking score
- discovery artifacts mark probe metrics as `fit_selected_windows` provenance rather than held-out evidence
- validation recomputes real-label discovery metrics on the validation run itself, checks artifact checkpoint/target-label provenance, reports a real baseline-sensitivity comparison, and records whether the reported metrics are sampled or full-split

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
- if a notebook subset should be used, change the subset scalars in the training notebook and rerun from the top
- if a resume checkpoint is missing, clear `training.resume_checkpoint` or point it at a real checkpoint
- if a checkpoint dataset mismatch is reported, use a checkpoint produced from the same dataset/config family
- if `training_summary.json` looks inconsistent with the latest epoch, remember it now intentionally describes the selected best checkpoint, not the latest completed epoch

Discovery problems:

- if discovery fails because the split does not provide both classes for the chosen decode target, inspect the emitted `*_decode_coverage.json` summary and rerun training with a larger or more suitable notebook subset
- if no candidate tokens are selected, lower `discovery.min_candidate_score` or increase `discovery.max_batches`
- if one session still dominates the discovered candidates, lower `discovery.candidate_session_balance_fraction`; set it to `1.0` only when you intentionally want the old global top-k behavior
- if clustering produces no motif clusters, reduce `discovery.min_cluster_size` or expand the discovery subset; singleton clusters are now rejected up front, so `min_cluster_size` must be at least `2`
- if `TRAINING_RUN_ID` points at a missing export, clear it to use the latest exported training run or rerun the training notebook so it writes a fresh `run_id`

Validation problems:

- if held-out motif similarity ROC-AUC / PR-AUC are weak, the discovered centroids are not separating positive vs negative test windows well on untouched data
- if the discovery artifact dataset, checkpoint path, or decode target does not match the validation inputs, rerun with aligned artifacts

## Verification

For repo-level validation:

```bash
.venv\Scripts\python.exe -m pytest -q
```
