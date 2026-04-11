# Predictive Circuit Coding

This repository is an Allen-first Neuropixels pipeline for predictive population-dynamics modeling. The workflow is intentionally split across two compute surfaces:

- local CPU for Allen data preparation, split planning, and inspection
- Google Colab A100 for training, evaluation, discovery, and validation

The current repo is Stage 7 complete: it supports `pcc-prepare-data`, `pcc-train`, `pcc-evaluate`, `pcc-discover`, `pcc-validate`, and `pcc-benchmark`, plus one supported Colab pipeline notebook that sits on top of the CLI/library surface.

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
pcc-train --config configs/pcc/predictive_circuit_coding_full.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml
pcc-evaluate --config configs/pcc/predictive_circuit_coding_full.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml --checkpoint artifacts/checkpoints/pcc_full_best.pt --split valid
pcc-discover --config configs/pcc/predictive_circuit_coding_full.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml --checkpoint artifacts/checkpoints/pcc_full_best.pt --split discovery
pcc-validate --config configs/pcc/predictive_circuit_coding_full.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml --checkpoint artifacts/checkpoints/pcc_full_best.pt --discovery-artifact artifacts/checkpoints/pcc_full_best_discovery_discovery.json
pcc-benchmark --config configs/pcc/predictive_circuit_coding_full.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml --checkpoint artifacts/checkpoints/pcc_full_best.pt --output-root artifacts/benchmarks
pcc-preview-notebook-ui --output-root artifacts/notebook_ui_preview --force
pcc-verify-full-run --pipeline-config configs/pcc/pipeline_cross_session_aug_full.yaml --output-root artifacts/full_run_verification/cross_session_aug_full
```

Each Stage 5-7 command writes:

- its main artifact
- a JSON run-manifest sidecar that records the config paths, split names, input artifacts, and output artifacts used for that run
- `pcc-discover` also writes a decode-coverage summary JSON plus companion cluster summary JSON and CSV outputs for easier inspection

## Dataset Selection

The canonical local dataset can contain the full processed Allen session store. Normal Colab runs are now config-first:

- choose a repo-native pipeline config such as `configs/pcc/pipeline_debug.yaml` or `configs/pcc/pipeline_full.yaml`
- each pipeline config points at an experiment config such as `predictive_circuit_coding_debug.yaml` or `predictive_circuit_coding_full.yaml`
- the unified notebook loads that config, writes stage outputs under one run root, and can resume a selected `run_id`

You can:

- process the full dataset once
- rebuild the rich session catalog cheaply
- define debug and full experiment panels directly in repo configs
- recompute split manifests for the selected subset without recreating `.h5` files

## Notebooks

The supported Colab entrypoint is:

- [run_predictive_circuit_coding_pipeline_colab.ipynb](/C:/Users/Jacob%20Poschl/Desktop/population-dynamics/notebooks/run_predictive_circuit_coding_pipeline_colab.ipynb)

It is intentionally thin and calls the same CLI/library surface listed above.

- the notebook asks for a single repo config path plus an optional `PIPELINE_RUN_ID`
- it restores or creates one `run_id`, then writes stage outputs under `pcc_colab_outputs/<run_id>/run_1/`
- it runs training, evaluation, representation benchmarks, motif benchmarks, and optional appendix diagnostics as explicit stages
- it writes compact stage state and manifest files under `pipeline/` so interrupted runs can resume without wasting storage
- repo-native config presets now include:
  `configs/pcc/pipeline_debug.yaml`, `configs/pcc/pipeline_full.yaml`,
  `configs/pcc/pipeline_cross_session_aug_debug.yaml`, `configs/pcc/pipeline_cross_session_aug_full.yaml`,
  `configs/pcc/predictive_circuit_coding_debug.yaml`, `configs/pcc/predictive_circuit_coding_full.yaml`,
  `configs/pcc/predictive_circuit_coding_cross_session_aug_debug.yaml`, and
  `configs/pcc/predictive_circuit_coding_cross_session_aug_full.yaml`
- representation benchmarks are now crossed by feature family x geometry mode, with raw versus whitened comparisons made explicit across count-based, PCA, and frozen-encoder feature families
- the cross-session auxiliary-loss experiment ships as a parallel config family rather than replacing the baseline full run; compare baseline and augmented runs across separate `run_id`s using the exported `training_variant_name` column in benchmark summaries
- run `pcc-verify-full-run` before launching a claim-facing Colab run; it validates the full config, checks local real-data label coverage, and blocks tasks that would otherwise degrade into NaN summaries
- the current claim-facing full task panel is `stimulus_change` plus `trials_go`; `stimulus_omitted` is excluded from the full configs because the current Familiar G split has no positive omitted windows in the benchmark coverage scan
- stimulus image-name decoding is supported as one-vs-rest tasks using `stimulus_presentations.image_name`; full pipeline configs set `image_identity_appendix: true` and `image_target_names: auto`, so the verifier will discover image names when the prepared sessions contain that interval and block clearly when they do not
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
- if a different subset or budget should be used, switch to another repo config or edit the relevant experiment / pipeline YAML
- if a resume checkpoint is missing, clear `training.resume_checkpoint` or point it at a real checkpoint
- if a checkpoint dataset mismatch is reported, use a checkpoint produced from the same dataset/config family
- if `training_summary.json` looks inconsistent with the latest epoch, remember it now intentionally describes the selected best checkpoint, not the latest completed epoch
- if you are running the cross-session auxiliary-loss variant, check `cross_session_geometry_monitor.json/csv` next to the training summary before spending time on a full benchmark run; the debug acceptance gate is a nonzero realized augmentation fraction plus a meaningful drop in raw session-neighbor enrichment without collapsing label enrichment
- if `pcc-verify-full-run` blocks a task, do not launch the full notebook with that task still enabled; either remove it from the claim-facing pipeline config or make a separate rare-event config that passes coverage

Discovery problems:

- if discovery fails because the split does not provide both classes for the chosen decode target, inspect the emitted `*_decode_coverage.json` summary and rerun training with a larger or more suitable notebook subset
- if no candidate tokens are selected, lower `discovery.min_candidate_score` or increase `discovery.max_batches`
- if one session still dominates the discovered candidates, lower `discovery.candidate_session_balance_fraction`; set it to `1.0` only when you intentionally want the old global top-k behavior
- if clustering produces no motif clusters, reduce `discovery.min_cluster_size` or expand the discovery subset; singleton clusters are now rejected up front, so `min_cluster_size` must be at least `2`
- if `PIPELINE_RUN_ID` points at a missing export, clear it to use the latest compatible exported run or rerun the unified notebook so it writes a fresh `run_id`

Validation problems:

- if held-out motif similarity ROC-AUC / PR-AUC are weak, the discovered centroids are not separating positive vs negative test windows well on untouched data
- if the discovery artifact dataset, checkpoint path, or decode target does not match the validation inputs, rerun with aligned artifacts

## Verification

For repo-level validation:

```bash
.venv\Scripts\python.exe -m pytest -q
```
