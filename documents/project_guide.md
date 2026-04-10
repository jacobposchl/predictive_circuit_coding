# Project Guide

## Project Overview

`predictive_circuit_coding` is an Allen-first Neuropixels project for learning predictive population-dynamics representations and using them to identify candidate task-relevant motifs in neural recordings.

The scientific goal is to sit between two common extremes:

- purely descriptive neural representations that smooth away trial-to-trial computation
- directly supervised decoders that recover task signals but often weaken circuit-level interpretability

The project instead learns a task-agnostic latent space from neural dynamics, then applies downstream decoding and clustering on frozen tokens to identify candidate motifs associated with a variable such as `stimulus_change`.

The project is intentionally conservative in what it claims. It aims to find candidate motifs that are:

- predictive
- recurring
- inspectable in raw neural terms

It does not claim automatic causal discovery from observation alone.

## Core Idea

### Dynamics-first learning

The encoder is trained on predictive dynamics and reconstruction anchoring. Task labels are not used in the encoder objective.

### Frozen-token interpretation

After training, the encoder is frozen. Task-specific analysis happens downstream by:

- extracting patch-level tokens
- fitting an additive decoder
- scoring token contributions
- selecting high-relevance candidates
- clustering those candidates in frozen latent space

Candidate selection is intentionally session-aware. The scorer ranks tokens by decoder relevance, but discovery now balances the top-k pool across sessions before any backfill so a single session cannot trivially swamp motif discovery. The corresponding experiment knob is `discovery.candidate_session_balance_fraction`; `1.0` restores the old pure global top-k behavior.

Discovery artifacts also retain score diagnostics for interpretation: the final `score` is the contrastive selection score used for ranking, while `raw_probe_score` and `negative_background_score` let you inspect whether a candidate was selected because of a positive-direction probe effect or because it merely exceeded a matched negative-window background.

### Provenance is required

Every token and every discovered candidate must remain traceable back to:

- session
- subject
- unit
- brain region
- probe depth
- window start and end
- patch index and patch timing
- aligned event structure when available


## Repo Map

### Core package

`predictive_circuit_coding/data/`

- data preparation contracts
- workspace layout
- rich session catalog, manifests, selection, and split logic
- processed session loading and session scanning
- `brainsets` runner integration
- `temporaldata` session writing

`predictive_circuit_coding/windowing/`

- `torch_brain` dataset config generation
- split-aware dataset bundle creation
- sequential and random fixed-window sampler helpers
- window inspection utilities

`predictive_circuit_coding/tokenization/`

- convert sampled windows into canonical training batches
- spike binning
- temporal patch construction
- provenance-preserving collation

`predictive_circuit_coding/models/`

- spatiotemporal encoder
- transformer blocks
- predictive head
- reconstruction head

`predictive_circuit_coding/objectives/`

- predictive target builders
- continuation baselines
- predictive and reconstruction losses
- optional cross-session region-rate auxiliary targets and donor-cache logic for the parallel augmented-training experiment

`predictive_circuit_coding/training/`

- typed experiment config
- runtime contracts
- training loop
- checkpoint and summary writing
- logging helpers

`predictive_circuit_coding/decoding/`

- frozen-token extraction
- label extraction
- additive probe fitting
- token scoring

`predictive_circuit_coding/discovery/`

- candidate selection
- clustering
- cluster-quality estimation
- cluster-level reporting

`predictive_circuit_coding/evaluation/`

- held-out evaluation on a chosen split
- metric aggregation

`predictive_circuit_coding/validation/`

- downstream falsification-style checks
- label shuffle control
- held-out motif-similarity discrimination
- provenance-integrity checks

`predictive_circuit_coding/utils/`

- dependency checks
- console helpers
- notebook-facing timing and preflight utilities

### Supporting repo surfaces

`configs/pcc/`

- preparation config for Allen data setup
- experiment config for model training and downstream analysis

`notebooks/`

- thin Colab orchestration only
- no core logic should live here
- the supported notebook is the unified stage-resume runner:
  `notebooks/run_predictive_circuit_coding_pipeline_colab.ipynb`
- legacy training / discovery / diagnostics notebooks may remain for reference, but they are no longer the primary experiment surface

`documents/`

- this guide plus artifact contracts

`tests/`

- synthetic unit tests
- workflow-level CLI tests
- hardening tests

`brainsets_local_pipelines/`

- repo-local Allen Visual Behavior Neuropixels pipeline used by `brainsets.runner`

`environments/`

- dedicated Allen prep environment materials

`scripts/`

- local environment setup scripts

## Environments And Compute Split

The project is split across two execution environments.

### Dedicated Allen prep environment

Use this for:

- AllenSDK / `brainsets` preparation
- raw dataset download and processing
- generation of prepared `.h5` sessions

Why it exists:

- AllenSDK and neural-data dependencies are more fragile than the main runtime stack
- local preparation is CPU-heavy but not GPU-dependent

### Main local environment

Use this for:

- normal package development
- tests
- synthetic workflow checks
- local window inspection
- notebook UI preview on synthetic data via `pcc-preview-notebook-ui`

### Colab A100

Use this for:

- training
- evaluation
- discovery
- validation
- artifact inspection during experiment runs

### Responsibility split

Local machine:

- prepare Allen data
- validate manifests and splits
- inspect windows before training
- upload processed data to Drive

Colab:

- install the repo
- load processed data from Drive
- run training and evaluation
- run discovery and validation
- inspect output artifacts

## Config Surfaces

There are two main config files a collaborator should know first.

### `configs/pcc/allen_visual_behavior_neuropixels_local.yaml`

This is the preparation config.

It controls:

- dataset/workspace paths
- raw, prepared, manifests, and splits directories
- split planning defaults
- `brainsets` pipeline wiring
- AllenSDK cache settings
- unit filtering defaults

Use this config when running:

- `pcc-prepare-data ...`

### `configs/pcc/predictive_circuit_coding_base.yaml`

This is the experiment config.

It controls:

- runtime data parameters such as bin width and patching
- model hyperparameters
- objective settings
- optimization settings
- training schedule
- execution settings
- evaluation settings
- discovery settings
- artifact output locations

Use this config when running:

- `pcc-train`
- `pcc-evaluate`
- `pcc-discover`
- `pcc-validate`

Parallel experiment configs also exist for the auxiliary-loss variant:

- `configs/pcc/predictive_circuit_coding_cross_session_aug_debug.yaml`
- `configs/pcc/predictive_circuit_coding_cross_session_aug_full.yaml`

Those configs keep the baseline encoder objective intact and add a cross-session region-rate auxiliary loss with donor-cache matching plus periodic geometry monitoring.

### Preparation config vs experiment config

Preparation config answers:

- where is the data
- how do we prepare it
- how do we split it

Experiment config answers:

- how do we batch windows
- how do we train the model
- how do we evaluate and discover motifs
- where do runtime artifacts go

## End-To-End Experiment Pipeline

This is the standard experiment path a collaborator should follow.

### 1. Local preparation

Purpose:

- process raw Allen sessions once and build the canonical local processed dataset view

Input:

- dedicated Allen prep environment
- `configs/pcc/allen_visual_behavior_neuropixels_local.yaml`

CLI:

```bash
pcc-prepare-data prepare-allen-visual-behavior-neuropixels --config configs/pcc/allen_visual_behavior_neuropixels_local.yaml
```

Outputs:

- prepared `.h5` sessions under `data/allen_visual_behavior_neuropixels/prepared/`
- `session_catalog.json`
- `session_catalog.csv`
- `session_manifest.json`
- `split_manifest.json`
- split-specific `torch_brain_*.yaml` files
- upload bundle manifest

The same workflow is also available as two explicit phases:

```bash
pcc-prepare-data process-allen-visual-behavior-neuropixels --config configs/pcc/allen_visual_behavior_neuropixels_local.yaml
pcc-prepare-data build-session-catalog --config configs/pcc/allen_visual_behavior_neuropixels_local.yaml
```

Use the split-phase commands when the processed `.h5` files are already correct and only metadata, catalogs, or split logic need to be rebuilt.

### 2. Local inspection

Purpose:

- confirm splits and sampling windows look sane before using expensive Colab compute

Input:

- prepared bundle from step 1

CLI:

```bash
pcc-prepare-data inspect-windows --config configs/pcc/allen_visual_behavior_neuropixels_local.yaml --split train --window-length 10.0 --step 10.0
```

Outputs:

- console inspection only

### 3. Upload to Drive

Purpose:

- make the processed bundle available to Colab

Input:

- `data/allen_visual_behavior_neuropixels/`

Outputs:

- Drive copy of processed sessions and support manifests

### Runtime subset selection

Subset experiments do not require reprocessing raw Allen sessions.

The repo now treats the full processed session store as canonical. The normal Colab workflow is config-first:

- the unified pipeline notebook takes a single repo-native pipeline config path plus an optional `PIPELINE_RUN_ID`
- the pipeline config points at a repo-native experiment config that defines the scientific subset and budgets
- the notebook restores or creates a single `run_id` and reuses completed config-matched stages automatically

The runtime subset bundle includes:

- `selected_session_catalog.json`
- `selected_session_catalog.csv`
- `selected_split_manifest.json`
- split-specific `torch_brain_runtime_*.yaml` files
- notebook profile and runtime metadata

Stage 5-7 commands consume those artifact-local subset assets automatically when `runtime_subset` is present in the runtime experiment config.

### 4. Colab training

Purpose:

- train the predictive encoder and save resumable checkpoints

Input:

- processed bundle on Drive
- experiment config
- unified notebook or CLI

Notebook:

- `notebooks/run_predictive_circuit_coding_pipeline_colab.ipynb`

CLI:

```bash
pcc-train --config configs/pcc/predictive_circuit_coding_full.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml
```

Outputs:

- checkpoint `.pt`
- training summary JSON
- optional `cross_session_geometry_monitor.json` and `cross_session_geometry_monitor.csv` when the auxiliary-loss training variant is enabled
- training run-manifest sidecar
- if a notebook runtime subset is active, the run-manifest records the artifact-local split manifest and session catalog paths
- the unified notebook also saves the realized runtime experiment config inside the run root so downstream stages can reuse the exact subset later

For the auxiliary-loss experiment, the train stage also records:

- `training_variant_name`
- `cross_session_aug_enabled`
- aggregated auxiliary-loss metrics in `training_summary.json`
- periodic raw-latent geometry diagnostics under the geometry-monitor artifact

### 5. Colab evaluation

Purpose:

- score a checkpoint on a held-out split and compare against predictive baselines

CLI:

```bash
pcc-evaluate --config configs/pcc/predictive_circuit_coding_full.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml --checkpoint artifacts/checkpoints/pcc_full_best.pt --split valid
```

Outputs:

- evaluation summary JSON
- evaluation run-manifest sidecar
- stage state under `pipeline/pipeline_state.json` when driven from the unified notebook

### 5b. Benchmark matrix

Purpose:

- run the final crossed experiment surface for the project claims
- compare feature family and geometry mode separately instead of collapsing them into one choice
- write compact benchmark summaries and per-arm artifacts under one run root

CLI:

```bash
pcc-benchmark --config configs/pcc/predictive_circuit_coding_full.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml --checkpoint artifacts/checkpoints/pcc_full_best.pt --output-root artifacts/benchmarks
```

Primary representation benchmark arms:

- `count_patch_mean_raw`
- `count_patch_mean_whitened`
- `count_patch_mean_pca_raw`
- `count_patch_mean_pca_whitened`
- `encoder_raw`
- `encoder_whitened`

Primary motif benchmark arms:

- `count_patch_mean_pca_raw`
- `count_patch_mean_pca_whitened`
- `encoder_raw`
- `encoder_whitened`

Primary task panel:

- `stimulus_change`
- `trials.go`
- `stimulus_presentations.omitted`

Optional appendix task:

- image identity one-vs-rest, only when `stimulus_presentations.image_name` exists in the prepared dataset

Outputs:

- representation benchmark summary JSON and CSV
- motif benchmark summary JSON and CSV
- final project summary JSON and CSV
- per-task/per-arm artifact folders under `benchmarks/representation/` and `benchmarks/motifs/`

### 6. Colab discovery

Purpose:

- scan the requested split with the same fixed-window geometry used for discovery
- label every scanned window for the chosen decode target
- support a capped `sequential` discovery pass or a `label_balanced` pass with explicit search and selection budgets
- default major Allen decode targets such as `stimulus_change` and `trials.go` to event-local onset labeling instead of broad overlap labeling
- build the selected discovery set according to `max_batches`, `search_max_batches`, `min_positive_windows`, and `negative_to_positive_ratio`
- write a decode-coverage summary before probe fitting
- extract frozen tokens
- fit the additive probe
- score candidates against positive-vs-negative discovery background
- cluster candidates with HDBSCAN in normalized embedding space
- estimate cluster stability with repeated bootstrap reclustering rounds
- write human-readable cluster summaries

Notebook:

- `notebooks/run_predictive_circuit_coding_pipeline_colab.ipynb`

CLI:

```bash
pcc-discover --config configs/pcc/predictive_circuit_coding_full.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml --checkpoint artifacts/checkpoints/pcc_full_best.pt --split discovery
```

Outputs:

- discovery decode-coverage summary JSON
- discovery artifact JSON
- cluster summary JSON
- cluster summary CSV
- discovery run-manifest sidecar

### 7. Colab validation

Purpose:

- run conservative downstream checks on the discovery result
- verify that the supplied checkpoint and decode target match the discovery artifact provenance
- recompute the real-label discovery metrics on the validation run itself instead of trusting artifact-supplied values
- evaluate fixed discovery-fit probe transfer on the untouched test split
- evaluate threshold-free held-out motif-similarity discrimination on the untouched test split
- report a real baseline-sensitivity comparison plus sampled-vs-full-split validation coverage metadata

CLI:

```bash
pcc-validate --config configs/pcc/predictive_circuit_coding_full.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml --checkpoint artifacts/checkpoints/pcc_full_best.pt --discovery-artifact artifacts/checkpoints/pcc_full_best_discovery_discovery.json
```

Outputs:

- validation summary JSON
- validation summary CSV
- validation run-manifest sidecar

## Notebook Role

The repo intentionally ends with one supported Colab notebook.

### Unified pipeline notebook

`notebooks/run_predictive_circuit_coding_pipeline_colab.ipynb`

Responsibilities:

- mount Drive
- install the repo and notebook extras
- run preflight checks
- choose a repo-native pipeline config such as `configs/pcc/pipeline_debug.yaml` or `configs/pcc/pipeline_full.yaml`
- auxiliary-loss runs use the parallel configs `configs/pcc/pipeline_cross_session_aug_debug.yaml` or `configs/pcc/pipeline_cross_session_aug_full.yaml`
- use the referenced experiment config as the source of truth for subset filters, split behavior, and budgets
- restore or create one `run_id`
- run training, evaluation, representation benchmarks, motif benchmarks, and optional appendix diagnostics as explicit stages
- reuse completed stages when config hashes and upstream inputs still match
- export the grouped run to `pcc_colab_outputs/<run_id>/run_1/`
- surface stage boundaries, elapsed time, resume status, realized split counts, and per-task benchmark leaderboards
- print concise “what we can claim / what we cannot claim yet” summaries from the benchmark outputs

### Notebook rules

- notebooks should not duplicate core implementation logic
- notebooks should install first and import repo helpers second
- notebooks should call the CLI or very thin library helpers
- any reusable notebook helper belongs in the package, not inline in multiple notebooks

## Artifacts In Workflow Context

The main artifact sequence is:

1. checkpoint  
   produced by training, used by evaluation, discovery, and validation

2. training summary  
   produced by training, summarizes epoch-level performance

2b. cross-session geometry monitor  
   produced by augmented training, tracks label/session/subject neighbor enrichment in raw encoder space during training

3. evaluation summary  
   produced by held-out evaluation

4. discovery decode-coverage summary  
   produced by discovery, records how many positive and negative windows were found and selected for the chosen decode target

5. representation benchmark summaries  
   produced by the benchmark stage, compare feature family x geometry mode across the task panel and retain `training_variant_name` so baseline and augmented runs can be merged cleanly

6. motif benchmark summaries  
   produced by the benchmark stage, compare motif quality for the selected arm subset

7. pipeline manifest/state  
   produced by the unified notebook runner, records stage status, config hashes, inputs, and outputs for resume behavior

5. discovery artifact  
   produced by discovery, contains token-level candidates and cluster assignments

6. cluster summary JSON / CSV  
   produced by discovery, gives human-readable cluster-level summaries

7. validation JSON / CSV  
   produced by validation, records shuffled-label controls, held-out probe transfer, and held-out motif-similarity discrimination

8. run-manifest sidecars  
   produced by all Stage 5-7 CLIs so a user can reconstruct which config, checkpoint, split, and outputs belonged to the run

9. artifact-local runtime subset bundle  
   produced by the training notebook so subset runs remain reproducible without touching canonical `.h5` files or canonical split directories

Use [artifact_contracts.md](/C:/Users/Jacob%20Poschl/Desktop/population-dynamics/documents/artifact_contracts.md) for exact shapes and required keys.

## Day-To-Day Development Workflow

Typical development loop:

1. reproduce with a synthetic or local test first when possible
2. keep core logic in the package
3. keep notebooks thin
4. update docs and tests when public-surface behavior changes
5. run the test suite before treating a change as complete

Practical habits:

- use the dedicated prep environment for Allen preparation work
- use the main environment for development and tests
- inspect windows locally before burning Colab time
- treat artifact format changes as public API changes
- prefer clearer logging and explicit artifacts over hidden notebook state

## Real-Run Acceptance Checklist

A full successful run should satisfy all of the following:

1. local preparation completes successfully
2. processed sessions exist
3. session and split manifests exist
4. local window inspection succeeds
5. the training notebook installs and runs in Colab
6. a checkpoint is written
7. checkpoint resume works at least once
8. evaluation summary is written
9. discovery artifact plus cluster summaries are written
10. validation JSON and CSV are written
11. generated artifacts are inspectable without manually reading raw JSON in Drive
