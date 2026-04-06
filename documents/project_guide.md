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
- stability estimation
- cluster-level reporting

`predictive_circuit_coding/evaluation/`

- held-out evaluation on a chosen split
- metric aggregation

`predictive_circuit_coding/validation/`

- downstream falsification-style checks
- label shuffle control
- recurrence checks
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

- thin Colab orchestration notebooks
- no core logic should live here

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
- dataset selection filters and subset split recomputation
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

### Preparation config vs experiment config

Preparation config answers:

- where is the data
- how do we prepare it
- how do we split it

Experiment config answers:

- how do we batch windows
- which subset of the canonical processed dataset should be used for this run
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

The repo now treats the full processed session store as canonical. The normal Colab workflow is notebook-first:

- the training notebook defines the subset with simple scalars such as `EXPERIENCE_LEVEL`, `MAX_SESSIONS`, and split fractions
- the notebook materializes the runtime subset automatically against the canonical processed store
- the discovery notebook reuses that saved runtime config and exact subset, then only overrides the decode target

The lower-level `dataset_selection` block in `configs/pcc/predictive_circuit_coding_base.yaml` is still supported for compatibility and scripting, but it is no longer the primary notebook UX.

Selection supports:

- exact inclusion via `session_ids`, `subject_ids`, and corresponding `*_file` options
- metadata filtering via fields like `experience_levels`, `session_types`, `image_sets`, `project_codes`, `session_numbers`, and `brain_regions_any`
- numeric filtering via `n_units`, `trial_count`, and `duration_s`
- exclusion lists for session and subject IDs

When a subset is active:

- the repo writes `selected_session_catalog.json`
- the repo recomputes a `selected_split_manifest.json`
- the repo writes derived `torch_brain_selected_*.yaml` files
- Stage 5-7 commands consume those derived artifacts automatically

You can still materialize the subset explicitly before training:

```bash
pcc-prepare-data materialize-runtime-selection --config configs/pcc/predictive_circuit_coding_base.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml
```

### 4. Colab training

Purpose:

- train the predictive encoder and save resumable checkpoints

Input:

- processed bundle on Drive
- experiment config
- training notebook or CLI

Notebook:

- `notebooks/train_predictive_circuit_coding_colab.ipynb`

CLI:

```bash
pcc-train --config configs/pcc/predictive_circuit_coding_base.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml
```

Outputs:

- checkpoint `.pt`
- training summary JSON
- training run-manifest sidecar
- if `dataset_selection` is active, the run-manifest also records the derived selected split manifest and selected session catalog paths
- the training notebook also saves the realized runtime experiment config inside `artifacts/` so discovery can reuse the exact subset later

### 5. Colab evaluation

Purpose:

- score a checkpoint on a held-out split and compare against predictive baselines

CLI:

```bash
pcc-evaluate --config configs/pcc/predictive_circuit_coding_base.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml --checkpoint artifacts/checkpoints/pcc_best.pt --split valid
```

Outputs:

- evaluation summary JSON
- evaluation run-manifest sidecar

### 6. Colab discovery

Purpose:

- scan the requested split with the same fixed-window geometry used for discovery
- label every scanned window for the chosen decode target
- build a deterministic balanced positive/negative discovery set when `sampling_strategy == label_balanced`
- write a decode-coverage summary before probe fitting
- extract frozen tokens
- fit the additive probe
- score candidates
- cluster candidates
- write human-readable cluster summaries

Notebook:

- `notebooks/discover_validate_inspect_colab.ipynb`

CLI:

```bash
pcc-discover --config configs/pcc/predictive_circuit_coding_base.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml --checkpoint artifacts/checkpoints/pcc_best.pt --split discovery
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

CLI:

```bash
pcc-validate --config configs/pcc/predictive_circuit_coding_base.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml --checkpoint artifacts/checkpoints/pcc_best.pt --discovery-artifact artifacts/checkpoints/pcc_best_discovery_discovery.json
```

Outputs:

- validation summary JSON
- validation summary CSV
- validation run-manifest sidecar

## Notebook Role

The repo intentionally ends with two notebooks only.

### Training notebook

`notebooks/train_predictive_circuit_coding_colab.ipynb`

Responsibilities:

- mount Drive
- install the repo and notebook extras
- run preflight checks
- define the subset with simple notebook scalars such as `EXPERIENCE_LEVEL`, `MAX_SESSIONS`, and split fractions
- materialize the runtime subset automatically
- launch training
- launch evaluation
- surface stage boundaries, elapsed time, checkpoint reminders, and realized split counts

### Discovery and validation notebook

`notebooks/discover_validate_inspect_colab.ipynb`

Responsibilities:

- reuse the saved training runtime config and subset
- select a decode target
- run discovery
- run validation
- inspect decode coverage, cluster summaries, and validation outputs directly in notebook tables

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

3. evaluation summary  
   produced by held-out evaluation

4. discovery decode-coverage summary  
   produced by discovery, records how many positive and negative windows were found and selected for the chosen decode target

5. discovery artifact  
   produced by discovery, contains token-level candidates and cluster assignments

6. cluster summary JSON / CSV  
   produced by discovery, gives human-readable cluster-level summaries

7. validation JSON / CSV  
   produced by validation, records shuffled-label and recurrence-style checks

8. run-manifest sidecars  
   produced by all Stage 5-7 CLIs so a user can reconstruct which config, checkpoint, split, and outputs belonged to the run

9. selected session catalogs and selected split manifests  
   produced only when `dataset_selection` is active so subset runs remain reproducible without touching canonical `.h5` files

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
