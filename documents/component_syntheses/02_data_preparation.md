# Data Preparation Synthesis

## What This Subsystem Owns

The data preparation subsystem turns Allen Visual Behavior Neuropixels sessions into repo-local processed sessions, catalogs, manifests, and splits that downstream stages can consume. It owns the boundary between external Allen raw/cache data and internal `temporaldata`/`torch_brain`-compatible artifacts.

There are two related preparation paths:

- Real Allen preparation through `brainsets.runner` and the repo-local pipeline in `brainsets_local_pipelines/allen_visual_behavior_neuropixels/pipeline.py`.
- Lightweight manifest/workspace preparation from CSV-style session tables through `predictive_circuit_coding/data/prepare.py`, used by tests and synthetic workflows.

The real Allen path is the canonical research path. It keeps the Allen raw cache outside the repository when `allen_sdk.cache_root` is configured, writes processed `.h5` sessions inside the project workspace, builds session catalogs, writes split manifests, and emits split-specific `torch_brain` YAML configs.

## Inputs And Outputs

Inputs:

- `configs/pcc/allen_visual_behavior_neuropixels_local.yaml`.
- AllenSDK cache root or local raw directory.
- Optional session ID filters and max-session limits.
- Unit filtering thresholds for Allen unit tables.

Outputs:

- Prepared `.h5` sessions under `data/<dataset_id>/prepared/`.
- `session_catalog.json` and `session_catalog.csv`.
- `session_manifest.json`.
- `split_manifest.json`.
- Split-specific `torch_brain_*.yaml` files.
- Upload bundle manifest for Colab/Drive workflows.

## Key Code/Config Anchors

- `predictive_circuit_coding/data/config.py`: typed preparation config loader.
- `predictive_circuit_coding/data/layout.py`: workspace directory layout.
- `predictive_circuit_coding/data/brainsets_runner.py`: command builder for `brainsets.runner`.
- `brainsets_local_pipelines/allen_visual_behavior_neuropixels/pipeline.py`: Allen session download/process logic.
- `predictive_circuit_coding/data/temporaldata_sessions.py`: helper for synthetic or direct `temporaldata` session construction.
- `predictive_circuit_coding/data/processed_sessions.py`: scans prepared sessions and applies split domains.
- `predictive_circuit_coding/data/catalog.py`: rich catalog construction and CSV/JSON writing.
- `predictive_circuit_coding/data/splits.py`: subject/session split manifest planning.
- `predictive_circuit_coding/data/selection.py`: runtime subset filtering and materialization.
- `predictive_circuit_coding/cli/prepare_data.py`: public `pcc-prepare-data` command surface.

## Workflow Dependencies

Data preparation feeds nearly every other subsystem. The windowing layer depends on split manifests and split-specific dataset configs. Training and downstream stages depend on session IDs, subject IDs, brain regions, unit metadata, and split assignments being stable. Artifact provenance depends on catalog records and prepared-session metadata being available later, not just at processing time.

The real Allen pipeline currently builds `temporaldata.Data` payloads with brainset, subject, session, device, units, spikes, domain, and optional stimulus/trial/optotagging intervals. The local `brainsets_runner` wrapper resolves the raw directory from `allen_sdk.cache_root` when provided and passes unit filtering flags into the pipeline.

## Refinement Opportunities

- Clarify in docs which preparation functions are production Allen workflow and which are synthetic/test support.
- Keep the "rebuild metadata/catalog/splits without reprocessing `.h5` files" path prominent, because it is central to local iteration.
- Consider adding a short preparation data-flow diagram in `project_guide.md` once this synthesis is merged.
- Continue tightening tests around catalog contract fields promoted from Allen metadata, since downstream subset selection depends on those fields.
- Document whole-session split semantics explicitly: current prepared sessions are assigned to train/valid/discovery/test by session or subject grouping rather than by slicing a session across multiple splits.
