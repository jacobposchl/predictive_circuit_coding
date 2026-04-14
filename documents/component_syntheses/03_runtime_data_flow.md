# Runtime Data Flow Synthesis

## What This Subsystem Owns

The runtime data-flow layer turns prepared sessions and split manifests into training/evaluation batches. It owns split-aware dataset construction, fixed-window sampling, spike binning, temporal patching, unit padding/masking, and provenance-preserving collation.

This layer is intentionally between data preparation and model training. It hides most `torch_brain` dataset mechanics from the model and objective code, while preserving the information needed later for frozen-token interpretation.

## Inputs And Outputs

Inputs:

- Prepared session paths from the data workspace or runtime subset bundle.
- `SplitManifest` assignments.
- Experiment `data_runtime` settings such as bin width, context bins, patch bins, max units, and interval inclusion flags.
- Split names such as train, valid, discovery, and test.

Outputs:

- `TorchBrainDatasetBundle` instances.
- Random or sequential fixed-window samplers.
- `PopulationWindowBatch` objects containing:
  - binned counts
  - temporal patch counts
  - unit and patch masks
  - bin width
  - token/window/unit/event provenance

## Key Code/Config Anchors

- `predictive_circuit_coding/windowing/dataset.py`: dataset config generation, bundle loading, random/sequential samplers, window descriptors.
- `predictive_circuit_coding/windowing/transforms.py`: window sample summaries for inspection.
- `predictive_circuit_coding/tokenization/batching.py`: event annotation extraction, spike binning, patch construction, and batch collation.
- `predictive_circuit_coding/training/contracts.py`: `TokenProvenanceBatch` and `PopulationWindowBatch`.
- `predictive_circuit_coding/training/factories.py`: tokenizer construction from experiment config.
- `configs/pcc/predictive_circuit_coding_*.yaml`: `data_runtime` settings.
- `tests/test_windowing_dataset.py` and `tests/test_tokenization_core.py`: focused coverage for sampling and collation behavior.

## Workflow Dependencies

Training, evaluation, benchmark extraction, discovery, and validation all rely on the same batch contract. That shared contract is what allows downstream frozen-token records to retain session, subject, unit, region, depth, window timing, patch timing, and event annotations.

Sampling strategy changes are scientifically meaningful. Random training windows, sequential evaluation windows, label-balanced discovery scans, and runtime subset materialization all depend on the same prepared-session and split infrastructure but use different traversal policies.

## Refinement Opportunities

- Document tensor shapes and dimensional conventions in one place, especially `counts`, `patch_counts`, `unit_mask`, and `patch_mask`.
- Keep label/event extraction close to provenance tests, because downstream task labels depend on how intervals overlap or onset-match sampled windows.
- Consider adding a small "batch contract" section to `artifact_contracts.md` or `project_guide.md`, since many downstream artifacts are derived from `PopulationWindowBatch`.
- Preserve the optional dependency boundary around `torch_brain` so simple config and contract tests can run without full data preparation.
- Keep sampler/window inspection commands easy to run locally before Colab work, since this is the cheapest place to detect split or label coverage problems.
