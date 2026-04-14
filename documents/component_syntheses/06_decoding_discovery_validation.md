# Decoding, Discovery, And Validation Synthesis

## What This Subsystem Owns

The decoding, discovery, and validation subsystem interprets frozen representations after training. It owns token extraction, label extraction, additive probe fitting, token scoring, candidate selection, clustering, stability estimation, cluster reporting, and conservative validation checks.

The design keeps downstream task analysis separate from encoder training. This is important for refinement: task labels should be used to interpret frozen tokens and evaluate candidate motifs, not to silently reshape the encoder objective unless an explicitly named training variant does so.

## Inputs And Outputs

Inputs:

- Trained checkpoint.
- Experiment config and data config.
- Discovery or test split windows.
- Target label configuration and label mode.
- Runtime subset assets when active.

Outputs:

- Discovery decode coverage summary JSON.
- Discovery artifact JSON with decoder summary, candidate tokens, cluster stats, and cluster quality.
- Cluster summary JSON and CSV.
- Validation summary JSON and CSV.
- Run-manifest sidecars.

## Key Code/Config Anchors

- `predictive_circuit_coding/decoding/extract.py`: discovery planning, selected-window extraction, frozen-token extraction, token shards.
- `predictive_circuit_coding/decoding/labels.py`: binary label extraction, aliases, dotted paths, event-local/default modes, image identity matching.
- `predictive_circuit_coding/decoding/probes.py`: additive probe fitting and evaluation.
- `predictive_circuit_coding/decoding/scoring.py`: token scoring and session-balanced candidate selection.
- `predictive_circuit_coding/decoding/geometry.py`: holdout splits, whitening/alignment transforms, neighbor enrichment summaries.
- `predictive_circuit_coding/discovery/run.py`: discovery orchestration.
- `predictive_circuit_coding/discovery/clustering.py`: HDBSCAN clustering.
- `predictive_circuit_coding/discovery/stability.py`: bootstrap stability estimates.
- `predictive_circuit_coding/discovery/reporting.py`: cluster summary JSON/CSV.
- `predictive_circuit_coding/validation/run.py`: artifact provenance, shuffle control, held-out transfer, motif similarity, baseline sensitivity.
- `predictive_circuit_coding/cli/discover.py` and `cli/validate.py`.
- `tests/test_decoding_*.py`, `tests/test_stage5_stage6_workflow.py`, and `tests/test_stage7_hardening.py`.

## Workflow Dependencies

Discovery depends on frozen tokens retaining provenance from the batch contract. Candidate records must remain traceable to recording, session, subject, unit, brain region, probe depth, window timing, and patch timing. Validation depends on discovery artifact provenance matching the checkpoint and target label used for the validation run.

Coverage summaries are first-class because many task panels can fail for non-model reasons, such as missing fields or no positive windows under a given split. Label-balanced discovery adds search and selection budgets that should remain visible in artifacts and docs.

## Refinement Opportunities

- Keep target-label mode semantics documented where users choose tasks, not only in label extraction tests.
- Continue making discovery coverage summaries easy to inspect before probe fitting and clustering.
- Consider a single provenance-validation helper shared by discovery, validation, and benchmark code if duplication grows.
- Keep candidate selection diagnostics separate from final selection scores so users can inspect sign direction and background effects.
- Preserve the conservative language around validation: these checks support observational motif confidence, not causal discovery.
