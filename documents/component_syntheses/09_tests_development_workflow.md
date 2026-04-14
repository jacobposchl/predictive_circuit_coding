# Tests And Development Workflow Synthesis

## What This Subsystem Owns

The tests and development workflow protect the project against regressions in data foundation, windowing, tokenization, model/objective behavior, training runtime config, decoding, discovery, validation, benchmarking, notebook helpers, and Allen preparation integration.

The development contract is simple: put core logic in `predictive_circuit_coding/`, keep notebooks thin, update docs/tests when behavior or artifact expectations change, and run `python -m pytest -q` after non-trivial changes.

## Inputs And Outputs

Inputs:

- Code changes in package modules, configs, docs, notebooks, and tests.
- Synthetic prepared workspaces used by workflow tests.
- Repo-native configs and notebook JSON.

Outputs:

- Unit-test coverage for contracts and helpers.
- Workflow-level CLI tests.
- Notebook orchestration tests.
- Hardening tests for failure modes and provenance checks.

## Key Code/Config Anchors

- `tests/test_data_foundation.py`: preparation config, workspace, manifest, split, provenance fields.
- `tests/test_allen_brainsets_prep.py` and `tests/test_allen_visual_behavior_pipeline.py`: Allen/brainsets preparation path.
- `tests/test_windowing_dataset.py` and `tests/test_tokenization_core.py`: sampling and batch collation.
- `tests/test_model_core.py`, `tests/test_training_runtime_config.py`, and `tests/test_cross_session_aug.py`: model, objective, config, checkpoint, and auxiliary-loss behavior.
- `tests/test_decoding_labels.py`, `test_decoding_scoring.py`, and `test_decoding_geometry.py`: labels, candidate selection, geometry transforms.
- `tests/test_stage5_stage6_workflow.py` and `tests/test_stage7_hardening.py`: CLI/workflow validation.
- `tests/test_benchmarks_pipeline.py`: benchmark matrices, full-run verifier, pipeline state, progress events.
- `tests/test_notebook_runtime_helpers.py` and `tests/test_notebook_ui_preview.py`: notebook helper and preview behavior.

## Workflow Dependencies

The test suite spans both small contract tests and synthetic end-to-end workflows. This is important because many bugs in this project are interface bugs: a config parses, but a downstream artifact field is missing; a notebook stage runs, but a sidecar does not include runtime subset provenance; a discovery task runs, but label coverage is absent.

Docs, configs, and artifact contracts should be treated as part of the development surface. A behavior change that affects commands, paths, field names, or artifact semantics should be accompanied by tests and doc updates.

## Refinement Opportunities

- Add lightweight doc/code drift checks for command names and config file names listed in the main docs.
- Add targeted artifact schema tests for summary CSV/JSON outputs that are not fully covered by dataclasses.
- Keep synthetic workflow tests fast enough to run locally, since Colab-only verification slows iteration.
- Maintain explicit tests for degraded/skipped task statuses, because they protect the pipeline from silent scientific overstatement.
- Consider a short contributor checklist that maps common change types to the minimum relevant tests.
