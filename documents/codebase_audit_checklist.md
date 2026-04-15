# Codebase Audit Checklist

This file is the working checklist for the full code-cleanup and correctness audit of `predictive_circuit_coding`.

The goal is not just to read files. The goal is to confirm that each audited surface is:

- clearly implemented
- bug-resistant and internally consistent
- necessary to the end-to-end pipeline
- easy to locate and understand in the codebase
- aligned with configs, docs, artifacts, and tests

For each item below, the audit will check:

- purpose in the full pipeline
- inputs, outputs, and artifact contracts
- naming and discoverability
- dead code, duplicate logic, or unclear layering
- validation and error handling
- consistency with tests and documentation

## 1. Canonical Project Contract

- [ ] `AGENTS.md`
- [ ] `README.md`
- [ ] `documents/project_guide.md`
- [ ] `documents/artifact_contracts.md`
- [ ] `pyproject.toml`

## 2. Config And Entry-Point Surface

- [ ] `configs/pcc/allen_visual_behavior_neuropixels_local.yaml`
- [ ] `configs/pcc/predictive_circuit_coding_refined_debug.yaml`
- [ ] `configs/pcc/predictive_circuit_coding_refined_full.yaml`
- [ ] `configs/pcc/predictive_circuit_coding_refined_recon000_debug.yaml`
- [ ] `configs/pcc/predictive_circuit_coding_refined_recon000_full.yaml`
- [ ] `configs/pcc/predictive_circuit_coding_refined_l2off_debug.yaml`
- [ ] `configs/pcc/predictive_circuit_coding_refined_l2off_full.yaml`
- [ ] `configs/pcc/predictive_circuit_coding_refined_countnorm_debug.yaml`
- [ ] `configs/pcc/predictive_circuit_coding_refined_countnorm_full.yaml`
- [ ] `configs/pcc/predictive_circuit_coding_refined_cls_debug.yaml`
- [ ] `configs/pcc/predictive_circuit_coding_refined_cls_full.yaml`
- [ ] `configs/pcc/pipeline_refined_debug.yaml`
- [ ] `configs/pcc/pipeline_refined_full.yaml`
- [ ] `predictive_circuit_coding/cli/`

## 3. Data Foundation And Preparation

- [ ] `predictive_circuit_coding/data/`
- [ ] `brainsets_local_pipelines/allen_visual_behavior_neuropixels/pipeline.py`
- [ ] `tests/test_prepare_cli.py`
- [ ] `tests/test_data_foundation.py`
- [ ] `tests/test_allen_brainsets_prep.py`
- [ ] `tests/test_allen_visual_behavior_pipeline.py`
- [ ] `tests/test_session_catalog_selection.py`

## 4. Windowing And Tokenization

- [ ] `predictive_circuit_coding/windowing/`
- [ ] `predictive_circuit_coding/tokenization/`
- [ ] `tests/test_windowing_dataset.py`
- [ ] `tests/test_tokenization_core.py`

## 5. Model, Objectives, And Training Runtime

- [ ] `predictive_circuit_coding/models/`
- [ ] `predictive_circuit_coding/objectives/`
- [ ] `predictive_circuit_coding/training/`
- [ ] `tests/test_model_core.py`
- [ ] `tests/test_training_runtime_config.py`

## 6. Evaluation, Decoding, Discovery, Refinement, And Validation

- [ ] `predictive_circuit_coding/evaluation/`
- [ ] `predictive_circuit_coding/decoding/`
- [ ] `predictive_circuit_coding/discovery/`
- [ ] `predictive_circuit_coding/benchmarks/`
- [ ] `predictive_circuit_coding/validation/`
- [ ] `tests/test_decoding_labels.py`
- [ ] `tests/test_decoding_geometry.py`
- [ ] `tests/test_decoding_scoring.py`
- [ ] `tests/test_refinement_core.py`
- [ ] `tests/test_stage5_stage6_workflow.py`
- [ ] `tests/test_stage7_hardening.py`

## 7. Workflow Orchestration, Notebook Path, And UX Helpers

- [ ] `predictive_circuit_coding/workflows/`
- [ ] `predictive_circuit_coding/utils/`
- [ ] `notebooks/run_predictive_circuit_coding_pipeline_colab.ipynb`
- [ ] `tests/test_workflow_pipeline.py`
- [ ] `tests/test_notebook_pipeline_progress.py`
- [ ] `tests/test_notebook_runtime_helpers.py`
- [ ] `tests/test_notebook_ui_preview.py`

## 8. Cross-Cutting Cleanup Passes

- [ ] CLI, config, and documentation consistency across the full `pcc-*` surface
- [ ] artifact schema consistency across code, docs, and tests
- [ ] provenance and traceability continuity from prepared sessions to final reports
- [ ] pipeline-stage clarity and end-to-end data-flow readability
- [ ] module naming, boundary clarity, and codebase discoverability
- [ ] dead code, stale helpers, unused branches, and duplicated logic
- [ ] runtime validation, failure modes, and error-message quality
- [ ] test coverage gaps and missing regression protection

## 9. Expected Audit Output

As the audit proceeds, each completed area should produce:

- a short summary of what the area is responsible for
- concrete findings on bugs, risks, unclear implementation, or cleanup needs
- the proposed or applied fix path
- any missing tests or documentation updates required to keep the area understandable
