# Predictive Circuit Coding

This repository is an Allen-first Neuropixels pipeline for predictive population-dynamics modeling. The workflow is intentionally split across two compute surfaces:

- local CPU for Allen data preparation, split planning, and inspection
- Google Colab A100 for training, evaluation, discovery, and validation

The current repo is Stage 7 complete: it supports `pcc-prepare-data`, `pcc-train`, `pcc-evaluate`, `pcc-discover`, and `pcc-validate`, plus two thin Colab notebooks that sit on top of the CLI/library surface.

## Workflow

Local machine:

1. Create or refresh the dedicated Allen prep environment.
2. Run `pcc-prepare-data prepare-allen-visual-behavior-neuropixels`.
3. Upload the processed bundle from `data/allen_visual_behavior_neuropixels/` to Google Drive.
4. Optionally inspect split windows locally before training.

Colab:

1. Mount Drive.
2. Install the repo.
3. Run the training notebook to produce checkpoints and evaluation summaries.
4. Run the discovery/validation notebook to produce motif and validation artifacts.

## Main Commands

```bash
pcc-prepare-data prepare-allen-visual-behavior-neuropixels --config configs/pcc/allen_visual_behavior_neuropixels_local.yaml
pcc-train --config configs/pcc/predictive_circuit_coding_base.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml
pcc-evaluate --config configs/pcc/predictive_circuit_coding_base.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml --checkpoint artifacts/checkpoints/pcc_best.pt --split valid
pcc-discover --config configs/pcc/predictive_circuit_coding_base.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml --checkpoint artifacts/checkpoints/pcc_best.pt --split discovery
pcc-validate --config configs/pcc/predictive_circuit_coding_base.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml --checkpoint artifacts/checkpoints/pcc_best.pt --discovery-artifact artifacts/checkpoints/pcc_best_discovery_discovery.json
```

Each Stage 5-7 command writes:

- its main artifact
- a JSON run-manifest sidecar that records the config paths, split names, input artifacts, and output artifacts used for that run
- `pcc-discover` also writes companion cluster summary JSON and CSV outputs for easier inspection

## Notebooks

The repo ends with two Colab notebooks:

- [train_predictive_circuit_coding_colab.ipynb](/C:/Users/Jacob%20Poschl/Desktop/population-dynamics/notebooks/train_predictive_circuit_coding_colab.ipynb)
- [discover_validate_inspect_colab.ipynb](/C:/Users/Jacob%20Poschl/Desktop/population-dynamics/notebooks/discover_validate_inspect_colab.ipynb)

They are intentionally thin and call the same CLI surface listed above.

## Documentation

Stage 7 docs live under `documents/`:

- [artifact_contracts.md](/C:/Users/Jacob%20Poschl/Desktop/population-dynamics/documents/artifact_contracts.md)
- [notebook_workflow.md](/C:/Users/Jacob%20Poschl/Desktop/population-dynamics/documents/notebook_workflow.md)
- [dev_workflows.md](/C:/Users/Jacob%20Poschl/Desktop/population-dynamics/documents/dev_workflows.md)
- [troubleshooting.md](/C:/Users/Jacob%20Poschl/Desktop/population-dynamics/documents/troubleshooting.md)
- [manual_acceptance_checklist.md](/C:/Users/Jacob%20Poschl/Desktop/population-dynamics/documents/manual_acceptance_checklist.md)
- [predictive_circuit_coding_design_doc.docx](/C:/Users/Jacob%20Poschl/Desktop/population-dynamics/documents/predictive_circuit_coding_design_doc.docx)

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

## Verification

For repo-level validation:

```bash
.venv\Scripts\python.exe -m pytest -q
```
