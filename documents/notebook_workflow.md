# Notebook Workflow

The repo uses two Colab notebooks and keeps them thin.

## Training Notebook

File:

- [train_predictive_circuit_coding_colab.ipynb](/C:/Users/Jacob%20Poschl/Desktop/population-dynamics/notebooks/train_predictive_circuit_coding_colab.ipynb)

Responsibilities:

- mount Drive
- install the repo
- install notebook extras before importing repo notebook helpers
- run preflight path checks
- launch `pcc-train`
- launch `pcc-evaluate`
- show stage banners, elapsed time, and checkpoint reminders

## Discovery And Validation Notebook

File:

- [discover_validate_inspect_colab.ipynb](/C:/Users/Jacob%20Poschl/Desktop/population-dynamics/notebooks/discover_validate_inspect_colab.ipynb)

Responsibilities:

- select a checkpoint and discovery artifact
- run `pcc-discover`
- run `pcc-validate`
- inspect generated JSON and CSV outputs without manually opening files in Drive
- load and display the companion cluster summary JSON and CSV outputs

## Notebook Rules

- notebooks do not duplicate training, evaluation, discovery, or validation logic
- notebooks call the CLI or library helpers
- progress UI should use `tqdm.auto`, `rich`, and small helper utilities from `predictive_circuit_coding.utils.notebook`
- long cells should state what they are doing and what artifact is expected next
