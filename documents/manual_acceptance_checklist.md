# Manual Acceptance Checklist

Use this checklist for one real prepared Allen Visual Behavior Neuropixels slice.

## Local

- run the dedicated Allen prep environment setup
- run `pcc-prepare-data prepare-allen-visual-behavior-neuropixels`
- confirm processed `.h5` files exist
- confirm `session_manifest.json`, `split_manifest.json`, and split YAML files exist
- inspect at least one split locally with `pcc-prepare-data inspect-windows`

## Colab Training

- mount Drive
- open the training notebook
- confirm the repo installs cleanly
- confirm the notebook preflight checks pass
- run `pcc-train`
- confirm a best checkpoint is written
- confirm the training run-manifest sidecar is written
- resume from an existing checkpoint at least once
- run `pcc-evaluate`
- confirm the evaluation summary and sidecar are written

## Colab Discovery And Validation

- open the discovery/validation notebook
- run `pcc-discover`
- confirm the discovery artifact and sidecar are written
- run `pcc-validate`
- confirm the validation JSON, CSV, and sidecar are written
- inspect the generated artifacts from notebook cells without manually opening JSON in Drive

## Final Acceptance

- all five CLI commands complete without ad hoc code edits
- all first-class artifacts are produced
- sidecars correctly identify the config paths, split names, checkpoint path, and output paths used in the run
