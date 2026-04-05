# Developer Workflows

## Environments

Use two environments:

- dedicated Allen prep environment for `brainsets` + AllenSDK data preparation
- main repo environment for model development, tests, and synthetic workflow checks

## Typical Local Sequence

1. Prepare Allen data locally.
2. Inspect windows locally with `pcc-prepare-data inspect-windows`.
3. Upload the processed bundle to Drive.
4. Run training/evaluation in Colab.
5. Run discovery/validation in Colab.
6. Pull artifacts back locally if needed for inspection.

## Common Commands

```bash
.venv\Scripts\python.exe -m pytest -q
pcc-prepare-data check-env
pcc-prepare-data inspect-windows --config configs/pcc/allen_visual_behavior_neuropixels_local.yaml --split train --window-length 10.0 --step 10.0
pcc-train --config configs/pcc/predictive_circuit_coding_base.yaml --data-config configs/pcc/allen_visual_behavior_neuropixels_local.yaml
```

## Expected Debug Pattern

- reproduce with a synthetic test first when possible
- keep notebook changes thin and push reusable logic back into the package
- treat artifact schema changes as public-surface changes and update docs/tests together
