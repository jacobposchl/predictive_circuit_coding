# Allen Visual Behavior Prep Environment

This environment is the dedicated local CPU-only environment for preparing Allen Visual Behavior Neuropixels data.

Why it exists:

- `allensdk==2.16.2` is not compatible with the newer `numpy`/`pynwb` stack used in the main project environment.
- Local Allen data prep and Colab model training are already separate stages in this project.
- The dedicated prep env lets `pcc-prepare-data` keep a single workflow while isolating the fragile Allen dependency stack.

Recommended setup:

```powershell
pwsh -File scripts/setup_allen_visual_behavior_prep_env.ps1
```

Then run the prep command from that environment:

```powershell
.venv-allen-prep\Scripts\python.exe -m predictive_circuit_coding.cli.prepare_data prepare-allen-visual-behavior-neuropixels --config configs/pcc/allen_visual_behavior_neuropixels_local.yaml
```

The prep CLI is now two-phase under the hood:

- `process-allen-visual-behavior-neuropixels`: expensive raw-to-processed `.h5` conversion
- `build-session-catalog`: cheap rebuild of rich session metadata, canonical splits, and upload manifests

If your processed `.h5` files are already correct and you only changed metadata handling or subset logic, rerun:

```powershell
.venv-allen-prep\Scripts\python.exe -m predictive_circuit_coding.cli.prepare_data build-session-catalog --config configs/pcc/allen_visual_behavior_neuropixels_local.yaml
```
