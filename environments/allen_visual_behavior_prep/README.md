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
