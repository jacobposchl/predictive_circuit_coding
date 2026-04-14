# Project Refinement Synthesis

The codebase has been simplified around refinement of a trained predictive population encoder.

Current refinement axes:

- normalized reconstruction targets
- reduced or removed reconstruction pressure
- token L2 ablations
- train-split count normalization
- per-patch population CLS pooling
- post-hoc token normalization
- probe-weighted candidate geometry
- oracle session alignment as a diagnostic

The refined workflow is designed to reduce compute spent on stale comparison work while keeping provenance, validation, and artifact contracts explicit.
