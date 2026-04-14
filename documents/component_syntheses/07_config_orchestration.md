# Config And Orchestration

Core configs are `predictive_circuit_coding_refined_debug.yaml` and `predictive_circuit_coding_refined_full.yaml`.

One-axis ablation configs use `extends` to override reconstruction weight, token L2 normalization, count normalization, or population-token pooling.

Notebook pipeline configs use `stages.refinement` and write refinement summaries under each run's reports directory.
