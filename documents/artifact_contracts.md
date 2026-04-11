# Artifact Contracts

This repo treats these outputs as first-class artifacts.

## Unified Pipeline Run Root

The unified Colab runner writes one stage-resumable run root under:

- `pcc_colab_outputs/<run_id>/run_1/`

Expected top-level subdirectories:

- `train/`
- `evaluation/`
- `benchmarks/representation/`
- `benchmarks/motifs/`
- `benchmarks/diagnostics/` when optional appendix stages are enabled
- `reports/`
- `pipeline/`

## Session Catalog JSON

Location:

- `data/<dataset_id>/manifests/session_catalog.json`

Required top-level keys:

- `dataset_id`
- `source_name`
- `records`

Each record must retain:

- processed-session scan fields such as `session_id`, `subject_id`, `duration_s`, `n_units`, `brain_regions`, `trial_count`, `prepared_session_path`, and `raw_data_path`
- promoted Allen session metadata such as `session_type`, `image_set`, `experience_level`, `session_number`, and `project_code` when available

## Runtime Subset Bundle

Notebook-driven subset runs materialize their realized subset under the run artifact root instead of under the canonical local data tree.

Typical location:

- `artifacts/runtime_subset/`

Required files:

- `selected_session_catalog.json`
- `selected_session_catalog.csv`
- `selected_split_manifest.json`
- `splits/torch_brain_runtime_train.yaml`
- `splits/torch_brain_runtime_valid.yaml`
- `splits/torch_brain_runtime_discovery.yaml`
- `splits/torch_brain_runtime_test.yaml`

Purpose:

- records the exact notebook-defined subset and recomputed split assignment used by the runtime experiment

## Training Checkpoint

Location:

- typically `artifacts/checkpoints/<prefix>_best.pt`
- in unified Colab runs, under `pcc_colab_outputs/<run_id>/run_1/train/checkpoints/<prefix>_best.pt`

Required payload:

- `epoch`
- `global_step`
- `best_metric`
- `best_epoch`
- `metadata`
- `model_state`
- `optimizer_state`
- `scheduler_state`
- `auxiliary_state` when the cross-session auxiliary-loss variant is enabled

Required metadata keys:

- `dataset_id`
- `split_name`
- `seed`
- `config_snapshot`
- `model_hparams`
- `continuation_baseline_type`
- `training_variant_name`
- `cross_session_aug_enabled`

## Training Summary JSON

Required keys:

- `dataset_id`
- `split_name`
- `epoch`
- `best_epoch`
- `metrics`
- `losses`
- `checkpoint_path`
- `training_variant_name`
- `cross_session_aug_enabled`
- `selection_reason`

Semantics:

- `training_summary.json` must describe the checkpoint named in `checkpoint_path`
- `epoch` and `best_epoch` therefore refer to the selected best checkpoint, not necessarily the latest completed epoch

Storage policy:

- notebook-driven final runs retain only the selected `best` and `latest` checkpoints by default
- verbose per-epoch checkpoints are not first-class artifacts in the compact final workflow

## Cross-Session Geometry Monitor JSON And CSV

Augmented training runs may emit a compact geometry-monitor artifact next to `training_summary.json`:

- `train/cross_session_geometry_monitor.json`
- `train/cross_session_geometry_monitor.csv`

Required row fields:

- `epoch`
- `training_variant_name`
- `cross_session_aug_enabled`
- `cross_session_aug_prob`
- `cross_session_region_loss_weight`
- `split_name`
- `sample_count`
- `neighbor_k`
- `label_neighbor_enrichment`
- `session_neighbor_enrichment`
- `subject_neighbor_enrichment`

Semantics:

- these rows are periodic diagnostics collected during training on raw encoder features
- the geometry monitor is the primary acceptance gate for the auxiliary-loss experiment before spending a full benchmark run

## Evaluation Summary JSON

Required keys:

- `dataset_id`
- `split_name`
- `checkpoint_path`
- `metrics`
- `losses`
- `window_count`

Expected metrics:

- `predictive_loss`
- `predictive_baseline_mse`
- `predictive_raw_mse`
- `predictive_improvement`
- `reconstruction_loss`

## Representation Benchmark Summary JSON And CSV

Unified runs emit one combined representation benchmark summary under:

- `reports/representation_benchmark_summary.json`
- `reports/representation_benchmark_summary.csv`

Each row must retain:

- `task_name`
- `target_label`
- `target_label_match_value`
- `arm_name`
- `training_variant_name`
- `cross_session_aug_enabled`
- `feature_family`
- `geometry_mode`
- `status`
- `cross_session_test_probe_accuracy`
- `cross_session_test_probe_bce`
- `cross_session_test_probe_roc_auc`
- `cross_session_test_probe_pr_auc`
- `within_session_holdout_probe_accuracy`
- `within_session_holdout_probe_bce`
- `within_session_holdout_probe_roc_auc`
- `within_session_holdout_probe_pr_auc`
- `label_neighbor_enrichment`
- `session_neighbor_enrichment`
- `subject_neighbor_enrichment`
- `summary_json_path`
- `transform_summary_json_path`

Semantics:

- the representation benchmark is the claim-facing crossed matrix over feature family x geometry mode
- optional tasks that cannot run because a field is absent must report `status = skipped_missing_field` instead of failing the whole run

## Motif Benchmark Summary JSON And CSV

Unified runs emit one combined motif benchmark summary under:

- `reports/motif_benchmark_summary.json`
- `reports/motif_benchmark_summary.csv`

Each row must retain:

- `task_name`
- `target_label`
- `target_label_match_value`
- `arm_name`
- `training_variant_name`
- `cross_session_aug_enabled`
- `feature_family`
- `geometry_mode`
- `status`
- `candidate_count`
- `cluster_count`
- `cluster_persistence_mean`
- `silhouette_score`
- `held_out_test_probe_accuracy`
- `held_out_test_probe_bce`
- `held_out_test_probe_roc_auc`
- `held_out_test_probe_pr_auc`
- `held_out_test_similarity_roc_auc`
- `held_out_test_similarity_pr_auc`
- `cluster_summary_json_path`
- `discovery_artifact_path`
- `validation_summary_json_path`

Semantics:

- motif benchmark rows correspond only to the selected arm subset for motif analysis, not the full representation matrix
- `status` must distinguish successful rows from degraded or skipped rows so the final report can stay compact and honest

## Discovery Artifact JSON

Required keys:

- `dataset_id`
- `split_name`
- `checkpoint_path`
- `config_snapshot`
- `decoder_summary`
- `candidates`
- `cluster_stats`
- `cluster_quality_summary`

`decoder_summary` must include:

- `target_label`
- `epochs`
- `learning_rate`
- `metrics`
- `probe_state`
- `metric_scope`

Each candidate token record must retain:

- session provenance
- unit provenance
- patch timing
- window timing
- label
- `score` as the final contrastive selection score used for ranking
- `raw_probe_score` as the signed additive-probe score before background subtraction
- `negative_background_score` as the matched negative-window background used in the contrastive score
- embedding

## Discovery Cluster Summary JSON And CSV

`pcc-discover` also emits a cluster summary JSON and CSV next to the main discovery artifact.

Required JSON keys:

- `dataset_id`
- `split_name`
- `checkpoint_path`
- `cluster_count`
- `candidate_count`
- `cluster_quality_summary`
- `clusters`

Required CSV columns:

- `cluster_id`
- `candidate_count`
- `cluster_persistence`
- `session_count`
- `subject_count`
- `mean_score`
- `max_score`
- `mean_raw_probe_score`
- `mean_negative_background_score`
- `positive_raw_probe_fraction`
- `mean_depth_um`
- `temporal_start_s`
- `temporal_end_s`
- `top_regions`
- `top_sessions`
- `top_subjects`
- `representative_candidate_id`
- `representative_recording_id`
- `representative_unit_id`
- `representative_patch_index`
- `representative_raw_probe_score`
- `representative_negative_background_score`

## Discovery Decode Coverage Summary JSON

`pcc-discover` also emits a decode-coverage summary JSON next to the main discovery artifact.

Required keys:

- `split_name`
- `target_label`
- `total_scanned_windows`
- `positive_window_count`
- `negative_window_count`
- `selected_positive_count`
- `selected_negative_count`
- `sessions_with_positive_windows`
- `sampling_strategy`
- `scan_max_batches`
- `selected_window_count`

## Validation Summary JSON

Required keys:

- `dataset_id`
- `checkpoint_path`
- `discovery_artifact_path`
- `real_label_metrics`
- `shuffled_label_metrics`
- `held_out_test_metrics`
- `held_out_similarity_summary`
- `baseline_sensitivity_summary`
- `candidate_count`
- `cluster_count`
- `cluster_quality_summary`
- `provenance_issues`
- `sampling_summary`

Validation semantics:

- `real_label_metrics` must be recomputed during validation from the same re-extracted discovery windows used for the shuffle control
- `baseline_sensitivity_summary` must contain a real comparison, not just placeholder metadata
- `sampling_summary` must distinguish sampled-window counts from full-split window counts when validation is batch-budgeted

`held_out_similarity_summary` must include:

- `window_roc_auc`
- `window_pr_auc`
- `positive_window_count`
- `negative_window_count`
- `per_session_roc_auc`

## Validation Summary CSV

Required columns:

- `dataset_id`
- `checkpoint_path`
- `discovery_artifact_path`
- `candidate_count`
- `cluster_count`
- `real_probe_accuracy`
- `shuffled_probe_accuracy`
- `held_out_test_probe_accuracy`
- `held_out_similarity_roc_auc`
- `held_out_similarity_pr_auc`
- `real_probe_bce`
- `shuffled_probe_bce`
- `held_out_test_probe_bce`
- `provenance_issue_count`

## Final Project Summary JSON And CSV

Unified runs also emit a compact final project report under:

- `reports/final_project_summary.json`
- `reports/final_project_summary.csv`

Each row must retain:

- `representation_row_count`
- `motif_row_count`
- `representation_completed_row_count`
- `motif_completed_row_count`
- `representation_mean_test_probe_pr_auc`
- `motif_mean_held_out_similarity_pr_auc`
- `training_variant_names`
- `best_representation_row`
- `best_motif_row`
- `claims`

Semantics:

- the final project summary is a compact run-level synthesis artifact, not a replacement for the benchmark and validation detail artifacts
- `training_variant_names` is the run-level bridge for comparing baseline and auxiliary-loss runs across separate `run_id`s

## Pipeline Manifest And State

Unified notebook runs write resumability metadata under:

- `pipeline/pipeline_manifest.json`
- `pipeline/pipeline_state.json`
- `pipeline/pipeline_config_snapshot.yaml`

`pipeline_manifest.json` required keys:

- `run_id`
- `dataset_id`
- `stage_order`
- `local_run_root`
- `drive_run_root`
- `config_snapshot_path`
- `created_at_utc`
- `updated_at_utc`

`pipeline_state.json` required structure:

- top-level `stages`
- one object per stage, keyed by stage name

Each stage object must retain:

- `stage_name`
- `status`
- `config_hash`
- `inputs`
- `outputs`
- `created_at_utc`
- `updated_at_utc`
- `error_message`

Semantics:

- stage reuse is valid only when `status == complete`, the config hash matches, upstream inputs still match, and declared outputs still exist
- interrupted or failed stages must remain visible in the state file instead of being silently discarded

## Full-Run Verification Summary

`pcc-verify-full-run` writes a no-training readiness report before a claim-facing Colab run:

- `full_run_verification_summary.json`
- `full_run_task_coverage.csv`

Required summary keys:

- `status`
- `pipeline_config_path`
- `experiment_config_path`
- `data_config_path`
- `training_num_epochs`
- `training_variant_name`
- `split_counts`
- `issues`
- `coverage_rows`

Required coverage CSV columns:

- `task_name`
- `target_label`
- `split_name`
- `status`
- `total_scanned_windows`
- `positive_window_count`
- `negative_window_count`
- `selected_positive_count`
- `selected_negative_count`
- `selected_window_count`
- `positive_session_count`
- `failure_reason`

Semantics:

- `status = ok` is required before launching the corresponding full notebook run
- task coverage must include both classes and at least two positive-window sessions for every configured claim-facing task/split

## Run-Manifest Sidecars

Stage 5-7 commands also emit sidecars next to their main output artifacts.

Required keys:

- `command_name`
- `created_at_utc`
- `dataset_id`
- `inputs`
- `outputs`

For notebook-subset runs, `inputs` should retain:

- `runtime_split_manifest_path`
- `runtime_session_catalog_path`
- `runtime_subset_active`

For `pcc-discover`, `outputs` must also include:

- `decode_coverage_summary_path`
