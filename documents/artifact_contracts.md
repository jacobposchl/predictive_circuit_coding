# Artifact Contracts

This document records the live artifact surfaces for the refinement-centered workflow.

## Training Checkpoint

Saved by `pcc-train`.

Required top-level keys:

- `epoch`
- `global_step`
- `best_metric`
- `best_epoch`
- `best_validation_metrics`
- `metadata`
- `model_state`
- `optimizer_state`
- `scheduler_state`

Metadata must include:

- `dataset_id`
- `split_name`
- `seed`
- `config_snapshot`
- `model_hparams`
- `continuation_baseline_type`
- `variant_name`
- `reconstruction_target_mode`
- `count_normalization_mode`
- `count_normalization_stats_path`

`auxiliary_state` may be present and may be `null`.

## Training Summary

Saved by `pcc-train`.

Required keys:

- `dataset_id`
- `split_name`
- `epoch`
- `best_epoch`
- `metrics`
- `losses`
- `checkpoint_path`
- `variant_name`
- `reconstruction_target_mode`
- `count_normalization_mode`
- `count_normalization_stats_path`
- `selection_reason`

## Training History

Saved by `pcc-train` beside the training summary as `training_history.json` and `training_history.csv`.

The JSON payload root is:

- `training_history`

Each row includes these stable columns:

- `epoch`
- `global_step`
- `variant_name`
- `reconstruction_target_mode`
- `count_normalization_mode`
- `train_split`
- `valid_split`
- `learning_rate`
- `evaluated`
- `became_best`
- `best_epoch_so_far`
- `best_predictive_improvement_so_far`

Additional `train_*` and `valid_*` metric columns are expected when those metrics are available for the epoch.

## Count Normalization Stats

Saved when `count_normalization.mode: log1p_train_zscore`.

Required keys:

- `mode`
- `mean`
- `std`
- `count`

Stats must be fit from the training split only.

## Evaluation Summary

Saved by `pcc-evaluate` and by the pipeline evaluation stage.

Required keys:

- `dataset_id`
- `split_name`
- `checkpoint_path`
- `metrics`
- `losses`
- `window_count`

## Discovery Decode Coverage Summary

Saved by `pcc-discover`.

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

## Refinement Summary

Saved by `pcc-refine` and the notebook pipeline as `reports/refinement_summary.json` and `.csv`.

Rows include:

- `task_name`
- `target_label`
- `target_label_match_value`
- `arm_name`
- `feature_family`
- `encoder_training_status`
- `encoder_checkpoint_loaded`
- `claim_safe`
- `supervision_level`
- `geometry_mode`
- `variant_name`
- `status`
- `failure_reason`
- `candidate_count`
- `cluster_count`
- `candidate_selection_fallback_used`
- `candidate_selection_effective_min_score`
- `fit_probe_accuracy`
- `fit_probe_bce`
- `shuffled_probe_accuracy`
- `shuffled_probe_bce`
- `held_out_test_probe_accuracy`
- `held_out_test_probe_bce`
- `held_out_test_probe_roc_auc`
- `held_out_test_probe_pr_auc`
- `held_out_similarity_roc_auc`
- `held_out_similarity_pr_auc`
- `cluster_persistence_mean`
- `silhouette_score`
- `max_cluster_session_spread`
- `max_cluster_subject_spread`
- artifact paths for row, cluster report, discovery artifact, and transform summary

## Discovery Artifact

Saved by `pcc-discover` and by `pcc-refine`, per task and refinement arm.

Required keys:

- `dataset_id`
- `split_name`
- `checkpoint_path`
- `config_snapshot`
- `decoder_summary`
- `candidates`
- `cluster_stats`
- `cluster_quality_summary`

Candidate records must preserve:

- `candidate_id`
- `cluster_id`
- `recording_id`
- `session_id`
- `subject_id`
- `unit_id`
- `unit_region`
- `unit_depth_um`
- `patch_index`
- `patch_start_s`
- `patch_end_s`
- `window_start_s`
- `window_end_s`
- `label`
- `score`
- `embedding`

Optional candidate fields currently written by refinement arms:

- `raw_probe_score`
- `negative_background_score`

`decoder_summary` must include:

- `target_label`
- `epochs`
- `learning_rate`
- `metrics`
- `probe_state`
- `metric_scope`

## Cluster Summary

Saved by `pcc-discover` and by `pcc-refine` as JSON and CSV.

Required top-level JSON keys:

- `dataset_id`
- `split_name`
- `checkpoint_path`
- `cluster_count`
- `candidate_count`
- `cluster_quality_summary`
- `clusters`

Each cluster row includes:

- `cluster_id`
- `cluster_persistence`
- `candidate_count`
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

## Transform Summary

Saved per refinement arm.

Required keys:

- `task_name`
- `arm_name`
- `variant_name`
- `claim_safe`
- `supervision_level`
- `status`
- `failure_reason`
- `pca_summary`
- `whitening_summary`
- `token_normalization_summary`
- `alignment_summary`
- `test_alignment_summary`
- `candidate_geometry_summary`
- `row`

Oracle alignment artifacts must set `claim_safe: false`.

## Final Project Summary

Saved by `pcc-refine` and by the pipeline final report stage as `final_project_summary.json` and `.csv`.

Required keys:

- `refinement_row_count`
- `motif_completed_row_count`
- `motif_mean_held_out_similarity_pr_auc`
- `variant_names`
- `best_motif_row`
- `notes`

## Validation Summary

Saved by `pcc-validate`.

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

The validation path remains conservative and observational.

## Validation CSV

Saved by `pcc-validate`.

Stable CSV columns are:

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

## Run-Manifest Sidecars

Saved by CLI commands beside their primary output artifact.

Required keys:

- `command_name`
- `created_at_utc`
- `dataset_id`
- `inputs`
- `outputs`

## Pipeline Manifest

Saved by `pcc-run-pipeline` as `pipeline/pipeline_manifest.json`.

Required keys:

- `run_id`
- `dataset_id`
- `stage_order`
- `local_run_root`
- `drive_run_root`
- `config_snapshot_path`
- `created_at_utc`
- `updated_at_utc`

## Pipeline State

Saved by `pcc-run-pipeline` as `pipeline/pipeline_state.json`.

The JSON payload root is:

- `stages`

Each stage state includes:

- `stage_name`
- `status`
- `config_hash`
- `inputs`
- `outputs`
- `created_at_utc`
- `updated_at_utc`
- `error_message`

Current stage names are:

- `train`
- `evaluate`
- `refinement`
- `alignment_diagnostic`
- `final_reports`

## Refinement Verification Summary

Saved by `pcc-verify-refinement` as `refinement_verification_summary.json`.

Required keys:

- `status`
- `pipeline_config_path`
- `experiment_config_path`
- `data_config_path`
- `variant_name`
- `split_counts`
- `issues`
- `coverage_rows`
- `summary_json_path`
- `coverage_csv_path`

Each verification issue includes:

- `gate`
- `severity`
- `message`

Each coverage row includes:

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
