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

## Count Normalization Stats

Saved when `count_normalization.mode: log1p_train_zscore`.

Required keys:

- `mode`
- `mean`
- `std`
- `count`

Stats must be fit from the training split only.

## Refinement Summary

Saved by `pcc-refine` and the notebook pipeline as `reports/refinement_summary.json` and `.csv`.

Rows include:

- `task_name`
- `target_label`
- `arm_name`
- `variant_name`
- `geometry_mode`
- `candidate_geometry_mode`
- `claim_safe`
- `supervision_level`
- `status`
- `failure_reason`
- `candidate_count`
- `cluster_count`
- `fit_probe_accuracy`
- `held_out_test_probe_pr_auc`
- `held_out_similarity_pr_auc`
- `cluster_persistence_mean`
- `silhouette_score`
- artifact paths for row, cluster report, discovery artifact, and transform summary

## Discovery Artifact

Saved per task and refinement arm.

Required keys:

- `dataset_id`
- `split_name`
- `checkpoint_path`
- `config_snapshot`
- `decoder_summary`
- `candidates`
- `cluster_stats`
- `cluster_quality_summary`

Candidate records must preserve session, subject, unit, patch, window, score, and embedding provenance.

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

Oracle alignment artifacts must set `claim_safe: false`.

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
