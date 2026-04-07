# Artifact Contracts

This repo treats these outputs as first-class artifacts.

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

Required payload:

- `epoch`
- `global_step`
- `best_metric`
- `metadata`
- `model_state`
- `optimizer_state`
- `scheduler_state`

Required metadata keys:

- `dataset_id`
- `split_name`
- `seed`
- `config_snapshot`
- `model_hparams`
- `continuation_baseline_type`

## Training Summary JSON

Required keys:

- `dataset_id`
- `split_name`
- `epoch`
- `best_epoch`
- `metrics`
- `losses`
- `checkpoint_path`

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

Each candidate token record must retain:

- session provenance
- unit provenance
- patch timing
- window timing
- label
- score
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
