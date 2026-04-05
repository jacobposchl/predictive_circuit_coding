# Artifact Contracts

This repo treats these outputs as first-class artifacts.

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
- `stability_summary`

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
- `clusters`

Required CSV columns:

- `cluster_id`
- `candidate_count`
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

## Validation Summary JSON

Required keys:

- `dataset_id`
- `checkpoint_path`
- `discovery_artifact_path`
- `real_label_metrics`
- `shuffled_label_metrics`
- `baseline_sensitivity_summary`
- `candidate_count`
- `cluster_count`
- `stability_summary`
- `recurrence_summary`
- `provenance_issues`

## Validation Summary CSV

Required columns:

- `dataset_id`
- `checkpoint_path`
- `discovery_artifact_path`
- `candidate_count`
- `cluster_count`
- `real_probe_accuracy`
- `shuffled_probe_accuracy`
- `real_probe_bce`
- `shuffled_probe_bce`
- `recurrence_rate`
- `provenance_issue_count`

## Run-Manifest Sidecars

Stage 7 commands also emit sidecars next to their main output artifacts.

Required keys:

- `command_name`
- `created_at_utc`
- `dataset_id`
- `inputs`
- `outputs`
