# Full-Pipeline Results Synthesis

This note summarizes the current claim-facing methodology, benchmark design, and results from the unified Colab pipeline configured by:

- `configs/pcc/pipeline_full.yaml`
- `configs/pcc/predictive_circuit_coding_full.yaml`

It is intended as a paper-planning document: a compact but in-depth statement of what we tested, what we found, and what we can honestly claim at the current stage of the project.

## Executive Summary

The strongest result so far is a **representation-learning result**, not a motif-discovery result:

- frozen features from the trained predictive encoder outperform simple count-based and PCA-based baselines for cross-session decoding of `stimulus_change` and `trials.go`
- whitening is **not** a universal improvement; in the final crossed benchmark it usually hurts held-out decoding
- motif behavior is more nuanced:
  - raw encoder features decode best in the motif benchmark
  - whitened encoder features produce better motif similarity than raw encoder features
  - however, cross-session motif transfer remains modest overall
- `stimulus_presentations.omitted` was degraded across the full benchmark run and should currently be treated as a blocked or unresolved task rather than a negative scientific result

The current strongest paper claim is therefore:

> A predictive population-dynamics encoder learns a frozen neural representation that outperforms simple fixed-window count and PCA baselines for held-out cross-session decoding of behaviorally relevant variables.

The current motif claim is more cautious:

> Motif recoverability depends strongly on latent geometry. Raw encoder features decode best, while whitened encoder features can modestly improve motif transfer, suggesting that geometry correction helps motif matching more than decoding.

## 1. Methodology

### 1.1 Dataset and split policy

The full pipeline currently uses the Allen Visual Behavior Neuropixels prepared dataset with the following selection:

- `experience_levels: [Familiar]`
- `image_sets: [G]`
- no fixed session list committed in the config
- split by `subject`, not by session

The runtime split fractions are:

- `train_fraction: 0.5`
- `valid_fraction: 0.2`
- `discovery_fraction: 0.15`
- `test_fraction: 0.15`

An important methodological detail is that the exact set of sessions is not hard-coded in `pipeline_full.yaml`. Instead, the run resolves all sessions matching the Familiar/G filter at runtime from the prepared session catalog, then performs the subject-level split. That is acceptable for current benchmarking, but a final paper-ready run would be easier to describe if it pinned the selected session IDs in a committed manifest.

### 1.2 Model and training configuration

The encoder used in the full benchmark is defined by `predictive_circuit_coding_full.yaml`:

- `d_model: 256`
- `num_heads: 8`
- `temporal_layers: 2`
- `spatial_layers: 2`
- `dropout: 0.1`
- `l2_normalize_tokens: true`

Training objective:

- predictive target: `delta`
- continuation baseline: `previous_patch`
- predictive loss: `mse`
- reconstruction loss: `mse`
- reconstruction weight: `0.2`

Optimization/training:

- batch size: `4`
- learning rate: `1e-4`
- weight decay: `1e-4`
- scheduler: cosine
- `num_epochs: 8`
- `train_steps_per_epoch: 128`
- `validation_steps: 16`

The scientific role of this model in the benchmarks is as a **frozen feature extractor**. After training, the encoder is not further fine-tuned for downstream tasks. All downstream results come from simple probes or motif-discovery procedures applied to frozen features.

### 1.3 Representation benchmark design

The representation benchmark asks a narrow but important question:

> Does the trained encoder provide a better frozen feature space than simple fixed-window baselines?

The crossed benchmark explicitly varies two factors:

1. feature family
2. geometry treatment

Representation arms:

- `count_patch_mean_raw`
- `count_patch_mean_whitened`
- `count_patch_mean_pca_raw`
- `count_patch_mean_pca_whitened`
- `encoder_raw`
- `encoder_whitened`

Operationally:

- `count_patch_mean_*` uses mean-pooled flattened patch-count tokens built directly from the tokenizer output
- `count_patch_mean_pca_*` applies global PCA to those count-based pooled features
- `encoder_*` uses mean-pooled frozen encoder token features
- `raw` means features are used directly
- `whitened` means per-session whitening is fit on the discovery-fit subset and applied downstream

Tasks included in the full representation benchmark:

- `stimulus_change`
- `trials.go`
- `stimulus_presentations.omitted`

Primary metrics:

- `test_probe_accuracy`
- `test_probe_bce`
- `test_probe_pr_auc`
- `test_probe_roc_auc`

Secondary diagnostics:

- `within_session_probe_pr_auc`
- `within_session_probe_roc_auc`
- neighbor enrichment summaries for label/session/subject structure

Interpretation rule:

- the cross-session test metrics are the main evidence for the representation claim
- the within-session metrics are secondary and help separate cross-session transfer from simpler within-session separability

### 1.4 Motif benchmark design

The motif benchmark asks a different question:

> Can the frozen representation support transferable motif discovery, rather than only probe-based decoding?

Motif arms:

- `count_patch_mean_pca_raw`
- `count_patch_mean_pca_whitened`
- `encoder_raw`
- `encoder_whitened`

The motif pipeline uses:

- `discovery.max_batches: 128`
- `top_k_candidates: 64`
- `min_candidate_score: 0.0`
- `min_cluster_size: 2`
- `stability_rounds: 8`
- a within-session held-out split of the discovery windows (`session_holdout_fraction: 0.5`)

For each task/arm, the pipeline:

1. extracts frozen features and token embeddings
2. fits the appropriate transforms
3. selects candidate tokens from discovery windows
4. clusters candidates
5. estimates clustering stability
6. evaluates the resulting motif summaries on held-out data

Primary motif metrics:

- `held_out_test_probe_pr_auc`
- `held_out_test_probe_roc_auc`
- `held_out_similarity_pr_auc`
- `held_out_similarity_roc_auc`
- `candidate_count`
- `cluster_count`
- `cluster_persistence_mean`
- `silhouette_score`

Interpretation rule:

- held-out probe metrics tell us whether the discovered windows still carry task signal
- held-out similarity metrics tell us whether the discovered clusters behave like transferable motifs
- a high probe score with weak similarity means “the representation contains the signal, but the motifs are not transferring”

## 2. Experimental Questions

The full benchmark is trying to resolve four questions.

### Q1. Does the predictive encoder beat simple fixed-window baselines?

This is the core representation question. The relevant comparison is:

- `encoder_raw` vs `count_patch_mean_raw`
- `encoder_raw` vs `count_patch_mean_pca_raw`

### Q2. Does whitening improve downstream performance?

This must be answered separately for:

- representation decoding
- motif discovery

The final crossed design is important here because it prevents conflating:

- “the encoder is better”
- with
- “whitening helps”

### Q3. Are motifs easier to recover than decoders?

This is really a gap question:

- if probe metrics are high but motif similarity is low, then the representation contains information that is not condensing into robust, reusable motifs

### Q4. Which task variables are actually supported by the current representation?

The current task panel includes:

- `stimulus_change`
- `trials.go`
- `stimulus_presentations.omitted`

This matters because different variables may be differently aligned with the encoder’s geometry and the motif-discovery method.

## 3. Results

## 3.1 Representation benchmark

Completed tasks:

- `stimulus_change`
- `trials.go`

Degraded task:

- `stimulus_presentations.omitted`

### 3.1.1 Stimulus change

Cross-session test results:

| Arm | Accuracy | PR-AUC | ROC-AUC |
| --- | ---: | ---: | ---: |
| `encoder_raw` | 0.652 | 0.545 | 0.721 |
| `count_patch_mean_raw` | 0.615 | 0.523 | 0.684 |
| `count_patch_mean_pca_raw` | 0.615 | 0.506 | 0.649 |
| `encoder_whitened` | 0.574 | 0.464 | 0.593 |
| `count_patch_mean_pca_whitened` | 0.555 | 0.437 | 0.541 |
| `count_patch_mean_whitened` | 0.520 | 0.411 | 0.522 |

Interpretation:

- `encoder_raw` is the strongest arm on the main held-out representation metrics
- the raw count baseline is competitive but consistently worse
- PCA does not improve the count baseline for this task
- whitening clearly hurts both count-based and encoder-based decoding here

### 3.1.2 Trials go

Cross-session test results:

| Arm | Accuracy | PR-AUC | ROC-AUC |
| --- | ---: | ---: | ---: |
| `encoder_raw` | 0.645 | 0.520 | 0.711 |
| `count_patch_mean_raw` | 0.621 | 0.497 | 0.678 |
| `count_patch_mean_pca_raw` | 0.621 | 0.455 | 0.628 |
| `encoder_whitened` | 0.590 | 0.470 | 0.600 |
| `count_patch_mean_pca_whitened` | 0.525 | 0.380 | 0.498 |
| `count_patch_mean_whitened` | 0.482 | 0.378 | 0.480 |

Interpretation:

- the same ranking pattern appears again
- `encoder_raw` is best on all three main held-out metrics
- whitening again hurts representation decoding
- the count-based baselines are not trivial, but they do not catch the encoder

### 3.1.3 Stimulus omitted

All representation rows for `stimulus_presentations.omitted` were degraded and reported `NaN` metrics.

This should currently be treated as:

- a blocked experiment
- not a negative scientific finding

Until the failure reason is inspected and corrected, the omitted-task rows should not be used in any paper-level claim.

### 3.1.4 Representation synthesis

The representation results are the cleanest part of the current project story:

- the trained predictive encoder outperforms simple count-based baselines on both completed tasks
- PCA does not close that gap
- whitening does not improve the main held-out representation metrics and usually harms them

The main representation claim is therefore already supported by the current data.

## 3.2 Motif benchmark

The motif benchmark tells a more nuanced story.

### 3.2.1 Stimulus change

| Arm | Probe PR-AUC | Probe ROC-AUC | Similarity PR-AUC | Similarity ROC-AUC | Clusters |
| --- | ---: | ---: | ---: | ---: | ---: |
| `encoder_raw` | 0.546 | 0.721 | 0.288 | 0.318 | 1 |
| `encoder_whitened` | 0.514 | 0.635 | 0.420 | 0.517 | 1 |
| `count_patch_mean_pca_raw` | 0.491 | 0.619 | 0.406 | 0.518 | 2 |
| `count_patch_mean_pca_whitened` | 0.439 | 0.547 | 0.389 | 0.472 | 3 |

Interpretation:

- `encoder_raw` has the strongest held-out probe performance
- but `encoder_raw` has the weakest held-out motif similarity by far
- `encoder_whitened` improves motif similarity substantially relative to `encoder_raw`
- the PCA count baselines are not obviously worse than the whitened encoder on similarity for this task

This is an important mismatch:

- the raw encoder representation contains strong task information
- but its discovered clusters do not behave like transferable motifs

### 3.2.2 Trials go

| Arm | Probe PR-AUC | Probe ROC-AUC | Similarity PR-AUC | Similarity ROC-AUC | Clusters |
| --- | ---: | ---: | ---: | ---: | ---: |
| `encoder_raw` | 0.520 | 0.711 | 0.286 | 0.328 | 1 |
| `encoder_whitened` | 0.501 | 0.626 | 0.409 | 0.507 | 1 |
| `count_patch_mean_pca_raw` | 0.502 | 0.653 | 0.388 | 0.498 | 1 |
| `count_patch_mean_pca_whitened` | 0.388 | 0.495 | 0.417 | 0.548 | 5 |

Interpretation:

- the same pattern repeats:
  - raw encoder probe metrics are strong
  - raw encoder similarity metrics are weak
- whitening improves encoder motif similarity
- the best similarity result on this task is actually `count_patch_mean_pca_whitened`, not an encoder arm

That means the motif story is currently weaker and more task-dependent than the representation story.

### 3.2.3 Stimulus omitted

All motif rows for `stimulus_presentations.omitted` were degraded:

- `candidate_count = 0`
- `cluster_count = 0`
- all downstream motif metrics are `NaN`

Again, this should be treated as an unresolved task, not as a scientific conclusion.

### 3.2.4 Motif synthesis

The motif benchmark currently supports the following synthesis:

- motif discovery is harder than frozen decoding
- raw encoder features are best for probe-style task readout
- whitened encoder features can improve motif similarity relative to raw encoder features
- however, overall motif similarity remains modest
- the current motif pipeline has not yet established a strong claim of robust canonical cross-session motifs

## 4. Cross-Cutting Interpretation

## 4.1 What the benchmark shows clearly

The most defensible conclusion is:

1. the predictive encoder is learning something useful
2. this usefulness is visible in frozen cross-session decoding
3. the learned feature space is better than simple count/PCA baselines on the completed tasks

That is the strongest current scientific claim.

## 4.2 What the benchmark shows about geometry

The current full benchmark does **not** support the blanket statement:

> whitening improves downstream performance

Instead, it supports a more careful statement:

- whitening hurts the main cross-session decoding metrics
- whitening can improve motif-transfer metrics, especially for encoder-based motifs

So whitening is not a general win. It is better understood as a geometry intervention with different effects on:

- linear readout
- motif matching

## 4.3 What the benchmark shows about motifs

The most interesting methodological finding is the mismatch between decoding and motif transfer:

- `encoder_raw` gives the strongest held-out probe metrics
- but `encoder_raw` gives poor held-out similarity metrics

That implies:

- the representation contains the relevant signal
- but the current cluster/similarity machinery is not extracting that signal into transferable motifs in raw latent space

This is not a failure of the entire project. It is a specific and informative result:

- representation learning is ahead of motif discovery

## 4.4 What the benchmark does not yet show

The current benchmark does **not** yet justify claims that:

- the model discovers robust universal motifs across all subjects/sessions
- whitening is broadly beneficial
- `stimulus_presentations.omitted` is unsupported by the encoder
- the predictive encoder is superior to all alternative learned models

The current baselines are simple but meaningful:

- direct count-based features
- PCA-reduced count features

They are appropriate for a first paper claim, but they are not an exhaustive model comparison.

## 5. Strongest Paper Claim Right Now

The strongest paper-ready claim is:

> A predictive population-dynamics encoder learns frozen neural features that outperform simple fixed-window count and PCA baselines for held-out cross-session decoding of behaviorally relevant variables, including `stimulus_change` and `trials.go`.

This claim is attractive because it is:

- clear
- supported on multiple tasks
- grounded in the primary held-out metrics
- not dependent on the weaker motif results

A good secondary claim is:

> Motif quality is geometry-sensitive: whitening improves motif similarity more than it improves decoding, indicating that the geometry that is best for readout is not necessarily the geometry that is best for transferable motif matching.

## 6. Limitations and Open Problems

### 6.1 Task coverage is incomplete

`stimulus_presentations.omitted` is not currently interpretable because the task degraded across both representation and motif benchmarks.

### 6.2 Motif transfer is still modest

Even the best motif-similarity numbers are only moderate:

- they are enough to be interesting
- they are not yet strong enough to support a major “canonical motif” claim

### 6.3 Exact full-run session membership is not pinned in config

The full config uses runtime selection from the Familiar/G pool rather than a fixed committed session list. That is acceptable for current research iteration but should ideally be frozen for a final paper run.

### 6.4 Baseline scope remains limited

The current representation claim is relative to:

- count features
- PCA count features

It is not yet a comparison against:

- untrained encoders
- alternate predictive objectives
- other self-supervised or sequence-model baselines

## 7. Recommended Next Steps

1. Resolve `stimulus_presentations.omitted` and rerun the full benchmark.
2. Freeze a final full-run session manifest so the exact cohort is fixed and easy to report.
3. Treat `encoder_raw` as the main representation arm for the decoding claim.
4. Treat `encoder_whitened` as a geometry-control arm for motif analyses, not as the default representation arm.
5. If paper scope allows, add one more learned baseline such as an untrained encoder or another simple sequence baseline.
6. Keep the motif claim modest unless a stronger cross-session motif result emerges.

## 8. Bottom Line

At the current stage of the project:

- the **representation-learning story is real and publishable**
- the **motif story is promising but still partial**
- the project’s main contribution is presently best framed as:
  - better frozen cross-session neural representations from predictive learning
  - plus evidence that motif discovery depends strongly on latent geometry

That is already a meaningful and coherent result, even before the motif story is fully resolved.
