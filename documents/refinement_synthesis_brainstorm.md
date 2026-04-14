# Refinement Synthesis: `predictive_circuit_coding`

**Status:** Implemented as the current refinement workflow  
**Scope:** Architecture and experiment cleanup for motif-transfer diagnosis

## 1. Current Direction

The project now centers on one compact loop:

1. train a predictive encoder
2. run trained-encoder refinement discovery arms
3. validate candidate discoveries conservatively
4. compare a small set of one-axis training ablations

The main research question is no longer whether a broad suite of alternative model families can be ranked. The immediate question is whether objective, normalization, geometry, and pooling refinements make the main predictive-encoder idea work cleanly enough to support transferable motif discovery.

## 2. Live Refinements

### 2.1 Trained-Encoder Refinement Arms

`pcc-refine` compares trained encoder outputs only:

- `encoder_raw`: direct trained-encoder geometry
- `encoder_token_normalized`: post-hoc L2 normalization of pooled features and token shards
- `encoder_probe_weighted`: candidate and held-out token embeddings scaled by additive-probe weight magnitude
- `encoder_aligned_oracle`: label-guided session alignment diagnostic

The oracle-aligned arm is intentionally marked `claim_safe=false`. It is useful for diagnosing session coordinate drift, but it should not be interpreted as claim-facing evidence when held-out labels are used to define held-out transforms.

### 2.2 Objective And Reconstruction

The refined core reduces reconstruction pressure and uses normalized reconstruction targets:

- `objective.reconstruction_weight: 0.05`
- `objective.reconstruction_target_mode: window_zscore`

The `refined_recon000` ablation sets reconstruction weight to zero. This keeps the comparison focused on whether reconstruction is regularizing useful structure or anchoring tokens too strongly to session-specific count scale.

### 2.3 Token And Count Normalization

Training-time token L2 normalization remains a first-class model knob:

- `refined_core`: token L2 enabled
- `refined_l2off`: token L2 disabled

Input count normalization is separated from token geometry:

- `count_normalization.mode: none`
- `count_normalization.mode: log1p_train_zscore`

The count-normalization ablation fits scalar log-count statistics from the training split only, writes a JSON stats artifact, and records the mode/path/hash in checkpoint metadata.

### 2.4 Population Pooling

The default discovery feature is still mean pooling over biological unit tokens. The `refined_cls` ablation enables:

- `model.population_token_mode: per_patch_cls`
- `discovery.pooled_feature_mode: cls_tokens`

The CLS-like population token is exposed for pooled discovery features while biological unit token shapes remain unchanged for prediction and reconstruction heads.

## 3. Interpretation Guardrails

- Probe-guided arms answer whether frozen predictive representations contain task-relevant directions that can guide motif extraction.
- Similarity metrics should be read alongside held-out probe metrics, cluster counts, and validation status.
- Diagnostic rows must remain explicitly marked as not claim-safe.
- Count normalization must remain leakage-safe: training statistics are fit on the train split and then reloaded for evaluation, discovery, and validation.
- One-axis ablations are preferred over combinatorial sweeps until the main idea is stable.

## 4. Remaining Refinement Questions

The current workflow is now ready to test whether:

- normalized reconstruction improves transfer without erasing predictive signal
- removing reconstruction entirely helps or destabilizes the encoder
- token L2 normalization is useful during training or only post hoc
- train-split count normalization reduces high-rate-unit dominance
- CLS pooling improves window-level probe quality and downstream candidate selection
- oracle alignment reveals a session-geometry ceiling worth converting into a claim-safe future method

## 5. Practical Sequence

Start with `refined_core` and the four refinement arms. Then run the one-axis ablations:

1. `refined_recon000`
2. `refined_l2off`
3. `refined_countnorm`
4. `refined_cls`

For each variant, compare the same trained-encoder refinement arms and keep the report focused on variant name, refinement arm, task label, probe metrics, cluster stability, held-out motif similarity, validation status, and claim-safety metadata.
