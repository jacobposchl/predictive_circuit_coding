# Troubleshooting

## Allen Prep Problems

`brainsets` or AllenSDK dependency conflicts:

- use the dedicated Allen prep environment instead of the main training env

No processed sessions were produced:

- confirm the Allen prep command completed successfully
- confirm the processed output directory contains `.h5` files
- rerun with a small subset using `--max-sessions 1`

## Training Problems

Split manifest missing:

- rerun local prep so `split_manifest.json` and the split YAML files exist

Resume checkpoint missing:

- clear `training.resume_checkpoint` in the experiment config or point it at an existing checkpoint

Checkpoint dataset mismatch:

- use a checkpoint produced from the same dataset named in the experiment config

## Discovery Problems

No positive `stimulus_change` labels:

- the sampled windows on the discovery split did not include positive events
- increase discovery coverage or adjust the sequential step / max batches

No candidate tokens:

- lower `discovery.min_candidate_score`
- increase `discovery.max_batches`

No clusters after clustering:

- lower `discovery.cluster_similarity_threshold`
- reduce `discovery.min_cluster_size`

## Validation Problems

No recurrence hits:

- this does not crash validation, but it weakens the held-out recurrence story
- check whether the discovery artifact is too small or too noisy

Discovery artifact dataset mismatch:

- ensure the artifact came from the same dataset/checkpoint family as the current run
