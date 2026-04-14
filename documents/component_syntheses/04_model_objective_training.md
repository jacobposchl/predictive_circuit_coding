# Model, Objective, And Training

The model stack contains the predictive encoder, predictive head, reconstruction head, and optional per-patch population tokens.

The objective combines predictive continuation loss with a configurable reconstruction loss. The refined core uses normalized reconstruction targets through `reconstruction_target_mode: window_zscore` and a reduced reconstruction weight.

Training artifacts now record `variant_name`, reconstruction target mode, and count-normalization provenance.
