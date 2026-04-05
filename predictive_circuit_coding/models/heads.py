from __future__ import annotations

from torch import nn


class PredictiveHead(nn.Module):
    def __init__(self, d_model: int, *, patch_bins: int):
        super().__init__()
        self.proj = nn.Linear(d_model, patch_bins)

    def forward(self, tokens):
        return self.proj(tokens)


class ReconstructionHead(nn.Module):
    def __init__(self, d_model: int, *, patch_bins: int):
        super().__init__()
        self.proj = nn.Linear(d_model, patch_bins)

    def forward(self, tokens):
        return self.proj(tokens)
