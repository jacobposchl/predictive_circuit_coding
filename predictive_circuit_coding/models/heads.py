from __future__ import annotations

import torch
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


class RegionRatePredictiveHead(nn.Module):
    def __init__(self, d_model: int, *, num_regions: int):
        super().__init__()
        self.proj = nn.Linear(d_model, num_regions)

    def forward(self, tokens: torch.Tensor, unit_mask: torch.Tensor) -> torch.Tensor:
        mask = unit_mask.to(dtype=tokens.dtype).unsqueeze(-1).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled_tokens = (tokens * mask).sum(dim=1) / denom
        predicted = self.proj(pooled_tokens)
        return predicted.permute(0, 2, 1)
