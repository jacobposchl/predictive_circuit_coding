from __future__ import annotations

import torch
from torch import nn


def _apply_unit_mask(tokens: torch.Tensor, unit_mask: torch.Tensor) -> torch.Tensor:
    return tokens * unit_mask.unsqueeze(-1).unsqueeze(-1).to(dtype=tokens.dtype)


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, *, mlp_ratio: float, dropout: float):
        super().__init__()
        hidden_dim = int(d_model * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalSelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, *, num_heads: int, dropout: float, mlp_ratio: float, norm_eps: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)
        self.ff = FeedForwardBlock(d_model, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, tokens: torch.Tensor, *, unit_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_units, num_patches, dim = tokens.shape
        flat_tokens = tokens.reshape(batch_size * num_units, num_patches, dim)
        flat_unit_mask = unit_mask.reshape(batch_size * num_units)
        output = flat_tokens.clone()
        if flat_unit_mask.any():
            valid_tokens = flat_tokens[flat_unit_mask]
            causal_mask = torch.triu(
                torch.ones((num_patches, num_patches), device=tokens.device, dtype=torch.bool),
                diagonal=1,
            )
            normed = self.norm1(valid_tokens)
            attn_output, _ = self.attn(normed, normed, normed, attn_mask=causal_mask, need_weights=False)
            valid_tokens = valid_tokens + attn_output
            valid_tokens = valid_tokens + self.ff(self.norm2(valid_tokens))
            output[flat_unit_mask] = valid_tokens
        output = output.reshape(batch_size, num_units, num_patches, dim)
        return _apply_unit_mask(output, unit_mask)


class SpatialAttentionBlock(nn.Module):
    def __init__(self, d_model: int, *, num_heads: int, dropout: float, mlp_ratio: float, norm_eps: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)
        self.ff = FeedForwardBlock(d_model, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, tokens: torch.Tensor, *, unit_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_units, num_patches, dim = tokens.shape
        transposed = tokens.permute(0, 2, 1, 3).reshape(batch_size * num_patches, num_units, dim)
        repeated_mask = (~unit_mask).unsqueeze(1).expand(batch_size, num_patches, num_units).reshape(batch_size * num_patches, num_units)
        normed = self.norm1(transposed)
        attn_output, _ = self.attn(
            normed,
            normed,
            normed,
            key_padding_mask=repeated_mask,
            need_weights=False,
        )
        transposed = transposed + attn_output
        transposed = transposed + self.ff(self.norm2(transposed))
        transposed = transposed.reshape(batch_size, num_patches, num_units, dim).permute(0, 2, 1, 3)
        return _apply_unit_mask(transposed, unit_mask)


class SpatiotemporalBlock(nn.Module):
    def __init__(self, d_model: int, *, num_heads: int, dropout: float, mlp_ratio: float, norm_eps: float):
        super().__init__()
        self.spatial = SpatialAttentionBlock(
            d_model,
            num_heads=num_heads,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            norm_eps=norm_eps,
        )
        self.temporal = TemporalSelfAttentionBlock(
            d_model,
            num_heads=num_heads,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            norm_eps=norm_eps,
        )

    def forward(self, tokens: torch.Tensor, *, unit_mask: torch.Tensor) -> torch.Tensor:
        tokens = self.spatial(tokens, unit_mask=unit_mask)
        return self.temporal(tokens, unit_mask=unit_mask)
