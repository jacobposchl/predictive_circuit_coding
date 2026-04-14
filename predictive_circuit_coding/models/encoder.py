from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from predictive_circuit_coding.models.blocks import SpatiotemporalBlock, TemporalSelfAttentionBlock
from predictive_circuit_coding.models.contracts import EncoderOutput
from predictive_circuit_coding.models.heads import PredictiveHead, ReconstructionHead
from predictive_circuit_coding.training.config import ModelConfig
from predictive_circuit_coding.training.contracts import ModelForwardOutput, PopulationWindowBatch


class PatchEmbedder(nn.Module):
    def __init__(self, *, patch_bins: int, d_model: int, num_patches: int):
        super().__init__()
        self.proj = nn.Linear(patch_bins, d_model)
        self.patch_positions = nn.Embedding(num_patches, d_model)

    def forward(self, patch_counts: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(patch_counts)
        patch_indices = torch.arange(patch_counts.shape[2], device=patch_counts.device)
        patch_positions = self.patch_positions(patch_indices).view(1, 1, patch_counts.shape[2], -1)
        return tokens + patch_positions


class PredictiveCircuitEncoder(nn.Module):
    def __init__(self, *, config: ModelConfig, patch_bins: int, num_patches: int):
        super().__init__()
        self.config = config
        self.embedder = PatchEmbedder(patch_bins=patch_bins, d_model=config.d_model, num_patches=num_patches)
        self.population_tokens = (
            nn.Parameter(torch.zeros(1, 1, num_patches, config.d_model))
            if config.population_token_mode == "per_patch_cls"
            else None
        )
        self.temporal_stack = nn.ModuleList(
            [
                TemporalSelfAttentionBlock(
                    config.d_model,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    mlp_ratio=config.mlp_ratio,
                    norm_eps=config.norm_eps,
                )
                for _ in range(config.temporal_layers)
            ]
        )
        self.spatiotemporal_stack = nn.ModuleList(
            [
                SpatiotemporalBlock(
                    config.d_model,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    mlp_ratio=config.mlp_ratio,
                    norm_eps=config.norm_eps,
                )
                for _ in range(config.spatial_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.norm_eps)

    def forward(self, batch: PopulationWindowBatch) -> EncoderOutput:
        tokens = self.embedder(batch.patch_counts)
        unit_mask = batch.unit_mask
        population_token_count = 0
        if self.population_tokens is not None:
            population_token_count = 1
            population_tokens = self.population_tokens.expand(tokens.shape[0], -1, -1, -1)
            tokens = torch.cat([population_tokens, tokens], dim=1)
            population_mask = torch.ones((unit_mask.shape[0], 1), dtype=unit_mask.dtype, device=unit_mask.device)
            unit_mask = torch.cat([population_mask, unit_mask], dim=1)
        for block in self.temporal_stack:
            tokens = block(tokens, unit_mask=unit_mask)
        for block in self.spatiotemporal_stack:
            tokens = block(tokens, unit_mask=unit_mask)
        tokens = self.final_norm(tokens)
        if self.config.l2_normalize_tokens:
            tokens = F.normalize(tokens, dim=-1)
        summary_tokens = None
        if population_token_count:
            summary_tokens = tokens[:, :population_token_count].squeeze(1)
            tokens = tokens[:, population_token_count:]
        tokens = tokens * batch.patch_mask.unsqueeze(-1).to(dtype=tokens.dtype)
        return EncoderOutput(
            tokens=tokens,
            unit_mask=batch.unit_mask,
            patch_mask=batch.patch_mask,
            summary_tokens=summary_tokens,
        )


class PredictiveCircuitModel(nn.Module):
    def __init__(self, *, model_config: ModelConfig, patch_bins: int, num_patches: int):
        super().__init__()
        self.encoder = PredictiveCircuitEncoder(
            config=model_config,
            patch_bins=patch_bins,
            num_patches=num_patches,
        )
        self.predictive_head = PredictiveHead(model_config.d_model, patch_bins=patch_bins)
        self.reconstruction_head = ReconstructionHead(model_config.d_model, patch_bins=patch_bins)

    def forward(self, batch: PopulationWindowBatch) -> ModelForwardOutput:
        encoder_output = self.encoder(batch)
        predictive_outputs = self.predictive_head(encoder_output.tokens)
        reconstruction_outputs = self.reconstruction_head(encoder_output.tokens)
        predictive_outputs = predictive_outputs * batch.patch_mask.unsqueeze(-1).to(dtype=predictive_outputs.dtype)
        reconstruction_outputs = reconstruction_outputs * batch.patch_mask.unsqueeze(-1).to(dtype=reconstruction_outputs.dtype)
        return ModelForwardOutput(
            tokens=encoder_output.tokens,
            predictive_outputs=predictive_outputs,
            reconstruction_outputs=reconstruction_outputs,
            unit_mask=encoder_output.unit_mask,
            patch_mask=encoder_output.patch_mask,
            summary_tokens=encoder_output.summary_tokens,
        )
