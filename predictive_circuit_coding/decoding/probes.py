from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class AdditiveTokenProbe(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        per_token_logits = self.linear(tokens).squeeze(-1)
        mask = token_mask.to(dtype=per_token_logits.dtype)
        token_counts = mask.sum(dim=1).clamp_min(1.0)
        sample_logits = (per_token_logits * mask).sum(dim=1) / token_counts
        return sample_logits, per_token_logits


@dataclass(frozen=True)
class ProbeFitResult:
    state_dict: dict
    metrics: dict[str, float]


def _compute_probe_metrics(
    *,
    model: AdditiveTokenProbe,
    tokens: torch.Tensor,
    token_mask: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        sample_logits, _ = model(tokens, token_mask)
        probabilities = torch.sigmoid(sample_logits)
        predictions = (probabilities >= 0.5).to(dtype=labels.dtype)
        accuracy = float((predictions == labels).to(dtype=torch.float32).mean().item())
        loss = float(F.binary_cross_entropy_with_logits(sample_logits, labels).item())
    return {
        "probe_accuracy": accuracy,
        "probe_bce": loss,
        "positive_rate": float(labels.mean().item()),
    }


def fit_additive_probe(
    *,
    tokens: torch.Tensor,
    token_mask: torch.Tensor,
    labels: torch.Tensor,
    epochs: int,
    learning_rate: float,
    label_name: str = "label",
) -> ProbeFitResult:
    if tokens.numel() == 0 or token_mask.numel() == 0 or labels.numel() == 0:
        raise ValueError("Cannot fit additive probe because no frozen-token batches were collected.")
    if float(labels.sum().item()) <= 0.0:
        raise ValueError(
            f"Cannot fit additive probe because no positive '{label_name}' labels were found in the sampled windows."
        )
    model = AdditiveTokenProbe(tokens.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        sample_logits, _ = model(tokens, token_mask)
        loss = F.binary_cross_entropy_with_logits(sample_logits, labels)
        loss.backward()
        optimizer.step()
    return ProbeFitResult(
        state_dict=model.state_dict(),
        metrics=_compute_probe_metrics(
            model=model,
            tokens=tokens,
            token_mask=token_mask,
            labels=labels,
        ),
    )


def evaluate_additive_probe(
    *,
    state_dict: dict,
    tokens: torch.Tensor,
    token_mask: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    if tokens.numel() == 0 or token_mask.numel() == 0 or labels.numel() == 0:
        raise ValueError("Cannot evaluate additive probe because no frozen-token batches were collected.")
    model = AdditiveTokenProbe(tokens.shape[-1])
    model.load_state_dict(state_dict)
    model.eval()
    return _compute_probe_metrics(
        model=model,
        tokens=tokens,
        token_mask=token_mask,
        labels=labels,
    )
