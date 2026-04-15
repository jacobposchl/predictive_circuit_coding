from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import random
from typing import Any

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
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


def _pooled_feature_tokens(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 2:
        raise ValueError(f"Expected pooled features with shape [windows, d_model], received {tuple(features.shape)}")
    return features.unsqueeze(1), torch.ones((features.shape[0], 1), dtype=torch.bool, device=features.device)


def fit_additive_probe(
    *,
    tokens: torch.Tensor,
    token_mask: torch.Tensor,
    labels: torch.Tensor,
    epochs: int,
    learning_rate: float,
    mini_batch_size: int | None = None,
    label_name: str = "label",
    seed: int | None = None,
) -> ProbeFitResult:
    if tokens.numel() == 0 or token_mask.numel() == 0 or labels.numel() == 0:
        raise ValueError("Cannot fit additive probe because no frozen-token batches were collected.")
    if float(labels.sum().item()) <= 0.0:
        raise ValueError(
            f"Cannot fit additive probe because no positive '{label_name}' labels were found in the sampled windows."
        )
    rng_context = torch.random.fork_rng(devices=[]) if seed is not None else nullcontext()
    with rng_context:
        if seed is not None:
            torch.manual_seed(seed)
        n = tokens.shape[0]
        model = AdditiveTokenProbe(tokens.shape[-1])
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        use_mini_batch = mini_batch_size is not None and mini_batch_size < n
        for _ in range(epochs):
            if use_mini_batch:
                indices = torch.randperm(n)
                for start in range(0, n, mini_batch_size):  # type: ignore[arg-type]
                    batch_idx = indices[start : start + mini_batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    sample_logits, _ = model(tokens[batch_idx], token_mask[batch_idx])
                    loss = F.binary_cross_entropy_with_logits(sample_logits, labels[batch_idx])
                    loss.backward()
                    optimizer.step()
            else:
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


def fit_additive_probe_features(
    *,
    features: torch.Tensor,
    labels: torch.Tensor,
    epochs: int,
    learning_rate: float,
    label_name: str = "label",
    seed: int | None = None,
) -> ProbeFitResult:
    tokens, token_mask = _pooled_feature_tokens(features)
    return fit_additive_probe(
        tokens=tokens,
        token_mask=token_mask,
        labels=labels,
        epochs=epochs,
        learning_rate=learning_rate,
        label_name=label_name,
        seed=seed,
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


def evaluate_additive_probe_features(
    *,
    state_dict: dict,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    tokens, token_mask = _pooled_feature_tokens(features)
    return evaluate_additive_probe(
        state_dict=state_dict,
        tokens=tokens,
        token_mask=token_mask,
        labels=labels,
    )


def probe_logits_from_features(*, state_dict: dict[str, Any], features: torch.Tensor) -> torch.Tensor:
    weight = state_dict["linear.weight"].detach().cpu().reshape(-1).to(dtype=torch.float32)
    bias = float(state_dict["linear.bias"].detach().cpu().item())
    feature_tensor = features.detach().cpu().to(dtype=torch.float32)
    return (feature_tensor @ weight) + bias


def probe_metrics_from_logits(*, sample_logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float | None]:
    probabilities = torch.sigmoid(sample_logits)
    predictions = (probabilities >= 0.5).to(dtype=labels.dtype)
    accuracy = float((predictions == labels).to(dtype=torch.float32).mean().item())
    bce = float(F.binary_cross_entropy_with_logits(sample_logits, labels).item())
    labels_np = labels.detach().cpu().numpy()
    probabilities_np = probabilities.detach().cpu().numpy()
    roc_auc = None
    pr_auc = None
    if len({int(value) for value in labels_np.tolist()}) >= 2:
        roc_auc = float(roc_auc_score(labels_np, probabilities_np))
        pr_auc = float(average_precision_score(labels_np, probabilities_np))
    return {
        "probe_accuracy": accuracy,
        "probe_bce": bce,
        "positive_rate": float(labels.mean().item()),
        "probe_roc_auc": roc_auc,
        "probe_pr_auc": pr_auc,
    }


def fit_shuffled_probe_features(
    *,
    features: torch.Tensor,
    labels: torch.Tensor,
    epochs: int,
    learning_rate: float,
    seed: int,
    label_name: str,
) -> ProbeFitResult:
    permutation = list(range(len(labels)))
    random.Random(int(seed)).shuffle(permutation)
    shuffled_labels = labels.clone()[torch.tensor(permutation, dtype=torch.long)]
    return fit_additive_probe_features(
        features=features,
        labels=shuffled_labels,
        epochs=epochs,
        learning_rate=learning_rate,
        label_name=label_name,
    )
