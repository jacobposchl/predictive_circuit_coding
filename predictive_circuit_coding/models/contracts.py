from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EncoderOutput:
    tokens: torch.Tensor
    unit_mask: torch.Tensor
    patch_mask: torch.Tensor
