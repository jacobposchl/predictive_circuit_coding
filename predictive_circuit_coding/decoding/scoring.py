from __future__ import annotations

import torch

from predictive_circuit_coding.training.contracts import FrozenTokenRecord


def score_token_records(
    *,
    records: tuple[FrozenTokenRecord, ...],
    probe_state_dict: dict,
) -> tuple[FrozenTokenRecord, ...]:
    weight = probe_state_dict["linear.weight"].detach().cpu().reshape(-1)
    bias = float(probe_state_dict["linear.bias"].detach().cpu().item())
    scored: list[FrozenTokenRecord] = []
    for record in records:
        embedding = torch.tensor(record.embedding, dtype=torch.float32)
        score = float(torch.dot(embedding, weight).item() + bias)
        scored.append(
            FrozenTokenRecord(
                recording_id=record.recording_id,
                session_id=record.session_id,
                subject_id=record.subject_id,
                unit_id=record.unit_id,
                unit_region=record.unit_region,
                unit_depth_um=record.unit_depth_um,
                patch_index=record.patch_index,
                patch_start_s=record.patch_start_s,
                patch_end_s=record.patch_end_s,
                window_start_s=record.window_start_s,
                window_end_s=record.window_end_s,
                label=record.label,
                score=score,
                embedding=record.embedding,
            )
        )
    return tuple(scored)
