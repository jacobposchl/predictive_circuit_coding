from __future__ import annotations

import heapq
from pathlib import Path

import torch

from predictive_circuit_coding.training.contracts import CandidateTokenRecord, FrozenTokenRecord


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


def select_candidate_tokens_from_shards(
    *,
    shard_paths: tuple[str | Path, ...],
    probe_state_dict: dict,
    top_k: int,
    min_score: float,
) -> tuple[CandidateTokenRecord, ...]:
    weight = probe_state_dict["linear.weight"].detach().cpu().reshape(-1)
    bias = float(probe_state_dict["linear.bias"].detach().cpu().item())
    heap: list[tuple[float, int, dict]] = []
    counter = 0

    for shard_path in shard_paths:
        payload = torch.load(Path(shard_path), map_location="cpu", weights_only=False)
        embeddings = payload["embeddings"].to(dtype=torch.float32)
        if embeddings.numel() == 0:
            continue
        scores = (embeddings @ weight) + bias
        keep_indices = torch.nonzero(scores >= float(min_score), as_tuple=False).flatten().tolist()
        for index in keep_indices:
            score = float(scores[index].item())
            row = {
                "recording_id": payload["recording_ids"][index],
                "session_id": payload["session_ids"][index],
                "subject_id": payload["subject_ids"][index],
                "unit_id": payload["unit_ids"][index],
                "unit_region": payload["unit_regions"][index],
                "unit_depth_um": float(payload["unit_depth_um"][index].item()),
                "patch_index": int(payload["patch_index"][index].item()),
                "patch_start_s": float(payload["patch_start_s"][index].item()),
                "patch_end_s": float(payload["patch_end_s"][index].item()),
                "window_start_s": float(payload["window_start_s"][index].item()),
                "window_end_s": float(payload["window_end_s"][index].item()),
                "embedding": tuple(float(value) for value in embeddings[index].tolist()),
                "score": score,
            }
            if top_k <= 0:
                continue
            if len(heap) < top_k:
                heapq.heappush(heap, (score, counter, row))
                counter += 1
                continue
            if score > heap[0][0]:
                heapq.heapreplace(heap, (score, counter, row))
                counter += 1

    ranked_rows = [item[2] for item in sorted(heap, key=lambda item: item[0], reverse=True)]
    return tuple(
        CandidateTokenRecord(
            candidate_id=f"candidate_{index:04d}",
            cluster_id=-1,
            recording_id=row["recording_id"],
            session_id=row["session_id"],
            subject_id=row["subject_id"],
            unit_id=row["unit_id"],
            unit_region=row["unit_region"],
            unit_depth_um=row["unit_depth_um"],
            patch_index=row["patch_index"],
            patch_start_s=row["patch_start_s"],
            patch_end_s=row["patch_end_s"],
            window_start_s=row["window_start_s"],
            window_end_s=row["window_end_s"],
            label=1,
            score=row["score"],
            embedding=row["embedding"],
        )
        for index, row in enumerate(ranked_rows)
    )
