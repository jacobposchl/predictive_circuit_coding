from __future__ import annotations

from predictive_circuit_coding.training.contracts import CandidateTokenRecord, FrozenTokenRecord


def select_candidate_tokens(
    *,
    scored_records: tuple[FrozenTokenRecord, ...],
    top_k: int,
    min_score: float,
) -> tuple[CandidateTokenRecord, ...]:
    ranked = sorted(
        (record for record in scored_records if record.label == 1 and record.score >= min_score),
        key=lambda item: item.score,
        reverse=True,
    )[:top_k]
    return tuple(
        CandidateTokenRecord(
            candidate_id=f"candidate_{index:04d}",
            cluster_id=-1,
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
            score=record.score,
            embedding=record.embedding,
        )
        for index, record in enumerate(ranked)
    )
