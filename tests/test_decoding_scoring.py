from __future__ import annotations

from pathlib import Path

import torch

from predictive_circuit_coding.decoding.scoring import select_candidate_tokens_from_shards


def _write_scoring_shard(
    path: Path,
    *,
    session_scores: dict[str, tuple[float, ...]],
) -> Path:
    session_ids: list[str] = []
    subject_ids: list[str] = []
    recording_ids: list[str] = []
    unit_ids: list[str] = []
    unit_regions: list[str] = []
    unit_depth_um: list[float] = []
    patch_index: list[int] = []
    patch_start_s: list[float] = []
    patch_end_s: list[float] = []
    window_start_s: list[float] = []
    window_end_s: list[float] = []
    labels: list[float] = []
    embeddings: list[tuple[float, float]] = []

    for session_id, scores in session_scores.items():
        for index, score in enumerate(scores):
            session_ids.append(session_id)
            subject_ids.append(f"subject_{session_id}")
            recording_ids.append(f"allen_visual_behavior_neuropixels/{session_id}")
            unit_ids.append(f"{session_id}_unit_{index}")
            unit_regions.append("VISp")
            unit_depth_um.append(100.0 + float(index))
            patch_index.append(0)
            patch_start_s.append(float(index))
            patch_end_s.append(float(index) + 0.5)
            window_start_s.append(float(index))
            window_end_s.append(float(index) + 1.0)
            labels.append(1.0)
            embeddings.append((float(score), 0.0))

    for negative_index in range(4):
        session_ids.append(f"negative_session_{negative_index}")
        subject_ids.append(f"negative_subject_{negative_index}")
        recording_ids.append(f"allen_visual_behavior_neuropixels/negative_session_{negative_index}")
        unit_ids.append(f"negative_unit_{negative_index}")
        unit_regions.append("VISp")
        unit_depth_um.append(200.0 + float(negative_index))
        patch_index.append(0)
        patch_start_s.append(10.0 + float(negative_index))
        patch_end_s.append(10.5 + float(negative_index))
        window_start_s.append(10.0 + float(negative_index))
        window_end_s.append(11.0 + float(negative_index))
        labels.append(0.0)
        embeddings.append((0.0, 0.0))

    torch.save(
        {
            "embeddings": torch.tensor(embeddings, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.float32),
            "recording_ids": recording_ids,
            "session_ids": session_ids,
            "subject_ids": subject_ids,
            "unit_ids": unit_ids,
            "unit_regions": unit_regions,
            "unit_depth_um": torch.tensor(unit_depth_um, dtype=torch.float32),
            "patch_index": torch.tensor(patch_index, dtype=torch.long),
            "patch_start_s": torch.tensor(patch_start_s, dtype=torch.float32),
            "patch_end_s": torch.tensor(patch_end_s, dtype=torch.float32),
            "window_start_s": torch.tensor(window_start_s, dtype=torch.float32),
            "window_end_s": torch.tensor(window_end_s, dtype=torch.float32),
        },
        path,
    )
    return path


def _probe_state_dict() -> dict[str, torch.Tensor]:
    return {
        "linear.weight": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        "linear.bias": torch.tensor([0.0], dtype=torch.float32),
    }


def test_select_candidate_tokens_from_shards_balances_sessions_before_backfill(tmp_path: Path) -> None:
    shard_path = _write_scoring_shard(
        tmp_path / "scoring_shard.pt",
        session_scores={
            "session_a": (9.0, 8.0, 7.0, 6.0),
            "session_b": (5.0, 4.0),
            "session_c": (3.0,),
        },
    )

    candidates = select_candidate_tokens_from_shards(
        shard_paths=(shard_path,),
        probe_state_dict=_probe_state_dict(),
        top_k=4,
        min_score=0.0,
        candidate_session_balance_fraction=0.25,
    )

    candidate_session_ids = [candidate.session_id for candidate in candidates]
    assert len(candidates) == 4
    assert candidate_session_ids.count("session_a") == 2
    assert "session_b" in candidate_session_ids
    assert "session_c" in candidate_session_ids
    assert all(candidate.raw_probe_score is not None for candidate in candidates)
    assert all(candidate.negative_background_score is not None for candidate in candidates)
    assert all(candidate.raw_probe_score >= candidate.score for candidate in candidates)


def test_select_candidate_tokens_from_shards_can_disable_session_balancing(tmp_path: Path) -> None:
    shard_path = _write_scoring_shard(
        tmp_path / "scoring_shard.pt",
        session_scores={
            "session_a": (9.0, 8.0, 7.0, 6.0),
            "session_b": (5.0, 4.0),
            "session_c": (3.0,),
        },
    )

    candidates = select_candidate_tokens_from_shards(
        shard_paths=(shard_path,),
        probe_state_dict=_probe_state_dict(),
        top_k=4,
        min_score=0.0,
        candidate_session_balance_fraction=1.0,
    )

    assert [candidate.session_id for candidate in candidates] == [
        "session_a",
        "session_a",
        "session_a",
        "session_a",
    ]
