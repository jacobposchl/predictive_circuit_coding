from __future__ import annotations

import heapq
import math
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from predictive_circuit_coding.training.contracts import CandidateTokenRecord, FrozenTokenRecord


_ScoredHeapEntry = tuple[float, int, dict[str, Any]]


def pooled_features_from_tokens(tokens: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    mask = token_mask.to(dtype=tokens.dtype).unsqueeze(-1)
    token_counts = mask.sum(dim=1).clamp_min(1.0)
    return (tokens * mask).sum(dim=1) / token_counts


def candidate_centroids(
    candidates: tuple[Any, ...] | list[Any],
    *,
    require_non_noise: bool = False,
    empty_message: str = "Discovery artifact does not contain any non-noise candidate clusters to validate.",
) -> list[torch.Tensor]:
    grouped: dict[int, list[torch.Tensor]] = {}
    for candidate in candidates:
        cluster_id = int(candidate["cluster_id"] if isinstance(candidate, dict) else candidate.cluster_id)
        if cluster_id == -1:
            continue
        embedding = candidate["embedding"] if isinstance(candidate, dict) else candidate.embedding
        grouped.setdefault(cluster_id, []).append(torch.tensor(embedding, dtype=torch.float32))
    if require_non_noise and not grouped:
        raise ValueError(empty_message)
    return [torch.stack(values, dim=0).mean(dim=0) for _, values in sorted(grouped.items())]


def window_similarity_scores(
    *,
    tokens: torch.Tensor,
    token_mask: torch.Tensor,
    centroids: list[torch.Tensor],
    require_centroids: bool = False,
) -> torch.Tensor:
    if tokens.numel() == 0 or token_mask.numel() == 0:
        return torch.empty((0,), dtype=torch.float32)
    if not centroids:
        if require_centroids:
            raise ValueError("Cannot compute motif similarity without at least one candidate centroid.")
        return torch.zeros((tokens.shape[0],), dtype=torch.float32)
    centroid_tensor = torch.stack(
        [centroid / centroid.norm().clamp_min(1.0e-8) for centroid in centroids],
        dim=0,
    )
    scores = torch.empty((tokens.shape[0],), dtype=torch.float32)
    for start in range(0, tokens.shape[0], 512):
        end = min(start + 512, tokens.shape[0])
        chunk = tokens[start:end]
        chunk_mask = token_mask[start:end].to(dtype=torch.bool)
        normalized_chunk = chunk / chunk.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        chunk_sims = torch.einsum("wtd,cd->wtc", normalized_chunk, centroid_tensor)
        chunk_sims = chunk_sims.masked_fill(~chunk_mask.unsqueeze(-1), float("-inf"))
        chunk_scores = chunk_sims.amax(dim=(1, 2)).to(dtype=torch.float32)
        scores[start:end] = torch.where(chunk_mask.any(dim=1), chunk_scores, torch.zeros_like(chunk_scores))
    return scores


def held_out_similarity_summary(
    *,
    labels: torch.Tensor,
    window_session_ids: tuple[str, ...],
    window_scores: torch.Tensor,
    missing_class_failure_reason: str | None = None,
    missing_class_error_message: str | None = None,
    include_comparison_available: bool = False,
) -> dict[str, Any]:
    positive_count = int((labels > 0.0).sum().item())
    negative_count = int((labels <= 0.0).sum().item())
    if positive_count <= 0 or negative_count <= 0:
        if missing_class_failure_reason is not None:
            return {
                "window_roc_auc": None,
                "window_pr_auc": None,
                "positive_window_count": positive_count,
                "negative_window_count": negative_count,
                "per_session_roc_auc": {},
                "comparison_available": False,
                "failure_reason": str(missing_class_failure_reason),
            }
        message = missing_class_error_message or (
            "Held-out motif similarity validation requires both positive and negative windows in the held-out subset."
        )
        raise ValueError(message)
    labels_np = labels.detach().cpu().numpy()
    scores_np = window_scores.detach().cpu().numpy()
    per_session_roc_auc: dict[str, float] = {}
    for session_id in sorted(set(window_session_ids)):
        indices = [index for index, value in enumerate(window_session_ids) if value == session_id]
        session_labels = labels_np[indices]
        if len({int(value) for value in session_labels.tolist()}) < 2:
            continue
        per_session_roc_auc[session_id] = float(roc_auc_score(session_labels, scores_np[indices]))
    summary: dict[str, Any] = {
        "window_roc_auc": float(roc_auc_score(labels_np, scores_np)),
        "window_pr_auc": float(average_precision_score(labels_np, scores_np)),
        "positive_window_count": positive_count,
        "negative_window_count": negative_count,
        "per_session_roc_auc": per_session_roc_auc,
    }
    if include_comparison_available:
        summary["comparison_available"] = True
    return summary


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
    candidate_session_balance_fraction: float = 0.2,
) -> tuple[CandidateTokenRecord, ...]:
    if top_k <= 0:
        return ()

    weight = probe_state_dict["linear.weight"].detach().cpu().reshape(-1)
    bias = float(probe_state_dict["linear.bias"].detach().cpu().item())
    counter = 0
    negative_score_sum: dict[tuple[str, int], float] = {}
    negative_score_count: dict[tuple[str, int], int] = {}
    global_negative_score_sum = 0.0
    global_negative_score_count = 0
    session_heaps: dict[str, list[_ScoredHeapEntry]] = {}

    def _push_bounded_heap(heap: list[_ScoredHeapEntry], entry: _ScoredHeapEntry, *, limit: int) -> None:
        if len(heap) < limit:
            heapq.heappush(heap, entry)
            return
        if entry[0] > heap[0][0]:
            heapq.heapreplace(heap, entry)

    def _rank_entries(entries: list[_ScoredHeapEntry]) -> list[_ScoredHeapEntry]:
        return sorted(entries, key=lambda item: (-item[0], item[1]))

    def _select_ranked_rows() -> list[dict[str, Any]]:
        per_session_ranked = {
            session_id: _rank_entries(entries)
            for session_id, entries in session_heaps.items()
            if entries
        }
        if not per_session_ranked:
            return []
        if candidate_session_balance_fraction >= 1.0 or len(per_session_ranked) <= 1:
            ranked_entries = _rank_entries([entry for entries in per_session_ranked.values() for entry in entries])
            return [entry[2] for entry in ranked_entries[:top_k]]

        max_per_session = max(1, int(math.ceil(float(top_k) * float(candidate_session_balance_fraction))))
        selected_entries: list[_ScoredHeapEntry] = []
        next_index_by_session = {session_id: 0 for session_id in per_session_ranked}
        selected_count_by_session = {session_id: 0 for session_id in per_session_ranked}

        while len(selected_entries) < top_k:
            round_entries: list[tuple[float, int, str]] = []
            for session_id, entries in per_session_ranked.items():
                next_index = next_index_by_session[session_id]
                if next_index >= len(entries):
                    continue
                if selected_count_by_session[session_id] >= max_per_session:
                    continue
                score, sort_index, _ = entries[next_index]
                round_entries.append((score, sort_index, session_id))
            if not round_entries:
                break
            round_entries.sort(key=lambda item: (-item[0], item[1], item[2]))
            for _, _, session_id in round_entries:
                if len(selected_entries) >= top_k:
                    break
                next_index = next_index_by_session[session_id]
                entries = per_session_ranked[session_id]
                if next_index >= len(entries):
                    continue
                if selected_count_by_session[session_id] >= max_per_session:
                    continue
                selected_entries.append(entries[next_index])
                next_index_by_session[session_id] += 1
                selected_count_by_session[session_id] += 1

        if len(selected_entries) < top_k:
            leftover_entries: list[_ScoredHeapEntry] = []
            for session_id, entries in per_session_ranked.items():
                next_index = next_index_by_session[session_id]
                leftover_entries.extend(entries[next_index:])
            selected_entries.extend(_rank_entries(leftover_entries)[: max(0, top_k - len(selected_entries))])

        ranked_entries = _rank_entries(selected_entries)[:top_k]
        return [entry[2] for entry in ranked_entries]

    for shard_path in shard_paths:
        payload = torch.load(Path(shard_path), map_location="cpu", weights_only=False)
        embeddings = payload["embeddings"].to(dtype=torch.float32)
        if embeddings.numel() == 0:
            continue
        labels = payload.get("labels")
        if labels is None:
            continue
        scores = (embeddings @ weight) + bias
        negative_indices = torch.nonzero(labels <= 0, as_tuple=False).flatten().tolist()
        for index in negative_indices:
            key = (str(payload["unit_regions"][index]), int(payload["patch_index"][index].item()))
            score = float(scores[index].item())
            negative_score_sum[key] = negative_score_sum.get(key, 0.0) + score
            negative_score_count[key] = negative_score_count.get(key, 0) + 1
            global_negative_score_sum += score
            global_negative_score_count += 1

    global_negative_mean = (
        global_negative_score_sum / float(global_negative_score_count)
        if global_negative_score_count > 0
        else 0.0
    )

    for shard_path in shard_paths:
        payload = torch.load(Path(shard_path), map_location="cpu", weights_only=False)
        embeddings = payload["embeddings"].to(dtype=torch.float32)
        if embeddings.numel() == 0:
            continue
        labels = payload.get("labels")
        if labels is None:
            continue
        scores = (embeddings @ weight) + bias
        keep_indices = torch.nonzero(labels > 0, as_tuple=False).flatten().tolist()
        for index in keep_indices:
            key = (str(payload["unit_regions"][index]), int(payload["patch_index"][index].item()))
            negative_background = (
                negative_score_sum[key] / float(negative_score_count[key])
                if key in negative_score_count and negative_score_count[key] > 0
                else global_negative_mean
            )
            raw_probe_score = float(scores[index].item())
            score = raw_probe_score - float(negative_background)
            if score < float(min_score):
                continue
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
                "raw_probe_score": raw_probe_score,
                "negative_background_score": float(negative_background),
            }
            session_id = str(row["session_id"])
            session_heap = session_heaps.setdefault(session_id, [])
            _push_bounded_heap(session_heap, (score, counter, row), limit=top_k)
            counter += 1

    ranked_rows = _select_ranked_rows()
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
            raw_probe_score=row["raw_probe_score"],
            negative_background_score=row["negative_background_score"],
        )
        for index, row in enumerate(ranked_rows)
    )
