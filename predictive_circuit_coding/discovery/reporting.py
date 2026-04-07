from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from predictive_circuit_coding.training.contracts import CandidateTokenRecord, DiscoveryArtifact
from predictive_circuit_coding.training.contracts import write_json_payload


def _top_items(values: list[str], *, limit: int = 3) -> list[dict[str, object]]:
    counter = Counter(value for value in values if value)
    return [
        {"value": key, "count": count}
        for key, count in counter.most_common(limit)
    ]


def build_discovery_cluster_report(artifact: DiscoveryArtifact) -> dict[str, object]:
    grouped: dict[int, list[CandidateTokenRecord]] = defaultdict(list)
    for candidate in artifact.candidates:
        if candidate.cluster_id == -1:
            continue
        grouped[int(candidate.cluster_id)].append(candidate)

    clusters: list[dict[str, object]] = []
    for cluster_id, members in sorted(grouped.items()):
        scores = [float(member.score) for member in members]
        regions = [member.unit_region for member in members]
        sessions = [member.session_id for member in members]
        subjects = [member.subject_id for member in members]
        representative = max(members, key=lambda item: item.score)
        clusters.append(
            {
                "cluster_id": cluster_id,
                "cluster_persistence": artifact.cluster_quality_summary.get("cluster_persistence_by_cluster", {}).get(str(cluster_id), artifact.cluster_quality_summary.get("cluster_persistence_by_cluster", {}).get(cluster_id)),
                "candidate_count": len(members),
                "session_count": len(set(sessions)),
                "subject_count": len(set(subjects)),
                "mean_score": sum(scores) / len(scores),
                "max_score": max(scores),
                "mean_depth_um": sum(float(member.unit_depth_um) for member in members) / len(members),
                "temporal_start_s": min(float(member.patch_start_s) for member in members),
                "temporal_end_s": max(float(member.patch_end_s) for member in members),
                "top_regions": _top_items(regions),
                "top_sessions": _top_items(sessions),
                "top_subjects": _top_items(subjects),
                "representative_candidate_id": representative.candidate_id,
                "representative_recording_id": representative.recording_id,
                "representative_unit_id": representative.unit_id,
                "representative_patch_index": representative.patch_index,
            }
        )

    return {
        "dataset_id": artifact.dataset_id,
        "split_name": artifact.split_name,
        "checkpoint_path": artifact.checkpoint_path,
        "cluster_count": len(clusters),
        "candidate_count": len(artifact.candidates),
        "cluster_quality_summary": artifact.cluster_quality_summary,
        "clusters": clusters,
    }


def write_discovery_cluster_report_json(report: dict[str, object], path: str | Path) -> Path:
    return write_json_payload(report, path)


def write_discovery_cluster_report_csv(report: dict[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cluster_id",
                "candidate_count",
                "cluster_persistence",
                "session_count",
                "subject_count",
                "mean_score",
                "max_score",
                "mean_depth_um",
                "temporal_start_s",
                "temporal_end_s",
                "top_regions",
                "top_sessions",
                "top_subjects",
                "representative_candidate_id",
                "representative_recording_id",
                "representative_unit_id",
                "representative_patch_index",
            ],
        )
        writer.writeheader()
        for cluster in report["clusters"]:
            writer.writerow(
                {
                    "cluster_id": cluster["cluster_id"],
                    "candidate_count": cluster["candidate_count"],
                    "cluster_persistence": cluster["cluster_persistence"],
                    "session_count": cluster["session_count"],
                    "subject_count": cluster["subject_count"],
                    "mean_score": cluster["mean_score"],
                    "max_score": cluster["max_score"],
                    "mean_depth_um": cluster["mean_depth_um"],
                    "temporal_start_s": cluster["temporal_start_s"],
                    "temporal_end_s": cluster["temporal_end_s"],
                    "top_regions": ", ".join(
                        f"{item['value']}:{item['count']}" for item in cluster["top_regions"]
                    ),
                    "top_sessions": ", ".join(
                        f"{item['value']}:{item['count']}" for item in cluster["top_sessions"]
                    ),
                    "top_subjects": ", ".join(
                        f"{item['value']}:{item['count']}" for item in cluster["top_subjects"]
                    ),
                    "representative_candidate_id": cluster["representative_candidate_id"],
                    "representative_recording_id": cluster["representative_recording_id"],
                    "representative_unit_id": cluster["representative_unit_id"],
                    "representative_patch_index": cluster["representative_patch_index"],
                }
            )
    return target
