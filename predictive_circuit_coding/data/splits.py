from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
from pathlib import Path

from predictive_circuit_coding.data.config import SplitPlanningConfig
from predictive_circuit_coding.data.manifest import SessionManifest

SPLIT_NAMES = ("train", "valid", "discovery", "test")


@dataclass(frozen=True)
class SplitAssignment:
    recording_id: str
    split: str
    group_id: str


@dataclass(frozen=True)
class SplitManifest:
    dataset_id: str
    seed: int
    primary_axis: str
    assignments: tuple[SplitAssignment, ...]


def build_split_manifest(
    manifest: SessionManifest,
    *,
    config: SplitPlanningConfig,
) -> SplitManifest:
    group_to_records: dict[str, list[str]] = {}
    for record in manifest.records:
        group_id = record.subject_id if config.primary_axis == "subject" else record.session_id
        group_to_records.setdefault(group_id, []).append(record.recording_id)

    groups = list(group_to_records)
    rng = random.Random(config.seed)
    rng.shuffle(groups)

    total = len(groups)
    split_targets = {
        "train": max(1, round(total * config.train_fraction)) if total else 0,
        "valid": max(1, round(total * config.valid_fraction)) if total >= 3 else 0,
        "discovery": max(1, round(total * config.discovery_fraction)) if total >= 4 else 0,
        "test": max(1, round(total * config.test_fraction)) if total >= 2 else 0,
    }
    consumed = sum(split_targets.values())
    while consumed > total and total > 0:
        for name in ("train", "valid", "discovery", "test"):
            if consumed <= total:
                break
            if split_targets[name] > (1 if name == "train" else 0):
                split_targets[name] -= 1
                consumed -= 1
    if consumed < total:
        split_targets["train"] += total - consumed

    ordered_splits = ("train", "valid", "discovery", "test")
    assignments: list[SplitAssignment] = []
    cursor = 0
    for split_name in ordered_splits:
        width = split_targets[split_name]
        for group_id in groups[cursor : cursor + width]:
            for recording_id in sorted(group_to_records[group_id]):
                assignments.append(
                    SplitAssignment(
                        recording_id=recording_id,
                        split=split_name,
                        group_id=group_id,
                    )
                )
        cursor += width
    return SplitManifest(
        dataset_id=manifest.dataset_id,
        seed=config.seed,
        primary_axis=config.primary_axis,
        assignments=tuple(sorted(assignments, key=lambda item: item.recording_id)),
    )


def write_split_manifest(manifest: SplitManifest, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_id": manifest.dataset_id,
        "seed": manifest.seed,
        "primary_axis": manifest.primary_axis,
        "assignments": [asdict(item) for item in manifest.assignments],
    }
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return target


def load_split_manifest(path: str | Path) -> SplitManifest:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return SplitManifest(
        dataset_id=str(payload["dataset_id"]),
        seed=int(payload["seed"]),
        primary_axis=str(payload["primary_axis"]),
        assignments=tuple(
            SplitAssignment(
                recording_id=str(item["recording_id"]),
                split=str(item["split"]),
                group_id=str(item["group_id"]),
            )
            for item in payload["assignments"]
        ),
    )
