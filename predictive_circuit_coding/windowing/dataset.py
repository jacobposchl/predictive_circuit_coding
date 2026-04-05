from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from predictive_circuit_coding.data.layout import PreparationWorkspace
from predictive_circuit_coding.data.splits import SplitManifest
from predictive_circuit_coding.utils.dependencies import ensure_optional_dependency


@dataclass(frozen=True)
class FixedWindowConfig:
    window_length_s: float
    step_s: float | None = None
    drop_short: bool = False
    seed: int = 0


@dataclass(frozen=True)
class WindowDescriptor:
    recording_id: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class TorchBrainDatasetBundle:
    root: Path
    config_path: Path
    split: str
    dataset: object


def _ensure_torch_brain() -> None:
    ensure_optional_dependency("torch_brain", package_name="pytorch_brain")


def split_session_ids(split_manifest: SplitManifest, split: str) -> list[str]:
    return sorted(
        assignment.recording_id.split("/", 1)[1]
        for assignment in split_manifest.assignments
        if assignment.split == split
    )


def build_torch_brain_config(
    *,
    workspace: PreparationWorkspace,
    dataset_id: str,
    session_ids: Iterable[str],
    split: str,
) -> Path:
    session_ids = sorted(dict.fromkeys(str(session_id) for session_id in session_ids))
    payload = [
        {
            "selection": [
                {
                    "brainset": str(dataset_id),
                    "sessions": session_ids,
                }
            ]
        }
    ]
    target = workspace.splits / f"torch_brain_{split}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return target


def load_torch_brain_dataset(
    *,
    root: str | Path,
    config_path: str | Path,
    split: str,
    transform=None,
):
    _ensure_torch_brain()
    from torch_brain.data import Dataset

    return Dataset(
        str(root),
        config=str(config_path),
        split=split,
        transform=transform,
    )


def build_dataset_bundle(
    *,
    workspace: PreparationWorkspace,
    split_manifest: SplitManifest,
    split: str,
    transform=None,
) -> TorchBrainDatasetBundle:
    session_ids = split_session_ids(split_manifest, split)
    config_path = build_torch_brain_config(
        workspace=workspace,
        dataset_id=split_manifest.dataset_id,
        session_ids=session_ids,
        split=split,
    )
    dataset = load_torch_brain_dataset(
        root=workspace.prepared,
        config_path=config_path,
        split=split,
        transform=transform,
    )
    return TorchBrainDatasetBundle(
        root=workspace.prepared,
        config_path=config_path,
        split=split,
        dataset=dataset,
    )


def build_random_fixed_window_sampler(dataset, *, window: FixedWindowConfig):
    _ensure_torch_brain()
    import torch
    from torch_brain.data.sampler import RandomFixedWindowSampler

    generator = torch.Generator().manual_seed(window.seed)
    return RandomFixedWindowSampler(
        sampling_intervals=dataset.get_sampling_intervals(),
        window_length=float(window.window_length_s),
        generator=generator,
        drop_short=bool(window.drop_short),
    )


def build_sequential_fixed_window_sampler(dataset, *, window: FixedWindowConfig):
    _ensure_torch_brain()
    from torch_brain.data.sampler import SequentialFixedWindowSampler

    return SequentialFixedWindowSampler(
        sampling_intervals=dataset.get_sampling_intervals(),
        window_length=float(window.window_length_s),
        step=None if window.step_s is None else float(window.step_s),
        drop_short=bool(window.drop_short),
    )


def describe_sampler_windows(sampler, *, limit: int = 5) -> list[WindowDescriptor]:
    rows: list[WindowDescriptor] = []
    for index, item in enumerate(sampler):
        rows.append(
            WindowDescriptor(
                recording_id=str(item.recording_id),
                start_s=float(item.start),
                end_s=float(item.end),
            )
        )
        if index + 1 >= limit:
            break
    return rows
