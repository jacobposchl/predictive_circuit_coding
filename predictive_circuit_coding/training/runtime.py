from __future__ import annotations

from collections.abc import Iterator

import torch


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def iter_sampler_batches(
    *,
    dataset,
    sampler,
    collator,
    batch_size: int,
    max_batches: int | None,
) -> Iterator:
    pending = []
    yielded = 0
    for item in sampler:
        pending.append(dataset.get(item.recording_id, item.start, item.end))
        if len(pending) == batch_size:
            yield collator(pending)
            yielded += 1
            pending = []
            if max_batches is not None and yielded >= max_batches:
                return
    if pending and (max_batches is None or yielded < max_batches):
        yield collator(pending)
