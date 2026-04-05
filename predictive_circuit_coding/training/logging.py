from __future__ import annotations

import time

from predictive_circuit_coding.utils import get_console


class StageLogger:
    def __init__(self, *, name: str):
        self.console = get_console()
        self.name = name
        self.start_time = time.perf_counter()

    def log(self, message: str) -> None:
        elapsed = time.perf_counter() - self.start_time
        self.console.print(f"[{self.name}] +{elapsed:0.1f}s {message}")

    def log_stage(self, stage_name: str, *, expected_next: str | None = None) -> None:
        message = f"Stage: {stage_name}"
        if expected_next:
            message += f" | next artifact: {expected_next}"
        self.log(message)

    def log_artifact(self, *, label: str, path) -> None:
        self.log(f"{label}: {path}")

    def log_metrics(self, *, prefix: str, metrics: dict[str, float]) -> None:
        parts = [f"{key}={value:0.4f}" for key, value in sorted(metrics.items())]
        self.log(f"{prefix}: " + ", ".join(parts))
