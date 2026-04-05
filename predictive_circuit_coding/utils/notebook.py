from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from predictive_circuit_coding.utils.console import get_console


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def verify_paths_exist(paths: dict[str, str | Path]) -> dict[str, bool]:
    return {label: Path(path).exists() for label, path in paths.items()}


@dataclass
class NotebookStageReporter:
    name: str
    expected_duration: str | None = None
    console: object = field(default_factory=get_console)
    started_at: float = field(default_factory=time.perf_counter)

    def banner(self, title: str, *, subtitle: str | None = None) -> None:
        self.console.print(f"[bold]{title}[/bold]")
        if subtitle:
            self.console.print(subtitle)

    def begin(self, stage_name: str, *, next_artifact: str | None = None) -> None:
        message = f"[{self.name}] starting {stage_name}"
        if self.expected_duration:
            message += f" | expected: {self.expected_duration}"
        if next_artifact:
            message += f" | next artifact: {next_artifact}"
        self.console.print(message)

    def finish(self, stage_name: str) -> None:
        elapsed = format_duration(time.perf_counter() - self.started_at)
        self.console.print(f"[{self.name}] finished {stage_name} in {elapsed}")

    def note_checkpoint(self, checkpoint_path: str | Path) -> None:
        self.console.print(f"[green]checkpoint saved[/green] {Path(checkpoint_path)}")

    def run_command(self, command: list[str], *, cwd: str | Path | None = None) -> subprocess.CompletedProcess[str]:
        self.console.print("$ " + " ".join(command))
        return subprocess.run(command, cwd=str(cwd) if cwd else None, check=True, text=True, capture_output=True)
