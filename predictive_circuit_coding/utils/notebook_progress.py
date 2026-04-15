from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, TextIO

from predictive_circuit_coding.utils.console import get_console

_STEP_LOG_PATTERN = re.compile(r"epoch=(?P<epoch>\d+) step=(?P<step>\d+):(?P<metrics>.*)")
_METRIC_CONTINUATION_PATTERN = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_]*=")
_NO_POSITIVE_LABELS_PATTERN = re.compile(r"no positive '.*?' labels")


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


@dataclass(frozen=True)
class NotebookProgressConfig:
    enabled: bool = True
    leave_pipeline_bar: bool = True
    leave_stage_bars: bool = False
    show_stage_summaries: bool = True
    show_artifact_paths: str = "compact"
    metric_snapshot_every_n: int | None = None


@dataclass(frozen=True)
class NotebookProgressEvent:
    scope: str
    event_type: str
    stage_name: str | None = None
    label: str | None = None
    current: int | None = None
    total: int | None = None
    message: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingProgressEvent:
    event_type: str
    epoch: int | None = None
    epoch_total: int | None = None
    step: int | None = None
    step_total: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    message: str | None = None
    checkpoint_path: str | None = None


@dataclass(frozen=True)
class EvaluationProgressEvent:
    event_type: str
    split_name: str
    current_batch: int | None = None
    total_batches: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    window_count: int | None = None
    message: str | None = None


@dataclass(frozen=True)
class BenchmarkProgressEvent:
    benchmark_name: str
    event_type: str
    task_name: str | None = None
    task_index: int | None = None
    task_total: int | None = None
    arm_name: str | None = None
    arm_index: int | None = None
    arm_total: int | None = None
    step_name: str | None = None
    current: int | None = None
    total: int | None = None
    status: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    message: str | None = None


@dataclass(frozen=True)
class NotebookStageSummary:
    stage_name: str
    status: str
    headline: str
    rows: tuple[dict[str, Any], ...] = ()
    notes: tuple[str, ...] = ()
    artifact_paths: dict[str, str] = field(default_factory=dict)
    debug_log_path: str | None = None


class NotebookProgressUI:
    def __init__(
        self,
        *,
        config: NotebookProgressConfig | None = None,
        stream: TextIO | None = None,
    ) -> None:
        self.config = config or NotebookProgressConfig()
        self.stream = stream or sys.stdout
        self._pipeline_bar = None
        self._stage_bar = None
        self._detail_bar = None
        self._stage_name: str | None = None
        self._last_detail_label: str | None = None
        self._last_training_epoch: int | None = None
        self._last_metric_snapshot: dict[str, int] = {}
        self._training_total_steps: int | None = None
        self._training_steps_per_epoch: int | None = None
        self._training_best_metric: float | None = None
        self._force_auto_tqdm = False

    def _tqdm_cls(self):
        if not self._force_auto_tqdm:
            try:
                from tqdm.notebook import IProgress, tqdm as notebook_tqdm

                if IProgress is None:
                    raise ImportError("ipywidgets progress unavailable")

                return notebook_tqdm
            except Exception:
                self._force_auto_tqdm = True
        try:
            from tqdm.auto import tqdm as auto_tqdm

            return auto_tqdm
        except Exception:
            from tqdm.notebook import tqdm as notebook_tqdm

            return notebook_tqdm

    def _display_markdown(self, text: str) -> None:
        try:
            from IPython.display import Markdown, display

            display(Markdown(text))
        except Exception:
            self.stream.write(text + "\n")

    def _display_rows(self, rows: Iterable[dict[str, Any]]) -> None:
        materialized = list(rows)
        if not materialized:
            return
        try:
            import pandas as pd
            from IPython.display import display

            display(pd.DataFrame(materialized))
        except Exception:
            for row in materialized:
                self.stream.write(str(row) + "\n")

    def _active_bar(self):
        for name in ("_detail_bar", "_stage_bar", "_pipeline_bar"):
            bar = getattr(self, name)
            if bar is not None:
                return bar
        return None

    def _write_line(self, text: str) -> None:
        line = str(text).rstrip()
        if not line:
            return
        bar = self._active_bar()
        if bar is not None and hasattr(bar, "write"):
            try:
                bar.write(line)
                return
            except Exception:
                pass
        self.stream.write(line + "\n")
        try:
            self.stream.flush()
        except Exception:
            pass

    def note(self, message: str) -> None:
        if not self.config.enabled:
            return
        self._write_line(message)

    def milestone(self, message: str) -> None:
        self.note(message)

    @staticmethod
    def _stage_display_name(stage_name: str | None, fallback: str | None = None) -> str:
        if stage_name == "train":
            return "training"
        if stage_name == "evaluate":
            return "evaluation"
        if stage_name == "final_reports":
            return "final reports"
        if stage_name == "alignment_diagnostic":
            return "alignment diagnostic"
        if stage_name:
            return stage_name.replace("_", " ")
        return (fallback or "stage").replace("_", " ").lower()

    @staticmethod
    def _format_metric(value: Any) -> str:
        if value is None:
            return "n/a"
        try:
            return f"{float(value):.4g}"
        except (TypeError, ValueError):
            return str(value)

    def _close_bar(self, name: str) -> None:
        bar = getattr(self, name)
        if bar is not None:
            bar.close()
            setattr(self, name, None)

    def _ensure_bar(
        self,
        *,
        current_bar_name: str,
        desc: str,
        total: int | None,
        position: int,
        leave: bool,
    ):
        tqdm_cls = self._tqdm_cls()
        bar = getattr(self, current_bar_name)
        if bar is None:
            try:
                bar = tqdm_cls(total=total, desc=desc, leave=leave, position=position, dynamic_ncols=True)
            except Exception:
                self._force_auto_tqdm = True
                tqdm_cls = self._tqdm_cls()
                bar = tqdm_cls(total=total, desc=desc, leave=leave, position=position, dynamic_ncols=True)
            setattr(self, current_bar_name, bar)
            return bar
        if total is not None and getattr(bar, "total", None) != total:
            bar.total = total
            bar.refresh()
        if desc and getattr(bar, "desc", None) != desc:
            bar.set_description_str(desc)
        return bar

    def start_pipeline(self, *, total_stages: int, completed_stages: int = 0) -> None:
        if not self.config.enabled:
            return
        bar = self._ensure_bar(
            current_bar_name="_pipeline_bar",
            desc="Pipeline",
            total=total_stages,
            position=0,
            leave=self.config.leave_pipeline_bar,
        )
        bar.n = int(completed_stages)
        bar.refresh()

    def start_stage(self, *, stage_name: str, total: int | None = None, description: str | None = None) -> None:
        if not self.config.enabled:
            return
        self._close_bar("_stage_bar")
        self._close_bar("_detail_bar")
        self._stage_name = stage_name
        self._last_detail_label = None
        self.milestone(f"Starting {self._stage_display_name(stage_name, description)}.")
        self._ensure_bar(
            current_bar_name="_stage_bar",
            desc=description or stage_name.replace("_", " ").title(),
            total=total,
            position=1,
            leave=self.config.leave_stage_bars,
        )

    def update_stage(
        self,
        *,
        current: int | None = None,
        total: int | None = None,
        description: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if not self.config.enabled or self._stage_bar is None:
            return
        bar = self._ensure_bar(
            current_bar_name="_stage_bar",
            desc=description or getattr(self._stage_bar, "desc", None) or "",
            total=total,
            position=1,
            leave=self.config.leave_stage_bars,
        )
        if current is not None:
            bar.n = int(current)
        if metrics:
            bar.set_postfix({key: value for key, value in metrics.items() if value is not None}, refresh=False)
        bar.refresh()

    def update_detail(
        self,
        *,
        label: str,
        current: int | None = None,
        total: int | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if not self.config.enabled:
            return
        if self._detail_bar is None or self._last_detail_label != label:
            self._close_bar("_detail_bar")
            self._last_detail_label = label
            self._ensure_bar(
                current_bar_name="_detail_bar",
                desc=label,
                total=total,
                position=2,
                leave=False,
            )
        bar = self._ensure_bar(
            current_bar_name="_detail_bar",
            desc=label,
            total=total,
            position=2,
            leave=False,
        )
        if current is not None:
            bar.n = int(current)
        if metrics:
            bar.set_postfix({key: value for key, value in metrics.items() if value is not None}, refresh=False)
        bar.refresh()

    def clear_detail(self) -> None:
        self._close_bar("_detail_bar")
        self._last_detail_label = None

    def advance_pipeline(self, steps: int = 1) -> None:
        if not self.config.enabled or self._pipeline_bar is None:
            return
        self._pipeline_bar.update(int(steps))
        self._pipeline_bar.refresh()

    def make_copy_callback(self, *, label: str) -> Callable[[int, int], None]:
        def _callback(current: int, total: int) -> None:
            self.update_detail(label=label, current=current, total=total)
            if total and current >= total:
                self.clear_detail()

        return _callback

    def render_artifacts(self, title: str, artifacts: dict[str, str | Path | None]) -> None:
        if not self.config.enabled or self.config.show_artifact_paths == "hidden":
            return
        materialized = {
            key: str(value)
            for key, value in artifacts.items()
            if value not in (None, "")
        }
        if not materialized:
            return
        self.note(title)
        for key, value in materialized.items():
            self.note(f"  {key}: {value}")

    def render_stage_summary(self, summary: NotebookStageSummary) -> None:
        if not self.config.enabled or not self.config.show_stage_summaries:
            return
        stage_name = self._stage_display_name(summary.stage_name)
        if summary.status == "complete":
            self.note(f"Finished {stage_name}.")
        elif summary.status == "reused":
            self.note(f"Reusing {stage_name}.")
        elif summary.status == "skipped":
            self.note(f"Skipping {stage_name}.")
        elif summary.status == "failed":
            self.note(f"Failed {stage_name}.")
        else:
            self.note(f"{stage_name.title()}: {summary.status}.")
        if summary.notes:
            for note in summary.notes:
                self.note(f"  {note}")
        if summary.rows:
            self._display_rows(summary.rows)

    def finish_stage(self, summary: NotebookStageSummary | None = None) -> None:
        if self.config.enabled:
            self._close_bar("_detail_bar")
            self._close_bar("_stage_bar")
        if summary is not None:
            self.render_stage_summary(summary)
        self._stage_name = None

    def fail_stage(
        self,
        *,
        stage_name: str,
        error_message: str,
        debug_log_path: str | None,
        tail_lines: Iterable[str] = (),
    ) -> None:
        self._close_bar("_detail_bar")
        self._close_bar("_stage_bar")
        headline = f"Failed with `{error_message}`."
        notes = tuple(line.rstrip() for line in tail_lines if str(line).strip())
        self.render_stage_summary(
            NotebookStageSummary(
                stage_name=stage_name,
                status="failed",
                headline=headline,
                notes=notes,
                artifact_paths=(
                    {"debug_log": debug_log_path}
                    if debug_log_path is not None
                    else {}
                ),
                debug_log_path=debug_log_path,
            )
        )
        self._stage_name = None

    def finish_pipeline(self) -> None:
        self._close_bar("_detail_bar")
        self._close_bar("_stage_bar")
        if self._pipeline_bar is not None:
            self._pipeline_bar.refresh()

    def _training_current_step(self, event: TrainingProgressEvent) -> int | None:
        if event.epoch is None or event.step is None:
            return None
        step_total = event.step_total or self._training_steps_per_epoch
        if step_total is None:
            return event.step
        return max(0, (int(event.epoch) - 1) * int(step_total) + int(event.step))

    def _training_postfix(self, event: TrainingProgressEvent) -> dict[str, str]:
        postfix: dict[str, str] = {}
        if event.epoch is not None and event.epoch_total is not None:
            postfix["epoch"] = f"{event.epoch}/{event.epoch_total}"
        metric_aliases = (
            ("loss", "total_loss"),
            ("pred_loss", "predictive_loss"),
            ("pred_imp", "predictive_improvement"),
        )
        for label, key in metric_aliases:
            if key in event.metrics:
                postfix[label] = self._format_metric(event.metrics.get(key))
        if self._training_best_metric is not None:
            postfix["best"] = self._format_metric(self._training_best_metric)
        return postfix

    def make_training_callback(self, *, stage_name: str) -> Callable[[TrainingProgressEvent], None]:
        def _callback(event: TrainingProgressEvent) -> None:
            if event.event_type == "setup_start":
                self.milestone(event.message or "Preparing training inputs.")
                return
            if event.event_type == "setup_complete":
                self.milestone(event.message or "Training data and model are ready.")
                return
            if event.event_type == "resume":
                self.milestone(event.message or "Resuming training.")
                return
            if event.event_type == "epoch_start":
                self.clear_detail()
                if event.step_total is not None:
                    self._training_steps_per_epoch = int(event.step_total)
                if event.epoch_total is not None and self._training_steps_per_epoch is not None:
                    self._training_total_steps = int(event.epoch_total) * int(self._training_steps_per_epoch)
                current = 0
                if event.epoch is not None and self._training_steps_per_epoch is not None:
                    current = max(0, (int(event.epoch) - 1) * int(self._training_steps_per_epoch))
                self.update_stage(
                    current=current,
                    total=self._training_total_steps,
                    description="Training",
                    metrics=self._training_postfix(event),
                )
                self._last_training_epoch = event.epoch
                return
            if event.event_type == "step":
                self.update_stage(
                    current=self._training_current_step(event),
                    total=self._training_total_steps,
                    description="Training",
                    metrics=self._training_postfix(event),
                )
                return
            if event.event_type == "validation_start":
                self.clear_detail()
                self.milestone(event.message or "Starting validation.")
                return
            if event.event_type == "validation_end":
                predictive_improvement = event.metrics.get("predictive_improvement")
                if predictive_improvement is not None:
                    value = float(predictive_improvement)
                    if math.isfinite(value) and (
                        self._training_best_metric is None or value > self._training_best_metric
                    ):
                        self._training_best_metric = value
                self.update_stage(
                    current=self._training_current_step(event),
                    total=self._training_total_steps,
                    description="Training",
                    metrics=self._training_postfix(event),
                )
                self.milestone(
                    "Validation complete"
                    if predictive_improvement is None
                    else f"Validation complete: predictive_improvement={self._format_metric(predictive_improvement)}."
                )
                return
            if event.event_type == "epoch_end":
                current = None
                if event.epoch is not None and self._training_steps_per_epoch is not None:
                    current = int(event.epoch) * int(self._training_steps_per_epoch)
                self.update_stage(
                    current=current,
                    total=self._training_total_steps,
                    description="Training",
                    metrics=self._training_postfix(event),
                )
                return
            if event.event_type == "checkpoint_saved":
                label = event.message or "checkpoint"
                if event.checkpoint_path:
                    self.milestone(f"Saved {label}: {event.checkpoint_path}")
                else:
                    self.milestone(f"Saved {label}.")
                return
            if event.event_type == "training_complete":
                self.clear_detail()
                self.update_stage(
                    current=self._training_total_steps,
                    total=self._training_total_steps,
                    description="Training",
                    metrics=self._training_postfix(event),
                )
                if event.checkpoint_path:
                    self.milestone(f"Training complete: best checkpoint at {event.checkpoint_path}")
                else:
                    self.milestone("Training complete.")

        return _callback

    def make_evaluation_callback(self, *, split_total: int) -> Callable[[EvaluationProgressEvent], None]:
        split_progress: dict[str, int] = {}

        def _callback(event: EvaluationProgressEvent) -> None:
            if event.event_type == "split_start":
                current_completed = len(split_progress)
                self.update_stage(current=current_completed, total=split_total, description="Evaluation")
                self.update_detail(
                    label=f"Split {event.split_name}",
                    current=0,
                    total=event.total_batches,
                )
                return
            if event.event_type == "batch":
                metrics = {}
                snapshot_every = self.config.metric_snapshot_every_n
                if snapshot_every is not None and event.current_batch is not None and event.current_batch % snapshot_every == 0:
                    metrics = {
                        key: event.metrics.get(key)
                        for key in ("predictive_improvement", "predictive_loss", "total_loss")
                    }
                self.update_detail(
                    label=f"Split {event.split_name}",
                    current=event.current_batch,
                    total=event.total_batches,
                    metrics=metrics or None,
                )
                return
            if event.event_type == "split_end":
                split_progress[event.split_name] = 1
                self.update_stage(current=len(split_progress), total=split_total, description="Evaluation")
                self.update_detail(
                    label=f"Split {event.split_name}",
                    current=event.current_batch,
                    total=event.total_batches,
                    metrics={
                        "predictive_improvement": event.metrics.get("predictive_improvement"),
                    },
                )

        return _callback

    def make_benchmark_callback(self, *, benchmark_name: str, total_arms: int | None = None) -> Callable[[BenchmarkProgressEvent], None]:
        def _callback(event: BenchmarkProgressEvent) -> None:
            if event.event_type == "task_start":
                self.update_stage(
                    current=event.arm_index or (event.arm_total and 0) or 0,
                    total=event.arm_total or total_arms,
                    description=f"{benchmark_name.title()} Benchmark",
                )
                return
            if event.event_type == "arm_start":
                self.update_stage(
                    current=(event.arm_index - 1) if event.arm_index is not None else None,
                    total=event.arm_total or total_arms,
                    description=f"{benchmark_name.title()} Benchmark",
                )
                self.update_detail(
                    label=f"{event.task_name} / {event.arm_name}",
                    current=0,
                    total=1,
                )
                return
            if event.event_type == "arm_step":
                label = f"{event.task_name} / {event.arm_name}: {event.step_name}"
                self.update_detail(
                    label=label,
                    current=event.current,
                    total=event.total,
                )
                return
            if event.event_type == "arm_end":
                self.update_stage(
                    current=event.arm_index,
                    total=event.arm_total or total_arms,
                    description=f"{benchmark_name.title()} Benchmark",
                    metrics={"status": event.status},
                )
                self.clear_detail()

        return _callback


def _extract_metric(metrics_blob: str, metric_name: str) -> str:
    match = re.search(rf"{metric_name}=(-?\d+(?:\.\d+)?)", metrics_blob)
    return match.group(1) if match else "n/a"


@dataclass
class _PendingStepMetrics:
    epoch: str
    step: int
    metrics_parts: list[str]
    should_emit: bool


class NotebookCommandStreamFormatter:
    def __init__(self, *, step_log_every: int = 16) -> None:
        self.step_log_every = max(1, int(step_log_every))
        self._pending: _PendingStepMetrics | None = None

    def _flush_pending(self) -> list[str]:
        if self._pending is None:
            return []
        pending = self._pending
        self._pending = None
        if not pending.should_emit:
            return []
        metrics_blob = " ".join(part.strip() for part in pending.metrics_parts if part.strip())
        return [
            (
                f"epoch={pending.epoch} step={pending.step}: "
                f"predictive_improvement={_extract_metric(metrics_blob, 'predictive_improvement')}, "
                f"predictive_loss={_extract_metric(metrics_blob, 'predictive_loss')}, "
                f"total_loss={_extract_metric(metrics_blob, 'total_loss')}\n"
            )
        ]

    def feed(self, line: str) -> list[str]:
        outputs: list[str] = []
        step_match = _STEP_LOG_PATTERN.search(line)
        if step_match is not None:
            outputs.extend(self._flush_pending())
            step = int(step_match.group("step"))
            self._pending = _PendingStepMetrics(
                epoch=step_match.group("epoch"),
                step=step,
                metrics_parts=[step_match.group("metrics")],
                should_emit=(step == 1 or step % self.step_log_every == 0),
            )
            return outputs
        if self._pending is not None and _METRIC_CONTINUATION_PATTERN.match(line):
            self._pending.metrics_parts.append(line)
            return outputs
        outputs.extend(self._flush_pending())
        outputs.append(line)
        return outputs

    def finalize(self) -> list[str]:
        return self._flush_pending()


def run_streaming_command(
    command: list[str],
    *,
    cwd: str | Path | None = None,
    step_log_every: int = 16,
    stream: TextIO | None = None,
) -> int:
    output_stream = stream or sys.stdout
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    formatter = NotebookCommandStreamFormatter(step_log_every=step_log_every)
    captured_output: list[str] = []
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert process.stdout is not None
    for line in process.stdout:
        for output_line in formatter.feed(line):
            captured_output.append(output_line)
            output_stream.write(output_line)
    for output_line in formatter.finalize():
        captured_output.append(output_line)
        output_stream.write(output_line)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command, output="".join(captured_output))
    return return_code


def output_indicates_missing_positive_labels(output: str) -> bool:
    return bool(_NO_POSITIVE_LABELS_PATTERN.search(output.lower()))


@dataclass
class NotebookStageReporter:
    name: str
    expected_duration: str | None = None
    console: object = field(default_factory=get_console)
    started_at: float = field(default_factory=time.perf_counter)
    current_stage_started_at: float | None = None

    def banner(self, title: str, *, subtitle: str | None = None) -> None:
        self.console.print(f"[bold]{title}[/bold]")
        if subtitle:
            self.console.print(subtitle)

    def begin(self, stage_name: str, *, next_artifact: str | None = None) -> None:
        self.current_stage_started_at = time.perf_counter()
        message = f"[{self.name}] starting {stage_name}"
        if self.expected_duration:
            message += f" | expected: {self.expected_duration}"
        if next_artifact:
            message += f" | next artifact: {next_artifact}"
        self.console.print(message)

    def finish(self, stage_name: str) -> None:
        total_elapsed = format_duration(time.perf_counter() - self.started_at)
        if self.current_stage_started_at is None:
            stage_elapsed = total_elapsed
        else:
            stage_elapsed = format_duration(time.perf_counter() - self.current_stage_started_at)
        self.console.print(f"[{self.name}] finished {stage_name} in {stage_elapsed} (total {total_elapsed})")
        self.current_stage_started_at = None

    def note_checkpoint(self, checkpoint_path: str | Path) -> None:
        self.console.print(f"[green]checkpoint saved[/green] {Path(checkpoint_path)}")

    def run_command(self, command: list[str], *, cwd: str | Path | None = None) -> subprocess.CompletedProcess[str]:
        self.console.print("$ " + " ".join(command))
        return subprocess.run(command, cwd=str(cwd) if cwd else None, check=True, text=True, capture_output=True)
