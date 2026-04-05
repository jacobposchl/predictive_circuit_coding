from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import sys


@dataclass(frozen=True)
class DependencyStatus:
    package_name: str
    import_name: str
    available: bool
    required_python: str | None = None
    note: str | None = None


def _is_available(import_name: str) -> bool:
    return importlib.util.find_spec(import_name) is not None


def collect_dependency_status() -> list[DependencyStatus]:
    current = f"{sys.version_info.major}.{sys.version_info.minor}"
    brainsets_note = None
    if sys.version_info >= (3, 12):
        brainsets_note = (
            "brainsets docs currently advertise Python 3.9-3.11 support; "
            f"current interpreter is {current}."
        )
    return [
        DependencyStatus(
            package_name="temporaldata",
            import_name="temporaldata",
            available=_is_available("temporaldata"),
            required_python=">=3.10",
        ),
        DependencyStatus(
            package_name="pytorch_brain",
            import_name="torch_brain",
            available=_is_available("torch_brain"),
            required_python=">=3.10",
        ),
        DependencyStatus(
            package_name="brainsets",
            import_name="brainsets",
            available=_is_available("brainsets"),
            required_python=">=3.9,<3.12",
            note=brainsets_note,
        ),
        DependencyStatus(
            package_name="allensdk",
            import_name="allensdk",
            available=_is_available("allensdk"),
            required_python=">=3.10",
            note="Required in the dedicated prep env for prepare-allen-visual-behavior-neuropixels.",
        ),
    ]


def ensure_optional_dependency(import_name: str, *, package_name: str | None = None) -> None:
    if _is_available(import_name):
        return
    package_name = package_name or import_name
    raise RuntimeError(
        f"Missing optional dependency '{package_name}'. Install the package before using this command."
    )
