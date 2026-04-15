from predictive_circuit_coding.validation.artifact_checks import (
    load_discovery_artifact,
    validate_discovery_artifact_identity,
)
from predictive_circuit_coding.validation.notebook import (
    NotebookValidationRunResult,
    default_validation_output_paths,
    flatten_comparison_validation_summary,
    run_notebook_validation,
)


def validate_discovery_artifact(*args, **kwargs):
    from predictive_circuit_coding.validation.run import validate_discovery_artifact as _validate_discovery_artifact

    return _validate_discovery_artifact(*args, **kwargs)


__all__ = [
    "NotebookValidationRunResult",
    "default_validation_output_paths",
    "flatten_comparison_validation_summary",
    "load_discovery_artifact",
    "run_notebook_validation",
    "validate_discovery_artifact_identity",
    "validate_discovery_artifact",
]
