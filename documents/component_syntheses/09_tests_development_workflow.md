# Tests And Development Workflow

The test suite now protects the live workflow: data prep, tokenization, training config loading, model/objective behavior, discovery, validation, and refinement-specific configs.

Removed experiment paths no longer carry dedicated tests. New coverage lives in `tests/test_refinement_core.py` and the existing workflow tests.
