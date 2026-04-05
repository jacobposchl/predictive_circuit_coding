from __future__ import annotations

from pathlib import Path

from predictive_circuit_coding.cli.preview_notebook import main as preview_main


def test_notebook_ui_preview_runs_and_writes_expected_artifacts(tmp_path: Path, capsys) -> None:
    output_root = tmp_path / "preview"
    preview_main(["--output-root", str(output_root)])

    artifact_root = output_root / "artifacts" / "checkpoints"
    assert (artifact_root / "pcc_preview_best.pt").is_file()
    assert (artifact_root / "pcc_preview_best_test_evaluation.json").is_file()
    assert (artifact_root / "pcc_preview_best_discovery_discovery.json").is_file()
    assert (artifact_root / "pcc_preview_best_discovery_discovery_cluster_summary.json").is_file()
    assert (artifact_root / "pcc_preview_best_discovery_discovery_cluster_summary.csv").is_file()
    assert (artifact_root / "pcc_preview_best_discovery_discovery_validation.json").is_file()
    assert (artifact_root / "pcc_preview_best_discovery_discovery_validation.csv").is_file()

    captured = capsys.readouterr()
    assert "Predictive Circuit Coding Notebook Preview" in captured.out
    assert "Evaluation preview" in captured.out
    assert "Discovery preview" in captured.out
    assert "Validation preview" in captured.out
