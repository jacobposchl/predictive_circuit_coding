from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


def write_summary_rows(
    rows: list[dict[str, Any]],
    *,
    output_json_path: str | Path,
    output_csv_path: str | Path,
    root_key: str,
) -> tuple[Path, Path]:
    json_path = Path(output_json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({root_key: rows}, indent=2), encoding="utf-8")
    csv_path = Path(output_csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return json_path, csv_path


def write_single_row_summary(
    row: dict[str, Any],
    *,
    output_json_path: str | Path,
    output_csv_path: str | Path,
    root_key: str = "summary",
) -> tuple[Path, Path]:
    return write_summary_rows(
        [row],
        output_json_path=output_json_path,
        output_csv_path=output_csv_path,
        root_key=root_key,
    )


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None and not math.isnan(float(row[key]))]
    if not values:
        return None
    return float(sum(values) / float(len(values)))


def build_final_project_summary(
    *,
    representation_rows: list[dict[str, Any]],
    motif_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    representation_ok = [row for row in representation_rows if row.get("status") == "ok"]
    motif_ok = [row for row in motif_rows if row.get("status") == "ok"]

    def _best_row(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
        scored_rows = [
            row
            for row in rows
            if row.get(key) is not None and not math.isnan(float(row[key]))
        ]
        if not scored_rows:
            return None
        return max(scored_rows, key=lambda row: float(row[key]))

    best_representation = _best_row(representation_ok, "test_probe_pr_auc")
    best_motif = _best_row(motif_ok, "held_out_similarity_pr_auc")

    claims: list[str] = []
    if best_representation is not None:
        claims.append(
            "representation: best held-out cross-session probe PR-AUC came from "
            f"{best_representation['arm_name']} on {best_representation['task_name']} "
            f"({best_representation.get('training_variant_name', 'baseline')})"
        )
    if best_motif is not None:
        claims.append(
            "motifs: best held-out motif PR-AUC came from "
            f"{best_motif['arm_name']} on {best_motif['task_name']} "
            f"({best_motif.get('training_variant_name', 'baseline')})"
        )
    if representation_ok:
        claims.append(
            "representation learning: compare encoder_raw against untrained_encoder_raw to isolate training from architecture/tokenization"
        )
        claims.append(
            "geometry: compare encoder_raw against encoder_whitened to isolate the effect of whitening on the trained representation"
        )
    if motif_ok:
        claims.append(
            "motif specificity: compare trained encoder motif rows against untrained_encoder_raw to test whether motifs depend on learned weights"
        )

    return {
        "representation_row_count": len(representation_rows),
        "motif_row_count": len(motif_rows),
        "representation_completed_row_count": len(representation_ok),
        "motif_completed_row_count": len(motif_ok),
        "representation_mean_test_probe_pr_auc": _mean_metric(representation_ok, "test_probe_pr_auc"),
        "motif_mean_held_out_similarity_pr_auc": _mean_metric(motif_ok, "held_out_similarity_pr_auc"),
        "training_variant_names": sorted(
            {
                str(row.get("training_variant_name", "baseline"))
                for row in (representation_rows + motif_rows)
                if row.get("training_variant_name") is not None
            }
        ),
        "best_representation_row": best_representation,
        "best_motif_row": best_motif,
        "claims": claims,
    }
